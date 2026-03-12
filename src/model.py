"""
LambdaMART implementation using XGBoost.
Supports standard LambdaMART and IPS (Inverse Propensity Scoring) for debiasing.
"""

import numpy as np
import xgboost as xgb
import pickle
import os
from typing import Dict, Tuple, Any, Optional, List
import warnings


class LambdaMARTModel:
    """
    LambdaMART model using XGBoost with support for IPS (Inverse Propensity Scoring).
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize LambdaMART model.
        
        Args:
            params: Model parameters
        """
        self.params = self._map_parameters(params) if params else self._get_default_params()
        self.model = None
        self.feature_importance = None
        self.training_history = None
        
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for XGBoost LambdaMART.
        
        Returns:
            Default parameters
        """
        return {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg',
            'booster': 'gbtree',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'alpha': 0.0,
            'lambda': 0.0,
            'nthread': -1,
            'verbosity': 1,
            'seed': 42
        }
    
    def _map_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map configuration parameters to XGBoost parameters.
        
        Args:
            params: Configuration parameters
            
        Returns:
            Mapped parameters for XGBoost
        """
        mapped = {}
        
        # Parameter mapping
        param_mapping = {
            'learning_rate': 'eta',
            'num_leaves': None,  # XGBoost uses max_depth instead
            'feature_fraction': 'colsample_bytree',
            'bagging_fraction': 'subsample',
            'lambda_l1': 'alpha',
            'lambda_l2': 'lambda',
            'num_threads': 'nthread',
            'verbose': 'verbosity',
            'silent': 'verbosity',  # silent is deprecated, map to verbosity
            'min_data_in_leaf': 'min_child_weight'
        }
        
        # LambdaRank and IPS specific parameters (pass through directly)
        lambdarank_params = {
            'lambdarank_pair_method',
            'lambdarank_num_pair_per_sample', 
            'lambdarank_normalization',
            'lambdarank_score_normalization',
            'lambdarank_unbiased',      # IPS parameter
            'lambdarank_bias_norm',     # IPS parameter
            'ndcg_exp_gain'
        }
        
        for key, value in params.items():
            if key in param_mapping:
                xgb_key = param_mapping[key]
                if xgb_key is not None:
                    # Special handling for silent -> verbosity conversion
                    if key == 'silent':
                        # silent=1 means quiet, so verbosity=0
                        # silent=0 means verbose, so verbosity=2
                        mapped['verbosity'] = 0 if value == 1 else 2
                    elif key == 'verbosity':
                        # Handle verbosity mapping from LightGBM to XGBoost
                        # LightGBM: -1 (silent), 0 (warning), 1 (info), 2+ (debug)
                        # XGBoost: 0 (silent), 1 (warning), 2 (info), 3 (debug)
                        if value == -1:
                            mapped['verbosity'] = 0  # Silent
                        elif value == 0:
                            mapped['verbosity'] = 1  # Warning
                        elif value == 1:
                            mapped['verbosity'] = 2  # Info
                        else:
                            mapped['verbosity'] = 3  # Debug
                    else:
                        mapped[xgb_key] = value
            elif key in lambdarank_params:
                # Pass LambdaRank/IPS parameters directly to XGBoost
                mapped[key] = value
            elif key == 'objective':
                # Handle objective mapping from LightGBM to XGBoost
                if value == 'lambdarank':
                    mapped['objective'] = 'rank:ndcg'
                else:
                    mapped[key] = value
            else:
                mapped[key] = value
        
        # Ensure correct objective for ranking
        if 'objective' in mapped:
            if mapped['objective'] not in ['rank:ndcg', 'rank:map', 'rank:pairwise']:
                warnings.warn(f"Objective {mapped['objective']} may not be suitable for ranking")
        
        return mapped

    def prepare_training_data(self, features: np.ndarray, labels: np.ndarray, 
                            query_ids: np.ndarray) -> Tuple[xgb.DMatrix, np.ndarray]:
        """
        Prepare training data for XGBoost.
        
        Args:
            features: Feature matrix
            labels: Labels (relevance scores)
            query_ids: Query IDs
            
        Returns:
            Tuple of (DMatrix, group sizes)
        """
        # Create DMatrix
        dtrain = xgb.DMatrix(features, label=labels)
        
        # Calculate group sizes for each query
        unique_queries = np.unique(query_ids)
        group_sizes = []
        for qid in unique_queries:
            group_size = np.sum(query_ids == qid)
            group_sizes.append(group_size)
        
        dtrain.set_group(group_sizes)
        return dtrain, np.array(group_sizes)

    def train(self, train_features: np.ndarray, train_labels: np.ndarray,
              train_qids: np.ndarray, val_features: Optional[np.ndarray] = None,
              val_labels: Optional[np.ndarray] = None, val_qids: Optional[np.ndarray] = None,
              num_boost_round: int = 100, early_stopping_rounds: Optional[int] = None,
              verbose_eval: bool = True) -> Dict:
        """
        Train the LambdaMART model.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            train_qids: Training query IDs
            val_features: Validation features (optional)
            val_labels: Validation labels (optional)
            val_qids: Validation query IDs (optional)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds (optional)
            verbose_eval: Whether to print evaluation results
            
        Returns:
            Training results dictionary
        """
        print("Preparing training data...")
        dtrain, train_group_sizes = self.prepare_training_data(
            train_features, train_labels, train_qids
        )
        
        # Prepare validation data if provided
        evallist = [(dtrain, 'train')]
        if val_features is not None and val_labels is not None and val_qids is not None:
            print("Preparing validation data...")
            dval, val_group_sizes = self.prepare_training_data(
                val_features, val_labels, val_qids
            )
            evallist.append((dval, 'eval'))
        
        # Set up callbacks
        callbacks = []
        if early_stopping_rounds is not None and len(evallist) > 1:
            callbacks.append(xgb.callback.EarlyStopping(rounds=early_stopping_rounds))
        
        print("Starting model training...")
        
        # Train model
        evals_result = {}
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evallist,
            evals_result=evals_result,
            callbacks=callbacks,
            verbose_eval=verbose_eval
        )
        
        # Store training results
        self.training_history = evals_result
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Determine best iteration
        best_iteration = getattr(self.model, 'best_iteration', num_boost_round - 1)
        if hasattr(self.model, 'best_iteration'):
            print(f"Training completed. Best iteration: {self.model.best_iteration}")
        else:
            if early_stopping_rounds is None:
                print(f"Training completed. Used all {num_boost_round} iterations (no early stopping)")
            else:
                print(f"Training completed. Best iteration: {best_iteration}")
        
        return {
            'best_iteration': best_iteration,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict scores for given features.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dtest = xgb.DMatrix(features)
        predictions = self.model.predict(dtest)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """
        Get model summary information.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "Model not trained"}
        
        summary = {
            "model_type": "XGBoost LambdaMART",
            "parameters": self.params,
            "feature_importance": self.feature_importance,
            "training_history": self.training_history
        }
        
        if hasattr(self.model, 'best_iteration'):
            summary["best_iteration"] = self.model.best_iteration
            summary["num_trees"] = self.model.best_iteration + 1
        else:
            summary["num_trees"] = self.model.num_boosted_rounds()
        
        return summary
