"""
LambdaMART model implementation using XGBoost.
Suporta IPS (Inverse Propensity Scoring) e configurações avançadas de LambdaRank.
"""

import numpy as np
import xgboost as xgb
from typing import Dict, Optional, Any, Tuple
import pickle
import os
from .model_factory import BaseLambdaMART


class XGBoostLambdaMART(BaseLambdaMART):
    """
    LambdaMART model implementation using XGBoost ranking objective.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize XGBoost LambdaMART model.
        
        Args:
            params: Dictionary of XGBoost parameters
        """
        super().__init__(params)
        self.default_params = {
            'objective': 'rank:ndcg',
            'eval_metric': 'ndcg',
            'booster': 'gbtree',
            'max_depth': 6,
            'eta': 0.1,  # learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'min_child_weight': 1,
            'silent': 1,
            'nthread': -1,
            'seed': 42,
            'alpha': 0.0,  # L1 regularization
            'lambda': 0.0,  # L2 regularization
        }
        
        if params:
            # Map common parameters to XGBoost equivalents
            mapped_params = self._map_parameters(params)
            self.default_params.update(mapped_params)
            
        self.params = self.default_params
        self.model = None
        self.feature_importance = None
        self.training_history = None
        
    def _map_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map common parameter names to XGBoost equivalents.
        Includes support for LambdaRank and IPS parameters.
        
        Args:
            params: Dictionary of parameters
            
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
                    else:
                        mapped[xgb_key] = value
            elif key in lambdarank_params:
                # Pass LambdaRank/IPS parameters directly to XGBoost
                mapped[key] = value
            else:
                mapped[key] = value
        
        # Ensure correct objective for ranking
        if 'objective' in mapped:
            if mapped['objective'] == 'lambdarank':
                mapped['objective'] = 'rank:ndcg'
        
        return mapped
    
    def prepare_training_data(self, features: np.ndarray, labels: np.ndarray, 
                            query_ids: np.ndarray) -> xgb.DMatrix:
        """
        Prepare XGBoost DMatrix for ranking.
        
        Args:
            features: Feature matrix
            labels: Relevance labels
            query_ids: Query IDs
            
        Returns:
            XGBoost DMatrix object
        """
        # Get group sizes (number of documents per query)
        unique_qids, indices = np.unique(query_ids, return_inverse=True)
        group_sizes = np.bincount(indices)
        
        # Create XGBoost DMatrix
        dtrain = xgb.DMatrix(features, label=labels)
        dtrain.set_group(group_sizes)
        
        return dtrain
    
    def train(self, train_features: np.ndarray, train_labels: np.ndarray,
              train_query_ids: np.ndarray, valid_features: Optional[np.ndarray] = None,
              valid_labels: Optional[np.ndarray] = None, 
              valid_query_ids: Optional[np.ndarray] = None,
              num_boost_round: int = 100, early_stopping_rounds: Optional[int] = 10) -> Dict:
        """
        Train the LambdaMART model.
        
        Args:
            train_features: Training features
            train_labels: Training labels
            train_query_ids: Training query IDs
            valid_features: Validation features (optional)
            valid_labels: Validation labels (optional)
            valid_query_ids: Validation query IDs (optional)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Training history and metrics
        """
        print("Preparing training data...")
        dtrain = self.prepare_training_data(train_features, train_labels, train_query_ids)
        
        # Prepare evaluation sets
        evallist = [(dtrain, 'train')]
        
        # Prepare validation data if provided
        if (valid_features is not None and valid_labels is not None 
            and valid_query_ids is not None):
            print("Preparing validation data...")
            dvalid = self.prepare_training_data(valid_features, valid_labels, valid_query_ids)
            evallist.append((dvalid, 'valid'))
        
        print("Starting model training...")
        
        # Training parameters
        evals_result = {}
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evallist,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=True
        )
        
        # Store training history
        self.training_history = evals_result
        
        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Convert feature importance to array format
        feature_importance_array = np.zeros(train_features.shape[1])
        for feat_name, importance in self.feature_importance.items():
            feat_idx = int(feat_name.replace('f', ''))
            feature_importance_array[feat_idx] = importance
        
        self.feature_importance = feature_importance_array
        
        # Get best iteration (only available with early stopping)
        try:
            best_iter = self.model.best_iteration
            print(f"Training completed. Best iteration: {best_iter}")
        except AttributeError:
            best_iter = num_boost_round
            print(f"Training completed. Used all {num_boost_round} iterations (no early stopping)")
        
        return {
            'best_iteration': best_iter,
            'training_history': evals_result,
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
            raise ValueError("Model not trained yet. Call train() first.")
            
        dtest = xgb.DMatrix(features)
        
        # Use best_iteration if available, otherwise use all iterations
        try:
            return self.model.predict(dtest, iteration_range=(0, self.model.best_iteration))
        except AttributeError:
            # No early stopping was used, predict with all iterations
            return self.model.predict(dtest)
    
    def predict_with_query_groups(self, features: np.ndarray, 
                                query_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Predict scores grouped by query ID.
        
        Args:
            features: Feature matrix
            query_ids: Query IDs
            
        Returns:
            Dictionary mapping query_id to predicted scores
        """
        scores = self.predict(features)
        
        # Group predictions by query ID
        predictions_by_query = {}
        for i, qid in enumerate(query_ids):
            if qid not in predictions_by_query:
                predictions_by_query[qid] = []
            predictions_by_query[qid].append(scores[i])
        
        # Convert lists to numpy arrays
        for qid in predictions_by_query:
            predictions_by_query[qid] = np.array(predictions_by_query[qid])
            
        return predictions_by_query
    
    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Feature importance array
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save XGBoost model
        model_path = filepath.replace('.txt', '.json')
        self.model.save_model(model_path)
        
        # Save additional information
        model_info = {
            'params': self.params,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history
        }
        
        info_filepath = filepath.replace('.txt', '_info.pkl')
        with open(info_filepath, 'wb') as f:
            pickle.dump(model_info, f)
            
        print(f"Model saved to {model_path}")
        print(f"Model info saved to {info_filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model from file.
        
        Args:
            filepath: Path to the saved model
        """
        model_path = filepath.replace('.txt', '.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load additional information
        info_filepath = filepath.replace('.txt', '_info.pkl')
        if os.path.exists(info_filepath):
            with open(info_filepath, 'rb') as f:
                model_info = pickle.load(f)
                self.params = model_info['params']
                self.feature_importance = model_info['feature_importance']
                self.training_history = model_info['training_history']
        
        print(f"Model loaded from {model_path}")
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of the trained model.
        
        Returns:
            Dictionary with model summary information
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Get number of features from feature importance
        num_features = len(self.feature_importance) if self.feature_importance is not None else 0
        
        # Get best iteration safely
        try:
            best_iter = self.model.best_iteration
        except AttributeError:
            # No early stopping, use total number of trees
            best_iter = self.model.num_boosted_rounds()
        
        return {
            'num_trees': best_iter,
            'num_features': num_features,
            'best_iteration': best_iter,
            'params': self.params,
            'feature_importance_top10': dict(enumerate(
                sorted(self.feature_importance, reverse=True)[:10]
            )) if self.feature_importance is not None else {}
        }


class ModelValidator:
    """
    Utility class for model validation and hyperparameter tuning.
    """
    
    @staticmethod
    def cross_validate_queries(features: np.ndarray, labels: np.ndarray,
                              query_ids: np.ndarray, params: Dict,
                              n_folds: int = 5) -> Dict:
        """
        Perform cross-validation at the query level.
        
        Args:
            features: Feature matrix
            labels: Labels
            query_ids: Query IDs
            params: Model parameters
            n_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        unique_queries = np.unique(query_ids)
        np.random.shuffle(unique_queries)
        
        fold_size = len(unique_queries) // n_folds
        cv_results = []
        
        for fold in range(n_folds):
            # Split queries into train/validation
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_folds - 1 else len(unique_queries)
            
            val_queries = unique_queries[start_idx:end_idx]
            train_queries = np.setdiff1d(unique_queries, val_queries)
            
            # Create train/validation splits
            train_mask = np.isin(query_ids, train_queries)
            val_mask = np.isin(query_ids, val_queries)
            
            # Train model on fold
            model = XGBoostLambdaMART(params)
            model.train(
                features[train_mask], labels[train_mask], query_ids[train_mask],
                features[val_mask], labels[val_mask], query_ids[val_mask]
            )
            
            # Get validation predictions
            val_predictions = model.predict(features[val_mask])
            
            cv_results.append({
                'fold': fold,
                'val_queries': val_queries,
                'predictions': val_predictions,
                'true_labels': labels[val_mask],
                'query_ids': query_ids[val_mask]
            })
        
        return cv_results

    def get_ips_info(self) -> Dict[str, Any]:
        """
        Get information about IPS (Inverse Propensity Scoring) configuration.
        
        Returns:
            Dictionary with IPS configuration details
        """
        ips_params = {}
        lambdarank_params = [
            'lambdarank_pair_method',
            'lambdarank_num_pair_per_sample', 
            'lambdarank_normalization',
            'lambdarank_score_normalization',
            'lambdarank_unbiased',
            'lambdarank_bias_norm',
            'ndcg_exp_gain'
        ]
        
        for param in lambdarank_params:
            if param in self.params:
                ips_params[param] = self.params[param]
        
        # Check if IPS is enabled
        ips_enabled = self.params.get('lambdarank_unbiased', False)
        
        return {
            'ips_enabled': ips_enabled,
            'lambdarank_params': ips_params,
            'description': self._get_ips_description(ips_enabled, ips_params)
        }
    
    def _get_ips_description(self, ips_enabled: bool, ips_params: Dict) -> str:
        """Generate description of IPS configuration."""
        if not ips_enabled:
            return "IPS (Inverse Propensity Scoring) está DESABILITADO. O modelo não aplica debiasing."
        
        description = [
            "IPS (Inverse Propensity Scoring) está HABILITADO.",
            "O modelo aplicará debiasing automático para reduzir viés de posição em dados de clique.",
            "",
            "Configurações ativas:"
        ]
        
        if 'lambdarank_bias_norm' in ips_params:
            description.append(f"- Norma de bias: {ips_params['lambdarank_bias_norm']}")
            
        if 'lambdarank_pair_method' in ips_params:
            method = ips_params['lambdarank_pair_method']
            num_pairs = ips_params.get('lambdarank_num_pair_per_sample', 1)
            if method == 'topk':
                description.append(f"- Método de pares: {method} (foco nos top-{num_pairs} documentos)")
            else:
                description.append(f"- Método de pares: {method} ({num_pairs} pares por documento)")
        
        description.extend([
            "",
            "Benefícios esperados:",
            "- Redução do viés de posição em dados de clique",
            "- Melhor generalização para consultas não vistas",
            "- Ranking mais justo e imparcial"
        ])
        
        return "\n".join(description)
    
    def print_ips_info(self):
        """Print IPS configuration information."""
        info = self.get_ips_info()
        print("="*60)
        print("CONFIGURAÇÃO IPS (INVERSE PROPENSITY SCORING)")
        print("="*60)
        print(info['description'])
        print("="*60)


if __name__ == "__main__":
    # Example usage
    print("LambdaMART Model (XGBoost) module loaded successfully!")
    
    # Create a simple test
    np.random.seed(42)
    n_samples, n_features = 100, 10
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 5, n_samples)
    query_ids = np.repeat(range(10), 10)  # 10 queries, 10 docs each
    
    # Initialize and train model
    model = XGBoostLambdaMART({})
    training_results = model.train(features, labels, query_ids)
    
    # Make predictions
    predictions = model.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    
    print("Model training test completed successfully!")