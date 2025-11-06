"""
LambdaMART model implementation using LightGBM.
Implementação baseline sem IPS para comparação.
"""

import numpy as np
import lightgbm as lgb
from typing import Dict, Optional, Any, Tuple
import pickle
import os
from .model_factory import BaseLambdaMART


class LightGBMLambdaMART(BaseLambdaMART):
    """
    LambdaMART model implementation using LightGBM ranking objective.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize LightGBM LambdaMART model.
        
        Args:
            params: Dictionary of LightGBM parameters
        """
        super().__init__(params)
        
        self.default_params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.0,
            'reg_lambda': 0.1,
            'num_threads': -1,
            'verbosity': -1,
            'seed': 42
        }
        
        # Merge with provided parameters
        if params:
            self.params = {**self.default_params, **self._map_parameters(params)}
        else:
            self.params = self.default_params.copy()
            
        self.model = None
        self.feature_importance = None
        self.training_history = None
        
    def _map_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map generic parameters to LightGBM specific parameters.
        
        Args:
            params: Input parameters dictionary
            
        Returns:
            Mapped parameters for LightGBM
        """
        mapped = {}
        
        # Parameter mapping from generic to LightGBM
        param_mapping = {
            'eta': 'learning_rate',
            'colsample_bytree': 'feature_fraction',
            'subsample': 'bagging_fraction',
            'alpha': 'reg_alpha',
            'lambda': 'reg_lambda',
            'nthread': 'num_threads',
            'min_child_weight': 'min_data_in_leaf',
            'eval_metric': 'metric',
            'silent': None,  # LightGBM uses verbosity
            'verbosity': 'verbosity'
        }
        
        for key, value in params.items():
            if key in param_mapping:
                lgb_key = param_mapping[key]
                if lgb_key is not None:
                    # Special handling for verbosity conversion
                    if key == 'verbosity' and isinstance(value, int):
                        # Convert XGBoost verbosity to LightGBM verbosity
                        # XGBoost: 0=silent, 1=warning, 2=info, 3=debug
                        # LightGBM: -1=fatal, 0=warning, 1=info, 2=debug
                        if value == 0:
                            mapped['verbosity'] = -1  # Silent
                        else:
                            mapped['verbosity'] = value - 1
                    else:
                        mapped[lgb_key] = value
            elif key == 'framework':
                continue  # Skip framework parameter
            elif key.startswith('lambdarank_') or key.startswith('ndcg_'):
                # LightGBM não suporta esses parâmetros específicos do XGBoost
                print(f"Warning: LightGBM não suporta o parâmetro '{key}' (específico do XGBoost)")
                continue
            elif key in ['xgboost', 'lightgbm']:
                # Handle framework-specific parameters
                if key == 'lightgbm' and isinstance(value, dict):
                    mapped.update(value)
                continue
            else:
                mapped[key] = value
        
        # Ensure correct objective for ranking
        if 'objective' in mapped and mapped['objective'] == 'rank:ndcg':
            mapped['objective'] = 'lambdarank'
            
        return mapped
        
    def prepare_training_data(self, features: np.ndarray, labels: np.ndarray, 
                            query_ids: np.ndarray) -> lgb.Dataset:
        """
        Prepare training data for LightGBM.
        
        Args:
            features: Feature matrix
            labels: Target labels (relevance scores)
            query_ids: Query IDs for grouping
            
        Returns:
            LightGBM Dataset
        """
        # Ensure data is sorted by query_id (required for LightGBM ranking)
        sort_indices = np.argsort(query_ids)
        sorted_features = features[sort_indices]
        sorted_labels = labels[sort_indices]
        sorted_qids = query_ids[sort_indices]
        
        # Calculate query group sizes in order
        unique_qids, group_sizes = np.unique(sorted_qids, return_counts=True)
        
        # Verify group sizes match data length
        if np.sum(group_sizes) != len(sorted_labels):
            raise ValueError(f"Group sizes sum ({np.sum(group_sizes)}) doesn't match data length ({len(sorted_labels)})")
        
        # Create LightGBM dataset
        train_data = lgb.Dataset(
            sorted_features, 
            label=sorted_labels,
            group=group_sizes
        )
        
        return train_data
        
    def train(self, train_features: np.ndarray, train_labels: np.ndarray,
              train_qids: np.ndarray, num_boost_round: int = 100,
              early_stopping_rounds: Optional[int] = None,
              validation_features: Optional[np.ndarray] = None,
              validation_labels: Optional[np.ndarray] = None,
              validation_qids: Optional[np.ndarray] = None) -> Dict:
        """
        Train the LightGBM LambdaMART model.
        
        Args:
            train_features: Training features
            train_labels: Training labels (relevance scores)
            train_qids: Training query IDs
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            validation_features: Validation features (optional)
            validation_labels: Validation labels (optional)
            validation_qids: Validation query IDs (optional)
            
        Returns:
            Training results dictionary
        """
        print("Preparing training data...")
        train_data = self.prepare_training_data(train_features, train_labels, train_qids)
        
        # Prepare validation data if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        use_validation = (validation_features is not None and 
                         validation_labels is not None and 
                         validation_qids is not None)
        
        if use_validation:
            valid_data = self.prepare_training_data(validation_features, validation_labels, validation_qids)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        print("Starting model training...")
        
        # Training callbacks
        callbacks = []
        # Only use early stopping if we have validation data
        if early_stopping_rounds and use_validation:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
        
        # Train model
        if use_validation:
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=callbacks
            )
        else:
            # Train without early stopping when no validation data
            self.model = lgb.train(
                self.params,
                train_data,
                num_boost_round=num_boost_round
            )
        
        # Store feature importance
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        # Skip training history due to LightGBM compatibility issues
        self.training_history = None
        
        best_iteration = getattr(self.model, 'best_iteration', num_boost_round)
        print(f"Training completed. Used {self.model.num_trees()} trees")
        
        return {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'best_iteration': self.model.best_iteration
        }
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features: Feature matrix for prediction
            
        Returns:
            Predicted scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        predictions = self.model.predict(features, num_iteration=self.model.best_iteration)
        return predictions
        
    def predict_with_query_groups(self, features: np.ndarray, 
                                 query_ids: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Make predictions grouped by query ID.
        
        Args:
            features: Feature matrix
            query_ids: Query IDs
            
        Returns:
            Dictionary mapping query_id to predictions
        """
        predictions = self.predict(features)
        
        # Group predictions by query
        grouped_predictions = {}
        unique_qids = np.unique(query_ids)
        
        for qid in unique_qids:
            mask = query_ids == qid
            grouped_predictions[qid] = predictions[mask]
            
        return grouped_predictions
        
    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """
        Get feature importance from trained model.
        
        Args:
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            Feature importance array
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
            
        return self.model.feature_importance(importance_type=importance_type)
        
    def save_model(self, filepath: str):
        """
        Save trained model to file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save LightGBM model
        model_path = filepath.replace('.txt', '.lgb')
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
        model_path = filepath.replace('.txt', '.lgb')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load LightGBM model
        self.model = lgb.Booster(model_file=model_path)
        
        # Load additional information
        info_filepath = filepath.replace('.txt', '_info.pkl')
        if os.path.exists(info_filepath):
            with open(info_filepath, 'rb') as f:
                model_info = pickle.load(f)
                self.params = model_info.get('params', {})
                self.feature_importance = model_info.get('feature_importance')
                self.training_history = model_info.get('training_history')
                
        print(f"Model loaded from {model_path}")
        
    def get_model_summary(self) -> Dict:
        """
        Get model summary including parameters and performance metrics.
        
        Returns:
            Dictionary with model information
        """
        summary = super().get_model_summary()
        summary.update({
            'framework': 'LightGBM',
            'best_iteration': self.model.best_iteration if self.model else None,
            'num_features': self.model.num_feature() if self.model else None,
            'supports_ips': False  # LightGBM doesn't support IPS natively
        })
        return summary


if __name__ == "__main__":
    # Example usage
    print("LightGBM LambdaMART Model module loaded successfully!")
    
    # Create a simple test
    np.random.seed(42)
    n_samples, n_features = 100, 10
    features = np.random.randn(n_samples, n_features)
    labels = np.random.randint(0, 5, n_samples)
    query_ids = np.repeat(range(10), 10)  # 10 queries, 10 docs each
    
    # Initialize and train model
    model = LightGBMLambdaMART({})
    training_results = model.train(features, labels, query_ids, num_boost_round=10)
    
    # Make predictions
    predictions = model.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    
    print("LightGBM model test completed successfully!")