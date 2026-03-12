"""
MLflow integration utilities for experiment tracking and model management.
Handles logging of parameters, metrics, artifacts, and model versioning.
"""

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class MLflowManager:
    """
    Manager for MLflow experiment tracking and model registration.
    """
    
    def __init__(self, experiment_name: str = "LambdaMART_Experiments",
                 tracking_uri: Optional[str] = None):
        """
        Initialize MLflow manager.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (defaults to local)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local file-based tracking
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                self.experiment_id = mlflow.create_experiment(experiment_name)
                print(f"Created new experiment: {experiment_name}")
            else:
                self.experiment_id = self.experiment.experiment_id
                # Check if experiment is active, if not try to restore it
                if self.experiment.lifecycle_stage == "deleted":
                    try:
                        mlflow.restore_experiment(self.experiment_id)
                        print(f"Restored deleted experiment: {experiment_name}")
                    except Exception as restore_error:
                        print(f"Could not restore experiment, creating new one: {restore_error}")
                        # Create new experiment with a timestamp to avoid conflicts
                        new_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        self.experiment_id = mlflow.create_experiment(new_name)
                        self.experiment_name = new_name
                        print(f"Created new experiment: {new_name}")
                else:
                    print(f"Using existing experiment: {experiment_name}")
            
            # Set the active experiment
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
        except Exception as e:
            print(f"Warning: Could not set experiment: {e}")
            # Fallback: create a unique experiment name
            fallback_name = f"LambdaMART_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            try:
                self.experiment_id = mlflow.create_experiment(fallback_name)
                self.experiment_name = fallback_name
                mlflow.set_experiment(experiment_id=self.experiment_id)
                print(f"Created fallback experiment: {fallback_name}")
            except Exception as fallback_error:
                print(f"Critical error: Could not create any experiment: {fallback_error}")
                self.experiment_id = None
        
        self.current_run = None
        
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict] = None) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Additional tags for the run
            
        Returns:
            Run ID
        """
        if self.experiment_id is None:
            raise Exception("No valid experiment ID available. MLflow setup failed.")
        
        # End any active run before starting a new one
        try:
            if mlflow.active_run() is not None:
                print("Ending previous active run...")
                mlflow.end_run()
        except Exception as e:
            print(f"Warning: Could not end previous run: {e}")
            
        if run_name is None:
            run_name = f"lambdamart_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Default tags
        default_tags = {
            "model_type": "LambdaMART",
            "framework": "XGBoost",
            "task": "Learning to Rank"
        }
        
        if tags:
            default_tags.update(tags)
        
        try:
            self.current_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name,
                tags=default_tags
            )
            return self.current_run.info.run_id
        except Exception as e:
            print(f"Error starting MLflow run: {e}")
            print(f"Experiment ID: {self.experiment_id}")
            print(f"Experiment Name: {self.experiment_name}")
            raise
    
    def log_model_parameters(self, model_params: Dict[str, Any]):
        """
        Log model hyperparameters.
        
        Args:
            model_params: Dictionary of model parameters
        """
        # Clean parameters for MLflow (remove None values, convert types)
        clean_params = {}
        for key, value in model_params.items():
            if value is not None:
                if isinstance(value, (list, dict)):
                    clean_params[key] = json.dumps(value)
                else:
                    clean_params[key] = value
        
        mlflow.log_params(clean_params)
    
    def log_dataset_info(self, train_info: Dict, test_info: Dict):
        """
        Log dataset information.
        
        Args:
            train_info: Training dataset information
            test_info: Test dataset information
        """
        dataset_params = {
            "train_samples": train_info.get("n_samples", 0),
            "train_features": train_info.get("n_features", 0),
            "train_queries": train_info.get("n_queries", 0),
            "test_samples": test_info.get("n_samples", 0),
            "test_features": test_info.get("n_features", 0),
            "test_queries": test_info.get("n_queries", 0)
        }
        
        mlflow.log_params(dataset_params)
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """
        Log evaluation metrics.
        
        Args:
            metrics: Dictionary of metric values
            prefix: Prefix for metric names (e.g., "train_", "test_")
        """
        prefixed_metrics = {}
        for metric_name, value in metrics.items():
            # Replace @ with _at_ for MLflow compatibility
            clean_metric_name = metric_name.replace('@', '_at_')
            prefixed_metrics[f"{prefix}{clean_metric_name}"] = value
        
        mlflow.log_metrics(prefixed_metrics)
    
    def log_per_query_metrics(self, per_query_df: 'pd.DataFrame', dataset_name: str = "test"):
        """
        Log per-query metrics as a CSV artifact.
        
        Args:
            per_query_df: DataFrame with per-query metrics (from RankingEvaluator.get_per_query_metrics_table)
            dataset_name: Name of dataset (e.g., "test", "train")
        """
        csv_path = f"{dataset_name}_per_query_metrics.csv"
        per_query_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        
        if os.path.exists(csv_path):
            os.remove(csv_path)
    
    def log_training_metrics(self, training_history: Dict, step_key: str = "iteration"):
        """
        Log training metrics over time.
        
        Args:
            training_history: Training history from LightGBM
            step_key: Key to use for step parameter
        """
        if not training_history:
            return
        
        # Log metrics for each dataset (train, valid)
        for dataset_name, metrics_dict in training_history.items():
            for metric_name, values in metrics_dict.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(
                        f"{dataset_name}_{metric_name}",
                        value,
                        step=step
                    )
    
    def log_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: Optional[List[str]] = None):
        """
        Log and visualize feature importance.
        
        Args:
            feature_importance: Array of feature importance values
            feature_names: Optional feature names
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Log top features as parameters
        top_features = importance_df.head(10)
        for idx, row in top_features.iterrows():
            mlflow.log_param(f"top_feature_{idx+1}", f"{row['feature']}:{row['importance']:.4f}")
        
        # Create and save feature importance plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        
        importance_plot_path = "feature_importance.png"
        plt.savefig(importance_plot_path)
        mlflow.log_artifact(importance_plot_path)
        plt.close()
        
        # Remove temporary file
        if os.path.exists(importance_plot_path):
            os.remove(importance_plot_path)
        
        # Save detailed feature importance
        importance_csv_path = "feature_importance.csv"
        importance_df.to_csv(importance_csv_path, index=False)
        mlflow.log_artifact(importance_csv_path)
        
        if os.path.exists(importance_csv_path):
            os.remove(importance_csv_path)
    
    def log_model_artifact(self, model, model_name: str = "lambdamart_model"):
        """
        Log the trained model as an artifact.
        
        Args:
            model: Trained model object
            model_name: Name for the model artifact
        """
        try:
            # Log XGBoost model
            if hasattr(model, 'model') and model.model is not None:
                mlflow.xgboost.log_model(
                    xgb_model=model.model,
                    artifact_path=model_name,
                    registered_model_name=f"{self.experiment_name}_{model_name}"
                )
            
            # Save additional model information
            model_info = {
                'model_type': 'LambdaMART',
                'framework': 'XGBoost',
                'parameters': model.params if hasattr(model, 'params') else {},
                'training_completed': True
            }
            
            model_info_path = "model_info.json"
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            mlflow.log_artifact(model_info_path)
            
            if os.path.exists(model_info_path):
                os.remove(model_info_path)
                
        except Exception as e:
            print(f"Warning: Could not log model artifact: {e}")
    
    def log_predictions(self, predictions: np.ndarray, true_labels: np.ndarray,
                       query_ids: np.ndarray, dataset_name: str = "test"):
        """
        Log prediction results and create visualizations.
        
        Args:
            predictions: Model predictions
            true_labels: True labels
            query_ids: Query IDs
            dataset_name: Name of the dataset
        """
        # Save predictions
        predictions_df = pd.DataFrame({
            'query_id': query_ids,
            'true_label': true_labels,
            'prediction': predictions
        })
        
        predictions_path = f"{dataset_name}_predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        mlflow.log_artifact(predictions_path)
        
        # Create prediction distribution plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(true_labels, bins=30, alpha=0.7, label='True Labels')
        plt.hist(predictions, bins=30, alpha=0.7, label='Predictions')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.title(f'{dataset_name.title()} - Score Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.scatter(true_labels, predictions, alpha=0.6)
        plt.xlabel('True Labels')
        plt.ylabel('Predictions')
        plt.title(f'{dataset_name.title()} - True vs Predicted')
        plt.plot([true_labels.min(), true_labels.max()], 
                [true_labels.min(), true_labels.max()], 'r--', alpha=0.8)
        
        plt.tight_layout()
        
        predictions_plot_path = f"{dataset_name}_predictions_analysis.png"
        plt.savefig(predictions_plot_path, dpi=150)
        mlflow.log_artifact(predictions_plot_path)
        plt.close()
        
        # Clean up temporary files
        for temp_file in [predictions_path, predictions_plot_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    def log_config_file(self, config_path: str):
        """
        Log configuration file as artifact.
        
        Args:
            config_path: Path to configuration file
        """
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
    
    def create_metrics_summary_plot(self, metrics: Dict[str, float]):
        """
        Create a summary plot of all evaluation metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        plt.figure(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Separate NDCG and MRR metrics
        ndcg_metrics = [(name, value) for name, value in zip(metric_names, metric_values) 
                       if name.startswith('ndcg')]
        mrr_metrics = [(name, value) for name, value in zip(metric_names, metric_values) 
                      if name.startswith('mrr')]
        
        if ndcg_metrics:
            plt.subplot(1, 2, 1)
            ndcg_names, ndcg_values = zip(*ndcg_metrics)
            plt.bar(ndcg_names, ndcg_values)
            plt.title('NDCG Metrics')
            plt.ylabel('NDCG Score')
            plt.xticks(rotation=45)
        
        if mrr_metrics:
            plt.subplot(1, 2, 2)
            mrr_names, mrr_values = zip(*mrr_metrics)
            plt.bar(mrr_names, mrr_values)
            plt.title('MRR Metrics')
            plt.ylabel('MRR Score')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        metrics_plot_path = "evaluation_metrics_summary.png"
        plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(metrics_plot_path)
        plt.close()
        
        if os.path.exists(metrics_plot_path):
            os.remove(metrics_plot_path)
    
    def end_run(self):
        """End the current MLflow run."""
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()
            self.current_run = None
        except Exception as e:
            print(f"Warning: Error ending MLflow run: {e}")
            self.current_run = None
    
    def get_experiment_runs(self) -> pd.DataFrame:
        """
        Get all runs from the current experiment.
        
        Returns:
            DataFrame with run information
        """
        runs = mlflow.search_runs(experiment_ids=[self.experiment_id])
        return runs
    
    def get_best_run(self, metric_name: str = "test_ndcg_at_10") -> Dict:
        """
        Get the best run based on a specific metric.
        
        Args:
            metric_name: Metric to optimize for (use underscore format, e.g., "test_ndcg_at_10")
            
        Returns:
            Dictionary with best run information
        """
        runs_df = self.get_experiment_runs()
        
        if f"metrics.{metric_name}" in runs_df.columns:
            best_run = runs_df.loc[runs_df[f"metrics.{metric_name}"].idxmax()]
            return best_run.to_dict()
        else:
            print(f"Metric {metric_name} not found in runs")
            return {}


class ExperimentComparison:
    """
    Utility class for comparing multiple experiments.
    """
    
    def __init__(self, mlflow_manager: MLflowManager):
        self.mlflow_manager = mlflow_manager
    
    def compare_runs(self, run_ids: List[str], metrics: List[str]) -> pd.DataFrame:
        """
        Compare specific runs and metrics.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            run_data = {'run_id': run_id, 'run_name': run.data.tags.get('mlflow.runName', run_id)}
            
            for metric in metrics:
                metric_value = run.data.metrics.get(metric, None)
                run_data[metric] = metric_value
            
            comparison_data.append(run_data)
        
        return pd.DataFrame(comparison_data)
    
    def create_comparison_plot(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Create visualization comparing runs.
        
        Args:
            comparison_df: DataFrame from compare_runs()
            save_path: Optional path to save the plot
        """
        metrics_columns = [col for col in comparison_df.columns 
                          if col not in ['run_id', 'run_name']]
        
        if not metrics_columns:
            print("No metrics found for comparison")
            return
        
        fig, axes = plt.subplots(len(metrics_columns), 1, figsize=(12, 4*len(metrics_columns)))
        
        if len(metrics_columns) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics_columns):
            ax = axes[idx]
            comparison_df.plot(x='run_name', y=metric, kind='bar', ax=ax)
            ax.set_title(f'Comparison: {metric}')
            ax.set_xlabel('Run')
            ax.set_ylabel(metric)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("MLflow Manager initialized successfully!")
    
    # Test basic functionality
    manager = MLflowManager("test_experiment")
    
    # Start a test run
    run_id = manager.start_run("test_run")
    print(f"Started run: {run_id}")
    
    # Log some test parameters and metrics
    manager.log_model_parameters({"learning_rate": 0.1, "num_leaves": 31})
    manager.log_evaluation_metrics({"ndcg@10": 0.85, "mrr": 0.72}, prefix="test_")
    
    # End run
    manager.end_run()
    
    print("MLflow Manager test completed successfully!")