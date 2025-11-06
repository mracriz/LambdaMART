"""
Evaluation metrics for ranking models.
Implements NDCG@k and MRR (Mean Reciprocal Rank) metrics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings


class RankingEvaluator:
    """
    Evaluator for ranking models with NDCG and MRR metrics.
    """
    
    def __init__(self):
        self.metrics_history = []
    
    @staticmethod
    def dcg_at_k(relevance_scores: np.ndarray, k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at k using the standard formula.
        Formula: DCG@k = sum((2^rel_i - 1) / log2(i + 2)) for i in [0, k-1]
        
        Args:
            relevance_scores: Array of relevance scores (sorted by rank)
            k: Cut-off rank
            
        Returns:
            DCG@k value
        """
        if k <= 0:
            return 0.0
            
        k = min(k, len(relevance_scores))
        if k == 0:
            return 0.0
            
        # DCG formula: sum((2^rel_i - 1) / log2(i + 2)) for i in [0, k-1]
        # This matches the formula used in allRank and many standard implementations
        discounts = np.log2(np.arange(len(relevance_scores[:k])) + 2)  # log2(2), log2(3), ..., log2(k+1)
        dcg = np.sum((np.power(2, relevance_scores[:k]) - 1) / discounts)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(true_relevance: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        
        Args:
            true_relevance: True relevance scores
            predicted_scores: Predicted scores
            k: Cut-off rank
            
        Returns:
            NDCG@k value
        """
        if len(true_relevance) == 0 or k <= 0:
            return 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        
        # Calculate DCG@k
        dcg_k = RankingEvaluator.dcg_at_k(sorted_relevance, k)
        
        # Calculate IDCG@k (Ideal DCG)
        ideal_relevance = np.sort(true_relevance)[::-1]  # Sort true relevance descending
        idcg_k = RankingEvaluator.dcg_at_k(ideal_relevance, k)
        
        # Avoid division by zero
        if idcg_k == 0:
            return 0.0
            
        return dcg_k / idcg_k
    
    @staticmethod
    def mrr_single_query(true_relevance: np.ndarray, predicted_scores: np.ndarray) -> float:
        """
        Calculate Mean Reciprocal Rank for a single query.
        
        Args:
            true_relevance: True relevance scores
            predicted_scores: Predicted scores
            
        Returns:
            Reciprocal rank (1/rank of first relevant document, 0 if none)
        """
        if len(true_relevance) == 0:
            return 0.0
        
        # Sort by predicted scores (descending)
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        
        # Find first relevant document (relevance > 0)
        relevant_positions = np.where(sorted_relevance > 0)[0]
        
        if len(relevant_positions) == 0:
            return 0.0
        
        # Return reciprocal of the rank (1-indexed)
        first_relevant_rank = relevant_positions[0] + 1
        return 1.0 / first_relevant_rank
    
    def evaluate_query(self, true_relevance: np.ndarray, predicted_scores: np.ndarray,
                      k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate a single query with NDCG@k and MRR metrics.
        
        Args:
            true_relevance: True relevance scores
            predicted_scores: Predicted scores
            k_values: List of k values for NDCG@k
            
        Returns:
            Dictionary with metric values
        """
        metrics = {}
        
        # Calculate NDCG@k for each k
        for k in k_values:
            ndcg_k = self.ndcg_at_k(true_relevance, predicted_scores, k)
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Calculate MRR
        mrr = self.mrr_single_query(true_relevance, predicted_scores)
        metrics['mrr'] = mrr
        
        return metrics
    
    def evaluate_ranking(self, true_labels: np.ndarray, predicted_scores: np.ndarray,
                        query_ids: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate ranking performance across multiple queries.
        
        Args:
            true_labels: True relevance labels
            predicted_scores: Predicted scores
            query_ids: Query IDs
            k_values: List of k values for NDCG@k
            
        Returns:
            Dictionary with averaged metric values
        """
        if len(true_labels) != len(predicted_scores) or len(true_labels) != len(query_ids):
            raise ValueError("Input arrays must have the same length")
        
        unique_queries = np.unique(query_ids)
        all_metrics = []
        
        for qid in unique_queries:
            # Get data for this query
            query_mask = query_ids == qid
            query_true_labels = true_labels[query_mask]
            query_predicted_scores = predicted_scores[query_mask]
            
            # Skip queries with no relevant documents
            if np.sum(query_true_labels) == 0:
                continue
            
            # Evaluate this query
            query_metrics = self.evaluate_query(
                query_true_labels, query_predicted_scores, k_values
            )
            all_metrics.append(query_metrics)
        
        if not all_metrics:
            warnings.warn("No queries with relevant documents found")
            return {f'ndcg@{k}': 0.0 for k in k_values} | {'mrr': 0.0}
        
        # Average metrics across queries
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            metric_values = [m[metric_name] for m in all_metrics]
            averaged_metrics[metric_name] = np.mean(metric_values)
        
        # Store evaluation results
        self.metrics_history.append({
            'metrics': averaged_metrics,
            'num_queries': len(all_metrics),
            'num_total_queries': len(unique_queries)
        })
        
        return averaged_metrics
    
    def evaluate_model_predictions(self, model_predictions: Dict[int, np.ndarray],
                                 true_labels_by_query: Dict[int, np.ndarray],
                                 k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate model predictions organized by query.
        
        Args:
            model_predictions: Dictionary mapping query_id to predicted scores
            true_labels_by_query: Dictionary mapping query_id to true labels
            k_values: List of k values for NDCG@k
            
        Returns:
            Dictionary with averaged metric values
        """
        all_metrics = []
        
        for qid in model_predictions.keys():
            if qid not in true_labels_by_query:
                continue
                
            predicted_scores = model_predictions[qid]
            true_labels = true_labels_by_query[qid]
            
            if len(predicted_scores) != len(true_labels):
                warnings.warn(f"Mismatch in lengths for query {qid}")
                continue
            
            # Skip queries with no relevant documents
            if np.sum(true_labels) == 0:
                continue
            
            # Evaluate this query
            query_metrics = self.evaluate_query(true_labels, predicted_scores, k_values)
            all_metrics.append(query_metrics)
        
        if not all_metrics:
            warnings.warn("No valid queries found for evaluation")
            return {f'ndcg@{k}': 0.0 for k in k_values} | {'mrr': 0.0}
        
        # Average metrics across queries
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            metric_values = [m[metric_name] for m in all_metrics]
            averaged_metrics[metric_name] = np.mean(metric_values)
        
        return averaged_metrics
    
    def get_detailed_evaluation(self, true_labels: np.ndarray, predicted_scores: np.ndarray,
                               query_ids: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Get detailed evaluation results including per-query metrics.
        
        Args:
            true_labels: True relevance labels
            predicted_scores: Predicted scores
            query_ids: Query IDs
            k_values: List of k values for NDCG@k
            
        Returns:
            Dictionary with detailed evaluation results
        """
        unique_queries = np.unique(query_ids)
        per_query_metrics = {}
        valid_queries = []
        
        for qid in unique_queries:
            query_mask = query_ids == qid
            query_true_labels = true_labels[query_mask]
            query_predicted_scores = predicted_scores[query_mask]
            
            # Skip queries with no relevant documents
            if np.sum(query_true_labels) == 0:
                continue
            
            query_metrics = self.evaluate_query(
                query_true_labels, query_predicted_scores, k_values
            )
            per_query_metrics[qid] = query_metrics
            valid_queries.append(qid)
        
        # Calculate overall metrics
        overall_metrics = self.evaluate_ranking(true_labels, predicted_scores, query_ids, k_values)
        
        return {
            'overall_metrics': overall_metrics,
            'per_query_metrics': per_query_metrics,
            'num_valid_queries': len(valid_queries),
            'num_total_queries': len(unique_queries),
            'valid_query_ids': valid_queries
        }
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple model evaluation results.
        
        Args:
            model_results: Dictionary mapping model_name to evaluation results
            
        Returns:
            Comparison summary
        """
        comparison = {
            'model_names': list(model_results.keys()),
            'metrics_comparison': {}
        }
        
        # Get all metric names
        first_model = list(model_results.values())[0]
        metric_names = first_model['overall_metrics'].keys()
        
        for metric in metric_names:
            comparison['metrics_comparison'][metric] = {}
            for model_name, results in model_results.items():
                comparison['metrics_comparison'][metric][model_name] = \
                    results['overall_metrics'][metric]
        
        # Find best model for each metric
        comparison['best_models'] = {}
        for metric in metric_names:
            best_model = max(model_results.keys(),
                           key=lambda m: model_results[m]['overall_metrics'][metric])
            comparison['best_models'][metric] = best_model
        
        return comparison
    
    def print_evaluation_summary(self, evaluation_results: Dict):
        """
        Print a formatted summary of evaluation results.
        
        Args:
            evaluation_results: Results from get_detailed_evaluation() or direct metrics
        """
        print("\n" + "="*60)
        print("RANKING EVALUATION SUMMARY")
        print("="*60)
        
        # Handle different input formats
        if 'overall_metrics' in evaluation_results:
            # Legacy format from get_detailed_evaluation()
            overall = evaluation_results['overall_metrics']
            num_queries = evaluation_results.get('num_valid_queries', 'N/A')
            total_queries = evaluation_results.get('num_total_queries', 'N/A')
        elif 'test' in evaluation_results:
            # New format - extract metrics directly
            overall = evaluation_results['test']
            num_queries = 'N/A'
            total_queries = 'N/A'
        else:
            # Direct metrics format
            overall = evaluation_results
            num_queries = 'N/A'
            total_queries = 'N/A'
        
        if num_queries != 'N/A':
            print(f"Queries evaluated: {num_queries}")
            print(f"Total queries: {total_queries}")
        
        print("\nOverall Metrics:")
        print("-" * 30)
        
        for metric, value in overall.items():
            if metric.startswith('ndcg'):
                print(f"{metric.upper():<12}: {value:.4f}")
            else:
                print(f"{metric.upper():<12}: {value:.4f}")
        
        print("="*60)


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Ranking Evaluator...")
    
    # Create sample data
    np.random.seed(42)
    
    # Sample query with known ranking
    true_relevance = np.array([3, 2, 1, 0, 2])  # Relevance scores
    predicted_scores = np.array([0.9, 0.7, 0.5, 0.2, 0.8])  # Predicted scores
    
    evaluator = RankingEvaluator()
    
    # Test single query evaluation
    metrics = evaluator.evaluate_query(true_relevance, predicted_scores)
    print("Single query metrics:", metrics)
    
    # Test multiple queries evaluation
    n_docs_per_query = 5
    n_queries = 3
    
    all_true_labels = np.tile(true_relevance, n_queries)
    all_predicted_scores = np.tile(predicted_scores, n_queries)
    all_query_ids = np.repeat(range(n_queries), n_docs_per_query)
    
    overall_metrics = evaluator.evaluate_ranking(
        all_true_labels, all_predicted_scores, all_query_ids
    )
    print("Overall metrics:", overall_metrics)
    
    print("Ranking Evaluator test completed successfully!")