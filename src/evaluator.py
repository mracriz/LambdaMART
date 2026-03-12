"""
Evaluation metrics for ranking models.
Implements NDCG@k and MRR (Mean Reciprocal Rank) metrics.
"""

import numpy as np
import pandas as pd
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
            
        discounts = np.log2(np.arange(len(relevance_scores[:k])) + 2)
        dcg = np.sum((np.power(2, relevance_scores[:k]) - 1) / discounts)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(true_relevance: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at k.
        """
        if len(true_relevance) == 0 or k <= 0:
            return 0.0
        
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        
        dcg_k = RankingEvaluator.dcg_at_k(sorted_relevance, k)
        ideal_relevance = np.sort(true_relevance)[::-1]
        idcg_k = RankingEvaluator.dcg_at_k(ideal_relevance, k)
        
        if idcg_k == 0:
            return 0.0
            
        return dcg_k / idcg_k
    
    @staticmethod
    def mrr_single_query(true_relevance: np.ndarray, predicted_scores: np.ndarray) -> float:
        """
        Calculate Mean Reciprocal Rank for a single query (full ranking).
        """
        if len(true_relevance) == 0:
            return 0.0
        
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        relevant_positions = np.where(sorted_relevance > 0)[0]
        
        if len(relevant_positions) == 0:
            return 0.0
        
        first_relevant_rank = relevant_positions[0] + 1
        return 1.0 / first_relevant_rank
    
    @staticmethod
    def mrr_at_k(true_relevance: np.ndarray, predicted_scores: np.ndarray, k: int) -> float:
        """
        Calculate Mean Reciprocal Rank at k (MRR@k).
        Returns reciprocal rank of first relevant document within top-k, 0 if none.
        """
        if len(true_relevance) == 0 or k <= 0:
            return 0.0
        
        sorted_indices = np.argsort(predicted_scores)[::-1]
        sorted_relevance = true_relevance[sorted_indices]
        relevant_positions = np.where(sorted_relevance[:k] > 0)[0]
        
        if len(relevant_positions) == 0:
            return 0.0
        
        first_relevant_rank = relevant_positions[0] + 1
        return 1.0 / first_relevant_rank
    
    def evaluate_query(self, true_relevance: np.ndarray, predicted_scores: np.ndarray,
                      k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate a single query with NDCG@k and MRR@k metrics.
        """
        metrics = {}
        
        # Calculate NDCG@k for each k
        for k in k_values:
            ndcg_k = self.ndcg_at_k(true_relevance, predicted_scores, k)
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # Calculate MRR (full ranking)
        mrr = self.mrr_single_query(true_relevance, predicted_scores)
        metrics['mrr'] = mrr
        
        # Calculate MRR@k for each k
        for k in k_values:
            mrr_k = self.mrr_at_k(true_relevance, predicted_scores, k)
            metrics[f'mrr@{k}'] = mrr_k
        
        return metrics
    
    def evaluate_ranking(self, true_labels: np.ndarray, predicted_scores: np.ndarray,
                        query_ids: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, float]:
        """
        Evaluate ranking performance across multiple queries.
        """
        if len(true_labels) != len(predicted_scores) or len(true_labels) != len(query_ids):
            raise ValueError("Input arrays must have the same length")
        
        unique_queries = np.unique(query_ids)
        all_metrics = []
        
        for qid in unique_queries:
            query_mask = query_ids == qid
            query_true_labels = true_labels[query_mask]
            query_predicted_scores = predicted_scores[query_mask]
            
            if np.sum(query_true_labels) == 0:
                continue
            
            query_metrics = self.evaluate_query(
                query_true_labels, query_predicted_scores, k_values
            )
            all_metrics.append(query_metrics)
        
        if not all_metrics:
            warnings.warn("No queries with relevant documents found")
            default_metrics = {f'ndcg@{k}': 0.0 for k in k_values}
            default_metrics.update({'mrr': 0.0, 'mrr@1': 0.0, 'mrr@3': 0.0, 'mrr@5': 0.0, 'mrr@10': 0.0})
            return default_metrics
        
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            metric_values = [m[metric_name] for m in all_metrics]
            averaged_metrics[metric_name] = np.mean(metric_values)
        
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
            
            if np.sum(true_labels) == 0:
                continue
            
            query_metrics = self.evaluate_query(true_labels, predicted_scores, k_values)
            all_metrics.append(query_metrics)
        
        if not all_metrics:
            warnings.warn("No valid queries found for evaluation")
            default_metrics = {f'ndcg@{k}': 0.0 for k in k_values}
            default_metrics.update({'mrr': 0.0, 'mrr@1': 0.0, 'mrr@3': 0.0, 'mrr@5': 0.0, 'mrr@10': 0.0})
            return default_metrics
        
        averaged_metrics = {}
        for metric_name in all_metrics[0].keys():
            metric_values = [m[metric_name] for m in all_metrics]
            averaged_metrics[metric_name] = np.mean(metric_values)
        
        return averaged_metrics
    
    def get_detailed_evaluation(self, true_labels: np.ndarray, predicted_scores: np.ndarray,
                               query_ids: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """
        Get detailed evaluation results including per-query metrics.
        """
        unique_queries = np.unique(query_ids)
        per_query_metrics = {}
        valid_queries = []
        
        for qid in unique_queries:
            query_mask = query_ids == qid
            query_true_labels = true_labels[query_mask]
            query_predicted_scores = predicted_scores[query_mask]
            
            if np.sum(query_true_labels) == 0:
                continue
            
            query_metrics = self.evaluate_query(
                query_true_labels, query_predicted_scores, k_values
            )
            per_query_metrics[qid] = query_metrics
            valid_queries.append(qid)
        
        overall_metrics = self.evaluate_ranking(true_labels, predicted_scores, query_ids, k_values)
        
        return {
            'overall_metrics': overall_metrics,
            'per_query_metrics': per_query_metrics,
            'num_valid_queries': len(valid_queries),
            'num_total_queries': len(unique_queries),
            'valid_query_ids': valid_queries
        }
    
    def get_per_query_metrics_table(self, true_labels: np.ndarray, predicted_scores: np.ndarray,
                                   query_ids: np.ndarray, k_values: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """
        Get per-query metrics as a pandas DataFrame for easy export to CSV.
        
        Columns:
        - query_id: Query identifier
        - num_docs: Number of documents for this query
        - num_relevant_docs: Number of relevant documents (relevance > 0)
        - ndcg@k: NDCG at k for each k in k_values
        - mrr: Mean Reciprocal Rank
        - mrr@1: MRR at position 1 (binary: relevant in top-1 or not)
        
        Args:
            true_labels: True relevance labels
            predicted_scores: Predicted scores
            query_ids: Query IDs
            k_values: List of k values for NDCG@k
            
        Returns:
            pandas DataFrame with per-query metrics
        """
        unique_queries = np.unique(query_ids)
        rows = []
        
        for qid in unique_queries:
            query_mask = query_ids == qid
            query_true_labels = true_labels[query_mask]
            query_predicted_scores = predicted_scores[query_mask]
            
            # Count documents and relevant documents
            num_docs = len(query_true_labels)
            num_relevant_docs = int(np.sum(query_true_labels > 0))
            
            # Skip queries with no relevant documents for metric calculation
            if num_relevant_docs == 0:
                # Still add row but with 0 metrics
                row = {'query_id': qid, 'num_docs': num_docs, 'num_relevant_docs': num_relevant_docs}
                for k in k_values:
                    row[f'ndcg@{k}'] = 0.0
                row['mrr'] = 0.0
                row['mrr@1'] = 0.0
                rows.append(row)
                continue
            
            # Calculate metrics for this query
            query_metrics = self.evaluate_query(query_true_labels, query_predicted_scores, k_values)
            
            # Build row for this query
            row = {
                'query_id': qid,
                'num_docs': num_docs,
                'num_relevant_docs': num_relevant_docs
            }
            
            # Add all metrics
            for metric_name, metric_value in query_metrics.items():
                row[metric_name] = float(metric_value)
            
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        
        # Reorder columns: query_id, num_docs, num_relevant_docs, then metrics
        metric_cols = [f'ndcg@{k}' for k in k_values] + ['mrr', 'mrr@1']
        cols = ['query_id', 'num_docs', 'num_relevant_docs'] + metric_cols
        df = df[cols]
        
        return df
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """
        Compare multiple model evaluation results.
        """
        comparison = {
            'model_names': list(model_results.keys()),
            'metrics_comparison': {}
        }
        
        first_model = list(model_results.values())[0]
        metric_names = first_model['overall_metrics'].keys()
        
        for metric in metric_names:
            comparison['metrics_comparison'][metric] = {}
            for model_name, results in model_results.items():
                comparison['metrics_comparison'][metric][model_name] = \
                    results['overall_metrics'][metric]
        
        comparison['best_models'] = {}
        for metric in metric_names:
            best_model = max(model_results.keys(),
                           key=lambda m: model_results[m]['overall_metrics'][metric])
            comparison['best_models'][metric] = best_model
        
        return comparison
    
    def print_evaluation_summary(self, evaluation_results: Dict):
        """
        Print a formatted summary of evaluation results.
        """
        print("\n" + "="*60)
        print("RANKING EVALUATION SUMMARY")
        print("="*60)
        
        if 'overall_metrics' in evaluation_results:
            overall = evaluation_results['overall_metrics']
            num_queries = evaluation_results.get('num_valid_queries', 'N/A')
            total_queries = evaluation_results.get('num_total_queries', 'N/A')
        elif 'test' in evaluation_results:
            overall = evaluation_results['test']
            num_queries = 'N/A'
            total_queries = 'N/A'
        else:
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
    
    np.random.seed(42)
    
    true_relevance = np.array([3, 2, 1, 0, 2])
    predicted_scores = np.array([0.9, 0.7, 0.5, 0.2, 0.8])
    
    evaluator = RankingEvaluator()
    
    metrics = evaluator.evaluate_query(true_relevance, predicted_scores)
    print("Single query metrics:", metrics)
    
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