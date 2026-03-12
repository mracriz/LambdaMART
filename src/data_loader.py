"""
Data loader module for SVM-Rank format files.
Handles loading and preprocessing of training and test data.
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.datasets import load_svmlight_file


class SVMRankDataLoader:
    """
    Data loader for SVM-Rank format files.
    
    SVM-Rank format:
    <label> qid:<query_id> <feature_1>:<value_1> ... <feature_n>:<value_n>
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.feature_names = None
        
    def load_svmrank_file(self, filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a single SVM-Rank format file.
        
        Args:
            filepath: Path to the SVM-Rank file
            
        Returns:
            Tuple of (features, labels, query_ids)
        """
        features = []
        labels = []
        query_ids = []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split(' ')
                
                # Extract label
                label = float(parts[0])
                labels.append(label)
                
                # Extract query ID
                qid_part = [p for p in parts if p.startswith('qid:')]
                if qid_part:
                    qid = int(qid_part[0].split(':')[1])
                else:
                    qid = 0  # Default if no qid found
                query_ids.append(qid)
                
                # Extract features
                feature_dict = {}
                for part in parts[1:]:
                    if ':' in part and not part.startswith('qid:'):
                        try:
                            feat_id, feat_val = part.split(':')
                            feature_dict[int(feat_id)] = float(feat_val)
                        except ValueError:
                            continue
                
                features.append(feature_dict)
        
        # Convert to dense matrix
        if features:
            max_feature_id = max(max(f.keys()) if f else [0] for f in features)
            dense_features = np.zeros((len(features), max_feature_id))
            
            for i, feat_dict in enumerate(features):
                for feat_id, feat_val in feat_dict.items():
                    dense_features[i, feat_id - 1] = feat_val  # Features are 1-indexed
        else:
            dense_features = np.array([])
            
        return dense_features, np.array(labels), np.array(query_ids)
    
    def load_directory(self, directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all SVM-Rank files from a directory.
        
        Args:
            directory: Directory containing .txt files in SVM-Rank format
            
        Returns:
            Tuple of (features, labels, query_ids)
        """
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        txt_files = glob.glob(os.path.join(directory, "*.txt"))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in directory: {directory}")
        
        all_features = []
        all_labels = []
        all_query_ids = []
        
        for filepath in txt_files:
            print(f"Loading file: {os.path.basename(filepath)}")
            features, labels, query_ids = self.load_svmrank_file(filepath)
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
                all_query_ids.append(query_ids)
        
        if not all_features:
            raise ValueError("No valid data found in any files")
        
        # Concatenate all data
        combined_features = np.vstack(all_features)
        combined_labels = np.concatenate(all_labels)
        combined_query_ids = np.concatenate(all_query_ids)
        
        return combined_features, combined_labels, combined_query_ids
    
    def load_data_source(self, source_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from either a directory or a single file.
        
        Args:
            source_path: Path to directory containing .txt files OR path to a single .txt file
            
        Returns:
            Tuple of (features, labels, query_ids)
        """
        if os.path.isfile(source_path):
            # Single file
            if not source_path.endswith('.txt'):
                raise ValueError(f"File must be a .txt file: {source_path}")
            print(f"Loading single file: {os.path.basename(source_path)}")
            return self.load_svmrank_file(source_path)
        elif os.path.isdir(source_path):
            # Directory
            return self.load_directory(source_path)
        else:
            raise FileNotFoundError(f"Path not found: {source_path}")
    
    def load_train_test_data(self, train_source: str, test_source: str) -> Dict:
        """
        Load training and test data from directories or files.
        
        Args:
            train_source: Directory containing training files OR path to training file
            test_source: Directory containing test files OR path to test file
            
        Returns:
            Dictionary with train/test features, labels, and query_ids
        """
        print("Loading training data...")
        train_features, train_labels, train_qids = self.load_data_source(train_source)
        
        print("Loading test data...")
        test_features, test_labels, test_qids = self.load_data_source(test_source)
        
        # Ensure feature dimensions match
        n_features = max(train_features.shape[1], test_features.shape[1])
        
        if train_features.shape[1] < n_features:
            padding = np.zeros((train_features.shape[0], n_features - train_features.shape[1]))
            train_features = np.hstack([train_features, padding])
            
        if test_features.shape[1] < n_features:
            padding = np.zeros((test_features.shape[0], n_features - test_features.shape[1]))
            test_features = np.hstack([test_features, padding])
        
        self.train_data = {
            'features': train_features,
            'labels': train_labels,
            'query_ids': train_qids
        }
        
        self.test_data = {
            'features': test_features,
            'labels': test_labels,
            'query_ids': test_qids
        }
        
        print(f"Training data: {train_features.shape[0]} samples, {train_features.shape[1]} features")
        print(f"Test data: {test_features.shape[0]} samples, {test_features.shape[1]} features")
        print(f"Unique queries in train: {len(np.unique(train_qids))}")
        print(f"Unique queries in test: {len(np.unique(test_qids))}")
        
        return {
            'train': self.train_data,
            'test': self.test_data
        }
    
    def get_query_groups(self, query_ids: np.ndarray) -> np.ndarray:
        """
        Get group sizes for each query (required for ranking algorithms).
        
        Args:
            query_ids: Array of query IDs
            
        Returns:
            Array of group sizes
        """
        unique_qids, counts = np.unique(query_ids, return_counts=True)
        return counts
    
    def create_sample_data(self, output_dir: str, n_queries: int = 5, n_docs_per_query: int = 10):
        """
        Create sample SVM-Rank format data for testing.
        
        Args:
            output_dir: Directory to save sample files
            n_queries: Number of queries to generate
            n_docs_per_query: Number of documents per query
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sample training data
        train_file = os.path.join(output_dir, "sample_train.txt")
        with open(train_file, 'w') as f:
            for qid in range(1, n_queries + 1):
                for doc in range(n_docs_per_query):
                    # Random relevance score (0-4)
                    label = np.random.randint(0, 5)
                    
                    # Random features (1-10)
                    features = []
                    for feat_id in range(1, 11):
                        feat_val = np.random.normal(0, 1)
                        features.append(f"{feat_id}:{feat_val:.6f}")
                    
                    line = f"{label} qid:{qid} " + " ".join(features) + "\n"
                    f.write(line)
        
        # Generate sample test data
        test_file = os.path.join(output_dir, "sample_test.txt")
        with open(test_file, 'w') as f:
            for qid in range(n_queries + 1, n_queries * 2 + 1):
                for doc in range(n_docs_per_query):
                    label = np.random.randint(0, 5)
                    features = []
                    for feat_id in range(1, 11):
                        feat_val = np.random.normal(0, 1)
                        features.append(f"{feat_id}:{feat_val:.6f}")
                    
                    line = f"{label} qid:{qid} " + " ".join(features) + "\n"
                    f.write(line)
        
        print(f"Sample data created in {output_dir}")
        print(f"Training file: {train_file}")
        print(f"Test file: {test_file}")


if __name__ == "__main__":
    # Example usage
    loader = SVMRankDataLoader()
    
    # Create sample data for testing
    loader.create_sample_data("../data/sample")
    
    # Load the sample data
    data = loader.load_train_test_data("../data/sample", "../data/sample")
    
    print("Data loaded successfully!")