"""
CSV Data Loader
Handles loading and preprocessing of CSV/tabular datasets.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class CSVDataLoader:
    """Loader for CSV/tabular datasets."""
    
    def __init__(self, 
                 target_column: Optional[str] = None,
                 handle_missing: str = 'mean',
                 scale_features: bool = True):
        """
        Initialize CSV loader.
        
        Args:
            target_column: Name of target column (if None, assumes last column)
            handle_missing: Strategy for handling missing values ('mean', 'median', 'most_frequent', 'drop')
            scale_features: Whether to scale/normalize features
        """
        self.target_column = target_column
        self.handle_missing = handle_missing
        self.scale_features = scale_features
    
    def load_from_uploaded_file(self, uploaded_file) -> Dict[str, Any]:
        """
        Load CSV data from uploaded file.
        
        Args:
            uploaded_file: Uploaded file object
            
        Returns:
            Dataset information dictionary
        """
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Basic validation
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Identify target column
            if self.target_column:
                if self.target_column not in df.columns:
                    raise ValueError(f"Target column '{self.target_column}' not found")
                target_col = self.target_column
            else:
                # Assume last column is target
                target_col = df.columns[-1]
            
            # Separate features and target
            feature_columns = [col for col in df.columns if col != target_col]
            X = df[feature_columns]
            y = df[target_col]
            
            # Handle missing values in features
            if self.handle_missing == 'drop':
                # Drop rows with any missing values
                mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[mask]
                y = y[mask]
            else:
                # Impute missing values
                numeric_columns = X.select_dtypes(include=[np.number]).columns
                categorical_columns = X.select_dtypes(exclude=[np.number]).columns
                
                # Handle numeric columns
                if len(numeric_columns) > 0:
                    numeric_imputer = SimpleImputer(strategy=self.handle_missing)
                    X[numeric_columns] = numeric_imputer.fit_transform(X[numeric_columns])
                
                # Handle categorical columns
                if len(categorical_columns) > 0:
                    categorical_imputer = SimpleImputer(strategy='most_frequent')
                    X[categorical_columns] = categorical_imputer.fit_transform(X[categorical_columns])
                
                # Handle missing target values
                y = y.dropna()
                X = X.loc[y.index]
            
            # Convert categorical features to numeric
            categorical_columns = X.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
            
            # Scale features if requested
            if self.scale_features:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Convert to numpy arrays
            X_array = X.values.astype(np.float32)
            y_array = y.values
            
            # Encode target if categorical
            if y.dtype == 'object' or isinstance(y.iloc[0], str):
                target_encoder = LabelEncoder()
                y_encoded = target_encoder.fit_transform(y_array)
            else:
                y_encoded = y_array
                target_encoder = None
            
            # Compute statistics
            feature_stats = self._compute_feature_statistics(X, y)
            
            # Get class distribution (if classification)
            if target_encoder:
                unique_targets, counts = np.unique(y_encoded, return_counts=True)
                class_distribution = dict(zip(target_encoder.classes_[unique_targets], counts.astype(int)))
            else:
                class_distribution = {}
            
            # Save processed data
            dataset_id = f"csv_{len(X_array)}_{X_array.shape[1]}"
            save_dir = f"data/{dataset_id}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save arrays
            np.save(f"{save_dir}/features.npy", X_array)
            np.save(f"{save_dir}/targets.npy", y_encoded)
            
            # Save metadata
            metadata = {
                'feature_columns': feature_columns,
                'target_column': target_col,
                'target_encoder_classes': target_encoder.classes_.tolist() if target_encoder else None,
                'feature_statistics': feature_stats
            }
            
            with open(f"{save_dir}/metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'type': 'csv',
                'dataset_id': dataset_id,
                'data_path': save_dir,
                'num_samples': len(X_array),
                'num_features': X_array.shape[1],
                'target_column': target_col,
                'class_distribution': class_distribution,
                'statistics': feature_stats,
                'preview_data': df.head(10)
            }
        
        except Exception as e:
            raise ValueError(f"Failed to process CSV file: {e}")
    
    def _compute_feature_statistics(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Compute feature statistics.
        
        Args:
            X: Feature dataframe
            y: Target series
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Basic statistics
        stats['basic'] = {
            'num_features': len(X.columns),
            'num_samples': len(X),
            'missing_values_per_column': X.isnull().sum().to_dict()
        }
        
        # Feature statistics
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        if len(numeric_features) > 0:
            stats['numeric_features'] = {}
            for col in numeric_features:
                stats['numeric_features'][col] = {
                    'mean': float(X[col].mean()),
                    'std': float(X[col].std()),
                    'min': float(X[col].min()),
                    'max': float(X[col].max()),
                    'median': float(X[col].median()),
                    'missing_count': int(X[col].isnull().sum())
                }
            
            # Correlation matrix
            if len(numeric_features) > 1:
                corr_matrix = X[numeric_features].corr()
                stats['correlation_matrix'] = corr_matrix.to_dict()
        
        return stats
    
    def load_processed_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load processed CSV data from disk.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Tuple of (X, y, metadata)
        """
        X = np.load(f"{data_path}/features.npy")
        y = np.load(f"{data_path}/targets.npy")
        
        with open(f"{data_path}/metadata.json") as f:
            metadata = json.load(f)
        
        return X, y, metadata
    
    def analyze_dataset(self, data_path: str) -> Dict[str, Any]:
        """
        Analyze CSV dataset and compute rich statistics.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Analysis results
        """
        X, y, metadata = self.load_processed_data(data_path)
        
        # Basic statistics
        stats = {
            'num_samples': len(X),
            'num_features': X.shape[1],
            'feature_columns': metadata['feature_columns'],
            'target_column': metadata['target_column']
        }
        
        # Feature statistics
        stats['feature_statistics'] = {}
        for i, col_name in enumerate(metadata['feature_columns']):
            feature_data = X[:, i]
            stats['feature_statistics'][col_name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data)),
                'percentile_25': float(np.percentile(feature_data, 25)),
                'percentile_75': float(np.percentile(feature_data, 75))
            }
        
        # Target statistics
        if metadata.get('target_encoder_classes'):
            # Classification target
            unique_targets, counts = np.unique(y, return_counts=True)
            class_names = [metadata['target_encoder_classes'][i] for i in unique_targets]
            stats['target_distribution'] = dict(zip(class_names, counts.astype(int)))
        else:
            # Regression target
            stats['target_statistics'] = {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y)),
                'median': float(np.median(y))
            }
        
        # Correlation analysis
        if X.shape[1] > 1:
            # Feature correlation matrix
            corr_matrix = np.corrcoef(X.T)
            stats['feature_correlation'] = corr_matrix.tolist()
            
            # Feature-target correlation
            if len(y.shape) == 1:  # Only for 1D targets
                feature_target_corr = []
                for i in range(X.shape[1]):
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr):
                        feature_target_corr.append(float(corr))
                    else:
                        feature_target_corr.append(0.0)
                
                stats['feature_target_correlation'] = feature_target_corr
        
        # Missing value analysis
        stats['missing_value_analysis'] = {
            'total_missing': int(np.sum(np.isnan(X))),
            'missing_percentage': float(np.sum(np.isnan(X)) / X.size * 100)
        }
        
        return stats
