"""
K-means Clustering Implementation for Insights Discovery Data
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """K-means clustering for employee Insights Discovery data"""
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_data = None
        self.scaled_data = None
        self.cluster_labels = None
        
    def fit(self, features: pd.DataFrame) -> 'KMeansClusterer':
        """Fit K-means clustering to the features"""
        self.feature_data = features.copy()
        
        # Scale features for clustering
        self.scaled_data = self.scaler.fit_transform(features)
        
        # Fit K-means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)
        
        logger.info(f"Clustered {len(features)} employees into {self.n_clusters} clusters")
        return self
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict cluster assignments for new data"""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        scaled_features = self.scaler.transform(features)
        return self.kmeans.predict(scaled_features)
    
    def get_cluster_assignments(self) -> np.ndarray:
        """Get cluster assignments for training data"""
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.cluster_labels
    
    def get_cluster_centroids(self) -> pd.DataFrame:
        """Get cluster centroids in original feature space"""
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Transform centroids back to original scale
        centroids_scaled = self.kmeans.cluster_centers_
        centroids_original = self.scaler.inverse_transform(centroids_scaled)
        
        return pd.DataFrame(
            centroids_original,
            columns=self.feature_data.columns,
            index=[f'Cluster_{i}' for i in range(self.n_clusters)]
        )
    
    def get_cluster_quality_metrics(self) -> Dict[str, float]:
        """Calculate cluster quality metrics"""
        if self.cluster_labels is None or self.scaled_data is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        metrics = {}
        
        # Silhouette score (higher is better, range: -1 to 1)
        if len(np.unique(self.cluster_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(
                self.scaled_data, self.cluster_labels
            )
        
        # Calinski-Harabasz score (higher is better)
        if len(np.unique(self.cluster_labels)) > 1:
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                self.scaled_data, self.cluster_labels
            )
        
        # Inertia (within-cluster sum of squares, lower is better)
        metrics['inertia'] = self.kmeans.inertia_
        
        # Cluster size distribution
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        metrics['cluster_size_std'] = np.std(counts)  # Lower is more balanced
        
        return metrics
    
    def get_cluster_summary(self) -> pd.DataFrame:
        """Get summary statistics for each cluster"""
        if self.feature_data is None or self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Add cluster labels to feature data
        data_with_clusters = self.feature_data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        # Calculate cluster statistics
        summary = data_with_clusters.groupby('cluster').agg({
            col: ['mean', 'std', 'count'] for col in self.feature_data.columns
        }).round(2)
        
        return summary
    
    def find_optimal_clusters(self, features: pd.DataFrame, 
                            max_clusters: int = 10) -> Dict[int, Dict[str, float]]:
        """Find optimal number of clusters using elbow method and silhouette analysis"""
        scores = {}
        
        scaled_features = self.scaler.fit_transform(features)
        
        for k in range(2, max_clusters + 1):
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels_temp = kmeans_temp.fit_predict(scaled_features)
            
            scores[k] = {
                'inertia': kmeans_temp.inertia_,
                'silhouette_score': silhouette_score(scaled_features, labels_temp),
                'calinski_harabasz_score': calinski_harabasz_score(scaled_features, labels_temp)
            }
        
        return scores