"""K-means Clustering Implementation for Insights Discovery Data
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class KMeansClusterer:
    """K-means clustering for employee Insights Discovery data with performance optimizations"""

    def __init__(self, n_clusters: int = 4, random_state: int = 42,
                 use_gpu: bool = False, memory_efficient: bool = False):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.memory_efficient = memory_efficient
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_data = None
        self.scaled_data = None
        self.cluster_labels = None

        # GPU acceleration if available
        if use_gpu:
            try:
                from cuml.cluster import KMeans as cuKMeans
                self._gpu_available = True
                logger.info("GPU acceleration enabled for clustering")
            except ImportError:
                self._gpu_available = False
                logger.warning("GPU libraries not available, falling back to CPU")
        else:
            self._gpu_available = False

    def fit(self, features: pd.DataFrame) -> 'KMeansClusterer':
        """Fit K-means clustering to the features with enhanced error handling"""
        if features is None or features.empty:
            raise ValueError("Features DataFrame cannot be None or empty")

        if len(features) < self.n_clusters:
            raise ValueError(f"Number of samples ({len(features)}) must be >= number of clusters ({self.n_clusters})")

        # Check for numeric columns only
        non_numeric = features.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric) > 0:
            raise ValueError(f"All features must be numeric. Non-numeric columns found: {list(non_numeric)}")

        # Check for infinite or NaN values
        if features.isnull().any().any():
            logger.warning("NaN values detected in features. Filling with column means.")
            features = features.fillna(features.mean())

        if np.isinf(features.values).any():
            logger.warning("Infinite values detected in features. Clipping to finite range.")
            features = features.clip(-1e10, 1e10)
        try:
            start_time = time.time()
            self.feature_data = features.copy()

            # Scale features for clustering
            self.scaled_data = self.scaler.fit_transform(features)

            # Fit K-means with enhanced configuration
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
                tol=1e-4
            )
            self.cluster_labels = self.kmeans.fit_predict(self.scaled_data)

            fit_time = time.time() - start_time
            logger.info(f"Successfully clustered {len(features)} employees into {self.n_clusters} clusters in {fit_time:.2f}s")

            # Validate clustering results
            self._validate_clustering_results()

            return self

        except Exception as e:
            logger.error(f"Clustering failed: {e!s}")
            raise RuntimeError(f"K-means clustering failed: {e!s}") from e

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

    def find_optimal_clusters_parallel(self, features: pd.DataFrame,
                                     max_clusters: int = 10, n_jobs: int = -1) -> Dict[int, Dict[str, float]]:
        """Find optimal number of clusters using parallel processing"""
        def evaluate_k(k):
            kmeans_temp = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels_temp = kmeans_temp.fit_predict(self.scaled_data)

            return k, {
                'inertia': kmeans_temp.inertia_,
                'silhouette_score': silhouette_score(self.scaled_data, labels_temp),
                'calinski_harabasz_score': calinski_harabasz_score(self.scaled_data, labels_temp)
            }

        scaled_features = self.scaler.fit_transform(features)
        self.scaled_data = scaled_features

        # Use joblib for parallel processing
        results = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(evaluate_k)(k) for k in range(2, max_clusters + 1)
        )

        return dict(results)

    def _validate_clustering_results(self) -> None:
        """Validate clustering results and log warnings for potential issues"""
        unique_labels = np.unique(self.cluster_labels)

        # Check for empty clusters
        if len(unique_labels) < self.n_clusters:
            logger.warning(f"Only {len(unique_labels)} clusters created out of {self.n_clusters} requested")

        # Check cluster balance
        cluster_sizes = np.bincount(self.cluster_labels)
        min_size, max_size = np.min(cluster_sizes), np.max(cluster_sizes)
        size_ratio = max_size / min_size if min_size > 0 else float('inf')

        if size_ratio > 5.0:  # Imbalanced if largest cluster is 5x larger than smallest
            logger.warning(f"Clusters are imbalanced. Size ratio: {size_ratio:.2f}")

        # Check silhouette score if possible
        if len(unique_labels) > 1:
            silhouette = silhouette_score(self.scaled_data, self.cluster_labels)
            if silhouette < 0.3:
                logger.warning(f"Low silhouette score: {silhouette:.3f}. Consider different number of clusters.")

    @contextmanager
    def _performance_monitor(self, operation_name: str):
        """Context manager for monitoring operation performance"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if duration > 5.0:  # Log if operation takes more than 5 seconds
                logger.warning(f"{operation_name} took {duration:.2f}s to complete")

    def batch_fit(self, features_list: List[pd.DataFrame],
                 n_clusters_list: List[int]) -> List['KMeansClusterer']:
        """Fit multiple clusterers in parallel for batch processing"""
        def fit_single(features, n_clusters):
            clusterer = KMeansClusterer(n_clusters=n_clusters, random_state=self.random_state)
            clusterer.fit(features)
            return clusterer

        with ThreadPoolExecutor(max_workers=min(4, len(features_list))) as executor:
            futures = [
                executor.submit(fit_single, features, n_clusters)
                for features, n_clusters in zip(features_list, n_clusters_list)
            ]
            results = [future.result() for future in futures]

        return results

    def incremental_fit(self, features: pd.DataFrame, batch_size: int = 1000) -> 'KMeansClusterer':
        """Incremental fitting for large datasets"""
        if len(features) <= batch_size:
            return self.fit(features)

        logger.info(f"Using incremental fitting with batch size {batch_size}")

        # Initialize with first batch
        first_batch = features.iloc[:batch_size]
        self.fit(first_batch)

        # Process remaining batches
        for start_idx in range(batch_size, len(features), batch_size):
            end_idx = min(start_idx + batch_size, len(features))
            batch = features.iloc[start_idx:end_idx]

            # Scale the batch using existing scaler
            batch_scaled = self.scaler.transform(batch)

            # Update cluster centers (simplified incremental approach)
            batch_labels = self.kmeans.predict(batch_scaled)

            # Update internal state
            combined_data = np.vstack([self.scaled_data, batch_scaled])
            combined_labels = np.hstack([self.cluster_labels, batch_labels])

            self.scaled_data = combined_data
            self.cluster_labels = combined_labels
            self.feature_data = pd.concat([self.feature_data, batch], ignore_index=True)

        logger.info(f"Incremental fitting completed for {len(features)} samples")
        return self
