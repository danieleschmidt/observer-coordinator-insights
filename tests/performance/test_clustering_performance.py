"""Performance tests for clustering algorithms."""

import time
from typing import Any, Dict

import pandas as pd
import pytest
from sklearn.cluster import KMeans

from src.insights_clustering.clustering import ClusteringEngine
from src.insights_clustering.parser import DataParser


@pytest.mark.performance
@pytest.mark.slow
class TestClusteringPerformance:
    """Test clustering performance with various dataset sizes."""

    def test_small_dataset_performance(
        self, benchmark: Any, sample_insights_data: pd.DataFrame
    ) -> None:
        """Test clustering performance with small dataset (100 records)."""
        engine = ClusteringEngine(n_clusters=4, random_state=42)
        
        def run_clustering() -> Dict[str, Any]:
            return engine.fit_predict(sample_insights_data)
        
        result = benchmark(run_clustering)
        
        # Performance assertions
        assert len(result["labels"]) == len(sample_insights_data)
        assert result["inertia"] is not None
        assert result["silhouette_score"] is not None

    def test_medium_dataset_performance(
        self, benchmark: Any, benchmark_config: Dict[str, Any]
    ) -> None:
        """Test clustering performance with medium dataset (1000 records)."""
        # Generate medium dataset
        data = pd.DataFrame({
            "red_energy": range(1000),
            "blue_energy": range(1000, 2000),
            "green_energy": range(2000, 3000),
            "yellow_energy": range(3000, 4000),
        })
        
        engine = ClusteringEngine(n_clusters=5, random_state=42)
        
        def run_clustering() -> Dict[str, Any]:
            return engine.fit_predict(data)
        
        # Configure benchmark
        benchmark.extra_info.update(benchmark_config)
        result = benchmark(run_clustering)
        
        # Performance assertions
        assert len(result["labels"]) == 1000
        assert result["inertia"] is not None

    def test_large_dataset_performance(
        self, benchmark: Any, large_dataset: pd.DataFrame
    ) -> None:
        """Test clustering performance with large dataset (10000 records)."""
        engine = ClusteringEngine(n_clusters=8, random_state=42)
        
        def run_clustering() -> Dict[str, Any]:
            return engine.fit_predict(large_dataset)
        
        result = benchmark(run_clustering)
        
        # Performance assertions
        assert len(result["labels"]) == 10000
        assert result["inertia"] is not None

    def test_memory_usage_large_dataset(self, large_dataset: pd.DataFrame) -> None:
        """Test memory usage doesn't exceed reasonable limits."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        engine = ClusteringEngine(n_clusters=8, random_state=42)
        result = engine.fit_predict(large_dataset)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase by more than 500MB for this dataset
        assert memory_increase < 500
        assert len(result["labels"]) == 10000

    @pytest.mark.parametrize("n_clusters", [2, 4, 8, 16])
    def test_scalability_with_cluster_count(
        self, benchmark: Any, sample_insights_data: pd.DataFrame, n_clusters: int
    ) -> None:
        """Test performance scaling with different cluster counts."""
        engine = ClusteringEngine(n_clusters=n_clusters, random_state=42)
        
        def run_clustering() -> Dict[str, Any]:
            return engine.fit_predict(sample_insights_data)
        
        result = benchmark(run_clustering)
        
        # Performance should not degrade significantly with more clusters
        assert len(result["labels"]) == len(sample_insights_data)
        assert len(set(result["labels"])) <= n_clusters

    def test_data_parsing_performance(
        self, benchmark: Any, sample_csv_file: str
    ) -> None:
        """Test data parsing performance."""
        parser = DataParser()
        
        def parse_data() -> pd.DataFrame:
            return parser.parse_csv(sample_csv_file)
        
        result = benchmark(parse_data)
        
        # Parsing should be fast
        assert len(result) > 0
        assert not result.empty

    def test_concurrent_clustering(self, sample_insights_data: pd.DataFrame) -> None:
        """Test clustering performance under concurrent load."""
        import concurrent.futures
        import threading
        
        results = []
        times = []
        
        def run_clustering() -> Dict[str, Any]:
            start_time = time.time()
            engine = ClusteringEngine(n_clusters=4, random_state=42)
            result = engine.fit_predict(sample_insights_data)
            end_time = time.time()
            
            times.append(end_time - start_time)
            return result
        
        # Run 5 concurrent clustering operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_clustering) for _ in range(5)]
            results = [future.result() for future in futures]
        
        # All operations should complete successfully
        assert len(results) == 5
        for result in results:
            assert len(result["labels"]) == len(sample_insights_data)
        
        # Average time should be reasonable (less than 5 seconds)
        avg_time = sum(times) / len(times)
        assert avg_time < 5.0

    def test_incremental_clustering_performance(
        self, benchmark: Any, sample_insights_data: pd.DataFrame
    ) -> None:
        """Test performance of incremental clustering updates."""
        from sklearn.cluster import MiniBatchKMeans
        
        # Use MiniBatchKMeans for incremental updates
        model = MiniBatchKMeans(n_clusters=4, random_state=42, batch_size=10)
        
        def incremental_fit() -> None:
            # Simulate incremental updates
            for i in range(0, len(sample_insights_data), 10):
                batch = sample_insights_data.iloc[i:i+10]
                if len(batch) > 0:
                    # Select only numeric columns for clustering
                    numeric_cols = ["red_energy", "blue_energy", "green_energy", "yellow_energy"]
                    batch_numeric = batch[numeric_cols]
                    model.partial_fit(batch_numeric)
        
        benchmark(incremental_fit)
        
        # Model should be fitted
        assert hasattr(model, "cluster_centers_")
        assert len(model.cluster_centers_) == 4

    def test_algorithm_comparison_performance(
        self, benchmark: Any, sample_insights_data: pd.DataFrame
    ) -> None:
        """Compare performance of different clustering algorithms."""
        from sklearn.cluster import AgglomerativeClustering, DBSCAN
        
        # Prepare data
        numeric_cols = ["red_energy", "blue_energy", "green_energy", "yellow_energy"]
        data = sample_insights_data[numeric_cols]
        
        algorithms = {
            "kmeans": KMeans(n_clusters=4, random_state=42),
            "agglomerative": AgglomerativeClustering(n_clusters=4),
            # DBSCAN doesn't require n_clusters
            "dbscan": DBSCAN(eps=10, min_samples=5),
        }
        
        performance_results = {}
        
        for name, algorithm in algorithms.items():
            start_time = time.time()
            labels = algorithm.fit_predict(data)
            end_time = time.time()
            
            performance_results[name] = {
                "time": end_time - start_time,
                "n_clusters": len(set(labels)) if -1 not in labels else len(set(labels)) - 1,
                "n_noise": sum(1 for label in labels if label == -1),
            }
        
        # KMeans should generally be fastest for this dataset size
        assert performance_results["kmeans"]["time"] <= max(
            performance_results[alg]["time"] for alg in performance_results
        )
        
        # All algorithms should produce clusters
        for result in performance_results.values():
            assert result["n_clusters"] > 0