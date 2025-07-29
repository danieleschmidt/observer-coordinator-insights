"""Performance benchmarks for clustering operations."""

import pytest
import pandas as pd
from src.insights_clustering.clustering import perform_clustering


class TestClusteringPerformance:
    """Benchmark tests for clustering algorithms."""

    def test_small_dataset_clustering_performance(self, small_dataset: pd.DataFrame, benchmark):
        """Benchmark clustering performance on small dataset (100 employees)."""
        result = benchmark(perform_clustering, small_dataset, n_clusters=4)
        
        assert result is not None
        assert len(result) == len(small_dataset)

    def test_medium_dataset_clustering_performance(self, medium_dataset: pd.DataFrame, benchmark):
        """Benchmark clustering performance on medium dataset (1K employees)."""
        result = benchmark(perform_clustering, medium_dataset, n_clusters=8)
        
        assert result is not None
        assert len(result) == len(medium_dataset)

    def test_large_dataset_clustering_performance(self, large_dataset: pd.DataFrame, benchmark):
        """Benchmark clustering performance on large dataset (10K employees)."""
        result = benchmark(perform_clustering, large_dataset, n_clusters=12)
        
        assert result is not None
        assert len(result) == len(large_dataset)

    def test_clustering_memory_efficiency(self, medium_dataset: pd.DataFrame, benchmark):
        """Test memory efficiency of clustering algorithm."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = benchmark(perform_clustering, medium_dataset, n_clusters=6)
        
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 100MB for 1K records)
        assert memory_growth < 100 * 1024 * 1024  # 100MB
        assert result is not None

    @pytest.mark.parametrize("n_clusters", [2, 4, 8, 12, 16])
    def test_clustering_scalability_by_clusters(self, medium_dataset: pd.DataFrame, n_clusters: int, benchmark):
        """Test how clustering performance scales with number of clusters."""
        result = benchmark(perform_clustering, medium_dataset, n_clusters=n_clusters)
        
        assert result is not None
        assert len(set(result)) <= n_clusters  # Should not exceed requested clusters

    @pytest.mark.slow
    def test_clustering_batch_processing(self, large_dataset: pd.DataFrame, benchmark):
        """Test batch processing performance for large datasets."""
        batch_size = 1000
        batches = [
            large_dataset[i:i + batch_size] 
            for i in range(0, len(large_dataset), batch_size)
        ]
        
        def process_batches():
            results = []
            for batch in batches:
                result = perform_clustering(batch, n_clusters=4)
                results.append(result)
            return results
        
        results = benchmark(process_batches)
        
        assert len(results) == len(batches)
        assert all(len(result) > 0 for result in results)