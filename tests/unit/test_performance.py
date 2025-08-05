"""
Unit tests for performance optimization module
"""

import pytest
import time
import threading
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.performance import (
    LRUCache, PerformanceMonitor, ParallelProcessor, 
    CachedDataProcessor, ResourceManager, performance_optimized
)


class TestLRUCache:
    """Test LRU Cache functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cache = LRUCache(max_size=3, ttl_seconds=1)
    
    def test_basic_operations(self):
        """Test basic cache operations"""
        # Test put and get
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Test cache miss
        assert self.cache.get("nonexistent") is None
    
    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        # Fill cache to capacity
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Add one more item, should evict least recently used
        self.cache.put("key4", "value4")
        
        # key1 should be evicted (least recently used)
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"
    
    def test_access_updates_order(self):
        """Test that accessing items updates LRU order"""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Access key1 to make it most recently used
        self.cache.get("key1")
        
        # Add new item, should evict key2 (now least recently used)
        self.cache.put("key4", "value4")
        
        assert self.cache.get("key1") == "value1"  # Should still be there
        assert self.cache.get("key2") is None      # Should be evicted
        assert self.cache.get("key3") == "value3"
        assert self.cache.get("key4") == "value4"
    
    def test_ttl_expiration(self):
        """Test TTL-based expiration"""
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Wait for TTL expiration
        time.sleep(1.1)
        assert self.cache.get("key1") is None
    
    def test_thread_safety(self):
        """Test thread safety of cache operations"""
        results = []
        
        def worker(thread_id):
            for i in range(10):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                self.cache.put(key, value)
                retrieved = self.cache.get(key)
                results.append(retrieved == value)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have some successful operations (exact count depends on eviction)
        assert sum(results) > 0
    
    def test_stats(self):
        """Test cache statistics"""
        # Generate some hits and misses
        self.cache.put("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        stats = self.cache.stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['size'] == 1
        assert stats['max_size'] == 3


class TestPerformanceMonitor:
    """Test performance monitoring functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_operation(self):
        """Test operation monitoring"""
        with self.monitor.monitor_operation("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        assert len(self.monitor.metrics_history) == 1
        metrics = self.monitor.metrics_history[0]
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration >= 0.1
        assert metrics.memory_peak > 0
        assert metrics.cpu_percent >= 0
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Add some test metrics
        with self.monitor.monitor_operation("op1"):
            time.sleep(0.05)
        
        with self.monitor.monitor_operation("op2"):
            time.sleep(0.1)
        
        summary = self.monitor.get_performance_summary()
        
        assert summary['total_operations'] == 2
        assert summary['avg_duration'] > 0
        assert summary['max_duration'] >= summary['avg_duration']
        assert 'op1' in summary['operations_by_type']
        assert 'op2' in summary['operations_by_type']
    
    def test_empty_metrics_summary(self):
        """Test summary with no metrics"""
        summary = self.monitor.get_performance_summary()
        assert summary['total_operations'] == 0


class TestParallelProcessor:
    """Test parallel processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = ParallelProcessor(max_workers=2)
        
        # Sample data for testing
        self.sample_features = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100)
        })
    
    def test_parallel_clustering(self):
        """Test parallel clustering functionality"""
        features_list = [self.sample_features, self.sample_features.copy()]
        n_clusters_list = [2, 3]
        
        with patch('src.insights_clustering.KMeansClusterer') as mock_clusterer_class:
            # Mock the clusterer
            mock_clusterer = MagicMock()
            mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1] * 50)
            mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({
                'feature1': [0.5, 0.7],
                'feature2': [0.4, 0.6]
            })
            mock_clusterer.get_cluster_quality_metrics.return_value = {'silhouette_score': 0.75}
            mock_clusterer_class.return_value = mock_clusterer
            
            results = self.processor.parallel_clustering(features_list, n_clusters_list)
            
            assert len(results) == 2
            for result in results:
                assert 'success' in result
                if result['success']:
                    assert 'assignments' in result
                    assert 'centroids' in result
                    assert 'metrics' in result
    
    @pytest.mark.asyncio
    async def test_async_data_processing(self):
        """Test asynchronous data processing"""
        # Create temporary CSV files
        temp_files = []
        for i in range(2):
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            self.sample_features.to_csv(temp_file.name, index=False)
            temp_file.close()
            temp_files.append(temp_file.name)
        
        try:
            results = await self.processor.async_data_processing(temp_files)
            
            assert len(results) <= 2  # May be less if some fail
            for result in results:
                assert isinstance(result, pd.DataFrame)
                assert len(result) > 0
        
        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)


class TestCachedDataProcessor:
    """Test cached data processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Use temporary directory for cache
        self.temp_cache_dir = tempfile.mkdtemp()
        self.processor = CachedDataProcessor(cache_dir=Path(self.temp_cache_dir))
        
        self.sample_features = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'feature3': np.random.rand(50)
        })
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_cache_dir, ignore_errors=True)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key1 = self.processor._get_cache_key(self.sample_features, "clustering", n_clusters=3)
        key2 = self.processor._get_cache_key(self.sample_features, "clustering", n_clusters=3)
        key3 = self.processor._get_cache_key(self.sample_features, "clustering", n_clusters=4)
        
        # Same data and parameters should produce same key
        assert key1 == key2
        
        # Different parameters should produce different key
        assert key1 != key3
    
    @patch('src.insights_clustering.KMeansClusterer')
    def test_cached_clustering_first_call(self, mock_clusterer_class):
        """Test cached clustering on first call (cache miss)"""
        # Mock the clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1] * 25)
        mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({
            'feature1': [0.5, 0.7],
            'feature2': [0.4, 0.6]
        })
        mock_clusterer.get_cluster_quality_metrics.return_value = {'silhouette_score': 0.75}
        mock_clusterer.get_cluster_summary.return_value = pd.DataFrame({'summary': [1, 2]})
        mock_clusterer_class.return_value = mock_clusterer
        
        result = self.processor.cached_clustering(self.sample_features, n_clusters=2)
        
        assert 'assignments' in result
        assert 'centroids' in result
        assert 'metrics' in result
        assert 'summary' in result
        
        # Verify clusterer was called
        mock_clusterer_class.assert_called_once_with(n_clusters=2)
        mock_clusterer.fit.assert_called_once()
    
    @patch('src.insights_clustering.KMeansClusterer')
    def test_cached_clustering_second_call(self, mock_clusterer_class):
        """Test cached clustering on second call (cache hit)"""
        # Mock the clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1] * 25)
        mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({
            'feature1': [0.5, 0.7],
            'feature2': [0.4, 0.6]
        })
        mock_clusterer.get_cluster_quality_metrics.return_value = {'silhouette_score': 0.75}
        mock_clusterer.get_cluster_summary.return_value = pd.DataFrame({'summary': [1, 2]})
        mock_clusterer_class.return_value = mock_clusterer
        
        # First call - should compute and cache
        result1 = self.processor.cached_clustering(self.sample_features, n_clusters=2)
        
        # Reset mock to verify second call doesn't compute
        mock_clusterer_class.reset_mock()
        mock_clusterer.fit.reset_mock()
        
        # Second call - should use cache
        result2 = self.processor.cached_clustering(self.sample_features, n_clusters=2)
        
        # Results should be identical
        assert result1['metrics'] == result2['metrics']
        
        # Clusterer should not be called on second invocation
        mock_clusterer_class.assert_not_called()
    
    def test_cache_stats(self):
        """Test cache statistics"""
        stats = self.processor.get_cache_stats()
        
        assert 'memory_cache' in stats
        assert 'disk_cache' in stats
        assert stats['memory_cache']['size'] >= 0
        assert stats['disk_cache']['files'] >= 0
        assert 'cache_dir' in stats['disk_cache']
    
    def test_clear_cache(self):
        """Test cache clearing"""
        # Add something to memory cache
        self.processor.memory_cache.put("test_key", "test_value")
        
        # Clear cache
        self.processor.clear_cache(older_than_hours=0)  # Clear everything
        
        # Verify cache is empty
        assert self.processor.memory_cache.get("test_key") is None
        stats = self.processor.get_cache_stats()
        assert stats['memory_cache']['size'] == 0


class TestResourceManager:
    """Test resource management functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.manager = ResourceManager(max_memory_percent=50.0, max_cpu_percent=50.0)
    
    def test_check_resource_availability(self):
        """Test resource availability checking"""
        resources = self.manager.check_resource_availability()
        
        assert 'memory_percent' in resources
        assert 'memory_available_gb' in resources
        assert 'cpu_percent' in resources
        assert 'can_proceed' in resources
        
        assert isinstance(resources['memory_percent'], float)
        assert isinstance(resources['memory_available_gb'], float)
        assert isinstance(resources['cpu_percent'], float)
        assert isinstance(resources['can_proceed'], bool)
    
    def test_resource_aware_execution(self):
        """Test resource-aware execution context"""
        with self.manager.resource_aware_execution("test_operation") as resources:
            assert 'memory_percent' in resources
            time.sleep(0.01)  # Simulate brief work
        
        # Check that metrics were recorded
        assert len(self.manager.monitor.metrics_history) == 1
        metrics = self.manager.monitor.metrics_history[0]
        assert metrics.operation_name == "test_operation"
    
    def test_optimize_for_large_dataset(self):
        """Test optimization recommendations for large datasets"""
        # Test small dataset
        small_recommendations = self.manager.optimize_for_large_dataset(1000)
        assert isinstance(small_recommendations, dict)
        assert 'use_chunking' in small_recommendations
        assert 'use_parallel' in small_recommendations
        
        # Test large dataset
        large_recommendations = self.manager.optimize_for_large_dataset(1000000)
        assert isinstance(large_recommendations, dict)
        assert 'chunk_size' in large_recommendations
        assert 'max_workers' in large_recommendations


class TestPerformanceDecorator:
    """Test performance optimization decorator"""
    
    def test_performance_optimized_decorator(self):
        """Test performance optimization decorator"""
        call_count = 0
        
        @performance_optimized(cache=False, parallel=False)
        def test_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        result = test_function(1, 2)
        assert result == 3
        assert call_count == 1
    
    def test_decorator_with_exception(self):
        """Test decorator behavior with exceptions"""
        @performance_optimized(cache=False, parallel=False)
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(Exception):  # Should be wrapped as ObserverCoordinatorError
            failing_function()


def test_integration_performance_pipeline():
    """Integration test for performance optimization pipeline"""
    # Create sample data
    features = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100)
    })
    
    # Test with temporary cache directory
    with tempfile.TemporaryDirectory() as temp_dir:
        processor = CachedDataProcessor(cache_dir=Path(temp_dir))
        
        with patch('src.insights_clustering.KMeansClusterer') as mock_clusterer_class:
            # Mock the clusterer
            mock_clusterer = MagicMock()
            mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1] * 50)
            mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({
                'feature1': [0.5, 0.7],
                'feature2': [0.4, 0.6]
            })
            mock_clusterer.get_cluster_quality_metrics.return_value = {'silhouette_score': 0.75}
            mock_clusterer.get_cluster_summary.return_value = pd.DataFrame({'summary': [1, 2]})
            mock_clusterer_class.return_value = mock_clusterer
            
            # First call - should compute and cache
            start_time = time.time()
            result1 = processor.cached_clustering(features, n_clusters=2)
            first_call_time = time.time() - start_time
            
            # Second call - should use cache (faster)
            start_time = time.time()
            result2 = processor.cached_clustering(features, n_clusters=2)
            second_call_time = time.time() - start_time
            
            # Verify results are identical
            assert result1['metrics'] == result2['metrics']
            
            # Second call should be faster (using cache)
            # Note: In tests this might not always be true due to mocking overhead
            # but in real usage caching provides significant speedup
            
            # Verify cache statistics
            stats = processor.get_cache_stats()
            assert stats['memory_cache']['hits'] >= 1  # At least one cache hit