"""
Performance optimization utilities for Observer Coordinator Insights
Handles caching, parallel processing, and resource management
"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import functools
import logging
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager
import psutil
import gc

logger = logging.getLogger(__name__)

# Import secure serializer from caching module
try:
    from .insights_clustering.caching import SecureSerializer
except ImportError:
    # Fallback if caching module not available
    class SecureSerializer:
        @staticmethod
        def serialize(obj):
            return json.dumps(obj, default=str).encode('utf-8')
        
        @staticmethod 
        def deserialize(data):
            return json.loads(data.decode('utf-8'))


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations"""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_peak: float
    cpu_percent: float
    cache_hits: int = 0
    cache_misses: int = 0


class LRUCache:
    """Thread-safe LRU cache with size and TTL limits"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.ttl_seconds
    
    def _evict_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache and not self._is_expired(key):
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.hits += 1
                return value
            
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any):
        """Put value in cache"""
        with self.lock:
            self._evict_expired()
            
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
                self.timestamps.pop(oldest_key, None)
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'utilization': len(self.cache) / self.max_size
            }


class PerformanceMonitor:
    """Monitor and track performance metrics"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_operation = None
        self.start_memory = None
        self.start_cpu = None
    
    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager for monitoring operation performance"""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_percent()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                memory_peak=max(start_memory, end_memory),
                cpu_percent=max(start_cpu, end_cpu)
            )
            
            self.metrics_history.append(metrics)
            
            if metrics.duration > 5.0:  # Log slow operations
                logger.warning(f"Slow operation: {operation_name} took {metrics.duration:.2f}s")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.metrics_history:
            return {'total_operations': 0}
        
        durations = [m.duration for m in self.metrics_history]
        memory_peaks = [m.memory_peak for m in self.metrics_history]
        
        return {
            'total_operations': len(self.metrics_history),
            'avg_duration': np.mean(durations),
            'max_duration': np.max(durations),
            'avg_memory_peak': np.mean(memory_peaks),
            'max_memory_peak': np.max(memory_peaks),
            'operations_by_type': {
                op_name: len([m for m in self.metrics_history if m.operation_name == op_name])
                for op_name in set(m.operation_name for m in self.metrics_history)
            }
        }


class ParallelProcessor:
    """Parallel processing utilities for CPU-intensive operations"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.monitor = PerformanceMonitor()
    
    def parallel_clustering(self, features_list: List[pd.DataFrame], 
                          n_clusters_list: List[int]) -> List[Dict[str, Any]]:
        """Perform clustering on multiple datasets in parallel"""
        from .insights_clustering import KMeansClusterer
        
        def cluster_single(args):
            features, n_clusters = args
            try:
                clusterer = KMeansClusterer(n_clusters=n_clusters)
                clusterer.fit(features)
                return {
                    'success': True,
                    'assignments': clusterer.get_cluster_assignments(),
                    'centroids': clusterer.get_cluster_centroids(),
                    'metrics': clusterer.get_cluster_quality_metrics()
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        with self.monitor.monitor_operation('parallel_clustering'):
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                args_list = list(zip(features_list, n_clusters_list))
                results = list(executor.map(cluster_single, args_list))
        
        return results
    
    def parallel_team_generation(self, employee_data_list: List[pd.DataFrame],
                                cluster_assignments_list: List[np.ndarray],
                                team_counts: List[int]) -> List[Dict[str, Any]]:
        """Generate teams for multiple scenarios in parallel"""
        from .team_simulator import TeamCompositionSimulator
        
        def generate_teams_single(args):
            employee_data, cluster_assignments, num_teams = args
            try:
                simulator = TeamCompositionSimulator()
                simulator.load_employee_data(employee_data, cluster_assignments)
                compositions = simulator.recommend_optimal_teams(num_teams, iterations=3)
                return {
                    'success': True,
                    'compositions': compositions
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        with self.monitor.monitor_operation('parallel_team_generation'):
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                args_list = list(zip(employee_data_list, cluster_assignments_list, team_counts))
                results = list(executor.map(generate_teams_single, args_list))
        
        return results
    
    async def async_data_processing(self, data_sources: List[str]) -> List[pd.DataFrame]:
        """Asynchronously process multiple data sources"""
        
        async def load_single_source(source_path: str) -> pd.DataFrame:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, pd.read_csv, source_path)
        
        with self.monitor.monitor_operation('async_data_processing'):
            tasks = [load_single_source(source) for source in data_sources]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and return successful results
        successful_results = [r for r in results if isinstance(r, pd.DataFrame)]
        return successful_results


class CachedDataProcessor:
    """Data processor with intelligent caching"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.observer_coordinator_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = LRUCache(max_size=100, ttl_seconds=1800)  # 30 minutes
        self.monitor = PerformanceMonitor()
    
    def _get_cache_key(self, data: Any, operation: str, **kwargs) -> str:
        """Generate cache key from data and operation parameters"""
        # Create hash from data and parameters using SHA-256 for security
        if isinstance(data, pd.DataFrame):
            data_hash = hashlib.sha256(str(data.shape).encode() + 
                                     str(data.columns.tolist()).encode()).hexdigest()
        else:
            data_hash = hashlib.sha256(str(data).encode()).hexdigest()
        
        params_hash = hashlib.sha256(str(sorted(kwargs.items())).encode()).hexdigest()
        return f"{operation}_{data_hash}_{params_hash}"
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get disk cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _save_to_disk_cache(self, cache_key: str, result: Any):
        """Save result to disk cache"""
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                data = SecureSerializer.serialize(result)
                f.write(data)
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _load_from_disk_cache(self, cache_key: str) -> Optional[Any]:
        """Load result from disk cache"""
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            if cache_path.exists():
                # Check if file is not too old (24 hours)
                if time.time() - cache_path.stat().st_mtime < 86400:
                    with open(cache_path, 'rb') as f:
                        data = f.read()
                        return SecureSerializer.deserialize(data)
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
        return None
    
    def cached_clustering(self, features: pd.DataFrame, n_clusters: int, 
                         **kwargs) -> Dict[str, Any]:
        """Perform clustering with caching"""
        cache_key = self._get_cache_key(features, 'clustering', 
                                      n_clusters=n_clusters, **kwargs)
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            logger.debug(f"Clustering result found in memory cache")
            return result
        
        # Try disk cache
        result = self._load_from_disk_cache(cache_key)
        if result is not None:
            logger.debug(f"Clustering result found in disk cache")
            self.memory_cache.put(cache_key, result)
            return result
        
        # Compute result
        with self.monitor.monitor_operation('cached_clustering'):
            from .insights_clustering import KMeansClusterer
            
            clusterer = KMeansClusterer(n_clusters=n_clusters, **kwargs)
            clusterer.fit(features)
            
            result = {
                'assignments': clusterer.get_cluster_assignments(),
                'centroids': clusterer.get_cluster_centroids(),
                'metrics': clusterer.get_cluster_quality_metrics(),
                'summary': clusterer.get_cluster_summary()
            }
        
        # Cache the result
        self.memory_cache.put(cache_key, result)
        self._save_to_disk_cache(cache_key, result)
        
        logger.debug(f"Clustering result computed and cached")
        return result
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear cache entries older than specified hours"""
        self.memory_cache.clear()
        
        # Clear disk cache
        cutoff_time = time.time() - (older_than_hours * 3600)
        for cache_file in self.cache_dir.glob('*.pkl'):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
        
        logger.info(f"Cleared cache entries older than {older_than_hours} hours")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_stats = self.memory_cache.stats()
        
        # Disk cache stats
        disk_files = list(self.cache_dir.glob('*.pkl'))
        total_disk_size = sum(f.stat().st_size for f in disk_files)
        
        return {
            'memory_cache': memory_stats,
            'disk_cache': {
                'files': len(disk_files),
                'total_size_mb': total_disk_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir)
            }
        }


class ResourceManager:
    """Manages system resources and prevents resource exhaustion"""
    
    def __init__(self, max_memory_percent: float = 80.0, max_cpu_percent: float = 90.0):
        self.max_memory_percent = max_memory_percent
        self.max_cpu_percent = max_cpu_percent
        self.monitor = PerformanceMonitor()
    
    def check_resource_availability(self) -> Dict[str, Any]:
        """Check current resource availability"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'cpu_percent': cpu_percent,
            'can_proceed': (
                memory.percent < self.max_memory_percent and 
                cpu_percent < self.max_cpu_percent
            )
        }
    
    @contextmanager
    def resource_aware_execution(self, operation_name: str):
        """Execute operation with resource monitoring"""
        resources = self.check_resource_availability()
        
        if not resources['can_proceed']:
            logger.warning(f"High resource usage detected before {operation_name}")
            logger.warning(f"Memory: {resources['memory_percent']:.1f}%, CPU: {resources['cpu_percent']:.1f}%")
            
            # Force garbage collection
            gc.collect()
            time.sleep(1)  # Brief pause to let system recover
        
        with self.monitor.monitor_operation(operation_name):
            yield resources
    
    def optimize_for_large_dataset(self, data_size: int) -> Dict[str, Any]:
        """Optimize settings for large dataset processing"""
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / (1024**3)
        
        # Estimate memory needs (rough heuristic)
        estimated_memory_gb = data_size * 8 / (1024**3)  # Assume 8 bytes per data point
        
        recommendations = {
            'use_chunking': estimated_memory_gb > available_memory_gb * 0.5,
            'chunk_size': max(1000, int(available_memory_gb * 1000 / estimated_memory_gb)) if estimated_memory_gb > 0 else 10000,
            'use_parallel': available_memory_gb > 4.0 and psutil.cpu_count() > 2,
            'max_workers': min(psutil.cpu_count(), int(available_memory_gb / 2)),
            'use_disk_cache': data_size > 100000,
            'memory_constraint': available_memory_gb < 2.0
        }
        
        logger.info(f"Optimization recommendations for dataset size {data_size}: {recommendations}")
        return recommendations


# Global instances for easy access
performance_monitor = PerformanceMonitor()
parallel_processor = ParallelProcessor()
cached_processor = CachedDataProcessor()
resource_manager = ResourceManager()


def performance_optimized(cache: bool = True, parallel: bool = False):
    """Decorator for performance optimization"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__name__}"
            
            # Resource check
            with resource_manager.resource_aware_execution(operation_name):
                if cache and hasattr(cached_processor, f"cached_{func.__name__}"):
                    # Use cached version if available
                    cached_func = getattr(cached_processor, f"cached_{func.__name__}")
                    return cached_func(*args, **kwargs)
                else:
                    # Execute original function
                    return func(*args, **kwargs)
        
        return wrapper
    return decorator


class MemoryMappedProcessor:
    """Generation 3 Memory-mapped file operations for large datasets"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / '.neuromorphic_mmap_cache'
        self.cache_dir.mkdir(exist_ok=True)
        self.active_mmaps = {}
        self.lock = threading.Lock()
    
    def create_mmap_file(self, data: np.ndarray, key: str) -> Path:
        """Create memory-mapped file for large numpy array"""
        file_path = self.cache_dir / f"{key}.mmap"
        
        # Save array to disk
        with open(file_path, 'w+b') as f:
            # Write header with shape and dtype info
            shape_bytes = np.array(data.shape).tobytes()
            dtype_bytes = str(data.dtype).encode('utf-8')
            
            # Write lengths and data
            f.write(len(shape_bytes).to_bytes(4, byteorder='little'))
            f.write(len(dtype_bytes).to_bytes(4, byteorder='little'))
            f.write(shape_bytes)
            f.write(dtype_bytes)
            f.write(data.tobytes())
        
        return file_path
    
    def load_mmap_array(self, file_path: Path) -> np.ndarray:
        """Load memory-mapped numpy array"""
        with self.lock:
            if str(file_path) in self.active_mmaps:
                return self.active_mmaps[str(file_path)]
        
        with open(file_path, 'rb') as f:
            # Read header
            shape_len = int.from_bytes(f.read(4), byteorder='little')
            dtype_len = int.from_bytes(f.read(4), byteorder='little')
            
            shape_bytes = f.read(shape_len)
            dtype_bytes = f.read(dtype_len)
            
            shape = tuple(np.frombuffer(shape_bytes, dtype=np.int64))
            dtype = np.dtype(dtype_bytes.decode('utf-8'))
            
            # Memory map the data
            offset = f.tell()
        
        # Create memory-mapped array
        mmap_array = np.memmap(file_path, dtype=dtype, mode='r', offset=offset, shape=shape)
        
        with self.lock:
            self.active_mmaps[str(file_path)] = mmap_array
        
        return mmap_array
    
    def process_large_dataset(self, data: Union[np.ndarray, pd.DataFrame], 
                            operation: Callable,
                            chunk_size: int = 10000) -> Any:
        """Process large dataset in chunks with memory mapping"""
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        # Create memory-mapped file if dataset is large
        if data_array.nbytes > 100 * 1024 * 1024:  # > 100MB
            key = hashlib.sha256(str(data_array.shape).encode()).hexdigest()
            mmap_file = self.create_mmap_file(data_array, key)
            mmap_array = self.load_mmap_array(mmap_file)
            data_to_process = mmap_array
        else:
            data_to_process = data_array
        
        # Process in chunks
        results = []
        for i in range(0, len(data_to_process), chunk_size):
            chunk = data_to_process[i:i + chunk_size]
            chunk_result = operation(chunk)
            results.append(chunk_result)
        
        # Combine results based on type
        if isinstance(results[0], np.ndarray):
            return np.concatenate(results)
        elif isinstance(results[0], list):
            return [item for sublist in results for item in sublist]
        else:
            return results
    
    def cleanup_mmap_files(self, older_than_hours: int = 24):
        """Clean up old memory-mapped files"""
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        for mmap_file in self.cache_dir.glob('*.mmap'):
            if mmap_file.stat().st_mtime < cutoff_time:
                # Remove from active mappings
                with self.lock:
                    self.active_mmaps.pop(str(mmap_file), None)
                
                mmap_file.unlink()
        
        logger.info(f"Cleaned up memory-mapped files older than {older_than_hours} hours")


class VectorizedOperations:
    """Generation 3 SIMD-optimized vectorized operations"""
    
    @staticmethod
    def vectorized_distance_matrix(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
        """Compute distance matrix using vectorized operations"""
        if metric == 'euclidean':
            # Optimized euclidean distance using broadcasting
            X_sqr = np.sum(X**2, axis=1, keepdims=True)
            Y_sqr = np.sum(Y**2, axis=1)
            XY = np.dot(X, Y.T)
            distances = np.sqrt(np.maximum(X_sqr - 2*XY + Y_sqr, 0))
        elif metric == 'cosine':
            # Cosine distance
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            Y_norm = Y / np.linalg.norm(Y, axis=1, keepdims=True)
            similarities = np.dot(X_norm, Y_norm.T)
            distances = 1 - similarities
        elif metric == 'manhattan':
            # Manhattan distance using broadcasting
            distances = np.sum(np.abs(X[:, np.newaxis, :] - Y[np.newaxis, :, :]), axis=2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        return distances
    
    @staticmethod
    def vectorized_clustering_update(data: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized K-means cluster update"""
        # Compute distances to all centroids
        distances = VectorizedOperations.vectorized_distance_matrix(data, centroids)
        
        # Assign points to closest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([
            data[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
            for k in range(len(centroids))
        ])
        
        return labels, new_centroids
    
    @staticmethod
    def vectorized_feature_scaling(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Vectorized feature scaling"""
        if method == 'standard':
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1  # Avoid division by zero
            scaled_data = (data - mean) / std
            params = {'mean': mean, 'std': std}
        elif method == 'minmax':
            min_vals = np.min(data, axis=0)
            max_vals = np.max(data, axis=0)
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1  # Avoid division by zero
            scaled_data = (data - min_vals) / range_vals
            params = {'min': min_vals, 'max': max_vals}
        elif method == 'robust':
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            mad[mad == 0] = 1  # Avoid division by zero
            scaled_data = (data - median) / mad
            params = {'median': median, 'mad': mad}
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
        
        return scaled_data, params


class AsyncProcessingManager:
    """Generation 3 Async processing manager for concurrent operations"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.active_tasks = set()
    
    async def process_batch_async(self, data_batches: List[Any], 
                                process_func: Callable,
                                *args, **kwargs) -> List[Any]:
        """Process multiple data batches asynchronously"""
        
        async def process_single_batch(batch_data):
            async with self.semaphore:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, process_func, batch_data, *args, **kwargs)
        
        # Create tasks
        tasks = []
        for batch in data_batches:
            task = asyncio.create_task(process_single_batch(batch))
            tasks.append(task)
            self.active_tasks.add(task)
        
        # Wait for completion
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
        finally:
            # Clean up completed tasks
            for task in tasks:
                self.active_tasks.discard(task)
    
    async def stream_process(self, data_stream, process_func: Callable, 
                           buffer_size: int = 100) -> AsyncIterable[Any]:
        """Process streaming data asynchronously"""
        buffer = []
        
        async for data_item in data_stream:
            buffer.append(data_item)
            
            if len(buffer) >= buffer_size:
                # Process buffer
                results = await self.process_batch_async(buffer, process_func)
                for result in results:
                    yield result
                buffer.clear()
        
        # Process remaining items
        if buffer:
            results = await self.process_batch_async(buffer, process_func)
            for result in results:
                yield result
    
    def get_active_task_count(self) -> int:
        """Get number of active tasks"""
        return len(self.active_tasks)


class PerformanceProfiler:
    """Generation 3 Advanced performance profiler with recommendations"""
    
    def __init__(self):
        self.profiles = {}
        self.recommendations_cache = {}
        self.profiling_enabled = True
    
    @contextmanager
    def profile_operation(self, operation_name: str, detailed: bool = False):
        """Profile operation with detailed metrics"""
        if not self.profiling_enabled:
            yield
            return
        
        import cProfile
        import pstats
        from io import StringIO
        
        profiler = cProfile.Profile() if detailed else None
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        if profiler:
            profiler.enable()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            if profiler:
                profiler.disable()
            
            # Store profile data
            profile_data = {
                'duration': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'start_time': start_time,
                'end_time': end_time
            }
            
            if profiler:
                stats_stream = StringIO()
                stats = pstats.Stats(profiler, stream=stats_stream)
                stats.sort_stats('cumulative').print_stats(10)
                profile_data['detailed_stats'] = stats_stream.getvalue()
            
            self.profiles[operation_name] = profile_data
    
    def analyze_performance_bottlenecks(self, operation_name: str) -> Dict[str, Any]:
        """Analyze performance bottlenecks and generate recommendations"""
        if operation_name not in self.profiles:
            return {'error': f'No profile data for {operation_name}'}
        
        profile = self.profiles[operation_name]
        recommendations = []
        
        # Memory analysis
        memory_mb = profile['memory_delta'] / (1024 * 1024)
        if memory_mb > 100:  # > 100MB
            recommendations.append({
                'type': 'memory',
                'issue': f'High memory usage: {memory_mb:.1f}MB',
                'recommendation': 'Consider using memory mapping or chunked processing',
                'priority': 'high'
            })
        
        # Duration analysis
        duration = profile['duration']
        if duration > 10:  # > 10 seconds
            recommendations.append({
                'type': 'performance',
                'issue': f'Slow operation: {duration:.1f}s',
                'recommendation': 'Consider GPU acceleration or parallel processing',
                'priority': 'high'
            })
        elif duration > 1:  # > 1 second
            recommendations.append({
                'type': 'performance',
                'issue': f'Moderate duration: {duration:.1f}s',
                'recommendation': 'Consider caching or vectorized operations',
                'priority': 'medium'
            })
        
        # Detailed stats analysis (if available)
        if 'detailed_stats' in profile:
            stats_text = profile['detailed_stats']
            if 'numpy' in stats_text:
                recommendations.append({
                    'type': 'optimization',
                    'issue': 'Heavy numpy usage detected',
                    'recommendation': 'Consider vectorized operations or GPU acceleration',
                    'priority': 'medium'
                })
        
        return {
            'operation': operation_name,
            'profile': profile,
            'recommendations': recommendations,
            'bottleneck_score': len([r for r in recommendations if r['priority'] == 'high'])
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.profiles:
            return {'total_operations': 0}
        
        total_duration = sum(p['duration'] for p in self.profiles.values())
        total_memory = sum(p['memory_delta'] for p in self.profiles.values()) / (1024 * 1024)  # MB
        
        slowest_op = max(self.profiles.items(), key=lambda x: x[1]['duration'])
        memory_intensive_op = max(self.profiles.items(), key=lambda x: x[1]['memory_delta'])
        
        return {
            'total_operations': len(self.profiles),
            'total_duration': total_duration,
            'total_memory_mb': total_memory,
            'avg_duration': total_duration / len(self.profiles),
            'slowest_operation': {
                'name': slowest_op[0],
                'duration': slowest_op[1]['duration']
            },
            'most_memory_intensive': {
                'name': memory_intensive_op[0],
                'memory_mb': memory_intensive_op[1]['memory_delta'] / (1024 * 1024)
            },
            'operations': list(self.profiles.keys())
        }


class Gen3PerformanceOptimizer:
    """Generation 3 comprehensive performance optimizer"""
    
    def __init__(self):
        self.mmap_processor = MemoryMappedProcessor()
        self.vectorized_ops = VectorizedOperations()
        self.async_manager = AsyncProcessingManager()
        self.profiler = PerformanceProfiler()
        
        # Configuration
        self.auto_optimization = True
        self.optimization_history = []
    
    def optimize_clustering_pipeline(self, features: pd.DataFrame, 
                                   n_clusters: int,
                                   optimization_level: str = 'balanced') -> Dict[str, Any]:
        """Optimize entire clustering pipeline"""
        start_time = time.time()
        
        with self.profiler.profile_operation('clustering_pipeline', detailed=True):
            # Step 1: Data preprocessing optimization
            features_array = features.values
            
            # Use vectorized scaling
            scaled_features, scaling_params = self.vectorized_ops.vectorized_feature_scaling(
                features_array, method='standard'
            )
            
            # Step 2: Memory optimization for large datasets
            if features_array.nbytes > 50 * 1024 * 1024:  # > 50MB
                logger.info("Using memory-mapped processing for large dataset")
                
                def clustering_operation(data_chunk):
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_clusters, n_init=1)
                    return kmeans.fit_predict(data_chunk)
                
                # Process in chunks
                chunk_results = self.mmap_processor.process_large_dataset(
                    scaled_features, clustering_operation, chunk_size=10000
                )
                cluster_labels = np.concatenate(chunk_results) if isinstance(chunk_results[0], np.ndarray) else chunk_results
            
            else:
                # Use vectorized clustering for smaller datasets
                from sklearn.cluster import KMeans
                initial_centroids = scaled_features[np.random.choice(len(scaled_features), n_clusters, replace=False)]
                
                # Vectorized K-means iterations
                centroids = initial_centroids.copy()
                for iteration in range(100):  # Max iterations
                    labels, new_centroids = self.vectorized_ops.vectorized_clustering_update(
                        scaled_features, centroids
                    )
                    
                    # Check convergence
                    if np.allclose(centroids, new_centroids, rtol=1e-4):
                        break
                    
                    centroids = new_centroids
                
                cluster_labels = labels
        
        total_time = time.time() - start_time
        
        # Generate performance analysis
        analysis = self.profiler.analyze_performance_bottlenecks('clustering_pipeline')
        
        return {
            'cluster_labels': cluster_labels,
            'scaling_params': scaling_params,
            'processing_time': total_time,
            'performance_analysis': analysis,
            'optimization_level': optimization_level,
            'memory_mapped': features_array.nbytes > 50 * 1024 * 1024
        }
    
    async def optimize_batch_processing(self, data_batches: List[pd.DataFrame],
                                      process_func: Callable) -> List[Any]:
        """Optimize batch processing using async operations"""
        with self.profiler.profile_operation('batch_processing'):
            results = await self.async_manager.process_batch_async(
                data_batches, process_func
            )
        
        return results
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system-wide optimization recommendations"""
        recommendations = []
        
        # Analyze all profiles
        for op_name in self.profiler.profiles:
            analysis = self.profiler.analyze_performance_bottlenecks(op_name)
            recommendations.extend(analysis.get('recommendations', []))
        
        # System-level recommendations
        memory_info = psutil.virtual_memory()
        if memory_info.percent > 80:
            recommendations.append({
                'type': 'system',
                'issue': f'High system memory usage: {memory_info.percent}%',
                'recommendation': 'Enable memory mapping and increase swap space',
                'priority': 'high'
            })
        
        cpu_count = psutil.cpu_count()
        if cpu_count > 4:
            recommendations.append({
                'type': 'parallelization',
                'issue': f'Underutilized CPU cores: {cpu_count} available',
                'recommendation': 'Enable parallel processing and async operations',
                'priority': 'medium'
            })
        
        return recommendations
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        performance_summary = self.profiler.get_performance_summary()
        recommendations = self.generate_optimization_recommendations()
        
        return {
            'performance_summary': performance_summary,
            'active_optimizations': {
                'memory_mapping': True,
                'vectorized_operations': True,
                'async_processing': True,
                'caching': True
            },
            'recommendations': recommendations,
            'system_status': {
                'memory_percent': psutil.virtual_memory().percent,
                'cpu_count': psutil.cpu_count(),
                'active_async_tasks': self.async_manager.get_active_task_count()
            }
        }


# Global Generation 3 instances
mmap_processor = MemoryMappedProcessor()
vectorized_ops = VectorizedOperations()
async_manager = AsyncProcessingManager()
performance_profiler = PerformanceProfiler()
gen3_optimizer = Gen3PerformanceOptimizer()