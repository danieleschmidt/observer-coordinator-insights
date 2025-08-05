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
import pickle
import os
from pathlib import Path
import numpy as np
import pandas as pd
from contextlib import contextmanager
import psutil
import gc

logger = logging.getLogger(__name__)


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
        # Create hash from data and parameters
        if isinstance(data, pd.DataFrame):
            data_hash = hashlib.md5(str(data.shape).encode() + 
                                  str(data.columns.tolist()).encode()).hexdigest()
        else:
            data_hash = hashlib.md5(str(data).encode()).hexdigest()
        
        params_hash = hashlib.md5(str(sorted(kwargs.items())).encode()).hexdigest()
        return f"{operation}_{data_hash}_{params_hash}"
    
    def _get_disk_cache_path(self, cache_key: str) -> Path:
        """Get disk cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _save_to_disk_cache(self, cache_key: str, result: Any):
        """Save result to disk cache"""
        try:
            cache_path = self._get_disk_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
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
                        return pickle.load(f)
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