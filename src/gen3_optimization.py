#!/usr/bin/env python3
"""
Generation 3 Optimization Implementation
Performance optimization, caching, concurrent processing, resource pooling, and auto-scaling
"""

import asyncio
import concurrent.futures
import functools
import hashlib
import json
import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"
    LFU = "lfu" 
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    size_bytes: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl_seconds


class IntelligentCache:
    """High-performance intelligent cache with multiple eviction strategies"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.lock = threading.RLock()
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments"""
        key_data = {
            'func': f"{func.__module__}.{func.__name__}",
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._estimate_size(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in obj.items())
            elif hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            else:
                return 64  # Default estimate
        except:
            return 64
    
    def _evict_entries(self):
        """Evict entries based on strategy"""
        if len(self.cache) <= self.max_size:
            return
        
        entries_to_remove = len(self.cache) - self.max_size + 1
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            for _ in range(entries_to_remove):
                if self.access_order:
                    oldest_key = self.access_order.popleft()
                    self.cache.pop(oldest_key, None)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: (x[1].access_count, x[1].last_accessed)
            )
            for key, _ in sorted_entries[:entries_to_remove]:
                self.cache.pop(key, None)
                self.access_frequency.pop(key, None)
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            for key in expired_keys:
                self.cache.pop(key, None)
            
            # If still need to evict more
            remaining_to_remove = max(0, len(self.cache) - self.max_size)
            if remaining_to_remove > 0:
                sorted_entries = sorted(
                    self.cache.items(),
                    key=lambda x: x[1].created_at
                )
                for key, _ in sorted_entries[:remaining_to_remove]:
                    self.cache.pop(key, None)
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on access patterns
            now = datetime.now()
            scored_entries = []
            
            for key, entry in self.cache.items():
                # Calculate adaptive score
                age_seconds = (now - entry.created_at).total_seconds()
                recency_score = 1.0 / (1.0 + age_seconds / 3600)  # Decay over hours
                frequency_score = entry.access_count / max(1, age_seconds / 60)  # Per minute
                size_penalty = entry.size_bytes / (1024 * 1024)  # MB penalty
                
                adaptive_score = (recency_score + frequency_score) - (size_penalty * 0.1)
                scored_entries.append((key, adaptive_score))
            
            # Remove lowest scoring entries
            scored_entries.sort(key=lambda x: x[1])
            for key, _ in scored_entries[:entries_to_remove]:
                self.cache.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            if entry.is_expired:
                self.cache.pop(key, None)
                self.miss_count += 1
                return None
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.access_frequency[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hit_count += 1
            return entry.value
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Put value in cache"""
        with self.lock:
            size_bytes = self._estimate_size(value)
            now = datetime.now()
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                last_accessed=now,
                access_count=1,
                ttl_seconds=ttl_seconds,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.access_frequency[key] = 1
            
            self._evict_entries()
    
    def clear_expired(self):
        """Clear all expired entries"""
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if v.is_expired]
            for key in expired_keys:
                self.cache.pop(key, None)
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_frequency.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total_requests)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value,
            'total_size_bytes': sum(entry.size_bytes for entry in self.cache.values())
        }


def cached(cache_instance: Optional[IntelligentCache] = None, ttl_seconds: Optional[int] = None):
    """Decorator for caching function results"""
    if cache_instance is None:
        cache_instance = global_cache
    
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_instance._generate_key(func, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result, ttl_seconds)
            return result
        
        return wrapper
    return decorator


class ResourcePool:
    """Generic resource pool for expensive objects"""
    
    def __init__(self, factory: Callable, max_size: int = 10, initial_size: int = 2):
        self.factory = factory
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
        self.lock = threading.Lock()
        
        # Create initial resources
        for _ in range(initial_size):
            resource = self.factory()
            self.pool.append(resource)
    
    def acquire(self, timeout: float = 5.0):
        """Acquire a resource from the pool"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            with self.lock:
                if self.pool:
                    resource = self.pool.pop()
                    self.in_use.add(id(resource))
                    return resource
                elif len(self.in_use) < self.max_size:
                    # Create new resource if pool not at max capacity
                    resource = self.factory()
                    self.in_use.add(id(resource))
                    return resource
            
            time.sleep(0.1)  # Wait before retrying
        
        raise Exception(f"Failed to acquire resource within {timeout}s")
    
    def release(self, resource):
        """Release a resource back to the pool"""
        with self.lock:
            resource_id = id(resource)
            if resource_id in self.in_use:
                self.in_use.remove(resource_id)
                if len(self.pool) < self.max_size:
                    self.pool.append(resource)
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'in_use': len(self.in_use),
                'max_size': self.max_size,
                'total_created': len(self.pool) + len(self.in_use)
            }


class ParallelProcessor:
    """High-performance parallel processing with intelligent load balancing"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (mp.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
        self.load_stats = defaultdict(list)
    
    def parallel_map(self, func: Callable, items: List[Any], chunk_size: Optional[int] = None, 
                    use_processes: bool = False) -> List[Any]:
        """Execute function in parallel across items with intelligent chunking"""
        if not items:
            return []
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        # Choose execution strategy
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Create chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Process chunks in parallel
        start_time = time.time()
        results = []
        
        future_to_chunk = {
            executor.submit(self._process_chunk, func, chunk): chunk 
            for chunk in chunks
        }
        
        for future in as_completed(future_to_chunk):
            chunk_results = future.result()
            results.extend(chunk_results)
        
        # Record performance stats
        execution_time = time.time() - start_time
        self.load_stats['parallel_map'].append({
            'items': len(items),
            'chunks': len(chunks),
            'execution_time': execution_time,
            'throughput': len(items) / execution_time
        })
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items"""
        return [func(item) for item in chunk]
    
    async def async_parallel_map(self, func: Callable, items: List[Any], 
                                concurrency: int = 50) -> List[Any]:
        """Async parallel processing with controlled concurrency"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def process_item(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, func, item)
        
        tasks = [process_item(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for operation, records in self.load_stats.items():
            if records:
                stats[operation] = {
                    'total_executions': len(records),
                    'avg_throughput': sum(r['throughput'] for r in records) / len(records),
                    'avg_execution_time': sum(r['execution_time'] for r in records) / len(records),
                    'max_throughput': max(r['throughput'] for r in records),
                    'min_throughput': min(r['throughput'] for r in records)
                }
        return stats
    
    def shutdown(self):
        """Shutdown executors"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class AdaptiveLoadBalancer:
    """Adaptive load balancer for distributing work across resources"""
    
    def __init__(self, resources: List[Any]):
        self.resources = resources
        self.load_metrics = {id(res): deque(maxlen=100) for res in resources}
        self.current_loads = {id(res): 0.0 for res in resources}
        self.lock = threading.Lock()
    
    def get_optimal_resource(self) -> Any:
        """Get the resource with the lowest current load"""
        with self.lock:
            # Calculate current load for each resource
            for resource_id in self.current_loads:
                recent_metrics = list(self.load_metrics[resource_id])
                if recent_metrics:
                    avg_load = sum(recent_metrics) / len(recent_metrics)
                    self.current_loads[resource_id] = avg_load
                else:
                    self.current_loads[resource_id] = 0.0
            
            # Select resource with minimum load
            min_load_id = min(self.current_loads, key=self.current_loads.get)
            return next(res for res in self.resources if id(res) == min_load_id)
    
    def record_load(self, resource: Any, load_metric: float):
        """Record load metric for a resource"""
        with self.lock:
            resource_id = id(resource)
            if resource_id in self.load_metrics:
                self.load_metrics[resource_id].append(load_metric)
    
    def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across resources"""
        with self.lock:
            total_load = sum(self.current_loads.values())
            if total_load == 0:
                return {f"resource_{i}": 0.0 for i in range(len(self.resources))}
            
            return {
                f"resource_{i}": load / total_load 
                for i, load in enumerate(self.current_loads.values())
            }


class AutoScaler:
    """Auto-scaling based on system metrics and load"""
    
    def __init__(self, min_workers: int = 1, max_workers: int = 16, 
                 scale_up_threshold: float = 0.7, scale_down_threshold: float = 0.3):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.current_workers = min_workers
        self.load_history = deque(maxlen=60)  # Keep 1 minute of history
        self.last_scale_time = datetime.now()
        self.scale_cooldown = timedelta(seconds=30)  # Prevent thrashing
        
    def record_load(self, cpu_percent: float, memory_percent: float, queue_size: int):
        """Record current system load"""
        load_score = (cpu_percent + memory_percent) / 200.0 + (queue_size / 100.0)
        self.load_history.append({
            'timestamp': datetime.now(),
            'load_score': load_score,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'queue_size': queue_size
        })
    
    def should_scale(self) -> tuple[bool, str, int]:
        """Determine if scaling is needed"""
        if len(self.load_history) < 5:  # Need some history
            return False, "insufficient_data", self.current_workers
        
        # Check cooldown
        if datetime.now() - self.last_scale_time < self.scale_cooldown:
            return False, "cooldown", self.current_workers
        
        # Calculate average load over recent history
        recent_loads = list(self.load_history)[-10:]  # Last 10 measurements
        avg_load = sum(metric['load_score'] for metric in recent_loads) / len(recent_loads)
        
        # Scale up if load is high and we can add workers
        if avg_load > self.scale_up_threshold and self.current_workers < self.max_workers:
            new_workers = min(self.max_workers, int(self.current_workers * 1.5))
            return True, "scale_up", new_workers
        
        # Scale down if load is low and we can remove workers
        if avg_load < self.scale_down_threshold and self.current_workers > self.min_workers:
            new_workers = max(self.min_workers, int(self.current_workers * 0.7))
            return True, "scale_down", new_workers
        
        return False, "stable", self.current_workers
    
    def execute_scaling(self, new_worker_count: int, reason: str):
        """Execute scaling decision"""
        logger.info(f"Auto-scaling: {reason} from {self.current_workers} to {new_worker_count} workers")
        self.current_workers = new_worker_count
        self.last_scale_time = datetime.now()
    
    def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get auto-scaling metrics"""
        recent_metrics = list(self.load_history)[-10:] if self.load_history else []
        avg_load = sum(m['load_score'] for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0.0
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'avg_load_score': avg_load,
            'scale_up_threshold': self.scale_up_threshold,
            'scale_down_threshold': self.scale_down_threshold,
            'history_size': len(self.load_history)
        }


# Global instances for Generation 3 optimization
global_cache = IntelligentCache(max_size=10000, strategy=CacheStrategy.ADAPTIVE)
parallel_processor = ParallelProcessor()
auto_scaler = AutoScaler()


def initialize_gen3_optimization():
    """Initialize Generation 3 optimization features"""
    logger.info("⚡ Initializing Generation 3 Optimization Features")
    
    # Start cache cleanup task
    def cache_cleanup_task():
        while True:
            time.sleep(300)  # Clean every 5 minutes
            global_cache.clear_expired()
    
    cleanup_thread = threading.Thread(target=cache_cleanup_task, daemon=True)
    cleanup_thread.start()
    
    logger.info("✅ Generation 3 Optimization Features Initialized")


if __name__ == "__main__":
    # Demo the optimization features
    initialize_gen3_optimization()
    
    # Test caching
    @cached(ttl_seconds=60)
    def expensive_computation(n: int) -> int:
        time.sleep(0.1)  # Simulate expensive operation
        return n * n
    
    # Test parallel processing
    start_time = time.time()
    results = parallel_processor.parallel_map(expensive_computation, list(range(100)))
    parallel_time = time.time() - start_time
    
    print(f"Parallel processing of 100 items took {parallel_time:.2f}s")
    print(f"Cache stats: {global_cache.get_stats()}")
    print(f"Performance stats: {parallel_processor.get_performance_stats()}")