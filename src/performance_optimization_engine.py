#!/usr/bin/env python3
"""
Performance Optimization Engine - Generation 3 Scalability
Advanced performance monitoring, optimization, and auto-scaling
"""

import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import json
import weakref
import cProfile
import pstats
import io
import gc
import sys
import psutil
import functools
import hashlib


class OptimizationLevel(Enum):
    """Performance optimization levels"""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    operation_name: str
    duration_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    cache_hit_rate: float
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation_name,
            'duration_ms': self.duration_ms,
            'cpu_percent': self.cpu_usage_percent,
            'memory_mb': self.memory_usage_mb,
            'throughput': self.throughput_ops_per_sec,
            'cache_hit_rate': self.cache_hit_rate,
            'timestamp': self.timestamp.isoformat(),
            **self.additional_metrics
        }


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation"""
    category: str  # caching, concurrency, algorithm, memory, etc.
    description: str
    impact_estimate: float  # 0-100 scale
    implementation_effort: int  # 1-10 scale
    code_changes_required: bool
    auto_applicable: bool
    recommendation_id: str
    priority: int  # 1-5 scale
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.recommendation_id,
            'category': self.category,
            'description': self.description,
            'impact_estimate': self.impact_estimate,
            'effort': self.implementation_effort,
            'code_changes': self.code_changes_required,
            'auto_applicable': self.auto_applicable,
            'priority': self.priority
        }


class AdvancedCache:
    """High-performance adaptive caching system"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: int = 3600,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.strategy = strategy
        
        # Storage
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._access_counts: Dict[str, int] = {}
        self._access_times: Dict[str, List[datetime]] = {}
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Adaptive tuning
        self._performance_history: List[float] = []
        self._auto_tune_enabled = True
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                self._misses += 1
                return None
            
            # Update access statistics
            self._access_counts[key] = self._access_counts.get(key, 0) + 1
            if key not in self._access_times:
                self._access_times[key] = []
            self._access_times[key].append(datetime.now())
            
            # Trim access history (keep last 100)
            if len(self._access_times[key]) > 100:
                self._access_times[key] = self._access_times[key][-100:]
            
            self._hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Put value in cache"""
        with self._lock:
            # Use custom TTL if provided
            actual_ttl = ttl if ttl is not None else self.ttl_seconds
            
            # Ensure space available
            if len(self._cache) >= self.max_size and key not in self._cache:
                if not self._evict_item():
                    return False  # Could not make space
            
            # Store value
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
            
            if key not in self._access_counts:
                self._access_counts[key] = 0
                self._access_times[key] = []
            
            return True
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self._timestamps:
            return True
        
        age = datetime.now() - self._timestamps[key]
        return age.total_seconds() > self.ttl_seconds
    
    def _evict_item(self) -> bool:
        """Evict item based on strategy"""
        if not self._cache:
            return False
        
        evict_key = None
        
        if self.strategy == CacheStrategy.LRU:
            # Least Recently Used
            oldest_time = min(
                self._access_times.get(key, [datetime.min])[-1] if self._access_times.get(key) 
                else datetime.min for key in self._cache.keys()
            )
            for key in self._cache.keys():
                last_access = (self._access_times.get(key, [datetime.min])[-1] 
                             if self._access_times.get(key) else datetime.min)
                if last_access == oldest_time:
                    evict_key = key
                    break
        
        elif self.strategy == CacheStrategy.LFU:
            # Least Frequently Used
            min_count = min(self._access_counts.get(key, 0) for key in self._cache.keys())
            for key in self._cache.keys():
                if self._access_counts.get(key, 0) == min_count:
                    evict_key = key
                    break
        
        elif self.strategy == CacheStrategy.TTL:
            # Oldest entry
            oldest_time = min(self._timestamps.values())
            for key, timestamp in self._timestamps.items():
                if timestamp == oldest_time:
                    evict_key = key
                    break
        
        elif self.strategy == CacheStrategy.ADAPTIVE:
            # Adaptive strategy based on performance
            evict_key = self._adaptive_eviction()
        
        if evict_key:
            self._remove_key(evict_key)
            self._evictions += 1
            return True
        
        return False
    
    def _adaptive_eviction(self) -> Optional[str]:
        """Adaptive eviction based on access patterns"""
        if not self._cache:
            return None
        
        # Score each key based on multiple factors
        scores = {}
        now = datetime.now()
        
        for key in self._cache.keys():
            score = 0
            
            # Frequency factor
            access_count = self._access_counts.get(key, 0)
            score += access_count * 0.3
            
            # Recency factor
            last_access = (self._access_times.get(key, [datetime.min])[-1] 
                         if self._access_times.get(key) else datetime.min)
            recency_hours = (now - last_access).total_seconds() / 3600
            score -= recency_hours * 0.1  # Penalize older items
            
            # Access pattern factor (consistent access gets bonus)
            access_times = self._access_times.get(key, [])
            if len(access_times) > 1:
                intervals = []
                for i in range(1, len(access_times)):
                    interval = (access_times[i] - access_times[i-1]).total_seconds()
                    intervals.append(interval)
                
                # Reward consistent access patterns
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                    consistency_score = 1.0 / (1.0 + variance / 1000)  # Normalize
                    score += consistency_score * 0.2
            
            scores[key] = score
        
        # Return key with lowest score
        return min(scores.keys(), key=lambda k: scores[k])
    
    def _remove_key(self, key: str):
        """Remove key and all associated data"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)
        self._access_times.pop(key, None)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / max(1, total_requests)
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'evictions': self._evictions,
            'hit_rate': hit_rate,
            'strategy': self.strategy.value,
            'memory_usage_estimation': sys.getsizeof(self._cache)
        }
    
    def clear(self):
        """Clear all cache data"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_counts.clear()
            self._access_times.clear()


class ParallelProcessingEngine:
    """Advanced parallel processing with adaptive scaling"""
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        
        # Determine optimal worker counts
        cpu_count = multiprocessing.cpu_count()
        if max_workers is None:
            if optimization_level == OptimizationLevel.CONSERVATIVE:
                max_workers = max(1, cpu_count // 2)
            elif optimization_level == OptimizationLevel.BALANCED:
                max_workers = cpu_count
            elif optimization_level == OptimizationLevel.AGGRESSIVE:
                max_workers = cpu_count * 2
            else:  # MAXIMUM
                max_workers = cpu_count * 4
        
        self.max_workers = max_workers
        
        # Executors
        self._thread_executor = ThreadPoolExecutor(max_workers=max_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=min(max_workers, cpu_count))
        
        # Performance tracking
        self._execution_times: List[float] = []
        self._throughput_history: List[float] = []
        
        # Adaptive scaling
        self._current_load = 0
        self._load_lock = threading.Lock()
    
    async def parallel_map(self, 
                          func: Callable,
                          items: List[Any],
                          use_processes: bool = False,
                          chunk_size: Optional[int] = None) -> List[Any]:
        """Execute function in parallel across items"""
        if not items:
            return []
        
        start_time = time.time()
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.max_workers * 4))
        
        # Choose execution strategy
        executor = self._process_executor if use_processes else self._thread_executor
        
        # Submit tasks
        with self._load_lock:
            self._current_load += 1
        
        try:
            loop = asyncio.get_event_loop()
            
            if use_processes and len(items) > chunk_size:
                # Chunk processing for large datasets
                chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
                
                def process_chunk(chunk):
                    return [func(item) for item in chunk]
                
                future_to_chunk = {
                    loop.run_in_executor(executor, process_chunk, chunk): chunk
                    for chunk in chunks
                }
                
                results = []
                for future in as_completed(future_to_chunk):
                    chunk_results = await future
                    results.extend(chunk_results)
            else:
                # Individual item processing
                future_to_item = {
                    loop.run_in_executor(executor, func, item): item
                    for item in items
                }
                
                results = []
                for future in as_completed(future_to_item):
                    result = await future
                    results.append(result)
            
            # Record performance
            execution_time = time.time() - start_time
            throughput = len(items) / max(0.001, execution_time)
            
            self._execution_times.append(execution_time)
            self._throughput_history.append(throughput)
            
            # Trim history
            if len(self._execution_times) > 100:
                self._execution_times = self._execution_times[-100:]
                self._throughput_history = self._throughput_history[-100:]
            
            return results
        
        finally:
            with self._load_lock:
                self._current_load = max(0, self._current_load - 1)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get parallel processing performance statistics"""
        if not self._execution_times:
            return {"message": "No execution history"}
        
        avg_execution_time = sum(self._execution_times) / len(self._execution_times)
        avg_throughput = sum(self._throughput_history) / len(self._throughput_history)
        
        return {
            'max_workers': self.max_workers,
            'current_load': self._current_load,
            'optimization_level': self.optimization_level.value,
            'avg_execution_time': avg_execution_time,
            'avg_throughput': avg_throughput,
            'total_executions': len(self._execution_times),
            'thread_executor_active': not self._thread_executor._shutdown,
            'process_executor_active': not self._process_executor._shutdown
        }
    
    def shutdown(self):
        """Shutdown executors"""
        self._thread_executor.shutdown(wait=True)
        self._process_executor.shutdown(wait=True)


class PerformanceProfiler:
    """Advanced performance profiling and analysis"""
    
    def __init__(self):
        self._profiles: Dict[str, cProfile.Profile] = {}
        self._active_profiles: Dict[str, cProfile.Profile] = {}
        self._profile_results: Dict[str, Dict[str, Any]] = {}
    
    def start_profiling(self, operation_name: str):
        """Start profiling an operation"""
        if operation_name in self._active_profiles:
            return  # Already profiling
        
        profile = cProfile.Profile()
        profile.enable()
        self._active_profiles[operation_name] = profile
    
    def stop_profiling(self, operation_name: str) -> Dict[str, Any]:
        """Stop profiling and analyze results"""
        if operation_name not in self._active_profiles:
            return {"error": "No active profile found"}
        
        profile = self._active_profiles.pop(operation_name)
        profile.disable()
        
        # Analyze results
        analysis = self._analyze_profile(profile)
        self._profile_results[operation_name] = analysis
        
        return analysis
    
    def _analyze_profile(self, profile: cProfile.Profile) -> Dict[str, Any]:
        """Analyze profiling results"""
        # Capture stats
        stats_stream = io.StringIO()
        stats = pstats.Stats(profile, stream=stats_stream)
        
        # Sort by cumulative time
        stats.sort_stats('cumulative')
        
        # Get top functions
        stats.print_stats(20)  # Top 20 functions
        profile_output = stats_stream.getvalue()
        
        # Extract key metrics
        total_calls = stats.total_calls
        total_time = stats.total_tt
        
        # Get function-level details
        function_stats = []
        for func_key, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_num, func_name = func_key
            
            function_stats.append({
                'function': func_name,
                'file': filename,
                'line': line_num,
                'total_calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'time_per_call': tt / max(1, nc)
            })
        
        # Sort by cumulative time and take top 10
        function_stats.sort(key=lambda x: x['cumulative_time'], reverse=True)
        top_functions = function_stats[:10]
        
        return {
            'total_calls': total_calls,
            'total_time': total_time,
            'top_functions': top_functions,
            'full_profile': profile_output,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def profile_function(self, func: Callable):
        """Decorator to profile a function"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            operation_name = f"{func.__module__}.{func.__name__}"
            self.start_profiling(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop_profiling(operation_name)
        
        return wrapper


class PerformanceOptimizer:
    """Main performance optimization engine"""
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        
        # Components
        self.cache = AdvancedCache(
            max_size=self._get_cache_size(),
            ttl_seconds=3600,
            strategy=CacheStrategy.ADAPTIVE
        )
        self.parallel_engine = ParallelProcessingEngine(
            optimization_level=optimization_level
        )
        self.profiler = PerformanceProfiler()
        
        # Performance monitoring
        self._metrics_history: List[PerformanceMetrics] = []
        self._baseline_established = False
        self._baseline_metrics: Optional[PerformanceMetrics] = None
        
        # Auto-optimization
        self._auto_optimize_enabled = True
        self._optimization_thread: Optional[threading.Thread] = None
        self._stop_optimization = threading.Event()
    
    def _get_cache_size(self) -> int:
        """Determine optimal cache size based on optimization level"""
        base_size = 1000
        
        if self.optimization_level == OptimizationLevel.CONSERVATIVE:
            return base_size
        elif self.optimization_level == OptimizationLevel.BALANCED:
            return base_size * 2
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return base_size * 4
        else:  # MAXIMUM
            return base_size * 8
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous performance monitoring"""
        if self._optimization_thread and self._optimization_thread.is_alive():
            return
        
        self._stop_optimization.clear()
        self._optimization_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._optimization_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self._stop_optimization.set()
        if self._optimization_thread:
            self._optimization_thread.join()
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop"""
        while not self._stop_optimization.wait(interval_seconds):
            try:
                self._collect_performance_metrics()
                if self._auto_optimize_enabled:
                    self._auto_optimize()
            except Exception as e:
                # Log error but continue monitoring
                pass
    
    def _collect_performance_metrics(self):
        """Collect current performance metrics"""
        process = psutil.Process()
        
        # Get cache stats
        cache_stats = self.cache.get_statistics()
        
        # Get parallel processing stats
        parallel_stats = self.parallel_engine.get_performance_stats()
        
        # Create metrics
        metrics = PerformanceMetrics(
            operation_name="system_monitoring",
            duration_ms=0,  # Not applicable for monitoring
            cpu_usage_percent=process.cpu_percent(),
            memory_usage_mb=process.memory_info().rss / 1024 / 1024,
            throughput_ops_per_sec=parallel_stats.get('avg_throughput', 0),
            cache_hit_rate=cache_stats.get('hit_rate', 0),
            additional_metrics={
                'cache_size': cache_stats.get('size', 0),
                'active_workers': parallel_stats.get('current_load', 0),
                'open_files': process.num_fds() if hasattr(process, 'num_fds') else 0
            }
        )
        
        self._metrics_history.append(metrics)
        
        # Establish baseline if needed
        if not self._baseline_established and len(self._metrics_history) >= 10:
            self._establish_baseline()
        
        # Trim history
        if len(self._metrics_history) > 1000:
            self._metrics_history = self._metrics_history[-1000:]
    
    def _establish_baseline(self):
        """Establish performance baseline"""
        if len(self._metrics_history) < 10:
            return
        
        # Average last 10 measurements for baseline
        recent_metrics = self._metrics_history[-10:]
        
        avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_throughput = sum(m.throughput_ops_per_sec for m in recent_metrics) / len(recent_metrics)
        avg_cache_hit = sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics)
        
        self._baseline_metrics = PerformanceMetrics(
            operation_name="baseline",
            duration_ms=avg_duration,
            cpu_usage_percent=avg_cpu,
            memory_usage_mb=avg_memory,
            throughput_ops_per_sec=avg_throughput,
            cache_hit_rate=avg_cache_hit
        )
        
        self._baseline_established = True
    
    def _auto_optimize(self):
        """Perform automatic optimizations"""
        if not self._baseline_established or not self._metrics_history:
            return
        
        current_metrics = self._metrics_history[-1]
        
        # Cache optimization
        if current_metrics.cache_hit_rate < 0.7:  # Below 70% hit rate
            # Increase cache size if memory allows
            if current_metrics.memory_usage_mb < 500:  # Below 500MB
                new_size = min(self.cache.max_size * 2, 10000)
                self.cache.max_size = new_size
        
        # Memory optimization
        if current_metrics.memory_usage_mb > 1000:  # Above 1GB
            # Force garbage collection
            gc.collect()
            
            # Reduce cache size
            self.cache.max_size = max(self.cache.max_size // 2, 100)
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if not self._metrics_history:
            return recommendations
        
        current_metrics = self._metrics_history[-1]
        
        # Cache recommendations
        if current_metrics.cache_hit_rate < 0.5:
            recommendations.append(OptimizationRecommendation(
                category="caching",
                description="Low cache hit rate detected. Consider increasing cache size or reviewing cache keys.",
                impact_estimate=75.0,
                implementation_effort=3,
                code_changes_required=False,
                auto_applicable=True,
                recommendation_id="cache_low_hit_rate",
                priority=4
            ))
        
        # Memory recommendations
        if current_metrics.memory_usage_mb > 500:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                description="High memory usage detected. Consider memory profiling and optimization.",
                impact_estimate=50.0,
                implementation_effort=6,
                code_changes_required=True,
                auto_applicable=False,
                recommendation_id="high_memory_usage",
                priority=3
            ))
        
        # CPU recommendations
        if current_metrics.cpu_usage_percent > 80:
            recommendations.append(OptimizationRecommendation(
                category="concurrency",
                description="High CPU usage detected. Consider parallel processing optimization.",
                impact_estimate=60.0,
                implementation_effort=5,
                code_changes_required=True,
                auto_applicable=False,
                recommendation_id="high_cpu_usage",
                priority=4
            ))
        
        # Throughput recommendations
        if self._baseline_established:
            baseline_throughput = self._baseline_metrics.throughput_ops_per_sec
            if baseline_throughput > 0 and current_metrics.throughput_ops_per_sec < baseline_throughput * 0.8:
                recommendations.append(OptimizationRecommendation(
                    category="performance",
                    description="Throughput below baseline. Review recent changes for performance regressions.",
                    impact_estimate=80.0,
                    implementation_effort=7,
                    code_changes_required=True,
                    auto_applicable=False,
                    recommendation_id="throughput_regression",
                    priority=5
                ))
        
        return recommendations
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self._metrics_history:
            return {"message": "No performance data available"}
        
        current_metrics = self._metrics_history[-1]
        recommendations = self.generate_optimization_recommendations()
        
        # Calculate trends
        trends = {}
        if len(self._metrics_history) >= 2:
            prev_metrics = self._metrics_history[-2]
            trends = {
                'cpu_trend': current_metrics.cpu_usage_percent - prev_metrics.cpu_usage_percent,
                'memory_trend': current_metrics.memory_usage_mb - prev_metrics.memory_usage_mb,
                'throughput_trend': current_metrics.throughput_ops_per_sec - prev_metrics.throughput_ops_per_sec,
                'cache_hit_trend': current_metrics.cache_hit_rate - prev_metrics.cache_hit_rate
            }
        
        return {
            'current_metrics': current_metrics.to_dict(),
            'baseline_metrics': self._baseline_metrics.to_dict() if self._baseline_metrics else None,
            'trends': trends,
            'cache_statistics': self.cache.get_statistics(),
            'parallel_processing': self.parallel_engine.get_performance_stats(),
            'recommendations': [rec.to_dict() for rec in recommendations],
            'optimization_level': self.optimization_level.value,
            'monitoring_active': self._optimization_thread.is_alive() if self._optimization_thread else False,
            'metrics_history_size': len(self._metrics_history),
            'generated_at': datetime.now().isoformat()
        }
    
    def shutdown(self):
        """Shutdown performance optimizer"""
        self.stop_monitoring()
        self.parallel_engine.shutdown()


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer(
    optimization_level=OptimizationLevel.BALANCED
)