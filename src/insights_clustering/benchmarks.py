"""
Generation 3 Comprehensive Performance Benchmarks
Advanced benchmarking suite for neuromorphic clustering with detailed analytics
"""

import asyncio
import logging
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
import psutil
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark result"""
    test_name: str
    dataset_size: int
    processing_time: float
    memory_peak_mb: float
    cpu_percent: float
    gpu_memory_mb: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_samples_per_sec: float = 0.0
    accuracy_score: float = 0.0
    error_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (throughput per resource unit)"""
        resource_usage = (self.memory_peak_mb / 1000) + (self.cpu_percent / 100)
        return self.throughput_samples_per_sec / max(resource_usage, 0.1)


@dataclass 
class BenchmarkSuite:
    """Comprehensive benchmark suite results"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    results: List[BenchmarkResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Total benchmark suite duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def total_samples_processed(self) -> int:
        """Total samples processed across all benchmarks"""
        return sum(r.dataset_size for r in self.results)
    
    @property
    def average_throughput(self) -> float:
        """Average throughput across all benchmarks"""
        throughputs = [r.throughput_samples_per_sec for r in self.results if r.throughput_samples_per_sec > 0]
        return statistics.mean(throughputs) if throughputs else 0.0
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak memory usage across all benchmarks"""
        return max((r.memory_peak_mb for r in self.results), default=0.0)


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking"""
    
    @staticmethod
    def generate_personality_data(n_samples: int, n_features: int = 4,
                                noise_level: float = 0.1,
                                cluster_separation: float = 2.0,
                                random_state: int = 42) -> pd.DataFrame:
        """Generate synthetic personality energy data"""
        np.random.seed(random_state)
        
        # Create cluster centers for RBGY energies
        cluster_centers = [
            [80, 20, 30, 70],  # Red-Yellow dominant
            [30, 70, 80, 20],  # Blue-Green dominant  
            [60, 60, 40, 40],  # Blue-Red balanced
            [40, 40, 60, 60]   # Green-Yellow balanced
        ]
        
        data_points = []
        cluster_labels = []
        
        samples_per_cluster = n_samples // len(cluster_centers)
        
        for i, center in enumerate(cluster_centers):
            for _ in range(samples_per_cluster):
                # Add noise to cluster center
                point = [
                    max(0, min(100, center[j] + np.random.normal(0, cluster_separation * noise_level * 10)))
                    for j in range(len(center))
                ]
                data_points.append(point)
                cluster_labels.append(i)
        
        # Add remaining samples randomly
        remaining = n_samples - len(data_points)
        for _ in range(remaining):
            random_center = np.random.choice(len(cluster_centers))
            center = cluster_centers[random_center]
            point = [
                max(0, min(100, center[j] + np.random.normal(0, cluster_separation * noise_level * 10)))
                for j in range(len(center))
            ]
            data_points.append(point)
            cluster_labels.append(random_center)
        
        # Create DataFrame
        columns = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        if n_features > 4:
            # Add synthetic features
            for i in range(4, n_features):
                columns.append(f'feature_{i}')
            
            for point in data_points:
                while len(point) < n_features:
                    point.append(np.random.uniform(0, 100))
        
        df = pd.DataFrame(data_points, columns=columns[:n_features])
        df['true_cluster'] = cluster_labels
        
        return df
    
    @staticmethod
    def generate_streaming_data(total_samples: int, 
                              batch_size: int = 100,
                              concept_drift_at: Optional[List[int]] = None):
        """Generate streaming data with optional concept drift"""
        concept_drift_points = concept_drift_at or []
        current_drift = 0
        
        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            current_batch_size = end_idx - start_idx
            
            # Check for concept drift
            if current_drift < len(concept_drift_points) and start_idx >= concept_drift_points[current_drift]:
                current_drift += 1
                # Modify cluster characteristics for drift
                noise_level = 0.1 + (current_drift * 0.05)
                separation = 2.0 - (current_drift * 0.2)
            else:
                noise_level = 0.1
                separation = 2.0
            
            batch_data = DatasetGenerator.generate_personality_data(
                current_batch_size,
                noise_level=noise_level,
                cluster_separation=separation,
                random_state=42 + start_idx
            )
            
            yield batch_data


class PerformanceMonitor:
    """Monitor system performance during benchmarks"""
    
    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.monitoring = False
        self.samples = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.monitoring = True
        self.samples = []
        
        import threading
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        if not self.samples:
            return {}
        
        # Calculate statistics
        cpu_values = [s['cpu_percent'] for s in self.samples]
        memory_values = [s['memory_percent'] for s in self.samples]
        memory_mb_values = [s['memory_mb'] for s in self.samples]
        
        stats = {
            'sample_count': len(self.samples),
            'duration': len(self.samples) * self.sample_interval,
            'cpu_percent': {
                'mean': statistics.mean(cpu_values),
                'max': max(cpu_values),
                'min': min(cpu_values),
                'std': statistics.stdev(cpu_values) if len(cpu_values) > 1 else 0
            },
            'memory_percent': {
                'mean': statistics.mean(memory_values),
                'max': max(memory_values),
                'min': min(memory_values),
                'std': statistics.stdev(memory_values) if len(memory_values) > 1 else 0
            },
            'memory_peak_mb': max(memory_mb_values)
        }
        
        # GPU stats if available
        gpu_samples = [s.get('gpu_memory_mb', 0) for s in self.samples if 'gpu_memory_mb' in s]
        if gpu_samples:
            stats['gpu_memory_mb'] = {
                'mean': statistics.mean(gpu_samples),
                'max': max(gpu_samples),
                'min': min(gpu_samples)
            }
        
        return stats
    
    def _monitor_loop(self):
        """Monitor system resources"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                system_memory = psutil.virtual_memory()
                
                sample = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_mb': memory_info.rss / (1024 * 1024),
                    'memory_percent': system_memory.percent
                }
                
                # Add GPU info if available
                try:
                    from .gpu_acceleration import gpu_ops
                    gpu_metrics = gpu_ops.gpu_manager.get_gpu_metrics()
                    if gpu_metrics:
                        sample['gpu_memory_mb'] = gpu_metrics.gpu_memory_used_mb
                except:
                    pass
                
                self.samples.append(sample)
                time.sleep(self.sample_interval)
                
            except Exception as e:
                logger.debug(f"Monitoring sample failed: {e}")
                time.sleep(self.sample_interval)


class NeuromorphicBenchmark:
    """Comprehensive neuromorphic clustering benchmark suite"""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.results_dir = Path.home() / '.neuromorphic_benchmarks'
        self.results_dir.mkdir(exist_ok=True)
    
    def run_single_benchmark(self, 
                           test_name: str,
                           clustering_func: Callable,
                           dataset: pd.DataFrame,
                           expected_clusters: Optional[np.ndarray] = None,
                           **kwargs) -> BenchmarkResult:
        """Run a single benchmark test"""
        logger.info(f"Running benchmark: {test_name}")
        
        # Start monitoring
        self.monitor.start_monitoring()
        start_time = time.perf_counter()
        
        error_count = 0
        result = None
        cache_hit_rate = 0.0
        
        try:
            # Run clustering
            result = clustering_func(dataset, **kwargs)
            
            # Get cache statistics if available
            try:
                from .caching import neuromorphic_cache
                cache_stats = neuromorphic_cache.get_stats()
                l1_stats = cache_stats.get('l1', {})
                if hasattr(l1_stats, 'hit_rate'):
                    cache_hit_rate = l1_stats.hit_rate
            except:
                pass
                
        except Exception as e:
            logger.error(f"Benchmark {test_name} failed: {e}")
            error_count = 1
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # Stop monitoring and get stats
        monitor_stats = self.monitor.stop_monitoring()
        
        # Calculate accuracy if expected clusters provided
        accuracy_score = 0.0
        if expected_clusters is not None and result is not None:
            try:
                if hasattr(result, 'cluster_assignments'):
                    predicted_labels = result.cluster_assignments
                elif isinstance(result, dict) and 'cluster_assignments' in result:
                    predicted_labels = result['cluster_assignments']
                else:
                    predicted_labels = result
                
                # Use adjusted rand index for accuracy
                from sklearn.metrics import adjusted_rand_score
                accuracy_score = adjusted_rand_score(expected_clusters, predicted_labels)
            except Exception as e:
                logger.debug(f"Accuracy calculation failed: {e}")
        
        # Calculate throughput
        throughput = len(dataset) / processing_time if processing_time > 0 else 0
        
        return BenchmarkResult(
            test_name=test_name,
            dataset_size=len(dataset),
            processing_time=processing_time,
            memory_peak_mb=monitor_stats.get('memory_peak_mb', 0),
            cpu_percent=monitor_stats.get('cpu_percent', {}).get('mean', 0),
            gpu_memory_mb=monitor_stats.get('gpu_memory_mb', {}).get('max', 0),
            cache_hit_rate=cache_hit_rate,
            throughput_samples_per_sec=throughput,
            accuracy_score=accuracy_score,
            error_count=error_count,
            metadata={
                'monitor_stats': monitor_stats,
                'dataset_shape': dataset.shape,
                'kwargs': kwargs
            }
        )
    
    def run_scalability_benchmark(self, 
                                dataset_sizes: List[int] = None,
                                methods: List[str] = None) -> BenchmarkSuite:
        """Run scalability benchmarks across different dataset sizes"""
        dataset_sizes = dataset_sizes or [100, 500, 1000, 5000, 10000]
        methods = methods or ['hybrid_reservoir', 'echo_state_network', 'kmeans_fallback']
        
        suite = BenchmarkSuite(
            suite_name='scalability_benchmark',
            start_time=datetime.now(),
            end_time=datetime.now(),  # Will update at end
            system_info=self._get_system_info()
        )
        
        for size in dataset_sizes:
            logger.info(f"Testing scalability with {size} samples")
            
            # Generate test dataset
            dataset = DatasetGenerator.generate_personality_data(
                n_samples=size,
                random_state=42
            )
            expected_clusters = dataset['true_cluster'].values
            dataset = dataset.drop('true_cluster', axis=1)
            
            for method in methods:
                test_name = f"scalability_{method}_{size}"
                
                def clustering_func(data, **kwargs):
                    if method == 'kmeans_fallback':
                        from sklearn.cluster import KMeans
                        kmeans = KMeans(n_clusters=4, random_state=42)
                        labels = kmeans.fit_predict(data.values)
                        return {'cluster_assignments': labels}
                    else:
                        from .neuromorphic_clustering import NeuromorphicClusterer
                        clusterer = NeuromorphicClusterer(
                            method=method,
                            n_clusters=4,
                            enable_fallback=True
                        )
                        clusterer.fit(data)
                        return {'cluster_assignments': clusterer.get_cluster_assignments()}
                
                result = self.run_single_benchmark(
                    test_name, clustering_func, dataset, expected_clusters
                )
                suite.results.append(result)
        
        suite.end_time = datetime.now()
        return suite
    
    def run_performance_comparison(self) -> BenchmarkSuite:
        """Run comprehensive performance comparison"""
        suite = BenchmarkSuite(
            suite_name='performance_comparison',
            start_time=datetime.now(),
            end_time=datetime.now(),
            system_info=self._get_system_info()
        )
        
        # Standard dataset
        dataset = DatasetGenerator.generate_personality_data(n_samples=2000, random_state=42)
        expected_clusters = dataset['true_cluster'].values
        dataset = dataset.drop('true_cluster', axis=1)
        
        # Test different configurations
        test_configs = [
            ('neuromorphic_baseline', {'method': 'hybrid_reservoir', 'enable_gpu': False, 'enable_caching': False}),
            ('neuromorphic_cached', {'method': 'hybrid_reservoir', 'enable_gpu': False, 'enable_caching': True}),
            ('neuromorphic_gpu', {'method': 'hybrid_reservoir', 'enable_gpu': True, 'enable_caching': False}),
            ('neuromorphic_optimized', {'method': 'hybrid_reservoir', 'enable_gpu': True, 'enable_caching': True}),
            ('traditional_kmeans', {'method': 'kmeans', 'n_clusters': 4}),
            ('traditional_dbscan', {'method': 'dbscan', 'eps': 0.5})
        ]
        
        for test_name, config in test_configs:
            def clustering_func(data, **kwargs):
                if config['method'] == 'kmeans':
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=config['n_clusters'], random_state=42)
                    labels = kmeans.fit_predict(data.values)
                    return {'cluster_assignments': labels}
                elif config['method'] == 'dbscan':
                    from sklearn.cluster import DBSCAN
                    dbscan = DBSCAN(eps=config['eps'], min_samples=5)
                    labels = dbscan.fit_predict(data.values)
                    return {'cluster_assignments': labels}
                else:
                    from .neuromorphic_clustering import NeuromorphicClusterer
                    clusterer = NeuromorphicClusterer(
                        method=config['method'],
                        n_clusters=4
                    )
                    clusterer.fit(data)
                    return {'cluster_assignments': clusterer.get_cluster_assignments()}
            
            result = self.run_single_benchmark(
                test_name, clustering_func, dataset, expected_clusters, **config
            )
            suite.results.append(result)
        
        suite.end_time = datetime.now()
        return suite
    
    def run_streaming_benchmark(self, 
                              total_samples: int = 5000,
                              batch_size: int = 100) -> BenchmarkSuite:
        """Run streaming data benchmark"""
        suite = BenchmarkSuite(
            suite_name='streaming_benchmark',
            start_time=datetime.now(),
            end_time=datetime.now(),
            system_info=self._get_system_info()
        )
        
        from .scaling import StreamingClusteringEngine
        streaming_engine = StreamingClusteringEngine(chunk_size=batch_size)
        
        # Generate streaming data with concept drift
        concept_drift_points = [total_samples // 3, 2 * total_samples // 3]
        data_stream = DatasetGenerator.generate_streaming_data(
            total_samples, batch_size, concept_drift_points
        )
        
        def streaming_clustering(data_generator, **kwargs):
            results = []
            start_time = time.time()
            
            streaming_engine.process_data_stream(data_generator)
            
            processing_time = time.time() - start_time
            stats = streaming_engine.get_streaming_stats()
            
            return {
                'processing_time': processing_time,
                'samples_processed': stats['seen_samples'],
                'final_chunk_size': stats['current_chunk_size'],
                'model_initialized': stats['model_initialized']
            }
        
        result = self.run_single_benchmark(
            'streaming_clustering',
            lambda x: streaming_clustering(data_stream),
            pd.DataFrame({'dummy': [1]})  # Placeholder
        )
        
        suite.results.append(result)
        suite.end_time = datetime.now()
        return suite
    
    def run_memory_benchmark(self,
                           dataset_sizes: List[int] = None) -> BenchmarkSuite:
        """Run memory efficiency benchmark"""
        dataset_sizes = dataset_sizes or [1000, 5000, 10000, 50000]
        
        suite = BenchmarkSuite(
            suite_name='memory_benchmark',
            start_time=datetime.now(),
            end_time=datetime.now(),
            system_info=self._get_system_info()
        )
        
        for size in dataset_sizes:
            # Test memory-mapped vs regular processing
            dataset = DatasetGenerator.generate_personality_data(n_samples=size)
            expected_clusters = dataset['true_cluster'].values
            dataset = dataset.drop('true_cluster', axis=1)
            
            # Regular processing
            def regular_clustering(data, **kwargs):
                from ..performance import gen3_optimizer
                result = gen3_optimizer.optimize_clustering_pipeline(
                    data, n_clusters=4, optimization_level='conservative'
                )
                return result
            
            result = self.run_single_benchmark(
                f'memory_regular_{size}',
                regular_clustering,
                dataset,
                expected_clusters
            )
            suite.results.append(result)
            
            # Memory-mapped processing (for large datasets)
            if size >= 5000:
                def mmap_clustering(data, **kwargs):
                    from ..performance import gen3_optimizer
                    result = gen3_optimizer.optimize_clustering_pipeline(
                        data, n_clusters=4, optimization_level='balanced'
                    )
                    return result
                
                result = self.run_single_benchmark(
                    f'memory_mmap_{size}',
                    mmap_clustering,
                    dataset,
                    expected_clusters
                )
                suite.results.append(result)
        
        suite.end_time = datetime.now()
        return suite
    
    async def run_concurrent_benchmark(self, 
                                     n_concurrent_jobs: int = 5) -> BenchmarkSuite:
        """Run concurrent processing benchmark"""
        suite = BenchmarkSuite(
            suite_name='concurrent_benchmark',
            start_time=datetime.now(),
            end_time=datetime.now(),
            system_info=self._get_system_info()
        )
        
        # Create multiple datasets
        datasets = []
        for i in range(n_concurrent_jobs):
            dataset = DatasetGenerator.generate_personality_data(
                n_samples=1000, random_state=42 + i
            )
            datasets.append(dataset.drop('true_cluster', axis=1))
        
        async def concurrent_clustering():
            from ..performance import async_manager
            
            def clustering_func(data):
                from .neuromorphic_clustering import NeuromorphicClusterer
                clusterer = NeuromorphicClusterer(n_clusters=4)
                clusterer.fit(data)
                return clusterer.get_cluster_assignments()
            
            results = await async_manager.process_batch_async(
                datasets, clustering_func
            )
            return results
        
        start_time = time.perf_counter()
        self.monitor.start_monitoring()
        
        try:
            results = await concurrent_clustering()
            success_count = len([r for r in results if r is not None])
        except Exception as e:
            logger.error(f"Concurrent benchmark failed: {e}")
            success_count = 0
        
        end_time = time.perf_counter()
        monitor_stats = self.monitor.stop_monitoring()
        
        processing_time = end_time - start_time
        total_samples = sum(len(d) for d in datasets)
        throughput = total_samples / processing_time if processing_time > 0 else 0
        
        result = BenchmarkResult(
            test_name='concurrent_clustering',
            dataset_size=total_samples,
            processing_time=processing_time,
            memory_peak_mb=monitor_stats.get('memory_peak_mb', 0),
            cpu_percent=monitor_stats.get('cpu_percent', {}).get('mean', 0),
            throughput_samples_per_sec=throughput,
            error_count=n_concurrent_jobs - success_count,
            metadata={
                'concurrent_jobs': n_concurrent_jobs,
                'successful_jobs': success_count,
                'monitor_stats': monitor_stats
            }
        )
        
        suite.results.append(result)
        suite.end_time = datetime.now()
        return suite
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, BenchmarkSuite]:
        """Run comprehensive benchmark suite"""
        logger.info("Starting comprehensive benchmark suite")
        
        suites = {}
        
        try:
            # Scalability benchmark
            logger.info("Running scalability benchmarks")
            suites['scalability'] = self.run_scalability_benchmark()
            
            # Performance comparison
            logger.info("Running performance comparison benchmarks")
            suites['performance'] = self.run_performance_comparison()
            
            # Memory benchmark
            logger.info("Running memory benchmarks")
            suites['memory'] = self.run_memory_benchmark()
            
            # Streaming benchmark
            logger.info("Running streaming benchmarks")
            suites['streaming'] = self.run_streaming_benchmark()
            
            # Concurrent benchmark
            logger.info("Running concurrent benchmarks")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            suites['concurrent'] = loop.run_until_complete(
                self.run_concurrent_benchmark()
            )
            loop.close()
            
        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
        
        # Save results
        self.save_benchmark_results(suites)
        
        return suites
    
    def save_benchmark_results(self, suites: Dict[str, BenchmarkSuite]):
        """Save benchmark results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for suite_name, suite in suites.items():
            # Save JSON results
            json_file = self.results_dir / f"benchmark_{suite_name}_{timestamp}.json"
            
            results_data = {
                'suite_name': suite.suite_name,
                'start_time': suite.start_time.isoformat(),
                'end_time': suite.end_time.isoformat(),
                'duration': suite.duration,
                'total_samples_processed': suite.total_samples_processed,
                'average_throughput': suite.average_throughput,
                'peak_memory_mb': suite.peak_memory_mb,
                'system_info': suite.system_info,
                'results': [
                    {
                        'test_name': r.test_name,
                        'dataset_size': r.dataset_size,
                        'processing_time': r.processing_time,
                        'memory_peak_mb': r.memory_peak_mb,
                        'cpu_percent': r.cpu_percent,
                        'gpu_memory_mb': r.gpu_memory_mb,
                        'cache_hit_rate': r.cache_hit_rate,
                        'throughput_samples_per_sec': r.throughput_samples_per_sec,
                        'accuracy_score': r.accuracy_score,
                        'efficiency_score': r.efficiency_score,
                        'error_count': r.error_count,
                        'timestamp': r.timestamp.isoformat(),
                        'metadata': r.metadata
                    }
                    for r in suite.results
                ]
            }
            
            with open(json_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            logger.info(f"Saved benchmark results to {json_file}")
    
    def generate_benchmark_report(self, suites: Dict[str, BenchmarkSuite]) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("# Neuromorphic Clustering Benchmark Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # System information
        if suites:
            system_info = list(suites.values())[0].system_info
            report.append("## System Information")
            report.append(f"- CPU: {system_info.get('cpu_count', 'Unknown')} cores")
            report.append(f"- Memory: {system_info.get('memory_gb', 'Unknown')} GB")
            report.append(f"- Python: {system_info.get('python_version', 'Unknown')}")
            report.append("")
        
        # Summary statistics
        report.append("## Summary Statistics")
        for suite_name, suite in suites.items():
            report.append(f"### {suite_name.title()} Benchmark")
            report.append(f"- Duration: {suite.duration:.2f} seconds")
            report.append(f"- Total samples: {suite.total_samples_processed:,}")
            report.append(f"- Average throughput: {suite.average_throughput:.2f} samples/sec")
            report.append(f"- Peak memory: {suite.peak_memory_mb:.2f} MB")
            report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        for suite_name, suite in suites.items():
            report.append(f"### {suite_name.title()} Results")
            report.append("")
            report.append("| Test | Samples | Time(s) | Memory(MB) | Throughput | Accuracy | Efficiency |")
            report.append("|------|---------|---------|------------|------------|----------|------------|")
            
            for result in suite.results:
                report.append(
                    f"| {result.test_name} | {result.dataset_size:,} | "
                    f"{result.processing_time:.2f} | {result.memory_peak_mb:.1f} | "
                    f"{result.throughput_samples_per_sec:.1f} | {result.accuracy_score:.3f} | "
                    f"{result.efficiency_score:.2f} |"
                )
            
            report.append("")
        
        # Performance insights
        report.append("## Performance Insights")
        
        # Find best performing tests
        all_results = []
        for suite in suites.values():
            all_results.extend(suite.results)
        
        if all_results:
            # Best throughput
            best_throughput = max(all_results, key=lambda r: r.throughput_samples_per_sec)
            report.append(f"- **Best Throughput**: {best_throughput.test_name} "
                         f"({best_throughput.throughput_samples_per_sec:.1f} samples/sec)")
            
            # Best accuracy
            best_accuracy = max(all_results, key=lambda r: r.accuracy_score)
            report.append(f"- **Best Accuracy**: {best_accuracy.test_name} "
                         f"({best_accuracy.accuracy_score:.3f})")
            
            # Best efficiency
            best_efficiency = max(all_results, key=lambda r: r.efficiency_score)
            report.append(f"- **Best Efficiency**: {best_efficiency.test_name} "
                         f"({best_efficiency.efficiency_score:.2f})")
            
            # Memory usage analysis
            memory_intensive = max(all_results, key=lambda r: r.memory_peak_mb)
            memory_efficient = min(all_results, key=lambda r: r.memory_peak_mb)
            report.append(f"- **Memory Range**: {memory_efficient.memory_peak_mb:.1f}MB - "
                         f"{memory_intensive.memory_peak_mb:.1f}MB")
        
        return "\n".join(report)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'timestamp': datetime.now().isoformat()
        }


# Global benchmark instance
benchmark_suite = NeuromorphicBenchmark()


def run_quick_benchmark() -> BenchmarkSuite:
    """Run a quick benchmark for basic performance testing"""
    suite = BenchmarkSuite(
        suite_name='quick_benchmark',
        start_time=datetime.now(),
        end_time=datetime.now()
    )
    
    # Quick test with small dataset
    dataset = DatasetGenerator.generate_personality_data(n_samples=500)
    expected_clusters = dataset['true_cluster'].values
    dataset = dataset.drop('true_cluster', axis=1)
    
    def quick_clustering(data, **kwargs):
        from .neuromorphic_clustering import NeuromorphicClusterer
        clusterer = NeuromorphicClusterer(n_clusters=4, enable_fallback=True)
        clusterer.fit(data)
        return {'cluster_assignments': clusterer.get_cluster_assignments()}
    
    result = benchmark_suite.run_single_benchmark(
        'quick_test', quick_clustering, dataset, expected_clusters
    )
    
    suite.results.append(result)
    suite.end_time = datetime.now()
    
    return suite


def benchmark_neuromorphic_vs_traditional() -> BenchmarkSuite:
    """Compare neuromorphic clustering with traditional methods"""
    return benchmark_suite.run_performance_comparison()