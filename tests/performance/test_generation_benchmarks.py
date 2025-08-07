"""
Comprehensive Performance Benchmarks for All Neuromorphic Generations
Includes regression detection, memory profiling, and scalability testing
"""

import pytest
import numpy as np
import pandas as pd
import time
import json
import sys
from pathlib import Path
from datetime import datetime
import gc
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import warnings
from contextlib import contextmanager
import psutil
import pickle

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer,
    NeuromorphicClusteringMethod,
    EchoStateNetwork,
    SpikingNeuralCluster,
    LiquidStateMachine
)
from insights_clustering.clustering import KMeansClusterer
from insights_clustering.parser import InsightsDataParser
from team_simulator.simulator import TeamCompositionSimulator

# Benchmark configuration
BENCHMARK_CONFIG = {
    'data_sizes': [50, 100, 200, 500, 1000, 2000],
    'cluster_counts': [3, 4, 6, 8, 10],
    'n_trials': 3,
    'warmup_trials': 1,
    'timeout_seconds': 600,
    'memory_limit_mb': 4096,
    'regression_threshold': 0.15,  # 15% performance degradation threshold
    'baseline_file': 'benchmark_baselines.json'
}

@dataclass
class BenchmarkResult:
    """Structure for benchmark results"""
    test_name: str
    method: str
    data_size: int
    n_clusters: int
    fit_time_mean: float
    fit_time_std: float
    memory_peak_mb: float
    memory_delta_mb: float
    silhouette_score: float
    throughput_samples_per_sec: float
    success_rate: float
    timestamp: str
    system_info: Dict
    
@dataclass
class RegressionResult:
    """Structure for regression test results"""
    test_name: str
    method: str
    baseline_time: float
    current_time: float
    performance_change_pct: float
    regression_detected: bool
    memory_change_mb: float
    quality_change: float


class PerformanceBenchmark:
    """Comprehensive performance benchmarking suite"""
    
    def setup_method(self):
        """Setup benchmarking environment"""
        np.random.seed(42)
        
        # System information
        self.system_info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'timestamp': datetime.now().isoformat()
        }
        
        # Load baseline results if they exist
        self.baselines = self._load_baselines()
        
        # Results storage
        self.current_results = []
        self.regression_results = []
        
    def _load_baselines(self) -> Dict:
        """Load baseline performance results"""
        baseline_path = Path(__file__).parent / BENCHMARK_CONFIG['baseline_file']
        if baseline_path.exists():
            try:
                with open(baseline_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {}
    
    def _save_baselines(self):
        """Save current results as new baselines"""
        baseline_path = Path(__file__).parent / BENCHMARK_CONFIG['baseline_file']
        
        # Convert current results to baseline format
        new_baselines = {}
        for result in self.current_results:
            key = f"{result.method}_{result.data_size}_{result.n_clusters}"
            new_baselines[key] = {
                'fit_time_mean': result.fit_time_mean,
                'memory_peak_mb': result.memory_peak_mb,
                'silhouette_score': result.silhouette_score,
                'timestamp': result.timestamp
            }
        
        with open(baseline_path, 'w') as f:
            json.dump(new_baselines, f, indent=2)
    
    @contextmanager
    def memory_profiler(self):
        """Context manager for memory profiling"""
        process = psutil.Process()
        gc.collect()  # Clean up before measurement
        
        memory_start = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = memory_start
        
        def monitor_memory():
            nonlocal peak_memory
            while True:
                try:
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.1)
                except:
                    break
        
        monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
        monitor_thread.start()
        
        try:
            yield
        finally:
            monitor_thread.join(timeout=0.1)
            memory_end = process.memory_info().rss / (1024 * 1024)
            
            # Store results in instance variables for access
            self._memory_start = memory_start
            self._memory_peak = peak_memory
            self._memory_delta = memory_end - memory_start
    
    def _create_benchmark_data(self, n_samples: int) -> pd.DataFrame:
        """Create benchmark data with known characteristics"""
        np.random.seed(42)  # Consistent data for benchmarking
        
        # Create diverse personality archetypes
        archetypes = [
            [80, 15, 10, 15],  # Strong Red
            [15, 80, 10, 15],  # Strong Blue  
            [10, 15, 80, 15],  # Strong Green
            [15, 10, 15, 80],  # Strong Yellow
            [50, 40, 20, 20],  # Red-Blue mix
            [25, 25, 50, 30],  # Green-Yellow mix
            [40, 30, 40, 30],  # Balanced complex
            [30, 30, 30, 30]   # Perfectly balanced
        ]
        
        data = []
        samples_per_archetype = n_samples // len(archetypes)
        remainder = n_samples % len(archetypes)
        
        for i, archetype in enumerate(archetypes):
            n_samples_this_type = samples_per_archetype + (1 if i < remainder else 0)
            
            for j in range(n_samples_this_type):
                # Add controlled noise for realism
                noise_scale = 5 + (i % 3) * 2  # Varying noise levels
                noise = np.random.randn(4) * noise_scale
                energies = np.array(archetype) + noise
                energies = np.clip(energies, 0.1, 100)
                
                # Normalize to sum to 100
                energies = (energies / np.sum(energies)) * 100
                
                data.append({
                    'employee_id': f'BENCH{i:02d}{j:05d}',
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'department': f'Dept_{i % 5}',
                    'experience_years': np.random.randint(0, 20)
                })
        
        return pd.DataFrame(data)
    
    def _benchmark_method(self, method_name: str, clusterer, features: pd.DataFrame, 
                         n_clusters: int, n_trials: int = 3) -> Dict:
        """Benchmark a specific clustering method"""
        fit_times = []
        memory_peaks = []
        memory_deltas = []
        silhouette_scores = []
        successful_runs = 0
        
        for trial in range(n_trials + BENCHMARK_CONFIG['warmup_trials']):
            try:
                with self.memory_profiler():
                    start_time = time.perf_counter()
                    
                    if hasattr(clusterer, 'random_state'):
                        clusterer.random_state = 42 + trial  # Vary seed slightly
                    
                    clusterer.fit(features)
                    
                    fit_time = time.perf_counter() - start_time
                
                # Skip warmup trials in results
                if trial >= BENCHMARK_CONFIG['warmup_trials']:
                    fit_times.append(fit_time)
                    memory_peaks.append(self._memory_peak)
                    memory_deltas.append(self._memory_delta)
                    
                    # Calculate clustering quality
                    labels = clusterer.get_cluster_assignments()
                    if len(np.unique(labels)) > 1:
                        from sklearn.metrics import silhouette_score
                        sil_score = silhouette_score(features.values, labels)
                        silhouette_scores.append(sil_score)
                    else:
                        silhouette_scores.append(-1.0)  # Poor clustering
                    
                    successful_runs += 1
                
                # Force cleanup between trials
                gc.collect()
                
            except Exception as e:
                if trial >= BENCHMARK_CONFIG['warmup_trials']:
                    # Record failure
                    fit_times.append(float('inf'))
                    memory_peaks.append(0)
                    memory_deltas.append(0)
                    silhouette_scores.append(-1.0)
                
                print(f"Trial {trial} failed for {method_name}: {str(e)}")
        
        if not fit_times or all(t == float('inf') for t in fit_times):
            return {
                'fit_time_mean': float('inf'),
                'fit_time_std': 0.0,
                'memory_peak_mb': 0.0,
                'memory_delta_mb': 0.0,
                'silhouette_score': -1.0,
                'success_rate': 0.0
            }
        
        valid_times = [t for t in fit_times if t != float('inf')]
        
        return {
            'fit_time_mean': np.mean(valid_times) if valid_times else float('inf'),
            'fit_time_std': np.std(valid_times) if len(valid_times) > 1 else 0.0,
            'memory_peak_mb': np.mean(memory_peaks),
            'memory_delta_mb': np.mean(memory_deltas),
            'silhouette_score': np.mean(silhouette_scores),
            'success_rate': successful_runs / n_trials
        }
    
    @pytest.mark.slow
    @pytest.mark.parametrize("data_size", [50, 100, 200, 500])
    @pytest.mark.parametrize("n_clusters", [4, 6])
    def test_neuromorphic_methods_benchmark(self, data_size, n_clusters):
        """Benchmark all neuromorphic clustering methods"""
        features = self._create_benchmark_data(data_size)
        clustering_features = features[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        # Methods to benchmark
        methods_config = {
            'kmeans': KMeansClusterer(n_clusters=n_clusters, random_state=42),
            'esn': NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=n_clusters,
                random_state=42,
                esn_params={'reservoir_size': 50}  # Smaller for speed
            ),
            'snn': NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
                n_clusters=n_clusters,
                random_state=42,
                snn_params={'n_neurons': 30}  # Smaller for speed
            ),
            'lsm': NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                n_clusters=n_clusters,
                random_state=42,
                lsm_params={'liquid_size': 40}  # Smaller for speed
            ),
            'hybrid': NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=n_clusters,
                random_state=42,
                esn_params={'reservoir_size': 40},
                snn_params={'n_neurons': 25},
                lsm_params={'liquid_size': 30}
            )
        }
        
        for method_name, clusterer in methods_config.items():
            print(f"Benchmarking {method_name} with {data_size} samples, {n_clusters} clusters...")
            
            benchmark_results = self._benchmark_method(
                method_name, clusterer, clustering_features, n_clusters,
                BENCHMARK_CONFIG['n_trials']
            )
            
            # Create benchmark result record
            result = BenchmarkResult(
                test_name='neuromorphic_methods_benchmark',
                method=method_name,
                data_size=data_size,
                n_clusters=n_clusters,
                fit_time_mean=benchmark_results['fit_time_mean'],
                fit_time_std=benchmark_results['fit_time_std'],
                memory_peak_mb=benchmark_results['memory_peak_mb'],
                memory_delta_mb=benchmark_results['memory_delta_mb'],
                silhouette_score=benchmark_results['silhouette_score'],
                throughput_samples_per_sec=data_size / benchmark_results['fit_time_mean'] if benchmark_results['fit_time_mean'] > 0 else 0,
                success_rate=benchmark_results['success_rate'],
                timestamp=datetime.now().isoformat(),
                system_info=self.system_info
            )
            
            self.current_results.append(result)
            
            # Performance assertions
            assert benchmark_results['success_rate'] >= 0.8, f"{method_name} had low success rate: {benchmark_results['success_rate']}"
            assert benchmark_results['fit_time_mean'] < BENCHMARK_CONFIG['timeout_seconds'], f"{method_name} exceeded timeout"
            assert benchmark_results['memory_peak_mb'] < BENCHMARK_CONFIG['memory_limit_mb'], f"{method_name} exceeded memory limit"
            
            # Check for regression
            self._check_regression(result)
    
    @pytest.mark.slow
    def test_scalability_benchmark(self):
        """Test scalability across different data sizes"""
        method_configs = {
            'kmeans': lambda n: KMeansClusterer(n_clusters=4, random_state=42),
            'esn': lambda n: NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=4,
                random_state=42,
                esn_params={'reservoir_size': min(50, max(20, n // 10))}  # Scale with data
            )
        }
        
        scalability_results = {}
        
        for method_name, method_factory in method_configs.items():
            method_results = []
            
            for data_size in BENCHMARK_CONFIG['data_sizes']:
                if data_size > 1000 and method_name != 'kmeans':
                    # Skip very large datasets for slower methods
                    continue
                    
                print(f"Testing {method_name} scalability with {data_size} samples...")
                
                features = self._create_benchmark_data(data_size)
                clustering_features = features[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
                
                clusterer = method_factory(data_size)
                
                benchmark_results = self._benchmark_method(
                    method_name, clusterer, clustering_features, 4, 2  # Fewer trials for speed
                )
                
                method_results.append({
                    'data_size': data_size,
                    'fit_time': benchmark_results['fit_time_mean'],
                    'memory_peak': benchmark_results['memory_peak_mb'],
                    'throughput': data_size / benchmark_results['fit_time_mean'] if benchmark_results['fit_time_mean'] > 0 else 0
                })
            
            scalability_results[method_name] = method_results
            
            # Analyze scalability
            if len(method_results) >= 3:
                self._analyze_scalability(method_name, method_results)
    
    def test_memory_efficiency_benchmark(self):
        """Test memory efficiency under various conditions"""
        data_sizes = [100, 300, 500]
        
        for data_size in data_sizes:
            features = self._create_benchmark_data(data_size)
            clustering_features = features[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
            
            # Test different reservoir sizes for memory impact
            reservoir_sizes = [25, 50, 100, 200]
            
            for reservoir_size in reservoir_sizes:
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    n_clusters=4,
                    random_state=42,
                    esn_params={'reservoir_size': reservoir_size}
                )
                
                with self.memory_profiler():
                    start_time = time.perf_counter()
                    clusterer.fit(clustering_features)
                    fit_time = time.perf_counter() - start_time
                
                # Memory efficiency metrics
                memory_per_sample = self._memory_delta / data_size
                memory_per_neuron = self._memory_delta / reservoir_size
                
                result = BenchmarkResult(
                    test_name='memory_efficiency_benchmark',
                    method=f'esn_reservoir_{reservoir_size}',
                    data_size=data_size,
                    n_clusters=4,
                    fit_time_mean=fit_time,
                    fit_time_std=0.0,
                    memory_peak_mb=self._memory_peak,
                    memory_delta_mb=self._memory_delta,
                    silhouette_score=0.0,  # Not calculated for efficiency test
                    throughput_samples_per_sec=data_size / fit_time,
                    success_rate=1.0,
                    timestamp=datetime.now().isoformat(),
                    system_info=self.system_info
                )
                
                self.current_results.append(result)
                
                # Memory efficiency assertions
                assert memory_per_sample < 1.0, f"High memory per sample: {memory_per_sample:.3f} MB"
                assert self._memory_peak < BENCHMARK_CONFIG['memory_limit_mb'], "Exceeded memory limit"
    
    @pytest.mark.slow
    def test_parallel_processing_benchmark(self):
        """Test performance with parallel processing"""
        data_size = 200
        features = self._create_benchmark_data(data_size)
        clustering_features = features[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        # Test sequential vs parallel processing
        n_jobs_options = [1, 2, 4]
        
        for n_jobs in n_jobs_options:
            if n_jobs > psutil.cpu_count():
                continue
                
            # Simulate parallel processing by running multiple clustering tasks
            def clustering_task():
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    n_clusters=4,
                    random_state=42
                )
                clusterer.fit(clustering_features)
                return clusterer.get_cluster_assignments()
            
            with self.memory_profiler():
                start_time = time.perf_counter()
                
                if n_jobs == 1:
                    # Sequential
                    results = [clustering_task() for _ in range(4)]
                else:
                    # Parallel
                    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                        futures = [executor.submit(clustering_task) for _ in range(4)]
                        results = [f.result() for f in futures]
                
                total_time = time.perf_counter() - start_time
            
            efficiency = (4 / total_time) / (4 / (4 * total_time / n_jobs))  # Parallel efficiency
            
            result = BenchmarkResult(
                test_name='parallel_processing_benchmark',
                method=f'parallel_{n_jobs}_jobs',
                data_size=data_size,
                n_clusters=4,
                fit_time_mean=total_time / 4,  # Average per task
                fit_time_std=0.0,
                memory_peak_mb=self._memory_peak,
                memory_delta_mb=self._memory_delta,
                silhouette_score=0.0,
                throughput_samples_per_sec=data_size * 4 / total_time,
                success_rate=1.0 if len(results) == 4 else len(results) / 4,
                timestamp=datetime.now().isoformat(),
                system_info=self.system_info
            )
            
            self.current_results.append(result)
            
            assert len(results) == 4, f"Not all parallel tasks completed successfully"
            
    def test_quality_vs_performance_tradeoff(self):
        """Test the tradeoff between clustering quality and performance"""
        data_size = 300
        features = self._create_benchmark_data(data_size)
        clustering_features = features[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        # Test different parameter configurations
        configs = [
            ('fast', {'reservoir_size': 25, 'n_neurons': 20, 'liquid_size': 25}),
            ('balanced', {'reservoir_size': 50, 'n_neurons': 30, 'liquid_size': 40}),
            ('quality', {'reservoir_size': 100, 'n_neurons': 50, 'liquid_size': 64})
        ]
        
        tradeoff_results = []
        
        for config_name, params in configs:
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4,
                random_state=42,
                esn_params={'reservoir_size': params['reservoir_size']},
                snn_params={'n_neurons': params['n_neurons']},
                lsm_params={'liquid_size': params['liquid_size']}
            )
            
            benchmark_results = self._benchmark_method(
                config_name, clusterer, clustering_features, 4, 2
            )
            
            tradeoff_results.append({
                'config': config_name,
                'fit_time': benchmark_results['fit_time_mean'],
                'quality': benchmark_results['silhouette_score'],
                'memory': benchmark_results['memory_peak_mb']
            })
        
        # Analyze tradeoffs
        for i, result in enumerate(tradeoff_results):
            print(f"{result['config']}: Time={result['fit_time']:.2f}s, "
                  f"Quality={result['quality']:.3f}, Memory={result['memory']:.1f}MB")
            
            if result['config'] == 'fast':
                # Fast config should be quickest but may have lower quality
                assert result['fit_time'] <= min(r['fit_time'] for r in tradeoff_results) * 1.2
            elif result['config'] == 'quality':
                # Quality config should have best silhouette score
                assert result['quality'] >= max(r['quality'] for r in tradeoff_results) * 0.9
    
    def _check_regression(self, current_result: BenchmarkResult):
        """Check for performance regression against baseline"""
        baseline_key = f"{current_result.method}_{current_result.data_size}_{current_result.n_clusters}"
        
        if baseline_key in self.baselines:
            baseline = self.baselines[baseline_key]
            
            # Performance change calculation
            baseline_time = baseline['fit_time_mean']
            current_time = current_result.fit_time_mean
            
            if baseline_time > 0 and current_time != float('inf'):
                performance_change = (current_time - baseline_time) / baseline_time
                
                regression_detected = performance_change > BENCHMARK_CONFIG['regression_threshold']
                
                # Memory change
                memory_change = current_result.memory_peak_mb - baseline.get('memory_peak_mb', 0)
                
                # Quality change
                quality_change = current_result.silhouette_score - baseline.get('silhouette_score', 0)
                
                regression_result = RegressionResult(
                    test_name=current_result.test_name,
                    method=current_result.method,
                    baseline_time=baseline_time,
                    current_time=current_time,
                    performance_change_pct=performance_change * 100,
                    regression_detected=regression_detected,
                    memory_change_mb=memory_change,
                    quality_change=quality_change
                )
                
                self.regression_results.append(regression_result)
                
                if regression_detected:
                    print(f"WARNING: Performance regression detected for {baseline_key}: "
                          f"{performance_change*100:.1f}% slower")
    
    def _analyze_scalability(self, method_name: str, results: List[Dict]):
        """Analyze scalability characteristics"""
        data_sizes = [r['data_size'] for r in results]
        fit_times = [r['fit_time'] for r in results if r['fit_time'] != float('inf')]
        
        if len(fit_times) >= 3:
            # Simple complexity analysis
            # Check if scaling is approximately linear, quadratic, etc.
            
            # Linear fit: time = a * size + b
            sizes_array = np.array(data_sizes[:len(fit_times)])
            times_array = np.array(fit_times)
            
            # Linear regression
            A = np.vstack([sizes_array, np.ones(len(sizes_array))]).T
            linear_coeffs, _, _, _ = np.linalg.lstsq(A, times_array, rcond=None)
            linear_fit = linear_coeffs[0] * sizes_array + linear_coeffs[1]
            linear_r2 = 1 - np.sum((times_array - linear_fit)**2) / np.sum((times_array - np.mean(times_array))**2)
            
            # Quadratic fit: time = a * size^2 + b * size + c
            A_quad = np.vstack([sizes_array**2, sizes_array, np.ones(len(sizes_array))]).T
            quad_coeffs, _, _, _ = np.linalg.lstsq(A_quad, times_array, rcond=None)
            quad_fit = quad_coeffs[0] * sizes_array**2 + quad_coeffs[1] * sizes_array + quad_coeffs[2]
            quad_r2 = 1 - np.sum((times_array - quad_fit)**2) / np.sum((times_array - np.mean(times_array))**2)
            
            print(f"Scalability analysis for {method_name}:")
            print(f"  Linear fit R²: {linear_r2:.3f}")
            print(f"  Quadratic fit R²: {quad_r2:.3f}")
            
            if quad_r2 > linear_r2 + 0.1:
                print(f"  {method_name} appears to scale quadratically")
                assert quad_coeffs[0] < 1e-6, f"Quadratic scaling coefficient too high: {quad_coeffs[0]}"
            else:
                print(f"  {method_name} appears to scale linearly")
                assert linear_coeffs[0] < 0.1, f"Linear scaling coefficient too high: {linear_coeffs[0]}"
    
    def teardown_method(self):
        """Save results and cleanup"""
        if self.current_results:
            # Save detailed results
            results_file = Path(__file__).parent / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w') as f:
                json.dump([asdict(r) for r in self.current_results], f, indent=2, default=str)
            
            # Save as new baselines (optional - uncomment to update baselines)
            # self._save_baselines()
            
            # Print summary
            print("\n" + "="*80)
            print("BENCHMARK SUMMARY")
            print("="*80)
            
            # Group by method
            methods = set(r.method for r in self.current_results)
            for method in sorted(methods):
                method_results = [r for r in self.current_results if r.method == method]
                
                if method_results:
                    avg_time = np.mean([r.fit_time_mean for r in method_results if r.fit_time_mean != float('inf')])
                    avg_memory = np.mean([r.memory_peak_mb for r in method_results])
                    avg_quality = np.mean([r.silhouette_score for r in method_results if r.silhouette_score > -1])
                    success_rate = np.mean([r.success_rate for r in method_results])
                    
                    print(f"{method:15s}: Time={avg_time:6.2f}s, Memory={avg_memory:6.1f}MB, "
                          f"Quality={avg_quality:5.3f}, Success={success_rate:5.1%}")
            
            # Print regression warnings
            if self.regression_results:
                print("\nREGRESSION WARNINGS:")
                for reg in self.regression_results:
                    if reg.regression_detected:
                        print(f"  {reg.method}: {reg.performance_change_pct:+.1f}% performance change")
        
        # Cleanup
        gc.collect()


if __name__ == '__main__':
    # Run benchmarks directly
    benchmark = PerformanceBenchmark()
    benchmark.setup_method()
    
    try:
        # Run key benchmarks
        benchmark.test_neuromorphic_methods_benchmark(200, 4)
        benchmark.test_scalability_benchmark()
        benchmark.test_memory_efficiency_benchmark()
        
    finally:
        benchmark.teardown_method()