"""
Benchmarking Suite for Neuromorphic vs Traditional Clustering
Comprehensive comparison of neuromorphic clustering approaches against K-means
"""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.datasets import make_blobs

from .clustering import KMeansClusterer
from .neuromorphic_clustering import (
    NeuromorphicClusterer, 
    NeuromorphicClusteringMethod,
    ClusteringMetrics
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    method_name: str
    dataset_name: str
    n_samples: int
    n_features: int
    n_clusters: int
    
    # Performance metrics
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    adjusted_rand_score: float
    normalized_mutual_info: float
    
    # Neuromorphic-specific metrics
    cluster_stability: Optional[float] = None
    interpretability_score: Optional[float] = None
    temporal_coherence: Optional[float] = None
    
    # Computational metrics
    fit_time: float = 0.0
    predict_time: float = 0.0
    memory_usage_mb: float = 0.0
    computational_efficiency: Optional[float] = None
    
    # Cluster quality
    cluster_balance: float = 0.0
    cluster_separation: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate_insights_discovery_data(self, n_samples: int = 200, 
                                       n_clusters: int = 4,
                                       cluster_std: float = 10.0) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic Insights Discovery personality data
        
        Args:
            n_samples: Number of employees
            n_clusters: Number of personality clusters
            cluster_std: Standard deviation within clusters
            
        Returns:
            DataFrame with personality data and true cluster labels
        """
        # Create cluster centers representing different personality profiles
        centers = np.array([
            [70, 20, 30, 30],  # Red-dominant (Director)
            [20, 70, 25, 25],  # Blue-dominant (Thinker)
            [25, 25, 70, 20],  # Green-dominant (Supporter)
            [30, 25, 20, 70],  # Yellow-dominant (Inspirational)
        ])[:n_clusters]
        
        # Generate clustered data
        X, y = make_blobs(n_samples=n_samples, centers=centers, 
                         cluster_std=cluster_std, random_state=self.random_state)
        
        # Ensure values are in valid range [0, 100]
        X = np.clip(X, 0, 100)
        
        # Normalize each row to sum to 100 (energy constraint)
        X = (X / X.sum(axis=1, keepdims=True)) * 100
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=['red_energy', 'blue_energy', 'green_energy', 'yellow_energy'])
        df['employee_id'] = [f'EMP{i:04d}' for i in range(n_samples)]
        
        return df, y
    
    def generate_noisy_data(self, n_samples: int = 200, 
                           noise_level: float = 0.3) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate data with significant noise to test robustness"""
        df, y = self.generate_insights_discovery_data(n_samples, n_clusters=4, cluster_std=15.0)
        
        # Add noise
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        noise = np.random.randn(n_samples, 4) * noise_level * 20
        
        for i, col in enumerate(energy_cols):
            df[col] += noise[:, i]
        
        # Ensure valid ranges and normalize
        df[energy_cols] = np.clip(df[energy_cols], 0, 100)
        row_sums = df[energy_cols].sum(axis=1)
        for col in energy_cols:
            df[col] = (df[col] / row_sums) * 100
        
        return df, y
    
    def generate_imbalanced_data(self, n_samples: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate imbalanced cluster data"""
        # Different cluster sizes
        cluster_sizes = [int(n_samples * p) for p in [0.5, 0.3, 0.15, 0.05]]
        
        all_data = []
        all_labels = []
        
        centers = np.array([
            [70, 20, 30, 30],  # Red-dominant
            [20, 70, 25, 25],  # Blue-dominant
            [25, 25, 70, 20],  # Green-dominant
            [30, 25, 20, 70],  # Yellow-dominant
        ])
        
        for cluster_id, (center, size) in enumerate(zip(centers, cluster_sizes)):
            if size == 0:
                continue
                
            # Generate cluster data
            cluster_data = np.random.multivariate_normal(
                center, np.eye(4) * 100, size
            )
            cluster_data = np.clip(cluster_data, 0, 100)
            
            all_data.append(cluster_data)
            all_labels.extend([cluster_id] * size)
        
        X = np.vstack(all_data)
        y = np.array(all_labels)
        
        # Normalize
        X = (X / X.sum(axis=1, keepdims=True)) * 100
        
        df = pd.DataFrame(X, columns=['red_energy', 'blue_energy', 'green_energy', 'yellow_energy'])
        df['employee_id'] = [f'EMP{i:04d}' for i in range(len(X))]
        
        return df, y
    
    def generate_temporal_data(self, n_samples: int = 200) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate data with temporal personality variations"""
        df, y = self.generate_insights_discovery_data(n_samples)
        
        # Add temporal variations to simulate personality dynamics
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        for idx in range(len(df)):
            # Simulate personality "breathing" with small oscillations
            time_factor = np.sin(2 * np.pi * idx / 50)  # Slow oscillation
            
            # Add temporal component that preserves cluster structure
            cluster_id = y[idx]
            if cluster_id == 0:  # Red cluster gets more aggressive over time
                df.loc[idx, 'red_energy'] += time_factor * 5
            elif cluster_id == 1:  # Blue cluster gets more analytical
                df.loc[idx, 'blue_energy'] += time_factor * 5
            elif cluster_id == 2:  # Green cluster gets more supportive
                df.loc[idx, 'green_energy'] += time_factor * 5
            elif cluster_id == 3:  # Yellow cluster gets more enthusiastic
                df.loc[idx, 'yellow_energy'] += time_factor * 5
        
        # Renormalize
        row_sums = df[energy_cols].sum(axis=1)
        for col in energy_cols:
            df[col] = (df[col] / row_sums) * 100
        
        return df, y


class ClusteringBenchmark:
    """Main benchmarking class"""
    
    def __init__(self, output_dir: str = "benchmark_results", random_state: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.random_state = random_state
        self.generator = DatasetGenerator(random_state)
        self.results = []
        
    def benchmark_method(self, method, method_name: str, data: pd.DataFrame, 
                        true_labels: np.ndarray, n_clusters: int = 4) -> BenchmarkResult:
        """
        Benchmark a single clustering method
        
        Args:
            method: Clustering method instance
            method_name: Name of the method
            data: Input data
            true_labels: Ground truth labels
            n_clusters: Number of clusters
            
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Benchmarking {method_name} on {len(data)} samples")
        
        # Memory usage before
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fit timing
        start_time = time.time()
        method.fit(data)
        fit_time = time.time() - start_time
        
        # Memory usage after
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = memory_after - memory_before
        
        # Get predictions
        predictions = method.get_cluster_assignments()
        
        # Prediction timing (on same data for consistency)
        if hasattr(method, 'predict'):
            start_time = time.time()
            _ = method.predict(data)
            predict_time = time.time() - start_time
        else:
            predict_time = 0.0
        
        # Calculate metrics
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        features_for_metrics = data[energy_cols].values
        
        # Standard metrics
        if len(np.unique(predictions)) > 1:
            sil_score = silhouette_score(features_for_metrics, predictions)
            ch_score = calinski_harabasz_score(features_for_metrics, predictions)
            db_score = davies_bouldin_score(features_for_metrics, predictions)
        else:
            sil_score = -1.0
            ch_score = 0.0
            db_score = 10.0
        
        # Comparison with ground truth
        ari_score = adjusted_rand_score(true_labels, predictions)
        nmi_score = normalized_mutual_info_score(true_labels, predictions)
        
        # Cluster quality metrics
        cluster_balance = self._calculate_cluster_balance(predictions)
        cluster_separation = self._calculate_cluster_separation(features_for_metrics, predictions)
        
        # Neuromorphic-specific metrics
        stability_score = None
        interpretability_score = None
        temporal_coherence = None
        computational_efficiency = None
        
        if hasattr(method, 'get_clustering_metrics'):
            try:
                neuromorphic_metrics = method.get_clustering_metrics()
                stability_score = neuromorphic_metrics.cluster_stability
                interpretability_score = neuromorphic_metrics.interpretability_score
                temporal_coherence = neuromorphic_metrics.temporal_coherence
                computational_efficiency = neuromorphic_metrics.computational_efficiency
            except Exception as e:
                logger.warning(f"Failed to get neuromorphic metrics: {e}")
        
        result = BenchmarkResult(
            method_name=method_name,
            dataset_name="synthetic",  # Will be updated by caller
            n_samples=len(data),
            n_features=len(energy_cols),
            n_clusters=n_clusters,
            silhouette_score=sil_score,
            calinski_harabasz_score=ch_score,
            davies_bouldin_score=db_score,
            adjusted_rand_score=ari_score,
            normalized_mutual_info=nmi_score,
            cluster_stability=stability_score,
            interpretability_score=interpretability_score,
            temporal_coherence=temporal_coherence,
            fit_time=fit_time,
            predict_time=predict_time,
            memory_usage_mb=memory_usage,
            computational_efficiency=computational_efficiency,
            cluster_balance=cluster_balance,
            cluster_separation=cluster_separation
        )
        
        return result
    
    def _calculate_cluster_balance(self, labels: np.ndarray) -> float:
        """Calculate how balanced the cluster sizes are"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        if len(counts) <= 1:
            return 0.0
        
        # Coefficient of variation (lower is more balanced)
        cv = np.std(counts) / np.mean(counts)
        
        # Convert to score (higher is better)
        balance_score = max(0.0, 1.0 - cv)
        
        return balance_score
    
    def _calculate_cluster_separation(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate separation between cluster centroids"""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) <= 1:
            return 0.0
        
        # Calculate centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = np.mean(features[mask], axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids)
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)
        
        if distances:
            # Normalize by maximum possible distance
            max_distance = np.linalg.norm([100, 100, 100, 100])
            avg_separation = np.mean(distances) / max_distance
        else:
            avg_separation = 0.0
        
        return avg_separation
    
    def run_comprehensive_benchmark(self) -> List[BenchmarkResult]:
        """
        Run comprehensive benchmark across multiple scenarios
        
        Returns:
            List of all benchmark results
        """
        logger.info("Starting comprehensive neuromorphic clustering benchmark")
        
        # Define test scenarios
        scenarios = [
            ("standard", lambda: self.generator.generate_insights_discovery_data(200, 4, 10.0)),
            ("large_dataset", lambda: self.generator.generate_insights_discovery_data(1000, 4, 10.0)),
            ("small_dataset", lambda: self.generator.generate_insights_discovery_data(50, 4, 10.0)),
            ("noisy_data", lambda: self.generator.generate_noisy_data(200, 0.3)),
            ("high_noise", lambda: self.generator.generate_noisy_data(200, 0.6)),
            ("imbalanced", lambda: self.generator.generate_imbalanced_data(200)),
            ("temporal", lambda: self.generator.generate_temporal_data(200)),
            ("few_clusters", lambda: self.generator.generate_insights_discovery_data(200, 2, 15.0)),
            ("many_clusters", lambda: self.generator.generate_insights_discovery_data(200, 6, 8.0)),
        ]
        
        # Define methods to test
        methods_to_test = [
            ("K-Means", lambda: KMeansClusterer(n_clusters=4, random_state=self.random_state)),
            ("ESN", lambda: NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=4, random_state=self.random_state
            )),
            ("SNN", lambda: NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
                n_clusters=4, random_state=self.random_state
            )),
            ("LSM", lambda: NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                n_clusters=4, random_state=self.random_state
            )),
            ("Hybrid", lambda: NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4, random_state=self.random_state
            )),
        ]
        
        all_results = []
        
        # Run benchmarks
        for scenario_name, data_generator in scenarios:
            logger.info(f"Testing scenario: {scenario_name}")
            
            try:
                data, true_labels = data_generator()
                n_clusters = len(np.unique(true_labels))
                
                for method_name, method_factory in methods_to_test:
                    try:
                        # Adjust n_clusters for the method
                        method = method_factory()
                        if hasattr(method, 'n_clusters'):
                            method.n_clusters = n_clusters
                        
                        result = self.benchmark_method(method, method_name, data, true_labels, n_clusters)
                        result.dataset_name = scenario_name
                        
                        all_results.append(result)
                        self.results.append(result)
                        
                        logger.info(f"  {method_name}: Silhouette={result.silhouette_score:.3f}, "
                                  f"ARI={result.adjusted_rand_score:.3f}, "
                                  f"Time={result.fit_time:.2f}s")
                        
                    except Exception as e:
                        logger.error(f"Failed to benchmark {method_name} on {scenario_name}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Failed to generate data for scenario {scenario_name}: {e}")
                continue
        
        # Save results
        self.save_results(all_results)
        
        logger.info(f"Benchmark completed. Total results: {len(all_results)}")
        return all_results
    
    def save_results(self, results: List[BenchmarkResult]):
        """Save benchmark results to JSON and CSV"""
        # Convert to dictionaries
        results_dict = [result.to_dict() for result in results]
        
        # Save as JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        csv_path = self.output_dir / "benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def generate_performance_report(self) -> Dict:
        """
        Generate comprehensive performance report
        
        Returns:
            Dictionary with performance analysis
        """
        if not self.results:
            return {"error": "No benchmark results available"}
        
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        report = {
            "summary": {
                "total_experiments": len(df),
                "methods_tested": df['method_name'].unique().tolist(),
                "scenarios_tested": df['dataset_name'].unique().tolist()
            },
            "performance_by_method": {},
            "performance_by_scenario": {},
            "key_insights": [],
            "recommendations": []
        }
        
        # Performance by method
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            
            report["performance_by_method"][method] = {
                "avg_silhouette_score": float(method_data['silhouette_score'].mean()),
                "avg_ari_score": float(method_data['adjusted_rand_score'].mean()),
                "avg_fit_time": float(method_data['fit_time'].mean()),
                "avg_memory_usage": float(method_data['memory_usage_mb'].mean()),
                "stability": float(method_data['cluster_stability'].mean()) if 'cluster_stability' in method_data.columns else None
            }
        
        # Performance by scenario
        for scenario in df['dataset_name'].unique():
            scenario_data = df[df['dataset_name'] == scenario]
            best_method = scenario_data.loc[scenario_data['silhouette_score'].idxmax(), 'method_name']
            
            report["performance_by_scenario"][scenario] = {
                "best_method": best_method,
                "best_silhouette": float(scenario_data['silhouette_score'].max()),
                "methods_tested": scenario_data['method_name'].tolist()
            }
        
        # Key insights
        overall_best = df.loc[df['silhouette_score'].idxmax()]
        report["key_insights"].append(
            f"Best overall performance: {overall_best['method_name']} on {overall_best['dataset_name']} "
            f"(Silhouette: {overall_best['silhouette_score']:.3f})"
        )
        
        # Speed comparison
        speed_ranking = df.groupby('method_name')['fit_time'].mean().sort_values()
        fastest = speed_ranking.index[0]
        report["key_insights"].append(f"Fastest method: {fastest} ({speed_ranking[fastest]:.2f}s avg)")
        
        # Memory efficiency
        memory_ranking = df.groupby('method_name')['memory_usage_mb'].mean().sort_values()
        most_efficient = memory_ranking.index[0]
        report["key_insights"].append(
            f"Most memory efficient: {most_efficient} ({memory_ranking[most_efficient]:.1f}MB avg)"
        )
        
        # Recommendations
        neuromorphic_avg = df[df['method_name'] != 'K-Means']['silhouette_score'].mean()
        kmeans_avg = df[df['method_name'] == 'K-Means']['silhouette_score'].mean()
        
        if neuromorphic_avg > kmeans_avg:
            improvement = ((neuromorphic_avg - kmeans_avg) / kmeans_avg) * 100
            report["recommendations"].append(
                f"Neuromorphic methods show {improvement:.1f}% average improvement over K-means"
            )
        else:
            degradation = ((kmeans_avg - neuromorphic_avg) / kmeans_avg) * 100
            report["recommendations"].append(
                f"K-means outperforms neuromorphic methods by {degradation:.1f}% on average"
            )
        
        return report
    
    def create_visualization_dashboard(self):
        """Create comprehensive visualization dashboard"""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        df = pd.DataFrame([result.to_dict() for result in self.results])
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Neuromorphic Clustering Benchmark Results', fontsize=16, fontweight='bold')
        
        # 1. Silhouette Score by Method
        df_grouped = df.groupby('method_name')['silhouette_score'].agg(['mean', 'std']).reset_index()
        ax = axes[0, 0]
        bars = ax.bar(df_grouped['method_name'], df_grouped['mean'], 
                     yerr=df_grouped['std'], capsize=5, alpha=0.7)
        ax.set_title('Average Silhouette Score by Method')
        ax.set_ylabel('Silhouette Score')
        ax.tick_params(axis='x', rotation=45)
        
        # Color bars by performance
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 2. Fit Time Comparison
        ax = axes[0, 1]
        df.boxplot(column='fit_time', by='method_name', ax=ax)
        ax.set_title('Training Time Distribution by Method')
        ax.set_ylabel('Fit Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax, xlabel='')
        
        # 3. Performance by Scenario Heatmap
        ax = axes[0, 2]
        pivot_data = df.pivot_table(values='silhouette_score', 
                                   index='method_name', 
                                   columns='dataset_name', 
                                   aggfunc='mean')
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        ax.set_title('Performance Heatmap (Silhouette Score)')
        
        # 4. Memory Usage vs Performance
        ax = axes[1, 0]
        for method in df['method_name'].unique():
            method_data = df[df['method_name'] == method]
            ax.scatter(method_data['memory_usage_mb'], method_data['silhouette_score'], 
                      label=method, alpha=0.7, s=60)
        
        ax.set_xlabel('Memory Usage (MB)')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Memory Usage vs Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. ARI Score Distribution
        ax = axes[1, 1]
        df.boxplot(column='adjusted_rand_score', by='method_name', ax=ax)
        ax.set_title('Adjusted Rand Index Distribution')
        ax.set_ylabel('ARI Score')
        ax.tick_params(axis='x', rotation=45)
        plt.setp(ax, xlabel='')
        
        # 6. Neuromorphic-specific Metrics (if available)
        ax = axes[1, 2]
        neuromorphic_data = df[df['cluster_stability'].notna()]
        if not neuromorphic_data.empty:
            stability_data = neuromorphic_data.groupby('method_name')['cluster_stability'].mean()
            interpretability_data = neuromorphic_data.groupby('method_name')['interpretability_score'].mean()
            
            x = np.arange(len(stability_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, stability_data.values, width, 
                          label='Cluster Stability', alpha=0.7)
            bars2 = ax.bar(x + width/2, interpretability_data.values, width,
                          label='Interpretability', alpha=0.7)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_title('Neuromorphic-Specific Metrics')
            ax.set_xticks(x)
            ax.set_xticklabels(stability_data.index, rotation=45)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No neuromorphic metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Neuromorphic-Specific Metrics')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = self.output_dir / "benchmark_dashboard.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dashboard saved to {plot_path}")
        
        plt.show()


def run_quick_benchmark() -> Dict:
    """
    Run a quick benchmark for testing purposes
    
    Returns:
        Performance report dictionary
    """
    logger.info("Running quick neuromorphic clustering benchmark")
    
    # Create benchmark instance
    benchmark = ClusteringBenchmark(output_dir="quick_benchmark_results")
    
    # Generate test data
    generator = DatasetGenerator()
    data, true_labels = generator.generate_insights_discovery_data(100, 4, 12.0)
    
    # Test methods
    methods = [
        ("K-Means", KMeansClusterer(n_clusters=4, random_state=42)),
        ("ESN", NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            n_clusters=4, random_state=42
        )),
        ("Hybrid", NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
            n_clusters=4, random_state=42
        ))
    ]
    
    results = []
    for method_name, method in methods:
        try:
            result = benchmark.benchmark_method(method, method_name, data, true_labels)
            result.dataset_name = "quick_test"
            results.append(result)
            benchmark.results.append(result)
        except Exception as e:
            logger.error(f"Failed to benchmark {method_name}: {e}")
    
    # Generate report
    if results:
        benchmark.save_results(results)
        report = benchmark.generate_performance_report()
        
        # Print summary
        print("\n" + "="*50)
        print("QUICK BENCHMARK RESULTS")
        print("="*50)
        
        for method, metrics in report["performance_by_method"].items():
            print(f"\n{method}:")
            print(f"  Silhouette Score: {metrics['avg_silhouette_score']:.3f}")
            print(f"  ARI Score: {metrics['avg_ari_score']:.3f}")
            print(f"  Fit Time: {metrics['avg_fit_time']:.2f}s")
            print(f"  Memory Usage: {metrics['avg_memory_usage']:.1f}MB")
            if metrics['stability']:
                print(f"  Stability: {metrics['stability']:.3f}")
        
        print("\nKey Insights:")
        for insight in report["key_insights"]:
            print(f"  • {insight}")
        
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
        return report
    else:
        logger.error("No successful benchmark runs")
        return {"error": "All benchmark runs failed"}


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive benchmark
    benchmark = ClusteringBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Generate performance report
    report = benchmark.generate_performance_report()
    
    # Create visualizations
    benchmark.create_visualization_dashboard()
    
    print("\nBenchmark completed successfully!")
    print(f"Results saved to: {benchmark.output_dir}")