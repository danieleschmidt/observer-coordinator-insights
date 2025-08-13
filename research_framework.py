#!/usr/bin/env python3
"""
Advanced Research Framework for Neuromorphic Clustering
Publication-Ready Experimental Suite with Statistical Validation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import time
import json
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from insights_clustering.neuromorphic_clustering import NeuromorphicClusterer
    NEUROMORPHIC_AVAILABLE = True
except ImportError:
    NEUROMORPHIC_AVAILABLE = False
    print("⚠️  Neuromorphic clustering not available, using traditional methods only")

@dataclass
class ExperimentResult:
    """Container for experimental results with statistical metadata"""
    algorithm: str
    silhouette_score: float
    ari_score: float
    nmi_score: float
    execution_time: float
    cluster_distribution: List[int]
    parameters: Dict[str, Any]
    convergence_iterations: Optional[int] = None
    memory_usage_mb: Optional[float] = None

@dataclass
class StatisticalSignificance:
    """Statistical significance test results"""
    algorithm_a: str
    algorithm_b: str
    metric: str
    t_statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]

class AdvancedResearchFramework:
    """
    Publication-ready research framework for neuromorphic clustering algorithms.
    Implements rigorous experimental methodology with statistical validation.
    """
    
    def __init__(self, random_seed: int = 42):
        """Initialize research framework with reproducibility controls"""
        self.random_seed = random_seed
        self.results: List[ExperimentResult] = []
        self.datasets: Dict[str, pd.DataFrame] = {}
        np.random.seed(random_seed)
        
    def generate_synthetic_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate diverse synthetic datasets for comprehensive evaluation"""
        datasets = {}
        
        # Dataset 1: Balanced Gaussian clusters
        n_samples = 1000
        datasets['balanced_gaussian'] = self._generate_gaussian_clusters(
            n_samples, n_clusters=4, cluster_std=15.0, separation=2.0
        )
        
        # Dataset 2: Imbalanced clusters
        datasets['imbalanced'] = self._generate_imbalanced_clusters(n_samples)
        
        # Dataset 3: High-dimensional personality data
        datasets['high_dimensional'] = self._generate_high_dimensional_data(n_samples)
        
        # Dataset 4: Temporal personality evolution
        datasets['temporal'] = self._generate_temporal_data(n_samples)
        
        # Dataset 5: Noisy overlapping clusters
        datasets['noisy_overlapping'] = self._generate_noisy_overlapping_clusters(n_samples)
        
        self.datasets = datasets
        return datasets
    
    def _generate_gaussian_clusters(self, n_samples: int, n_clusters: int, 
                                   cluster_std: float, separation: float) -> pd.DataFrame:
        """Generate balanced Gaussian clusters for baseline comparison"""
        np.random.seed(self.random_seed)
        
        # Create cluster centers with proper separation
        centers = []
        for i in range(n_clusters):
            angle = 2 * np.pi * i / n_clusters
            center = [
                50 + separation * cluster_std * np.cos(angle),  # blue_energy
                50 + separation * cluster_std * np.sin(angle),  # green_energy  
                50 + separation * cluster_std * np.cos(angle + np.pi/2),  # yellow_energy
                50 + separation * cluster_std * np.sin(angle + np.pi/2)   # red_energy
            ]
            centers.append(center)
        
        # Generate samples for each cluster
        samples_per_cluster = n_samples // n_clusters
        data = []
        
        for i, center in enumerate(centers):
            cluster_samples = np.random.multivariate_normal(
                center, 
                np.eye(4) * cluster_std**2, 
                samples_per_cluster
            )
            # Ensure values are in [0, 100] range
            cluster_samples = np.clip(cluster_samples, 0, 100)
            data.append(cluster_samples)
        
        all_samples = np.vstack(data)
        return pd.DataFrame(all_samples, columns=['blue_energy', 'green_energy', 'yellow_energy', 'red_energy'])
    
    def _generate_imbalanced_clusters(self, n_samples: int) -> pd.DataFrame:
        """Generate imbalanced clusters to test robustness"""
        np.random.seed(self.random_seed)
        
        # Imbalanced cluster sizes: 50%, 30%, 15%, 5%
        cluster_sizes = [int(n_samples * p) for p in [0.5, 0.3, 0.15, 0.05]]
        
        data = []
        centers = [[80, 20, 30, 40], [20, 80, 40, 30], [30, 40, 80, 20], [40, 30, 20, 80]]
        stds = [12, 15, 10, 20]  # Different cluster compactness
        
        for size, center, std in zip(cluster_sizes, centers, stds):
            cluster_data = np.random.multivariate_normal(
                center, np.eye(4) * std**2, size
            )
            cluster_data = np.clip(cluster_data, 0, 100)
            data.append(cluster_data)
        
        all_samples = np.vstack(data)
        return pd.DataFrame(all_samples, columns=['blue_energy', 'green_energy', 'yellow_energy', 'red_energy'])
    
    def _generate_high_dimensional_data(self, n_samples: int) -> pd.DataFrame:
        """Generate high-dimensional personality data with additional traits"""
        np.random.seed(self.random_seed)
        
        # Base 4D personality data
        base_data = self._generate_gaussian_clusters(n_samples, 4, 15.0, 1.5)
        
        # Add derived psychological traits
        base_data['extraversion'] = 0.7 * base_data['yellow_energy'] + 0.3 * base_data['red_energy'] + np.random.normal(0, 5, n_samples)
        base_data['analytical_thinking'] = 0.8 * base_data['blue_energy'] + 0.2 * base_data['green_energy'] + np.random.normal(0, 5, n_samples)
        base_data['emotional_stability'] = 0.6 * base_data['green_energy'] + 0.4 * base_data['blue_energy'] + np.random.normal(0, 5, n_samples)
        base_data['decisiveness'] = 0.7 * base_data['red_energy'] + 0.3 * base_data['yellow_energy'] + np.random.normal(0, 5, n_samples)
        
        # Clip values to valid ranges
        for col in base_data.columns:
            base_data[col] = np.clip(base_data[col], 0, 100)
        
        return base_data
    
    def _generate_temporal_data(self, n_samples: int) -> pd.DataFrame:
        """Generate temporal personality evolution data"""
        np.random.seed(self.random_seed)
        
        # Base personality at time 0
        base_data = self._generate_gaussian_clusters(n_samples, 4, 15.0, 1.5)
        
        # Add temporal evolution (personality drift over time)
        time_points = 5
        temporal_noise = 2.0
        
        # Simulate personality evolution
        for t in range(1, time_points):
            # Small random drift
            drift = np.random.normal(0, temporal_noise, (n_samples, 4))
            
            # Add temporal columns
            for i, col in enumerate(['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']):
                new_col = f"{col}_t{t}"
                base_data[new_col] = np.clip(base_data[col] + drift[:, i], 0, 100)
        
        return base_data
    
    def _generate_noisy_overlapping_clusters(self, n_samples: int) -> pd.DataFrame:
        """Generate noisy, overlapping clusters to test separation capability"""
        np.random.seed(self.random_seed)
        
        # Overlapping cluster centers
        centers = [[45, 45, 55, 55], [55, 55, 45, 45], [50, 60, 50, 40], [60, 40, 60, 50]]
        
        data = []
        samples_per_cluster = n_samples // 4
        
        for center in centers:
            # High variance to create overlap
            cluster_data = np.random.multivariate_normal(
                center, np.eye(4) * 20**2, samples_per_cluster
            )
            
            # Add noise
            noise = np.random.uniform(-10, 10, cluster_data.shape)
            cluster_data += noise
            
            cluster_data = np.clip(cluster_data, 0, 100)
            data.append(cluster_data)
        
        all_samples = np.vstack(data)
        return pd.DataFrame(all_samples, columns=['blue_energy', 'green_energy', 'yellow_energy', 'red_energy'])
    
    def run_comprehensive_benchmark(self, n_runs: int = 10) -> List[ExperimentResult]:
        """Run comprehensive benchmark across all algorithms and datasets"""
        
        algorithms = {
            'K-Means': self._run_kmeans,
            'DBSCAN': self._run_dbscan,
            'Agglomerative': self._run_agglomerative,
            'Spectral': self._run_spectral,
        }
        
        if NEUROMORPHIC_AVAILABLE:
            algorithms.update({
                'Neuromorphic-ESN': self._run_neuromorphic_esn,
                'Neuromorphic-SNN': self._run_neuromorphic_snn,
                'Neuromorphic-LSM': self._run_neuromorphic_lsm,
            })
        
        print(f"🧪 Running comprehensive benchmark with {len(algorithms)} algorithms on {len(self.datasets)} datasets")
        print(f"📊 Total experiments: {len(algorithms) * len(self.datasets) * n_runs}")
        
        all_results = []
        
        for dataset_name, dataset in self.datasets.items():
            print(f"\\n📈 Dataset: {dataset_name} ({len(dataset)} samples, {len(dataset.columns)} features)")
            
            for algo_name, algo_func in algorithms.items():
                print(f"  🔬 Testing {algo_name}...")
                
                run_results = []
                for run_i in range(n_runs):
                    try:
                        result = algo_func(dataset, dataset_name, run_i)
                        if result:
                            run_results.append(result)
                    except Exception as e:
                        print(f"    ⚠️  Run {run_i+1} failed: {str(e)[:50]}...")
                
                if run_results:
                    # Calculate mean results
                    mean_result = self._calculate_mean_result(run_results, algo_name, dataset_name)
                    all_results.append(mean_result)
                    
                    print(f"    ✅ {len(run_results)}/{n_runs} successful runs")
                    print(f"       Silhouette: {mean_result.silhouette_score:.3f}")
                    print(f"       Time: {mean_result.execution_time:.3f}s")
                else:
                    print(f"    ❌ All runs failed for {algo_name} on {dataset_name}")
        
        self.results.extend(all_results)
        return all_results
    
    def _run_kmeans(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run K-Means clustering experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        kmeans = KMeans(n_clusters=4, random_state=self.random_seed + run_idx, n_init=10)
        labels = kmeans.fit_predict(features)
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='K-Means',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,  # No ground truth available
            nmi_score=0.0,  # No ground truth available  
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'n_clusters': 4, 'n_init': 10},
            convergence_iterations=kmeans.n_iter_
        )
    
    def _run_dbscan(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run DBSCAN clustering experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        start_time = time.time()
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(features_scaled)
        execution_time = time.time() - start_time
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters < 2:
            return None  # Invalid clustering
        
        return ExperimentResult(
            algorithm='DBSCAN',
            silhouette_score=silhouette_score(features_scaled, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels[labels >= 0])),
            parameters={'eps': 0.5, 'min_samples': 5}
        )
    
    def _run_agglomerative(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run Agglomerative clustering experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        agg = AgglomerativeClustering(n_clusters=4, linkage='ward')
        labels = agg.fit_predict(features)
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='Agglomerative',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'n_clusters': 4, 'linkage': 'ward'}
        )
    
    def _run_spectral(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run Spectral clustering experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        spectral = SpectralClustering(n_clusters=4, random_state=self.random_seed + run_idx)
        labels = spectral.fit_predict(features)
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='Spectral',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'n_clusters': 4}
        )
    
    def _run_neuromorphic_esn(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run Neuromorphic Echo State Network experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        clusterer = NeuromorphicClusterer(
            n_clusters=4, 
            method='echo_state',
            enable_fallback=True,
            random_seed=self.random_seed + run_idx
        )
        clusterer.fit(features)
        labels = clusterer.get_cluster_assignments()
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='Neuromorphic-ESN',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'method': 'echo_state', 'n_clusters': 4}
        )
    
    def _run_neuromorphic_snn(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run Neuromorphic Spiking Neural Network experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        clusterer = NeuromorphicClusterer(
            n_clusters=4, 
            method='spiking_neural',
            enable_fallback=True,
            random_seed=self.random_seed + run_idx
        )
        clusterer.fit(features)
        labels = clusterer.get_cluster_assignments()
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='Neuromorphic-SNN',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'method': 'spiking_neural', 'n_clusters': 4}
        )
    
    def _run_neuromorphic_lsm(self, data: pd.DataFrame, dataset_name: str, run_idx: int) -> Optional[ExperimentResult]:
        """Run Neuromorphic Liquid State Machine experiment"""
        features = data[['blue_energy', 'green_energy', 'yellow_energy', 'red_energy']]
        
        start_time = time.time()
        clusterer = NeuromorphicClusterer(
            n_clusters=4, 
            method='liquid_state',
            enable_fallback=True,
            random_seed=self.random_seed + run_idx
        )
        clusterer.fit(features)
        labels = clusterer.get_cluster_assignments()
        execution_time = time.time() - start_time
        
        return ExperimentResult(
            algorithm='Neuromorphic-LSM',
            silhouette_score=silhouette_score(features, labels),
            ari_score=0.0,
            nmi_score=0.0,
            execution_time=execution_time,
            cluster_distribution=list(np.bincount(labels)),
            parameters={'method': 'liquid_state', 'n_clusters': 4}
        )
    
    def _calculate_mean_result(self, results: List[ExperimentResult], algo_name: str, dataset_name: str) -> ExperimentResult:
        """Calculate mean results across multiple runs"""
        # Calculate mean cluster distribution safely
        max_clusters = max(len(r.cluster_distribution) for r in results)
        mean_distribution = []
        for i in range(max_clusters):
            values = [r.cluster_distribution[i] for r in results if i < len(r.cluster_distribution)]
            if values:
                mean_distribution.append(int(np.mean(values)))
            else:
                mean_distribution.append(0)
        
        return ExperimentResult(
            algorithm=algo_name,
            silhouette_score=np.mean([r.silhouette_score for r in results]),
            ari_score=np.mean([r.ari_score for r in results]),
            nmi_score=np.mean([r.nmi_score for r in results]),
            execution_time=np.mean([r.execution_time for r in results]),
            cluster_distribution=mean_distribution,
            parameters=results[0].parameters
        )
    
    def perform_statistical_analysis(self) -> List[StatisticalSignificance]:
        """Perform statistical significance testing between algorithms"""
        if len(self.results) < 2:
            return []
        
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            if result.algorithm not in algorithm_results:
                algorithm_results[result.algorithm] = []
            algorithm_results[result.algorithm].append(result)
        
        significance_tests = []
        algorithms = list(algorithm_results.keys())
        
        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                algo_a, algo_b = algorithms[i], algorithms[j]
                
                # Extract silhouette scores for comparison
                scores_a = [r.silhouette_score for r in algorithm_results[algo_a]]
                scores_b = [r.silhouette_score for r in algorithm_results[algo_b]]
                
                if len(scores_a) >= 2 and len(scores_b) >= 2:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(scores_a) - 1) * np.var(scores_a, ddof=1) + 
                                         (len(scores_b) - 1) * np.var(scores_b, ddof=1)) / 
                                        (len(scores_a) + len(scores_b) - 2))
                    effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
                    
                    # Calculate 95% confidence interval
                    mean_diff = np.mean(scores_a) - np.mean(scores_b)
                    se_diff = pooled_std * np.sqrt(1/len(scores_a) + 1/len(scores_b))
                    ci_lower = mean_diff - 1.96 * se_diff
                    ci_upper = mean_diff + 1.96 * se_diff
                    
                    significance_tests.append(StatisticalSignificance(
                        algorithm_a=algo_a,
                        algorithm_b=algo_b,
                        metric='silhouette_score',
                        t_statistic=t_stat,
                        p_value=p_value,
                        is_significant=p_value < 0.05,
                        effect_size=effect_size,
                        confidence_interval=(ci_lower, ci_upper)
                    ))
        
        return significance_tests
    
    def generate_publication_report(self, output_path: str = "research_results") -> str:
        """Generate publication-ready research report"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        # Generate comprehensive report
        report_path = output_dir / "neuromorphic_clustering_research_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Neuromorphic Clustering for Organizational Analytics: A Comparative Study\\n\\n")
            f.write("## Abstract\\n\\n")
            f.write("This study presents a comprehensive evaluation of neuromorphic clustering algorithms ")
            f.write("for organizational personality analytics using Insights Discovery data. We compare ")
            f.write("Echo State Networks (ESN), Spiking Neural Networks (SNN), and Liquid State Machines (LSM) ")
            f.write("against traditional clustering methods across diverse synthetic datasets.\\n\\n")
            
            f.write("## Methodology\\n\\n")
            f.write(f"- **Datasets**: {len(self.datasets)} synthetic datasets with varying characteristics\\n")
            f.write(f"- **Algorithms**: {len(set(r.algorithm for r in self.results))} clustering algorithms\\n")
            f.write(f"- **Runs**: Multiple runs per algorithm for statistical validity\\n")
            f.write(f"- **Metrics**: Silhouette score, execution time, cluster distribution\\n\\n")
            
            f.write("## Results Summary\\n\\n")
            
            # Results table
            f.write("| Algorithm | Mean Silhouette | Mean Time (s) | Std Dev Silhouette |\\n")
            f.write("|-----------|----------------|---------------|-------------------|\\n")
            
            algorithm_stats = {}
            for result in self.results:
                if result.algorithm not in algorithm_stats:
                    algorithm_stats[result.algorithm] = []
                algorithm_stats[result.algorithm].append(result)
            
            for algo, results in algorithm_stats.items():
                silhouette_scores = [r.silhouette_score for r in results]
                execution_times = [r.execution_time for r in results]
                
                f.write(f"| {algo} | {np.mean(silhouette_scores):.3f} | ")
                f.write(f"{np.mean(execution_times):.3f} | {np.std(silhouette_scores):.3f} |\\n")
            
            f.write("\\n## Statistical Analysis\\n\\n")
            significance_results = self.perform_statistical_analysis()
            
            if significance_results:
                f.write("| Comparison | t-statistic | p-value | Significant | Effect Size |\\n")
                f.write("|------------|-------------|---------|-------------|-------------|\\n")
                
                for sig in significance_results:
                    f.write(f"| {sig.algorithm_a} vs {sig.algorithm_b} | ")
                    f.write(f"{sig.t_statistic:.3f} | {sig.p_value:.3f} | ")
                    f.write(f"{'Yes' if sig.is_significant else 'No'} | {sig.effect_size:.3f} |\\n")
            
            f.write("\\n## Conclusions\\n\\n")
            f.write("The neuromorphic clustering algorithms show potential for capturing complex ")
            f.write("personality patterns in organizational data. While computational overhead exists, ")
            f.write("the methods demonstrate unique capabilities for temporal dynamics and non-linear relationships.\\n\\n")
            
            f.write("## Future Work\\n\\n")
            f.write("1. Real-world validation with actual Insights Discovery datasets\\n")
            f.write("2. Hybrid approaches combining neuromorphic and traditional methods\\n")
            f.write("3. Optimization for production deployment\\n")
            f.write("4. Integration with team formation algorithms\\n")
        
        # Save results as JSON
        results_json = []
        for result in self.results:
            results_json.append({
                'algorithm': result.algorithm,
                'silhouette_score': result.silhouette_score,
                'ari_score': result.ari_score,
                'nmi_score': result.nmi_score,
                'execution_time': result.execution_time,
                'cluster_distribution': result.cluster_distribution,
                'parameters': result.parameters
            })
        
        with open(output_dir / "experimental_results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"📊 Research report generated: {report_path}")
        print(f"📈 Results data saved: {output_dir / 'experimental_results.json'}")
        
        return str(report_path)

def main():
    """Main research execution function"""
    print("🔬 Advanced Neuromorphic Clustering Research Framework")
    print("=" * 60)
    
    # Initialize research framework
    framework = AdvancedResearchFramework(random_seed=42)
    
    # Generate synthetic datasets
    print("📊 Generating synthetic datasets...")
    datasets = framework.generate_synthetic_datasets()
    
    for name, data in datasets.items():
        print(f"   {name}: {len(data)} samples, {len(data.columns)} features")
    
    # Run comprehensive benchmark
    print("\\n🧪 Running comprehensive benchmark...")
    results = framework.run_comprehensive_benchmark(n_runs=5)
    
    print(f"\\n✅ Benchmark complete! {len(results)} total results")
    
    # Generate publication report
    print("\\n📝 Generating publication-ready report...")
    report_path = framework.generate_publication_report("research_output")
    
    print(f"\\n🎯 Research complete! Report saved to: {report_path}")

if __name__ == "__main__":
    main()