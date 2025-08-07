"""
Enhanced Integration Tests for Full Clustering Pipeline
Tests end-to-end workflow with comprehensive validation and error scenarios
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import logging
import time
import json
from datetime import datetime
from unittest.mock import Mock, patch
import gc

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.parser import InsightsDataParser
from insights_clustering.clustering import KMeansClusterer
from insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer, 
    NeuromorphicClusteringMethod,
    NeuromorphicException,
    ClusteringMetrics
)
from insights_clustering.validator import DataValidator
from team_simulator.simulator import TeamCompositionSimulator
from insights_clustering.config import Config
from insights_clustering.monitoring import ClusteringMonitor

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline test configuration
PIPELINE_CONFIG = {
    'data_quality_threshold': 0.8,
    'clustering_quality_threshold': 0.6,
    'team_balance_threshold': 40.0,
    'performance_timeout': 300,  # 5 minutes max per test
    'memory_limit_mb': 2048
}


class TestFullClusteringPipeline:
    """Comprehensive integration tests for the complete clustering pipeline"""
    
    def setup_method(self):
        """Setup test fixtures and monitoring"""
        np.random.seed(42)
        
        # Create comprehensive test datasets
        self.datasets = {
            'small': self._create_test_dataset(50),
            'medium': self._create_test_dataset(200), 
            'large': self._create_test_dataset(500),
            'realistic': self._create_realistic_dataset(300)
        }
        
        # Initialize monitoring
        self.monitor = ClusteringMonitor()
        self.performance_metrics = []
        
    def _create_test_dataset(self, n_samples):
        """Create test dataset with known cluster structure"""
        data = []
        
        # Define 6 personality archetypes for more realistic testing
        archetypes = [
            [80, 15, 10, 15],  # Strong Red (Director)
            [15, 80, 10, 15],  # Strong Blue (Thinker)
            [10, 15, 80, 15],  # Strong Green (Supporter)
            [15, 10, 15, 80],  # Strong Yellow (Inspirational)
            [45, 45, 20, 20],  # Red-Blue mix (Analytical Driver)
            [25, 25, 45, 45]   # Green-Yellow mix (Supportive Inspirational)
        ]
        
        samples_per_type = n_samples // len(archetypes)
        remainder = n_samples % len(archetypes)
        
        for i, archetype in enumerate(archetypes):
            # Add extra samples to first archetypes if there's remainder
            current_samples = samples_per_type + (1 if i < remainder else 0)
            
            for j in range(current_samples):
                # Add realistic noise
                noise_scale = np.random.uniform(3, 8)
                noise = np.random.randn(4) * noise_scale
                energies = np.array(archetype) + noise
                energies = np.clip(energies, 0.1, 100)
                
                # Normalize to sum to 100
                energies = (energies / np.sum(energies)) * 100
                
                data.append({
                    'employee_id': f'EMP{i:02d}{j:04d}',
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'department': np.random.choice([
                        'Engineering', 'Marketing', 'Sales', 'HR', 
                        'Finance', 'Operations', 'Product', 'Design'
                    ]),
                    'experience_years': np.random.randint(0, 25),
                    'position_level': np.random.choice(['Junior', 'Mid', 'Senior', 'Lead', 'Manager']),
                    'true_archetype': i,
                    'hire_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 2000))
                })
        
        return pd.DataFrame(data)
    
    def _create_realistic_dataset(self, n_samples):
        """Create realistic dataset based on actual organizational patterns"""
        data = []
        
        # Realistic department distributions and personality correlations
        dept_personality_prefs = {
            'Engineering': ([25, 60, 20, 25], 8),      # Blue preference, moderate variation
            'Sales': ([55, 20, 25, 45], 10),           # Red-Yellow preference, high variation  
            'HR': ([20, 30, 60, 40], 6),               # Green preference, low variation
            'Marketing': ([35, 30, 25, 55], 12),       # Yellow preference, high variation
            'Finance': ([30, 70, 25, 20], 7),          # Blue preference, low variation
            'Operations': ([45, 35, 45, 25], 9)        # Red-Green preference, moderate variation
        }
        
        dept_sizes = {
            'Engineering': int(n_samples * 0.25),
            'Sales': int(n_samples * 0.20),
            'Marketing': int(n_samples * 0.15),
            'HR': int(n_samples * 0.10),
            'Finance': int(n_samples * 0.15),
            'Operations': int(n_samples * 0.15)
        }
        
        # Adjust for exact sample count
        total_allocated = sum(dept_sizes.values())
        dept_sizes['Engineering'] += n_samples - total_allocated
        
        employee_count = 0
        
        for dept, size in dept_sizes.items():
            base_profile, variation = dept_personality_prefs[dept]
            
            for i in range(size):
                # Generate personality with departmental bias
                noise = np.random.randn(4) * variation
                energies = np.array(base_profile) + noise
                energies = np.clip(energies, 1, 99)
                
                # Add some cross-correlations (realistic personality interactions)
                if energies[0] > 50:  # High red
                    energies[2] *= 0.8  # Lower green (less supportive when assertive)
                if energies[1] > 50:  # High blue
                    energies[3] *= 0.9  # Slightly lower yellow (less spontaneous when analytical)
                
                # Normalize
                energies = (energies / np.sum(energies)) * 100
                
                # Experience correlations
                if dept == 'Engineering':
                    exp_mean, exp_std = 8, 6
                elif dept in ['Sales', 'Marketing']:
                    exp_mean, exp_std = 6, 4
                else:
                    exp_mean, exp_std = 10, 7
                    
                experience = max(0, int(np.random.normal(exp_mean, exp_std)))
                
                # Position level based on experience
                if experience >= 15:
                    position = np.random.choice(['Senior', 'Lead', 'Manager'], p=[0.4, 0.3, 0.3])
                elif experience >= 8:
                    position = np.random.choice(['Mid', 'Senior', 'Lead'], p=[0.3, 0.5, 0.2])
                elif experience >= 3:
                    position = np.random.choice(['Junior', 'Mid'], p=[0.3, 0.7])
                else:
                    position = 'Junior'
                
                data.append({
                    'employee_id': f'REAL{employee_count:05d}',
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'department': dept,
                    'experience_years': experience,
                    'position_level': position,
                    'hire_date': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 3000)),
                    'performance_rating': np.random.choice(['Exceeds', 'Meets', 'Below'], p=[0.2, 0.7, 0.1])
                })
                
                employee_count += 1
        
        return pd.DataFrame(data)
    
    @pytest.mark.parametrize("dataset_name", ['small', 'medium', 'realistic'])
    def test_complete_pipeline_flow(self, dataset_name):
        """Test complete pipeline from data to team recommendations"""
        dataset = self.datasets[dataset_name]
        
        with self.monitor.performance_monitor(f"complete_pipeline_{dataset_name}"):
            # Step 1: Data parsing and validation
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            dataset.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                # Parse data
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                
                assert len(parsed_data) == len(dataset)
                
                # Validate data quality
                validator = DataValidator()
                validation_results = validator.validate_data_quality(parsed_data)
                
                assert validation_results['is_valid']
                assert validation_results['quality_score'] >= PIPELINE_CONFIG['data_quality_threshold']
                
                # Extract clustering features
                features = parser.get_clustering_features()
                assert features.shape[0] == len(dataset)
                assert features.shape[1] == 4
                
                # Step 2: Test multiple clustering approaches
                clustering_results = {}
                
                # Test K-means baseline
                kmeans_clusterer = KMeansClusterer(n_clusters=4, random_state=42)
                kmeans_clusterer.fit(features)
                kmeans_labels = kmeans_clusterer.get_cluster_assignments()
                kmeans_metrics = kmeans_clusterer.get_clustering_metrics()
                
                clustering_results['kmeans'] = {
                    'labels': kmeans_labels,
                    'metrics': kmeans_metrics,
                    'silhouette_score': kmeans_metrics.silhouette_score
                }
                
                # Test neuromorphic methods
                neuro_methods = [
                    NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    NeuromorphicClusteringMethod.HYBRID_RESERVOIR
                ]
                
                for method in neuro_methods:
                    neuro_clusterer = NeuromorphicClusterer(
                        method=method,
                        n_clusters=4,
                        random_state=42,
                        enable_fallback=True
                    )
                    
                    neuro_clusterer.fit(features)
                    neuro_labels = neuro_clusterer.get_cluster_assignments()
                    neuro_metrics = neuro_clusterer.get_clustering_metrics()
                    
                    clustering_results[method.value] = {
                        'labels': neuro_labels,
                        'metrics': neuro_metrics,
                        'silhouette_score': neuro_metrics.silhouette_score,
                        'fallback_used': neuro_clusterer.fallback_used
                    }
                
                # Step 3: Compare clustering quality
                best_method = max(clustering_results.keys(), 
                                key=lambda k: clustering_results[k]['silhouette_score'])
                best_labels = clustering_results[best_method]['labels']
                best_score = clustering_results[best_method]['silhouette_score']
                
                assert best_score >= PIPELINE_CONFIG['clustering_quality_threshold']
                
                # Step 4: Team simulation and optimization
                simulator = TeamCompositionSimulator(min_team_size=4, max_team_size=8)
                simulator.load_employee_data(parsed_data, best_labels)
                
                # Generate multiple team configurations
                team_configs = []
                for i in range(3):
                    teams = simulator.generate_balanced_teams(num_teams=min(5, len(dataset)//8))
                    if teams:
                        avg_balance = np.mean([team['balance_score'] for team in teams])
                        team_configs.append({
                            'teams': teams,
                            'average_balance': avg_balance,
                            'config_id': i
                        })
                
                assert len(team_configs) > 0
                
                # Step 5: Select best team configuration
                best_config = max(team_configs, key=lambda x: x['average_balance'])
                assert best_config['average_balance'] >= PIPELINE_CONFIG['team_balance_threshold']
                
                # Step 6: Generate insights and recommendations
                interpretations = {}
                if best_method != 'kmeans':
                    # Get neuromorphic cluster interpretations
                    method_obj = next((m for m in neuro_methods if m.value == best_method), None)
                    if method_obj:
                        neuro_clusterer = NeuromorphicClusterer(
                            method=method_obj, n_clusters=4, random_state=42
                        )
                        neuro_clusterer.fit(features)
                        interpretations = neuro_clusterer.get_cluster_interpretation()
                
                # Verify final pipeline outputs
                pipeline_results = {
                    'dataset_size': len(dataset),
                    'data_quality_score': validation_results['quality_score'],
                    'best_clustering_method': best_method,
                    'clustering_quality_score': best_score,
                    'num_teams_generated': len(best_config['teams']),
                    'average_team_balance': best_config['average_balance'],
                    'cluster_interpretations': len(interpretations),
                    'pipeline_success': True
                }
                
                # Log results
                logger.info(f"Pipeline results for {dataset_name}: {pipeline_results}")
                
                # Assertions for pipeline success
                assert pipeline_results['pipeline_success']
                assert pipeline_results['num_teams_generated'] > 0
                assert pipeline_results['cluster_interpretations'] >= 0
                
            finally:
                temp_path.unlink()
    
    def test_pipeline_error_handling_and_recovery(self):
        """Test pipeline behavior under various error conditions"""
        # Test with corrupted data
        corrupted_data = self.datasets['small'].copy()
        
        # Introduce various data quality issues
        corrupted_data.loc[0:5, 'red_energy'] = np.nan  # Missing values
        corrupted_data.loc[10:15, 'blue_energy'] = -50   # Invalid negative values
        corrupted_data.loc[20:25, 'yellow_energy'] = 500  # Invalid high values
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        corrupted_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            
            # Should handle corrupted data gracefully
            try:
                parsed_data = parser.parse_csv(temp_path)
                
                # Data validation should detect issues
                validator = DataValidator()
                validation_results = validator.validate_data_quality(parsed_data)
                
                # Should either fix the data or report low quality
                if validation_results['is_valid']:
                    # If marked as valid, quality score should reflect issues
                    assert validation_results['quality_score'] < 0.9
                else:
                    # If marked as invalid, should provide details
                    assert 'issues' in validation_results
                
            except Exception as e:
                # Parsing failures should be handled gracefully
                assert any(keyword in str(e).lower() for keyword in 
                          ['data', 'quality', 'validation', 'corrupt'])
                
        finally:
            temp_path.unlink()
    
    def test_pipeline_performance_benchmarks(self):
        """Test pipeline performance across different data sizes"""
        performance_results = {}
        
        for dataset_name, dataset in self.datasets.items():
            if dataset_name == 'large':  # Skip large dataset in regular testing
                continue
                
            start_time = time.time()
            memory_start = self._get_memory_usage()
            
            # Run streamlined pipeline for performance testing
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            dataset.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                # Parse and validate
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                features = parser.get_clustering_features()
                
                # Fast clustering method
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    n_clusters=4,
                    random_state=42
                )
                
                clustering_start = time.time()
                clusterer.fit(features)
                clustering_time = time.time() - clustering_start
                
                labels = clusterer.get_cluster_assignments()
                
                # Team generation
                simulator = TeamCompositionSimulator()
                simulator.load_employee_data(parsed_data, labels)
                teams = simulator.generate_balanced_teams(num_teams=3)
                
                total_time = time.time() - start_time
                memory_peak = self._get_memory_usage()
                memory_used = memory_peak - memory_start
                
                performance_results[dataset_name] = {
                    'data_size': len(dataset),
                    'total_time': total_time,
                    'clustering_time': clustering_time,
                    'memory_used_mb': memory_used,
                    'teams_generated': len(teams),
                    'throughput_samples_per_sec': len(dataset) / total_time
                }
                
            finally:
                temp_path.unlink()
                gc.collect()  # Clean up memory
        
        # Performance assertions
        for dataset_name, results in performance_results.items():
            # Should complete within reasonable time
            assert results['total_time'] < PIPELINE_CONFIG['performance_timeout']
            
            # Memory usage should be reasonable
            assert results['memory_used_mb'] < PIPELINE_CONFIG['memory_limit_mb']
            
            # Should generate teams
            assert results['teams_generated'] > 0
            
            # Throughput should be reasonable
            assert results['throughput_samples_per_sec'] > 1.0
        
        # Log performance summary
        logger.info("Pipeline Performance Results:")
        for dataset_name, results in performance_results.items():
            logger.info(f"  {dataset_name}: {results}")
    
    def test_pipeline_scalability_stress(self):
        """Test pipeline scalability under stress conditions"""
        # Create stress test with challenging data
        stress_data = self._create_stress_test_data(300)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        stress_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            with self.monitor.performance_monitor("stress_test"):
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                features = parser.get_clustering_features()
                
                # Test multiple clustering methods simultaneously
                methods_to_test = [
                    NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE
                ]
                
                results = {}
                for method in methods_to_test:
                    try:
                        clusterer = NeuromorphicClusterer(
                            method=method,
                            n_clusters=6,  # More clusters for stress test
                            random_state=42,
                            enable_fallback=True
                        )
                        
                        clusterer.fit(features)
                        labels = clusterer.get_cluster_assignments()
                        metrics = clusterer.get_clustering_metrics()
                        
                        results[method.value] = {
                            'success': True,
                            'labels': labels,
                            'silhouette_score': metrics.silhouette_score,
                            'fallback_used': clusterer.fallback_used
                        }
                        
                    except Exception as e:
                        results[method.value] = {
                            'success': False,
                            'error': str(e),
                            'fallback_used': False
                        }
                
                # At least one method should succeed
                successful_methods = [name for name, result in results.items() if result['success']]
                assert len(successful_methods) > 0
                
                # Use best successful method for team generation
                best_method = max(successful_methods, 
                                key=lambda m: results[m].get('silhouette_score', -1))
                
                simulator = TeamCompositionSimulator()
                simulator.load_employee_data(parsed_data, results[best_method]['labels'])
                teams = simulator.generate_balanced_teams(num_teams=8)
                
                assert len(teams) > 0
                
        finally:
            temp_path.unlink()
    
    def test_pipeline_data_quality_thresholds(self):
        """Test pipeline behavior with various data quality levels"""
        quality_levels = {
            'high': 0.95,
            'medium': 0.75,
            'low': 0.45
        }
        
        for quality_name, target_quality in quality_levels.items():
            test_data = self._create_quality_controlled_data(100, target_quality)
            
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            test_data.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                
                validator = DataValidator()
                validation_results = validator.validate_data_quality(parsed_data)
                
                # Quality score should approximate target
                actual_quality = validation_results['quality_score']
                assert abs(actual_quality - target_quality) < 0.2
                
                if actual_quality >= PIPELINE_CONFIG['data_quality_threshold']:
                    # High enough quality - should proceed with clustering
                    features = parser.get_clustering_features()
                    
                    clusterer = NeuromorphicClusterer(
                        method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                        n_clusters=4,
                        random_state=42,
                        enable_fallback=True
                    )
                    
                    clusterer.fit(features)
                    assert clusterer.trained
                else:
                    # Low quality - should either reject or provide warnings
                    logger.warning(f"Low quality data detected: {actual_quality:.2f}")
                    
            finally:
                temp_path.unlink()
    
    def test_cross_method_consistency(self):
        """Test consistency across different clustering methods"""
        dataset = self.datasets['medium']
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        dataset.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Test multiple methods with same data
            methods = [
                ('kmeans', KMeansClusterer(n_clusters=4, random_state=42)),
                ('esn', NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    n_clusters=4, random_state=42
                )),
                ('hybrid', NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                    n_clusters=4, random_state=42
                ))
            ]
            
            clustering_results = {}
            
            for method_name, clusterer in methods:
                clusterer.fit(features)
                labels = clusterer.get_cluster_assignments()
                
                # Generate teams with this clustering
                simulator = TeamCompositionSimulator()
                simulator.load_employee_data(parsed_data, labels)
                teams = simulator.generate_balanced_teams(num_teams=4)
                
                clustering_results[method_name] = {
                    'labels': labels,
                    'teams': teams,
                    'avg_balance': np.mean([t['balance_score'] for t in teams]) if teams else 0
                }
            
            # All methods should produce reasonable results
            for method_name, results in clustering_results.items():
                assert len(results['labels']) == len(dataset)
                assert len(results['teams']) > 0
                assert results['avg_balance'] > 20  # Minimum reasonable balance
            
            # Compare clustering agreement between methods
            method_names = list(clustering_results.keys())
            for i in range(len(method_names)):
                for j in range(i + 1, len(method_names)):
                    labels1 = clustering_results[method_names[i]]['labels']
                    labels2 = clustering_results[method_names[j]]['labels']
                    
                    # Calculate adjusted rand index for agreement
                    from sklearn.metrics import adjusted_rand_score
                    agreement = adjusted_rand_score(labels1, labels2)
                    
                    # Methods should have some agreement (>= 0.1) but don't need to be identical
                    assert agreement >= 0.1, f"Low agreement between {method_names[i]} and {method_names[j]}: {agreement}"
                    
        finally:
            temp_path.unlink()
    
    def _create_stress_test_data(self, n_samples):
        """Create challenging data for stress testing"""
        data = []
        
        # Create overlapping clusters and edge cases
        for i in range(n_samples):
            # Some samples with extreme values
            if i % 50 == 0:
                energies = [95, 2, 2, 1]  # Very extreme
            elif i % 47 == 0:
                energies = [25, 25, 25, 25]  # Perfectly balanced (hard to cluster)
            else:
                # Normal samples with some overlap between clusters
                base_type = np.random.randint(0, 4)
                if base_type == 0:
                    base = [60, 30, 20, 30]  # Red with blue overlap
                elif base_type == 1:
                    base = [30, 60, 20, 30]  # Blue with red overlap
                elif base_type == 2:
                    base = [20, 20, 60, 40]  # Green with yellow overlap
                else:
                    base = [30, 20, 40, 60]  # Yellow with green overlap
                
                # Add significant noise for challenge
                noise = np.random.randn(4) * 12
                energies = np.array(base) + noise
                energies = np.clip(energies, 0.1, 100)
            
            # Normalize
            energies = np.array(energies)
            energies = (energies / np.sum(energies)) * 100
            
            data.append({
                'employee_id': f'STRESS{i:05d}',
                'red_energy': round(energies[0], 2),
                'blue_energy': round(energies[1], 2),
                'green_energy': round(energies[2], 2),
                'yellow_energy': round(energies[3], 2),
                'department': 'Testing',
                'experience_years': np.random.randint(0, 30)
            })
        
        return pd.DataFrame(data)
    
    def _create_quality_controlled_data(self, n_samples, target_quality):
        """Create data with specific quality level"""
        data = []
        
        # Calculate how many samples should have issues based on target quality
        n_issues = int(n_samples * (1 - target_quality))
        issue_indices = set(np.random.choice(n_samples, n_issues, replace=False))
        
        for i in range(n_samples):
            if i in issue_indices:
                # Introduce data quality issues
                if np.random.rand() < 0.3:
                    # Missing values
                    energies = [np.nan, 50, 25, 25]
                elif np.random.rand() < 0.3:
                    # Invalid values
                    energies = [-10, 150, 25, 25]
                else:
                    # Inconsistent values (don't sum to 100)
                    energies = [200, 50, 25, 25]
            else:
                # Good quality data
                archetype = np.random.randint(0, 4)
                if archetype == 0:
                    base = [70, 20, 15, 20]
                elif archetype == 1:
                    base = [20, 70, 15, 20]
                elif archetype == 2:
                    base = [15, 20, 70, 20]
                else:
                    base = [20, 15, 20, 70]
                
                noise = np.random.randn(4) * 5
                energies = np.array(base) + noise
                energies = np.clip(energies, 1, 99)
                energies = (energies / np.sum(energies)) * 100
            
            data.append({
                'employee_id': f'QUAL{i:05d}',
                'red_energy': energies[0] if not np.isnan(energies[0]) else energies[0],
                'blue_energy': energies[1],
                'green_energy': energies[2],
                'yellow_energy': energies[3],
                'department': 'Quality',
                'experience_years': np.random.randint(0, 20)
            })
        
        return pd.DataFrame(data)
    
    def _get_memory_usage(self):
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0  # If psutil not available, return 0
    
    def teardown_method(self):
        """Cleanup after tests"""
        # Force garbage collection to free memory
        gc.collect()
        
        # Log any performance metrics collected
        if self.performance_metrics:
            logger.info("Test performance summary:")
            for metric in self.performance_metrics:
                logger.info(f"  {metric}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])  # Stop on first failure for debugging