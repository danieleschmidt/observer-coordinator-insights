"""
Integration tests for neuromorphic clustering with existing team simulation system
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.parser import InsightsDataParser
from insights_clustering.clustering import KMeansClusterer
from insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer, 
    NeuromorphicClusteringMethod
)
from insights_clustering.validator import DataValidator
from team_simulator.simulator import TeamCompositionSimulator

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Suppress info logs during tests


class TestNeuromorphicIntegration:
    """Test neuromorphic clustering integration with existing systems"""
    
    def setup_method(self):
        """Set up test fixtures"""
        np.random.seed(42)
        self.n_employees = 100
        
        # Create comprehensive test dataset
        self.test_data = self._create_comprehensive_test_data()
        
    def _create_comprehensive_test_data(self) -> pd.DataFrame:
        """Create realistic test data with personality clusters"""
        data = []
        
        # Define personality archetypes based on Insights Discovery
        archetypes = [
            # Red-dominant (Directors): High red, moderate others
            {"red": (60, 80), "blue": (15, 25), "green": (10, 20), "yellow": (15, 25)},
            # Blue-dominant (Thinkers): High blue, moderate others  
            {"red": (15, 25), "blue": (60, 80), "green": (10, 20), "yellow": (15, 25)},
            # Green-dominant (Supporters): High green, moderate others
            {"red": (10, 20), "blue": (15, 25), "green": (60, 80), "yellow": (15, 25)},
            # Yellow-dominant (Inspirational): High yellow, moderate others
            {"red": (15, 25), "blue": (10, 20), "green": (15, 25), "yellow": (60, 80)},
        ]
        
        # Generate employees for each archetype
        employees_per_type = self.n_employees // len(archetypes)
        
        for i, archetype in enumerate(archetypes):
            for j in range(employees_per_type):
                employee_id = f'EMP{i:02d}{j:03d}'
                
                # Generate energy values within archetype ranges
                red_energy = np.random.randint(archetype["red"][0], archetype["red"][1])
                blue_energy = np.random.randint(archetype["blue"][0], archetype["blue"][1])
                green_energy = np.random.randint(archetype["green"][0], archetype["green"][1])
                yellow_energy = np.random.randint(archetype["yellow"][0], archetype["yellow"][1])
                
                # Normalize to sum to 100
                total = red_energy + blue_energy + green_energy + yellow_energy
                red_energy = (red_energy / total) * 100
                blue_energy = (blue_energy / total) * 100
                green_energy = (green_energy / total) * 100
                yellow_energy = (yellow_energy / total) * 100
                
                data.append({
                    'employee_id': employee_id,
                    'red_energy': round(red_energy, 1),
                    'blue_energy': round(blue_energy, 1),
                    'green_energy': round(green_energy, 1),
                    'yellow_energy': round(yellow_energy, 1),
                    'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR']),
                    'experience_years': np.random.randint(1, 15),
                    'true_archetype': i
                })
        
        return pd.DataFrame(data)
    
    def test_neuromorphic_pipeline_integration(self):
        """Test complete pipeline from data parsing to team simulation"""
        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            # Step 1: Parse data
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            
            assert len(parsed_data) == self.n_employees
            assert all(col in parsed_data.columns for col in 
                      ['employee_id', 'red_energy', 'blue_energy', 'green_energy', 'yellow_energy'])
            
            # Step 2: Validate data quality
            validator = DataValidator()
            validation_results = validator.validate_data_quality(parsed_data)
            
            assert validation_results['is_valid']
            assert validation_results['quality_score'] > 80
            
            # Step 3: Extract features
            features = parser.get_clustering_features()
            assert features.shape == (self.n_employees, 4)
            
            # Step 4: Test neuromorphic clustering
            neuromorphic_clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4,
                random_state=42
            )
            
            neuromorphic_clusterer.fit(features)
            neuro_clusters = neuromorphic_clusterer.get_cluster_assignments()
            
            # Verify clustering results
            assert len(neuro_clusters) == self.n_employees
            assert len(np.unique(neuro_clusters)) <= 4
            assert all(0 <= cluster <= 3 for cluster in neuro_clusters)
            
            # Step 5: Test team simulation with neuromorphic clusters
            simulator = TeamCompositionSimulator(min_team_size=4, max_team_size=8)
            simulator.load_employee_data(parsed_data, neuro_clusters)
            
            teams = simulator.generate_balanced_teams(num_teams=5)
            
            # Verify team generation worked
            assert len(teams) > 0
            assert all('team_id' in team for team in teams)
            assert all('balance_score' in team for team in teams)
            
            # Verify teams use neuromorphic cluster assignments
            for team in teams:
                team_df = pd.DataFrame(team['members'])
                assert 'cluster' in team_df.columns
                # Check that cluster values match our neuromorphic assignments
                assert all(cluster in neuro_clusters for cluster in team_df['cluster'])
            
        finally:
            temp_path.unlink()
    
    def test_neuromorphic_vs_kmeans_integration(self):
        """Compare neuromorphic and K-means clustering in full pipeline"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            # Parse data once
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Test K-means clustering
            kmeans_clusterer = KMeansClusterer(n_clusters=4, random_state=42)
            kmeans_clusterer.fit(features)
            kmeans_clusters = kmeans_clusterer.get_cluster_assignments()
            
            # Test neuromorphic clustering
            neuromorphic_clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=4,
                random_state=42
            )
            neuromorphic_clusterer.fit(features)
            neuro_clusters = neuromorphic_clusterer.get_cluster_assignments()
            
            # Both should produce valid clusters
            assert len(kmeans_clusters) == len(neuro_clusters) == self.n_employees
            
            # Test team simulation with both clustering methods
            simulator_kmeans = TeamCompositionSimulator()
            simulator_kmeans.load_employee_data(parsed_data, kmeans_clusters)
            teams_kmeans = simulator_kmeans.generate_balanced_teams(num_teams=3)
            
            simulator_neuro = TeamCompositionSimulator()
            simulator_neuro.load_employee_data(parsed_data, neuro_clusters)
            teams_neuro = simulator_neuro.generate_balanced_teams(num_teams=3)
            
            # Both should generate teams successfully
            assert len(teams_kmeans) > 0
            assert len(teams_neuro) > 0
            
            # Compare team balance scores
            kmeans_scores = [team['balance_score'] for team in teams_kmeans]
            neuro_scores = [team['balance_score'] for team in teams_neuro]
            
            # Log comparison results
            print(f"K-means average balance score: {np.mean(kmeans_scores):.2f}")
            print(f"Neuromorphic average balance score: {np.mean(neuro_scores):.2f}")
            
            # Both methods should produce reasonable balance scores
            assert all(score > 40 for score in kmeans_scores)  # Minimum threshold
            assert all(score > 40 for score in neuro_scores)  # Minimum threshold
            
        finally:
            temp_path.unlink()
    
    def test_all_neuromorphic_methods_integration(self):
        """Test integration of all neuromorphic clustering methods"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        neuromorphic_methods = [
            NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
            NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
            NeuromorphicClusteringMethod.HYBRID_RESERVOIR
        ]
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            results = {}
            
            for method in neuromorphic_methods:
                method_name = method.value
                print(f"Testing {method_name}...")
                
                # Test clustering
                clusterer = NeuromorphicClusterer(
                    method=method,
                    n_clusters=4,
                    random_state=42
                )
                
                clusterer.fit(features)
                clusters = clusterer.get_cluster_assignments()
                
                # Test metrics calculation
                metrics = clusterer.get_clustering_metrics()
                
                # Test team simulation
                simulator = TeamCompositionSimulator()
                simulator.load_employee_data(parsed_data, clusters)
                teams = simulator.generate_balanced_teams(num_teams=2)
                
                # Store results
                results[method_name] = {
                    'clusters': clusters,
                    'metrics': metrics,
                    'teams': teams,
                    'avg_balance_score': np.mean([team['balance_score'] for team in teams])
                }
                
                # Verify results
                assert len(clusters) == self.n_employees
                assert len(teams) > 0
                assert metrics.silhouette_score >= -1.0
                assert metrics.silhouette_score <= 1.0
            
            # Compare methods
            print("\nMethod Comparison:")
            for method_name, result in results.items():
                print(f"{method_name}:")
                print(f"  Silhouette Score: {result['metrics'].silhouette_score:.3f}")
                print(f"  Stability: {result['metrics'].cluster_stability:.3f}")
                print(f"  Interpretability: {result['metrics'].interpretability_score:.3f}")
                print(f"  Avg Team Balance: {result['avg_balance_score']:.2f}")
            
            # All methods should produce valid results
            for method_name, result in results.items():
                assert result['metrics'].silhouette_score > -0.5  # Reasonable clustering
                assert result['avg_balance_score'] > 30  # Reasonable team balance
            
        finally:
            temp_path.unlink()
    
    def test_neuromorphic_cluster_interpretation(self):
        """Test psychological interpretation of neuromorphic clusters"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4,
                random_state=42
            )
            
            clusterer.fit(features)
            interpretations = clusterer.get_cluster_interpretation()
            
            # Verify interpretations
            assert len(interpretations) <= 4
            
            for cluster_id, traits in interpretations.items():
                # Check that all expected traits are present
                expected_traits = ['assertiveness', 'analytical', 'supportive', 'enthusiastic', 
                                 'complexity', 'stability']
                assert all(trait in traits for trait in expected_traits)
                
                # Check trait value ranges
                for trait, value in traits.items():
                    assert 0 <= value <= 1, f"Trait {trait} value {value} out of range [0,1]"
                
                # Verify at least one dominant trait
                personality_traits = ['assertiveness', 'analytical', 'supportive', 'enthusiastic']
                max_trait_value = max(traits[trait] for trait in personality_traits)
                assert max_trait_value > 0.2, f"No dominant trait in cluster {cluster_id}"
            
            print("Cluster Interpretations:")
            for cluster_id, traits in interpretations.items():
                dominant_trait = max(traits.items(), key=lambda x: x[1] if x[0] in 
                                   ['assertiveness', 'analytical', 'supportive', 'enthusiastic'] else 0)
                print(f"Cluster {cluster_id}: Dominant trait = {dominant_trait[0]} ({dominant_trait[1]:.3f})")
            
        finally:
            temp_path.unlink()
    
    def test_neuromorphic_performance_metrics(self):
        """Test neuromorphic-specific performance metrics"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4,
                random_state=42
            )
            
            clusterer.fit(features)
            metrics = clusterer.get_clustering_metrics()
            
            # Test metric ranges and validity
            assert -1.0 <= metrics.silhouette_score <= 1.0
            assert metrics.calinski_harabasz_score >= 0
            assert 0 <= metrics.cluster_stability <= 1.0
            assert 0 <= metrics.interpretability_score <= 1.0
            assert 0 <= metrics.temporal_coherence <= 1.0
            assert 0 <= metrics.computational_efficiency <= 1.0
            
            # Test that neuromorphic metrics provide additional insights
            # These should be different from what standard K-means would provide
            assert metrics.temporal_coherence is not None
            assert metrics.cluster_stability is not None
            assert metrics.interpretability_score is not None
            
            print(f"Neuromorphic Metrics:")
            print(f"  Silhouette Score: {metrics.silhouette_score:.3f}")
            print(f"  Cluster Stability: {metrics.cluster_stability:.3f}")
            print(f"  Interpretability: {metrics.interpretability_score:.3f}")
            print(f"  Temporal Coherence: {metrics.temporal_coherence:.3f}")
            print(f"  Computational Efficiency: {metrics.computational_efficiency:.3f}")
            
        finally:
            temp_path.unlink()
    
    def test_team_recommendation_with_neuromorphic(self):
        """Test team recommendation system with neuromorphic clustering"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Use neuromorphic clustering
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                n_clusters=4,
                random_state=42
            )
            
            clusterer.fit(features)
            clusters = clusterer.get_cluster_assignments()
            
            # Test team recommendations
            simulator = TeamCompositionSimulator(min_team_size=5, max_team_size=10)
            simulator.load_employee_data(parsed_data, clusters)
            
            # Generate multiple team composition options
            recommendations = simulator.recommend_optimal_teams(target_teams=3, iterations=5)
            
            assert len(recommendations) == 5  # 5 iterations
            assert all('teams' in rec for rec in recommendations)
            assert all('average_balance_score' in rec for rec in recommendations)
            
            # Test recommendations summary
            summary = simulator.get_team_recommendations_summary(recommendations)
            
            assert 'recommended_composition' in summary
            assert 'key_insights' in summary
            assert 'improvement_suggestions' in summary
            
            # Best recommendation should be first (highest score)
            best_rec = recommendations[0]
            assert best_rec['average_balance_score'] >= recommendations[-1]['average_balance_score']
            
            print(f"Best team composition average balance: {best_rec['average_balance_score']:.2f}")
            print(f"Number of insights: {len(summary['key_insights'])}")
            print(f"Number of suggestions: {len(summary['improvement_suggestions'])}")
            
        finally:
            temp_path.unlink()
    
    def test_error_handling_integration(self):
        """Test error handling in neuromorphic integration"""
        # Test with invalid data
        invalid_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002'],
            'red_energy': [50, 60],
            'blue_energy': [30, None],  # Missing value
            'green_energy': [20, 25],
            # Missing yellow_energy column
        })
        
        # Test neuromorphic clustering error handling
        clusterer = NeuromorphicClusterer(n_clusters=2, random_state=42)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            clusterer.fit(invalid_data)
        
        # Test with too few samples
        tiny_data = pd.DataFrame({
            'employee_id': ['EMP001'],
            'red_energy': [50],
            'blue_energy': [30],
            'green_energy': [20],
            'yellow_energy': [25]
        })
        
        # Should handle gracefully (might reduce clusters or give warning)
        try:
            clusterer.fit(tiny_data)
            clusters = clusterer.get_cluster_assignments()
            assert len(clusters) == 1
        except Exception as e:
            # Some error is expected with insufficient data
            assert "sample" in str(e).lower() or "cluster" in str(e).lower()
    
    @pytest.mark.slow
    def test_scalability_integration(self):
        """Test neuromorphic clustering with larger datasets"""
        # Generate larger dataset
        large_n = 500
        large_data = self._create_large_test_data(large_n)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        large_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Test with faster neuromorphic method for scalability
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=6,  # More clusters for larger dataset
                random_state=42,
                esn_params={'reservoir_size': 50}  # Smaller reservoir for speed
            )
            
            import time
            start_time = time.time()
            clusterer.fit(features)
            fit_time = time.time() - start_time
            
            clusters = clusterer.get_cluster_assignments()
            
            # Verify results
            assert len(clusters) == large_n
            assert len(np.unique(clusters)) <= 6
            
            # Test team simulation scalability
            simulator = TeamCompositionSimulator()
            simulator.load_employee_data(parsed_data, clusters)
            teams = simulator.generate_balanced_teams(num_teams=10)
            
            assert len(teams) > 0
            
            print(f"Processed {large_n} employees in {fit_time:.2f}s")
            print(f"Generated {len(teams)} teams")
            
        finally:
            temp_path.unlink()
    
    def _create_large_test_data(self, n_samples: int) -> pd.DataFrame:
        """Create larger test dataset for scalability testing"""
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # Random personality profile
            energies = np.random.dirichlet([1, 1, 1, 1]) * 100  # Sum to 100
            
            data.append({
                'employee_id': f'EMP{i:05d}',
                'red_energy': round(energies[0], 1),
                'blue_energy': round(energies[1], 1),
                'green_energy': round(energies[2], 1),
                'yellow_energy': round(energies[3], 1),
                'department': np.random.choice(['Eng', 'Marketing', 'Sales', 'HR', 'Finance']),
                'experience_years': np.random.randint(0, 20)
            })
        
        return pd.DataFrame(data)


if __name__ == '__main__':
    # Run specific test
    test_instance = TestNeuromorphicIntegration()
    test_instance.setup_method()
    
    print("Running neuromorphic integration tests...")
    
    # Run key integration tests
    test_instance.test_neuromorphic_pipeline_integration()
    print("✓ Pipeline integration test passed")
    
    test_instance.test_neuromorphic_vs_kmeans_integration()
    print("✓ K-means comparison test passed")
    
    test_instance.test_neuromorphic_cluster_interpretation()
    print("✓ Cluster interpretation test passed")
    
    test_instance.test_team_recommendation_with_neuromorphic()
    print("✓ Team recommendation test passed")
    
    print("\nAll integration tests passed successfully!")