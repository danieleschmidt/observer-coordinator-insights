"""
Integration tests for the complete clustering pipeline
"""

import pytest
import pandas as pd
import tempfile
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.parser import InsightsDataParser
from insights_clustering.clustering import KMeansClusterer
from insights_clustering.validator import DataValidator


class TestClusteringPipeline:
    """Test complete clustering pipeline integration"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create sample dataset
        np.random.seed(42)
        n_employees = 50
        
        self.test_data = {
            'employee_id': [f'EMP{i:03d}' for i in range(1, n_employees + 1)],
            'red_energy': np.random.randint(15, 40, n_employees),
            'blue_energy': np.random.randint(15, 40, n_employees),
            'green_energy': np.random.randint(15, 40, n_employees),
            'yellow_energy': np.random.randint(15, 40, n_employees),
            'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n_employees),
            'experience_years': np.random.randint(1, 15, n_employees)
        }
    
    def create_temp_csv(self, data_dict):
        """Helper to create temporary CSV file"""
        df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        return Path(temp_file.name)
    
    def test_complete_pipeline(self):
        """Test complete pipeline from CSV to clustering"""
        # Create test CSV
        csv_file = self.create_temp_csv(self.test_data)
        
        try:
            # Step 1: Parse data
            parser = InsightsDataParser()
            data = parser.parse_csv(csv_file)
            
            # Step 2: Validate data
            validator = DataValidator()
            validation_results = validator.validate_data_quality(data)
            
            assert validation_results['is_valid']
            assert validation_results['quality_score'] > 80
            
            # Step 3: Extract features for clustering
            features = parser.get_clustering_features()
            
            # Step 4: Perform clustering
            clusterer = KMeansClusterer(n_clusters=4)
            clusterer.fit(features)
            
            # Step 5: Get results
            cluster_assignments = clusterer.get_cluster_assignments()
            centroids = clusterer.get_cluster_centroids()
            quality_metrics = clusterer.get_cluster_quality_metrics()
            
            # Assertions
            assert len(cluster_assignments) == len(self.test_data['employee_id'])
            assert centroids.shape == (4, 4)  # 4 clusters, 4 features
            assert 'silhouette_score' in quality_metrics
            assert 'inertia' in quality_metrics
            
            # Check cluster assignments are valid
            unique_clusters = np.unique(cluster_assignments)
            assert len(unique_clusters) <= 4
            assert all(0 <= cluster <= 3 for cluster in unique_clusters)
            
        finally:
            csv_file.unlink()
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline handles missing data gracefully"""
        # Introduce missing values
        data_with_missing = self.test_data.copy()
        data_with_missing['red_energy'][0] = None
        data_with_missing['blue_energy'][1] = None
        
        csv_file = self.create_temp_csv(data_with_missing)
        
        try:
            parser = InsightsDataParser()
            data = parser.parse_csv(csv_file)
            
            # Should handle missing data
            assert data['red_energy'].isnull().sum() == 0
            assert data['blue_energy'].isnull().sum() == 0
            
            # Should still be able to cluster
            features = parser.get_clustering_features()
            clusterer = KMeansClusterer(n_clusters=3)
            clusterer.fit(features)
            
            cluster_assignments = clusterer.get_cluster_assignments()
            assert len(cluster_assignments) == len(data)
            
        finally:
            csv_file.unlink()
    
    def test_optimal_cluster_detection(self):
        """Test optimal cluster number detection"""
        csv_file = self.create_temp_csv(self.test_data)
        
        try:
            parser = InsightsDataParser()
            data = parser.parse_csv(csv_file)
            features = parser.get_clustering_features()
            
            clusterer = KMeansClusterer()
            scores = clusterer.find_optimal_clusters(features, max_clusters=8)
            
            # Should return scores for k=2 to k=8
            assert len(scores) == 7
            assert all(k in scores for k in range(2, 9))
            assert all('silhouette_score' in scores[k] for k in scores)
            assert all('inertia' in scores[k] for k in scores)
            
        finally:
            csv_file.unlink()
    
    def test_data_profile_generation(self):
        """Test comprehensive data profiling"""
        csv_file = self.create_temp_csv(self.test_data)
        
        try:
            parser = InsightsDataParser()
            data = parser.parse_csv(csv_file)
            
            validator = DataValidator()
            profile = validator.get_data_profile(data)
            
            # Check profile completeness
            assert 'shape' in profile
            assert 'columns' in profile
            assert 'numeric_summary' in profile
            assert 'categorical_summary' in profile
            
            # Check numeric summaries for energy columns
            energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
            for col in energy_cols:
                assert col in profile['numeric_summary']
                assert 'mean' in profile['numeric_summary'][col]
                assert 'std' in profile['numeric_summary'][col]
            
            # Check categorical summaries
            assert 'department' in profile['categorical_summary']
            assert 'unique_count' in profile['categorical_summary']['department']
            
        finally:
            csv_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])