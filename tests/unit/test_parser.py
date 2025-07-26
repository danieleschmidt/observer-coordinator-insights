"""
Unit tests for Insights Discovery data parser
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
import io
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.parser import InsightsDataParser


class TestInsightsDataParser:
    """Test cases for InsightsDataParser"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.parser = InsightsDataParser()
        
        # Sample valid data
        self.valid_data = {
            'employee_id': ['EMP001', 'EMP002', 'EMP003'],
            'red_energy': [30, 25, 35],
            'blue_energy': [20, 35, 15],
            'green_energy': [25, 20, 30],
            'yellow_energy': [25, 20, 20]
        }
    
    def create_temp_csv(self, data_dict):
        """Helper to create temporary CSV file"""
        df = pd.DataFrame(data_dict)
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        return Path(temp_file.name)
    
    def test_parse_valid_csv(self):
        """Test parsing valid CSV data"""
        csv_file = self.create_temp_csv(self.valid_data)
        
        result = self.parser.parse_csv(csv_file)
        
        assert len(result) == 3
        assert list(result.columns) == list(self.valid_data.keys())
        assert result['employee_id'].tolist() == ['EMP001', 'EMP002', 'EMP003']
        
        # Clean up
        csv_file.unlink()
    
    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError"""
        invalid_data = {
            'employee_id': ['EMP001'],
            'red_energy': [30]
            # Missing blue, green, yellow energies
        }
        csv_file = self.create_temp_csv(invalid_data)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            self.parser.parse_csv(csv_file)
        
        csv_file.unlink()
    
    def test_get_clustering_features(self):
        """Test extraction of clustering features"""
        csv_file = self.create_temp_csv(self.valid_data)
        self.parser.parse_csv(csv_file)
        
        features = self.parser.get_clustering_features()
        
        expected_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        assert list(features.columns) == expected_cols
        assert len(features) == 3
        
        csv_file.unlink()
    
    def test_get_employee_metadata(self):
        """Test extraction of employee metadata"""
        # Add extra metadata column
        data_with_meta = self.valid_data.copy()
        data_with_meta['department'] = ['Engineering', 'Marketing', 'Sales']
        
        csv_file = self.create_temp_csv(data_with_meta)
        self.parser.parse_csv(csv_file)
        
        metadata = self.parser.get_employee_metadata()
        
        assert 'employee_id' in metadata.columns
        assert 'department' in metadata.columns
        assert 'red_energy' not in metadata.columns  # Should be excluded
        
        csv_file.unlink()
    
    def test_data_cleaning_fills_missing_values(self):
        """Test that missing energy values are filled"""
        data_with_missing = self.valid_data.copy()
        data_with_missing['red_energy'][1] = None  # Add missing value
        
        csv_file = self.create_temp_csv(data_with_missing)
        result = self.parser.parse_csv(csv_file)
        
        # Should have no null values after cleaning
        assert result['red_energy'].isnull().sum() == 0
        
        csv_file.unlink()
    
    def test_energy_normalization(self):
        """Test that energy values are normalized to sum to 100"""
        csv_file = self.create_temp_csv(self.valid_data)
        result = self.parser.parse_csv(csv_file)
        
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        energy_sums = result[energy_cols].sum(axis=1)
        
        # All rows should sum to approximately 100
        assert all(abs(sum_val - 100.0) < 0.1 for sum_val in energy_sums)
        
        csv_file.unlink()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])