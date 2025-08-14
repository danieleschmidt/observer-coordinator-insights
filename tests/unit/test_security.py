"""
Unit tests for security module
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.security import (
    EnhancedDataAnonymizer, InputValidator, EnhancedSecurityAuditor, 
    SecureDataProcessor
)


class TestEnhancedDataAnonymizer:
    """Test data anonymization functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.anonymizer = EnhancedDataAnonymizer()
        self.sample_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
            'email': ['john@company.com', 'jane@company.com', 'bob@company.com'],
            'red_energy': [30, 25, 35],
            'blue_energy': [20, 30, 15],
            'green_energy': [25, 25, 25],
            'yellow_energy': [25, 20, 25]
        })
    
    def test_anonymize_employee_id(self):
        """Test employee ID anonymization"""
        original_id = "EMP001"
        anonymized = self.anonymizer.anonymize_employee_id(original_id)
        
        assert anonymized.startswith("EMP_")
        assert len(anonymized) == 16  # EMP_ + 12 character hash
        assert anonymized != original_id
        
        # Test consistency - same ID should produce same result
        second_anonymized = self.anonymizer.anonymize_employee_id(original_id)
        assert anonymized == second_anonymized
    
    def test_anonymize_employee_id_invalid_input(self):
        """Test employee ID anonymization with invalid input"""
        with pytest.raises(ValueError):
            self.anonymizer.anonymize_employee_id("")
        
        with pytest.raises(ValueError):
            self.anonymizer.anonymize_employee_id(None)
    
    def test_anonymize_dataframe(self):
        """Test DataFrame anonymization"""
        anonymized_df = self.anonymizer.anonymize_dataframe(self.sample_data)
        
        # Check that PII columns are handled
        assert 'name' not in anonymized_df.columns
        assert 'email' not in anonymized_df.columns
        
        # Check that employee IDs are anonymized
        assert all(emp_id.startswith("EMP_") for emp_id in anonymized_df['employee_id'])
        
        # Check that energy columns are preserved
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        for col in energy_cols:
            assert col in anonymized_df.columns
            assert anonymized_df[col].equals(self.sample_data[col])
    
    def test_anonymize_dataframe_empty(self):
        """Test anonymization with empty DataFrame"""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            self.anonymizer.anonymize_dataframe(empty_df)
    
    def test_encrypt_decrypt_data(self):
        """Test data encryption and decryption"""
        original_data = "sensitive information"
        encrypted = self.anonymizer.encrypt_sensitive_data(original_data)
        decrypted = self.anonymizer.decrypt_sensitive_data(encrypted)
        
        assert encrypted != original_data
        assert decrypted == original_data
    
    def test_encrypt_empty_data(self):
        """Test encryption with empty data"""
        result = self.anonymizer.encrypt_sensitive_data("")
        assert result == ""


class TestInputValidator:
    """Test input validation functionality"""
    
    def test_validate_file_path_valid(self):
        """Test valid file path validation"""
        valid_paths = [
            "data.csv",
            "employees.xlsx",
            "results.json",
            "folder/data.csv"
        ]
        
        for path in valid_paths:
            assert InputValidator.validate_file_path(path) is True
    
    def test_validate_file_path_invalid(self):
        """Test invalid file path validation"""
        invalid_paths = [
            "",
            None,
            "../../../etc/passwd",
            "/absolute/path/file.csv",
            "file.txt",  # Not allowed extension
            "file.exe"   # Not allowed extension
        ]
        
        for path in invalid_paths:
            assert InputValidator.validate_file_path(path) is False
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        test_cases = [
            ("normal_file.csv", "normal_file.csv"),
            ("file with spaces.csv", "file_with_spaces.csv"),
            ("file@#$%^&*().csv", "file_________.csv"),
            ("", "unnamed_file"),
            (".hidden_file", "file_hidden_file"),
            ("very_long_filename_that_exceeds_the_maximum_allowed_length_for_filenames_and_should_be_truncated.csv", 
             "very_long_filename_that_exceeds_the_maximum_allowed_length_for_filenames_and_should_be_tr")
        ]
        
        for input_name, expected in test_cases:
            result = InputValidator.sanitize_filename(input_name)
            assert result == expected or len(result) <= 100
    
    def test_validate_cluster_count(self):
        """Test cluster count validation"""
        # Valid cases
        assert InputValidator.validate_cluster_count(3, 100) is True
        assert InputValidator.validate_cluster_count(2, 10) is True
        
        # Invalid cases
        assert InputValidator.validate_cluster_count(1, 100) is False  # Less than 2
        assert InputValidator.validate_cluster_count(50, 10) is False  # More than data size
        assert InputValidator.validate_cluster_count("3", 100) is False  # Non-integer
    
    def test_validate_energy_values(self):
        """Test energy values validation"""
        # Valid data
        valid_data = pd.DataFrame({
            'red_energy': [25, 30, 20],
            'blue_energy': [25, 20, 30],
            'green_energy': [25, 25, 25],
            'yellow_energy': [25, 25, 25]
        })
        
        result = InputValidator.validate_energy_values(valid_data)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid data - missing column
        invalid_data = pd.DataFrame({
            'red_energy': [25, 30, 20],
            'blue_energy': [25, 20, 30],
            'green_energy': [25, 25, 25]
            # Missing yellow_energy
        })
        
        result = InputValidator.validate_energy_values(invalid_data)
        assert result['is_valid'] is False
        assert any('yellow_energy' in error for error in result['errors'])
        
        # Data with out-of-range values
        out_of_range_data = pd.DataFrame({
            'red_energy': [125, 30, -10],  # Out of range values
            'blue_energy': [25, 20, 30],
            'green_energy': [25, 25, 25],
            'yellow_energy': [25, 25, 25]
        })
        
        result = InputValidator.validate_energy_values(out_of_range_data)
        assert len(result['warnings']) > 0


class TestSecurityAuditor:
    """Test security auditing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.auditor = SecurityAuditor()
    
    def test_log_data_access(self):
        """Test data access logging"""
        self.auditor.log_data_access(
            user_id="test_user",
            data_type="employee_data",
            record_count=100,
            operation="read"
        )
        
        assert len(self.auditor.audit_log) == 1
        log_entry = self.auditor.audit_log[0]
        
        assert log_entry['user_id'] == "test_user"
        assert log_entry['data_type'] == "employee_data"
        assert log_entry['record_count'] == 100
        assert log_entry['operation'] == "read"
        assert 'timestamp' in log_entry
        assert 'session_id' in log_entry
    
    def test_log_clustering_operation(self):
        """Test clustering operation logging"""
        parameters = {'n_clusters': 4, 'data_size': 100}
        result_summary = {'silhouette_score': 0.75, 'clusters_created': 4}
        
        self.auditor.log_clustering_operation(parameters, result_summary)
        
        assert len(self.auditor.audit_log) == 1
        log_entry = self.auditor.audit_log[0]
        
        assert log_entry['operation'] == 'clustering'
        assert log_entry['parameters'] == parameters
        assert log_entry['result_summary'] == result_summary
    
    def test_get_audit_summary(self):
        """Test audit summary generation"""
        # Add some test logs
        self.auditor.log_data_access("user1", "type1", 50, "read")
        self.auditor.log_data_access("user2", "type2", 75, "write")
        
        summary = self.auditor.get_audit_summary(days=30)
        
        assert summary['total_operations'] == 2
        assert summary['unique_users'] == 2
        assert 'type1' in summary['data_types_accessed']
        assert 'type2' in summary['data_types_accessed']
        assert summary['period_days'] == 30
    
    def test_check_data_retention_compliance(self):
        """Test data retention compliance checking"""
        from datetime import datetime, timedelta
        
        # Recent data should be compliant
        recent_date = datetime.utcnow() - timedelta(days=30)
        assert self.auditor.check_data_retention_compliance(recent_date, 180) is True
        
        # Old data should not be compliant
        old_date = datetime.utcnow() - timedelta(days=200)
        assert self.auditor.check_data_retention_compliance(old_date, 180) is False


class TestSecureDataProcessor:
    """Test secure data processing functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = SecureDataProcessor()
        
        # Create temporary test file
        self.test_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003'],
            'red_energy': [30, 25, 35],
            'blue_energy': [20, 30, 15],
            'green_energy': [25, 25, 25],
            'yellow_energy': [25, 20, 25]
        })
        
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.test_data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_secure_load_data(self):
        """Test secure data loading"""
        # Get just the filename (not full path) for security validation
        filename = os.path.basename(self.temp_file.name)
        
        with patch('src.security.InputValidator.validate_file_path', return_value=True):
            with patch('pandas.read_csv', return_value=self.test_data):
                result = self.processor.secure_load_data(filename, "test_user")
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # Check that data was anonymized
        assert all(emp_id.startswith("EMP_") for emp_id in result['employee_id'])
    
    def test_secure_load_data_invalid_path(self):
        """Test secure data loading with invalid path"""
        with pytest.raises(ValueError, match="Invalid or unsafe file path"):
            self.processor.secure_load_data("../../../etc/passwd", "test_user")
    
    @patch('src.insights_clustering.clustering.KMeansClusterer')
    def test_secure_clustering_pipeline(self, mock_clusterer_class):
        """Test secure clustering pipeline"""
        # Mock the clusterer
        mock_clusterer = MagicMock()
        mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1, 0])
        mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({'feature1': [1, 2]})
        mock_clusterer.get_cluster_quality_metrics.return_value = {'silhouette_score': 0.75}
        mock_clusterer_class.return_value = mock_clusterer
        
        result = self.processor.secure_clustering_pipeline(self.test_data, 2, "test_user")
        
        assert 'cluster_assignments' in result
        assert 'centroids' in result
        assert 'quality_metrics' in result
        assert 'data_summary' in result
        
        # Verify clusterer was called correctly
        mock_clusterer_class.assert_called_once_with(n_clusters=2)
        mock_clusterer.fit.assert_called_once()
    
    def test_secure_clustering_pipeline_invalid_clusters(self):
        """Test secure clustering pipeline with invalid cluster count"""
        with pytest.raises(ValueError, match="Invalid cluster configuration"):
            self.processor.secure_clustering_pipeline(self.test_data, 10, "test_user")  # Too many clusters


@pytest.fixture
def sample_employee_data():
    """Fixture providing sample employee data"""
    return pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
        'red_energy': [30, 25, 35, 20],
        'blue_energy': [20, 30, 15, 25],
        'green_energy': [25, 25, 25, 30],
        'yellow_energy': [25, 20, 25, 25]
    })


def test_integration_secure_processing_pipeline(sample_employee_data):
    """Integration test for complete secure processing pipeline"""
    processor = SecureDataProcessor()
    
    # Test the full pipeline with valid data
    with patch('src.security.InputValidator.validate_file_path', return_value=True):
        with patch('pandas.read_csv', return_value=sample_employee_data):
            with patch('src.insights_clustering.clustering.KMeansClusterer') as mock_clusterer_class:
                # Mock the clusterer
                mock_clusterer = MagicMock()
                mock_clusterer.get_cluster_assignments.return_value = np.array([0, 1, 0, 1])
                mock_clusterer.get_cluster_centroids.return_value = pd.DataFrame({
                    'red_energy': [25, 30],
                    'blue_energy': [25, 25]
                })
                mock_clusterer.get_cluster_quality_metrics.return_value = {
                    'silhouette_score': 0.75,
                    'inertia': 100.0
                }
                mock_clusterer_class.return_value = mock_clusterer
                
                # Load data securely
                data = processor.secure_load_data("test.csv", "test_user")
                
                # Perform secure clustering
                results = processor.secure_clustering_pipeline(data, 2, "test_user")
                
                # Verify results
                assert len(results['cluster_assignments']) == 4
                assert results['quality_metrics']['silhouette_score'] == 0.75
                assert results['data_summary']['total_employees'] == 4
                
                # Verify audit logs were created
                assert len(processor.auditor.audit_log) == 2  # One for load, one for clustering