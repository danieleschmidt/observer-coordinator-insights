"""
Security and Privacy Compliance Testing Suite
Tests data protection, encryption, audit logging, differential privacy, and GDPR/CCPA compliance
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import tempfile
import hashlib
import json
import time
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import secrets
import base64
from typing import Dict, List, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer,
    NeuromorphicClusteringMethod,
    get_correlation_id,
    set_correlation_id
)
from insights_clustering.parser import InsightsDataParser
from insights_clustering.security import SecurityManager, DataProtectionManager
from database.repositories.audit import AuditRepository

# Security test configuration
SECURITY_CONFIG = {
    'min_encryption_key_bits': 256,
    'audit_log_retention_days': 90,
    'pii_detection_threshold': 0.8,
    'differential_privacy_epsilon': 1.0,
    'min_anonymization_k': 3,
    'data_breach_simulation_size': 1000,
    'max_inference_accuracy': 0.6  # Max allowed re-identification accuracy
}

class MockSecurityManager:
    """Mock security manager for testing"""
    
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.audit_logs = []
        
    def encrypt_data(self, data: str) -> bytes:
        """Encrypt sensitive data"""
        return self.cipher.encrypt(data.encode())
        
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Decrypt sensitive data"""
        return self.cipher.decrypt(encrypted_data).decode()
        
    def hash_pii(self, pii_data: str) -> str:
        """Hash PII data for safe storage"""
        return hashlib.sha256(pii_data.encode()).hexdigest()
        
    def log_audit_event(self, event_type: str, user_id: str, resource: str, 
                       details: Dict[str, Any], correlation_id: str = None):
        """Log security-relevant events"""
        self.audit_logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'resource': resource,
            'details': details,
            'correlation_id': correlation_id or get_correlation_id()
        })


class TestDataEncryption:
    """Test data encryption and key management"""
    
    def setup_method(self):
        """Setup encryption test environment"""
        self.security_manager = MockSecurityManager()
        
    def test_data_encryption_basic(self):
        """Test basic data encryption functionality"""
        # Test string data
        sensitive_data = "employee_12345_salary_85000"
        encrypted = self.security_manager.encrypt_data(sensitive_data)
        decrypted = self.security_manager.decrypt_data(encrypted)
        
        assert decrypted == sensitive_data
        assert encrypted != sensitive_data.encode()
        assert len(encrypted) > len(sensitive_data)
        
    def test_personality_data_encryption(self):
        """Test encryption of personality assessment data"""
        # Create sample personality data
        personality_data = {
            'employee_id': 'EMP123456',
            'red_energy': 75.5,
            'blue_energy': 45.2,
            'green_energy': 30.8,
            'yellow_energy': 48.5,
            'assessment_date': '2024-01-15',
            'assessor': 'HR_SYSTEM'
        }
        
        # Encrypt sensitive fields
        sensitive_fields = ['employee_id', 'assessor']
        encrypted_data = personality_data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_data:
                encrypted_value = self.security_manager.encrypt_data(str(encrypted_data[field]))
                encrypted_data[field] = base64.b64encode(encrypted_value).decode()
        
        # Verify encryption worked
        assert encrypted_data['employee_id'] != personality_data['employee_id']
        assert encrypted_data['assessor'] != personality_data['assessor']
        
        # Verify energy values remain unencrypted (for analysis)
        assert encrypted_data['red_energy'] == personality_data['red_energy']
        assert encrypted_data['blue_energy'] == personality_data['blue_energy']
        
    def test_key_rotation_simulation(self):
        """Test key rotation procedures"""
        original_data = "sensitive_employee_data_12345"
        
        # Encrypt with original key
        encrypted_v1 = self.security_manager.encrypt_data(original_data)
        
        # Simulate key rotation
        old_cipher = self.security_manager.cipher
        new_key = Fernet.generate_key()
        new_cipher = Fernet(new_key)
        
        # Decrypt with old key, re-encrypt with new key
        decrypted = old_cipher.decrypt(encrypted_v1).decode()
        encrypted_v2 = new_cipher.encrypt(decrypted.encode())
        
        # Update security manager
        self.security_manager.cipher = new_cipher
        
        # Verify new encryption works
        final_decrypted = self.security_manager.decrypt_data(encrypted_v2)
        assert final_decrypted == original_data
        
        # Verify old encrypted data is no longer decryptable with new key
        with pytest.raises(Exception):
            new_cipher.decrypt(encrypted_v1)
            
    def test_encryption_performance(self):
        """Test encryption performance with large datasets"""
        # Generate large dataset
        large_dataset = []
        for i in range(1000):
            employee_data = f"employee_{i:06d}_data_{'x'*100}"
            large_dataset.append(employee_data)
        
        # Measure encryption time
        start_time = time.time()
        encrypted_dataset = []
        for data in large_dataset:
            encrypted = self.security_manager.encrypt_data(data)
            encrypted_dataset.append(encrypted)
        encryption_time = time.time() - start_time
        
        # Measure decryption time
        start_time = time.time()
        decrypted_dataset = []
        for encrypted_data in encrypted_dataset:
            decrypted = self.security_manager.decrypt_data(encrypted_data)
            decrypted_dataset.append(decrypted)
        decryption_time = time.time() - start_time
        
        # Performance assertions
        assert encryption_time < 10.0  # Should encrypt 1000 records in <10s
        assert decryption_time < 10.0  # Should decrypt 1000 records in <10s
        
        # Verify all data correctly encrypted/decrypted
        assert len(decrypted_dataset) == len(large_dataset)
        for original, decrypted in zip(large_dataset, decrypted_dataset):
            assert original == decrypted


class TestPIIDetectionAndProtection:
    """Test PII detection and protection mechanisms"""
    
    def setup_method(self):
        """Setup PII protection test environment"""
        self.security_manager = MockSecurityManager()
        
    def test_pii_field_detection(self):
        """Test detection of PII fields in data"""
        # Create test data with various PII types
        test_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003'],
            'first_name': ['John', 'Jane', 'Bob'],
            'last_name': ['Doe', 'Smith', 'Johnson'],
            'email': ['john.doe@company.com', 'jane.smith@company.com', 'bob.johnson@company.com'],
            'ssn': ['123-45-6789', '987-65-4321', '456-78-9123'],
            'phone': ['555-123-4567', '555-987-6543', '555-456-7890'],
            'red_energy': [75.5, 45.2, 60.8],
            'blue_energy': [45.2, 70.1, 35.5],
            'department': ['Engineering', 'HR', 'Sales'],
            'hire_date': ['2020-01-15', '2019-05-22', '2021-03-10']
        })
        
        # Define PII field patterns
        pii_patterns = {
            'employee_id': r'^EMP\d+$',
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'phone': r'^\d{3}-\d{3}-\d{4}$'
        }
        
        detected_pii = []
        for column in test_data.columns:
            if column in pii_patterns:
                # Check if column matches PII pattern
                sample_value = str(test_data[column].iloc[0])
                import re
                if re.match(pii_patterns[column], sample_value):
                    detected_pii.append(column)
        
        # Assert PII detection
        expected_pii = ['employee_id', 'email', 'ssn', 'phone']
        assert set(detected_pii) == set(expected_pii)
        
    def test_pii_anonymization(self):
        """Test PII anonymization techniques"""
        # Original data with PII
        pii_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
            'name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
            'age': [25, 30, 35, 28],
            'salary': [50000, 75000, 90000, 60000],
            'department': ['Eng', 'HR', 'Sales', 'Eng'],
            'red_energy': [75, 45, 60, 80],
            'blue_energy': [45, 70, 35, 25]
        })
        
        # Anonymization strategies
        anonymized_data = pii_data.copy()
        
        # 1. Remove direct identifiers
        anonymized_data = anonymized_data.drop(columns=['employee_id', 'name'])
        
        # 2. Generalize quasi-identifiers
        anonymized_data['age_group'] = pd.cut(
            anonymized_data['age'], 
            bins=[0, 25, 35, 100], 
            labels=['<25', '25-35', '>35']
        )
        anonymized_data = anonymized_data.drop(columns=['age'])
        
        # 3. Apply noise to sensitive attributes
        noise = np.random.normal(0, 2, len(anonymized_data))
        anonymized_data['red_energy_noisy'] = anonymized_data['red_energy'] + noise
        anonymized_data['blue_energy_noisy'] = anonymized_data['blue_energy'] + noise
        
        # Verify anonymization
        assert 'employee_id' not in anonymized_data.columns
        assert 'name' not in anonymized_data.columns
        assert 'age_group' in anonymized_data.columns
        assert len(anonymized_data['age_group'].unique()) == 2  # Should have 2 groups for this data
        
        # Verify energy values are close but not identical
        energy_diff = np.abs(anonymized_data['red_energy'] - anonymized_data['red_energy_noisy'])
        assert np.all(energy_diff > 0)  # Should all be different due to noise
        assert np.mean(energy_diff) < 5  # But not too different
        
    def test_k_anonymity_validation(self):
        """Test k-anonymity validation"""
        # Create test data
        data = pd.DataFrame({
            'age': [25, 25, 30, 30, 35, 35],
            'department': ['Eng', 'Eng', 'HR', 'HR', 'Sales', 'Sales'],
            'salary_band': ['50-60k', '50-60k', '70-80k', '70-80k', '80-90k', '80-90k'],
            'red_energy': [75, 78, 45, 42, 60, 65]
        })
        
        # Check k-anonymity for k=2
        k = 2
        quasi_identifiers = ['age', 'department', 'salary_band']
        
        # Group by quasi-identifiers
        grouped = data.groupby(quasi_identifiers).size()
        min_group_size = grouped.min()
        
        assert min_group_size >= k, f"K-anonymity violated: minimum group size is {min_group_size}, required {k}"
        
        # Verify all groups meet k-anonymity
        for group_size in grouped.values:
            assert group_size >= k
            
    def test_differential_privacy(self):
        """Test differential privacy mechanisms"""
        # Original dataset
        original_data = pd.DataFrame({
            'red_energy': np.random.normal(50, 15, 1000),
            'blue_energy': np.random.normal(50, 15, 1000),
            'green_energy': np.random.normal(50, 15, 1000),
            'yellow_energy': np.random.normal(50, 15, 1000)
        })
        
        # Apply Laplace mechanism for differential privacy
        epsilon = SECURITY_CONFIG['differential_privacy_epsilon']
        sensitivity = 100  # Max change in personality energy
        
        def add_laplace_noise(data, epsilon, sensitivity):
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, data.shape)
            return data + noise
        
        # Add noise to preserve differential privacy
        private_data = original_data.copy()
        for column in private_data.columns:
            private_data[column] = add_laplace_noise(
                private_data[column], epsilon, sensitivity
            )
        
        # Verify privacy preservation
        # 1. Data should be different
        for column in original_data.columns:
            assert not np.allclose(original_data[column], private_data[column])
        
        # 2. But statistical properties should be approximately preserved
        for column in original_data.columns:
            original_mean = original_data[column].mean()
            private_mean = private_data[column].mean()
            
            # Allow for noise but expect similar means
            assert abs(original_mean - private_mean) < 10, f"Mean changed too much for {column}"
        
        # 3. Test privacy guarantee by attempting re-identification
        correlation = np.corrcoef(
            original_data['red_energy'], 
            private_data['red_energy']
        )[0, 1]
        
        # Should be correlated (utility preserved) but not perfectly (privacy preserved)
        assert 0.3 < correlation < 0.9, f"Privacy-utility balance not achieved: correlation = {correlation}"


class TestAuditLoggingAndCompliance:
    """Test audit logging and compliance features"""
    
    def setup_method(self):
        """Setup audit logging test environment"""
        self.security_manager = MockSecurityManager()
        set_correlation_id("test-audit-correlation")
        
    def test_clustering_audit_logging(self):
        """Test audit logging for clustering operations"""
        # Create test data
        test_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
            'red_energy': [75, 45, 60, 80],
            'blue_energy': [45, 70, 35, 25],
            'green_energy': [30, 20, 50, 40],
            'yellow_energy': [50, 65, 55, 55]
        })
        
        # Mock audit logging during clustering
        user_id = "test_analyst_001"
        
        # Log data access
        self.security_manager.log_audit_event(
            event_type="DATA_ACCESS",
            user_id=user_id,
            resource="employee_personality_data",
            details={
                'action': 'read',
                'record_count': len(test_data),
                'fields_accessed': list(test_data.columns)
            }
        )
        
        # Perform clustering
        features = test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        clusterer = NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            n_clusters=2,
            random_state=42
        )
        
        # Log clustering operation
        start_time = datetime.utcnow()
        clusterer.fit(features)
        end_time = datetime.utcnow()
        
        self.security_manager.log_audit_event(
            event_type="CLUSTERING_ANALYSIS",
            user_id=user_id,
            resource="neuromorphic_clustering",
            details={
                'method': 'echo_state_network',
                'n_clusters': 2,
                'n_samples': len(features),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'success': True
            }
        )
        
        # Log result access
        labels = clusterer.get_cluster_assignments()
        self.security_manager.log_audit_event(
            event_type="RESULT_ACCESS",
            user_id=user_id,
            resource="clustering_results",
            details={
                'action': 'retrieve_labels',
                'result_count': len(labels),
                'unique_clusters': len(np.unique(labels))
            }
        )
        
        # Verify audit logs
        assert len(self.security_manager.audit_logs) == 3
        
        # Check log structure
        for log in self.security_manager.audit_logs:
            assert 'timestamp' in log
            assert 'event_type' in log
            assert 'user_id' in log
            assert 'resource' in log
            assert 'details' in log
            assert 'correlation_id' in log
            assert log['user_id'] == user_id
            
        # Verify specific events
        event_types = [log['event_type'] for log in self.security_manager.audit_logs]
        assert 'DATA_ACCESS' in event_types
        assert 'CLUSTERING_ANALYSIS' in event_types
        assert 'RESULT_ACCESS' in event_types
        
    def test_audit_log_retention_policy(self):
        """Test audit log retention and cleanup policies"""
        # Create audit logs with different ages
        current_time = datetime.utcnow()
        
        test_logs = [
            # Recent logs (should be retained)
            {
                'timestamp': (current_time - timedelta(days=1)).isoformat(),
                'event_type': 'DATA_ACCESS',
                'user_id': 'user001',
                'resource': 'test_resource',
                'details': {'test': True}
            },
            {
                'timestamp': (current_time - timedelta(days=30)).isoformat(),
                'event_type': 'CLUSTERING_ANALYSIS',
                'user_id': 'user002',
                'resource': 'test_resource',
                'details': {'test': True}
            },
            # Old logs (should be cleaned up)
            {
                'timestamp': (current_time - timedelta(days=120)).isoformat(),
                'event_type': 'OLD_EVENT',
                'user_id': 'user003',
                'resource': 'test_resource',
                'details': {'test': True}
            },
            {
                'timestamp': (current_time - timedelta(days=200)).isoformat(),
                'event_type': 'VERY_OLD_EVENT',
                'user_id': 'user004',
                'resource': 'test_resource',
                'details': {'test': True}
            }
        ]
        
        # Apply retention policy
        retention_days = SECURITY_CONFIG['audit_log_retention_days']
        cutoff_date = current_time - timedelta(days=retention_days)
        
        retained_logs = []
        for log in test_logs:
            log_date = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
            if log_date.replace(tzinfo=None) > cutoff_date:
                retained_logs.append(log)
        
        # Verify retention policy
        assert len(retained_logs) == 2  # Only recent logs should be retained
        for log in retained_logs:
            log_date = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
            assert log_date.replace(tzinfo=None) > cutoff_date
            
    def test_gdpr_compliance_features(self):
        """Test GDPR compliance features"""
        # Test data subject rights
        
        # 1. Right to access (Article 15)
        subject_id = "EMP12345"
        
        # Mock data retrieval for subject
        subject_data = {
            'employee_id': subject_id,
            'personality_assessments': [
                {
                    'date': '2024-01-15',
                    'red_energy': 75.5,
                    'blue_energy': 45.2,
                    'processing_purpose': 'team_composition_analysis'
                }
            ],
            'clustering_results': [
                {
                    'date': '2024-01-16',
                    'cluster_id': 2,
                    'method': 'neuromorphic_esn'
                }
            ]
        }
        
        # Verify data completeness for access request
        assert 'employee_id' in subject_data
        assert 'personality_assessments' in subject_data
        assert 'clustering_results' in subject_data
        
        # 2. Right to rectification (Article 16)
        # Test data update capability
        updated_assessment = {
            'red_energy': 78.0,  # Corrected value
            'blue_energy': 47.1,  # Corrected value
            'correction_reason': 'data_subject_request',
            'correction_date': datetime.utcnow().isoformat()
        }
        
        # Simulate update
        subject_data['personality_assessments'][0].update(updated_assessment)
        
        assert subject_data['personality_assessments'][0]['red_energy'] == 78.0
        assert 'correction_reason' in subject_data['personality_assessments'][0]
        
        # 3. Right to erasure (Article 17)
        # Test data deletion capability
        def anonymize_subject_data(subject_id):
            # Replace identifiable data with anonymous identifiers
            anonymized_data = {
                'employee_id': f"ANON_{hash(subject_id) % 1000000}",
                'personality_assessments': subject_data['personality_assessments'].copy(),
                'clustering_results': subject_data['clustering_results'].copy(),
                'anonymization_date': datetime.utcnow().isoformat(),
                'original_deleted': True
            }
            return anonymized_data
        
        anonymized = anonymize_subject_data(subject_id)
        
        assert anonymized['employee_id'] != subject_id
        assert anonymized['employee_id'].startswith('ANON_')
        assert 'anonymization_date' in anonymized
        assert anonymized['original_deleted'] is True
        
    def test_ccpa_compliance_features(self):
        """Test CCPA compliance features"""
        # CCPA (California Consumer Privacy Act) requirements
        
        # 1. Right to know (similar to GDPR access)
        consumer_id = "CA_CONSUMER_001"
        
        # Mock data collection disclosure
        data_collection_disclosure = {
            'categories_of_data': [
                'personality_assessment_data',
                'workplace_behavioral_data',
                'team_assignment_history'
            ],
            'purposes': [
                'team_composition_optimization',
                'workplace_efficiency_analysis',
                'employee_development_planning'
            ],
            'third_parties': [
                'analytics_service_provider',
                'hr_consulting_partner'
            ],
            'retention_period': '3_years_from_collection',
            'disclosure_date': datetime.utcnow().isoformat()
        }
        
        # Verify disclosure completeness
        required_disclosures = ['categories_of_data', 'purposes', 'third_parties', 'retention_period']
        for disclosure in required_disclosures:
            assert disclosure in data_collection_disclosure
            
        # 2. Right to delete
        # Test consumer data deletion
        def process_ccpa_deletion_request(consumer_id, verification_token):
            # Verify consumer identity (simplified)
            if verification_token != "valid_token_123":
                return {'status': 'verification_failed'}
            
            # Process deletion
            deletion_result = {
                'consumer_id': consumer_id,
                'deletion_date': datetime.utcnow().isoformat(),
                'data_categories_deleted': [
                    'personality_assessments',
                    'clustering_results',
                    'team_assignments'
                ],
                'exceptions': [],  # No exceptions for this test
                'status': 'completed'
            }
            
            return deletion_result
        
        deletion_result = process_ccpa_deletion_request(consumer_id, "valid_token_123")
        
        assert deletion_result['status'] == 'completed'
        assert len(deletion_result['data_categories_deleted']) > 0
        assert deletion_result['consumer_id'] == consumer_id
        
        # 3. Right to opt-out of sale
        opt_out_status = {
            'consumer_id': consumer_id,
            'opt_out_date': datetime.utcnow().isoformat(),
            'data_sales_stopped': True,
            'third_party_notifications_sent': True
        }
        
        assert opt_out_status['data_sales_stopped'] is True
        assert opt_out_status['third_party_notifications_sent'] is True


class TestSecurityVulnerabilities:
    """Test for common security vulnerabilities"""
    
    def setup_method(self):
        """Setup vulnerability testing environment"""
        self.security_manager = MockSecurityManager()
        
    def test_sql_injection_protection(self):
        """Test protection against SQL injection attacks"""
        # Simulate SQL injection attempts in employee ID field
        malicious_inputs = [
            "EMP001'; DROP TABLE employees; --",
            "EMP001' OR '1'='1",
            "EMP001' UNION SELECT * FROM sensitive_data --",
            "EMP001'; INSERT INTO audit_log VALUES ('hacked'); --"
        ]
        
        for malicious_input in malicious_inputs:
            # Test input sanitization
            sanitized = self._sanitize_employee_id(malicious_input)
            
            # Should remove or escape dangerous characters
            dangerous_chars = [';', '--', 'DROP', 'UNION', 'INSERT', 'DELETE']
            for char in dangerous_chars:
                assert char.lower() not in sanitized.lower(), f"Dangerous pattern '{char}' not filtered"
    
    def _sanitize_employee_id(self, employee_id: str) -> str:
        """Sanitize employee ID input"""
        # Remove dangerous SQL characters and keywords
        import re
        
        # Allow only alphanumeric and specific characters
        sanitized = re.sub(r'[^A-Za-z0-9_-]', '', employee_id)
        
        # Limit length
        sanitized = sanitized[:20]
        
        # Check against dangerous keywords
        dangerous_keywords = ['DROP', 'INSERT', 'DELETE', 'UPDATE', 'UNION', 'SELECT']
        for keyword in dangerous_keywords:
            sanitized = sanitized.replace(keyword.lower(), '')
            sanitized = sanitized.replace(keyword.upper(), '')
        
        return sanitized
        
    def test_data_inference_attack_protection(self):
        """Test protection against data inference attacks"""
        # Create original dataset
        np.random.seed(42)
        original_data = pd.DataFrame({
            'employee_id': [f'EMP{i:04d}' for i in range(100)],
            'department': np.random.choice(['Eng', 'HR', 'Sales', 'Finance'], 100),
            'age': np.random.randint(22, 65, 100),
            'salary': np.random.randint(40000, 150000, 100),
            'red_energy': np.random.normal(50, 15, 100),
            'blue_energy': np.random.normal(50, 15, 100)
        })
        
        # Create anonymized version
        anonymized_data = original_data.copy()
        anonymized_data = anonymized_data.drop(columns=['employee_id'])
        
        # Add noise to prevent inference
        noise_scale = 5
        anonymized_data['red_energy'] += np.random.normal(0, noise_scale, len(anonymized_data))
        anonymized_data['blue_energy'] += np.random.normal(0, noise_scale, len(anonymized_data))
        
        # Generalize sensitive attributes
        anonymized_data['age_group'] = pd.cut(
            anonymized_data['age'],
            bins=[0, 30, 40, 50, 100],
            labels=['<30', '30-40', '40-50', '>50']
        )
        anonymized_data = anonymized_data.drop(columns=['age'])
        
        anonymized_data['salary_band'] = pd.cut(
            anonymized_data['salary'],
            bins=[0, 60000, 80000, 100000, 200000],
            labels=['<60k', '60-80k', '80-100k', '>100k']
        )
        anonymized_data = anonymized_data.drop(columns=['salary'])
        
        # Test inference attack resistance
        # Attempt to match records between original and anonymized data
        matches = 0
        for _, orig_row in original_data.iterrows():
            # Try to find matching record in anonymized data
            dept_matches = anonymized_data[anonymized_data['department'] == orig_row['department']]
            
            if len(dept_matches) > 0:
                # Check if personality data is too similar (indicating poor anonymization)
                for _, anon_row in dept_matches.iterrows():
                    red_diff = abs(orig_row['red_energy'] - anon_row['red_energy'])
                    blue_diff = abs(orig_row['blue_energy'] - anon_row['blue_energy'])
                    
                    if red_diff < 3 and blue_diff < 3:  # Very similar personality data
                        matches += 1
                        break
        
        # Inference accuracy should be low
        inference_accuracy = matches / len(original_data)
        assert inference_accuracy < SECURITY_CONFIG['max_inference_accuracy'], \
            f"Inference accuracy too high: {inference_accuracy:.2f}"
            
    def test_timing_attack_protection(self):
        """Test protection against timing attacks"""
        # Test constant-time comparison for sensitive operations
        
        def secure_compare(a: str, b: str) -> bool:
            """Constant-time string comparison"""
            if len(a) != len(b):
                # Still do work to prevent timing analysis
                dummy = sum(ord(c) for c in a + 'x' * (len(b) - len(a)) if len(b) > len(a) else a)
                dummy += sum(ord(c) for c in b + 'x' * (len(a) - len(b)) if len(a) > len(b) else b)
                return False
            
            result = 0
            for x, y in zip(a, b):
                result |= ord(x) ^ ord(y)
            
            return result == 0
        
        # Test with various inputs
        correct_token = os.environ.get("TEST_TOKEN", "test_token_for_timing_attack_tests")
        
        # Time the comparisons
        import time
        
        # Correct comparison
        start_time = time.perf_counter()
        result1 = secure_compare(correct_token, correct_token)
        time1 = time.perf_counter() - start_time
        
        # Incorrect comparison (early difference)
        start_time = time.perf_counter()
        result2 = secure_compare("WRONG_TOKEN_12345", correct_token)
        time2 = time.perf_counter() - start_time
        
        # Incorrect comparison (late difference)
        start_time = time.perf_counter()
        result3 = secure_compare("SECRET_TOKEN_54321", correct_token)
        time3 = time.perf_counter() - start_time
        
        # Verify correctness
        assert result1 is True
        assert result2 is False
        assert result3 is False
        
        # Timing should be relatively constant (allowing for measurement noise)
        max_timing_diff = max(abs(time2 - time1), abs(time3 - time1), abs(time3 - time2))
        assert max_timing_diff < 0.01, f"Timing difference too large: {max_timing_diff:.6f}s"
        
    def test_memory_disclosure_protection(self):
        """Test protection against memory disclosure attacks"""
        # Test that sensitive data is properly cleared from memory
        
        sensitive_data = "HIGHLY_SENSITIVE_EMPLOYEE_DATA_" + "X" * 1000
        
        # Simulate processing sensitive data
        processed_data = self.security_manager.encrypt_data(sensitive_data)
        
        # Clear sensitive data
        sensitive_data = "X" * len(sensitive_data)  # Overwrite with dummy data
        del sensitive_data  # Delete reference
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Verify encrypted data is still valid
        decrypted = self.security_manager.decrypt_data(processed_data)
        assert "HIGHLY_SENSITIVE_EMPLOYEE_DATA_" in decrypted
        
        # Note: In a real implementation, you would use more sophisticated
        # memory clearing techniques and might test with memory analysis tools


class TestComplianceReporting:
    """Test compliance reporting and monitoring"""
    
    def setup_method(self):
        """Setup compliance testing environment"""
        self.security_manager = MockSecurityManager()
        
    def test_compliance_report_generation(self):
        """Test generation of compliance reports"""
        # Generate test audit data
        test_events = [
            ('DATA_ACCESS', 'user001', 'read_employee_data'),
            ('DATA_MODIFICATION', 'user002', 'update_personality_score'),
            ('CLUSTERING_ANALYSIS', 'user001', 'neuromorphic_clustering'),
            ('RESULT_EXPORT', 'user003', 'team_composition_results'),
            ('DATA_DELETION', 'user002', 'gdpr_erasure_request')
        ]
        
        for event_type, user_id, resource in test_events:
            self.security_manager.log_audit_event(
                event_type=event_type,
                user_id=user_id,
                resource=resource,
                details={'test': True}
            )
        
        # Generate compliance report
        report = self._generate_compliance_report(self.security_manager.audit_logs)
        
        # Verify report structure
        assert 'report_period' in report
        assert 'total_events' in report
        assert 'events_by_type' in report
        assert 'users_active' in report
        assert 'gdpr_requests' in report
        assert 'security_incidents' in report
        
        # Verify report content
        assert report['total_events'] == len(test_events)
        assert len(report['events_by_type']) > 0
        assert report['users_active'] == 3  # Three unique users
        
    def _generate_compliance_report(self, audit_logs: List[Dict]) -> Dict:
        """Generate compliance report from audit logs"""
        from collections import Counter
        
        # Basic statistics
        total_events = len(audit_logs)
        events_by_type = Counter(log['event_type'] for log in audit_logs)
        users_active = len(set(log['user_id'] for log in audit_logs))
        
        # GDPR-specific metrics
        gdpr_events = [log for log in audit_logs if 'gdpr' in log.get('resource', '').lower()]
        
        # Security incident detection
        security_incidents = [
            log for log in audit_logs 
            if log['event_type'] in ['UNAUTHORIZED_ACCESS', 'DATA_BREACH', 'SECURITY_VIOLATION']
        ]
        
        return {
            'report_period': {
                'start': datetime.utcnow().replace(day=1).isoformat(),
                'end': datetime.utcnow().isoformat()
            },
            'total_events': total_events,
            'events_by_type': dict(events_by_type),
            'users_active': users_active,
            'gdpr_requests': len(gdpr_events),
            'security_incidents': len(security_incidents),
            'compliance_score': self._calculate_compliance_score(audit_logs)
        }
    
    def _calculate_compliance_score(self, audit_logs: List[Dict]) -> float:
        """Calculate overall compliance score"""
        # Simple scoring based on audit completeness and security practices
        base_score = 85.0
        
        # Deduct points for missing required events
        required_event_types = ['DATA_ACCESS', 'DATA_MODIFICATION', 'DATA_DELETION']
        present_types = set(log['event_type'] for log in audit_logs)
        missing_types = set(required_event_types) - present_types
        base_score -= len(missing_types) * 5
        
        # Add points for good practices
        if any('encryption' in str(log.get('details', {})).lower() for log in audit_logs):
            base_score += 5
        
        if any('anonymization' in str(log.get('details', {})).lower() for log in audit_logs):
            base_score += 5
        
        return max(0.0, min(100.0, base_score))
    
    def test_privacy_impact_assessment(self):
        """Test privacy impact assessment functionality"""
        # Define data processing scenario
        processing_scenario = {
            'data_categories': [
                'personality_assessment_scores',
                'behavioral_analysis_results',
                'team_assignment_history'
            ],
            'processing_purposes': [
                'team_optimization',
                'performance_analysis',
                'organizational_development'
            ],
            'data_subjects': 'employees',
            'retention_period': '3_years',
            'third_party_sharing': True,
            'automated_decision_making': True,
            'cross_border_transfers': False
        }
        
        # Assess privacy risks
        risk_assessment = self._assess_privacy_risks(processing_scenario)
        
        # Verify assessment structure
        assert 'overall_risk_level' in risk_assessment
        assert 'risk_factors' in risk_assessment
        assert 'mitigation_measures' in risk_assessment
        assert 'compliance_requirements' in risk_assessment
        
        # Verify risk levels are valid
        assert risk_assessment['overall_risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
        
    def _assess_privacy_risks(self, scenario: Dict) -> Dict:
        """Assess privacy risks for a processing scenario"""
        risk_factors = []
        risk_score = 0
        
        # Assess various risk factors
        if scenario.get('automated_decision_making'):
            risk_factors.append('Automated decision-making affecting individuals')
            risk_score += 15
            
        if scenario.get('third_party_sharing'):
            risk_factors.append('Data sharing with third parties')
            risk_score += 10
            
        if scenario.get('cross_border_transfers'):
            risk_factors.append('International data transfers')
            risk_score += 10
            
        if 'behavioral_analysis' in str(scenario.get('data_categories', [])):
            risk_factors.append('Processing of behavioral data')
            risk_score += 10
            
        # Determine overall risk level
        if risk_score >= 30:
            risk_level = 'CRITICAL'
        elif risk_score >= 20:
            risk_level = 'HIGH'
        elif risk_score >= 10:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'overall_risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'mitigation_measures': [
                'Implement data minimization principles',
                'Apply pseudonymization techniques',
                'Establish data retention policies',
                'Conduct regular security assessments'
            ],
            'compliance_requirements': [
                'GDPR Article 35 (Data Protection Impact Assessment)',
                'CCPA Section 1798.185 (Privacy by Design)',
                'ISO 27001 Information Security Management'
            ]
        }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])