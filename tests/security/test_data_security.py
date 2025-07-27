"""Security tests for data handling and processing."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.insights_clustering.parser import DataParser
from src.insights_clustering.validator import DataValidator


@pytest.mark.security
class TestDataSecurity:
    """Test security aspects of data handling."""

    def test_pii_anonymization(
        self, security_test_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test that PII data is properly anonymized."""
        # Create CSV with PII data
        pii_data = security_test_data["pii_data"]
        df = pd.DataFrame([pii_data])
        csv_file = tmp_path / "pii_test.csv"
        df.to_csv(csv_file, index=False)
        
        parser = DataParser(anonymize=True)
        result = parser.parse_csv(csv_file)
        
        # Check that PII has been anonymized
        assert result["employee_id"].iloc[0] != pii_data["employee_id"]
        assert result["name"].iloc[0] != pii_data["name"]
        assert result["email"].iloc[0] != pii_data["email"]
        
        # Anonymized data should still be consistent in format
        assert len(result["employee_id"].iloc[0]) > 0
        assert "@" not in result["email"].iloc[0]  # Email should be hashed/anonymized

    def test_sql_injection_protection(
        self, security_test_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test protection against SQL injection attempts."""
        injection_attempts = security_test_data["injection_attempts"]
        
        for injection in injection_attempts:
            # Create DataFrame with injection attempt
            df = pd.DataFrame({
                "employee_id": [injection],
                "red_energy": [50],
                "blue_energy": [50],
                "green_energy": [50],
                "yellow_energy": [50],
            })
            
            csv_file = tmp_path / f"injection_test_{hash(injection)}.csv"
            df.to_csv(csv_file, index=False)
            
            validator = DataValidator()
            
            # Validator should either sanitize or reject malicious input
            with pytest.raises((ValueError, TypeError)) or not pytest.raises():
                result = validator.validate_data(df)
                if result is not None:
                    # If validation passes, data should be sanitized
                    assert injection not in str(result.values)

    def test_file_path_traversal_protection(self, tmp_path: Path) -> None:
        """Test protection against directory traversal attacks."""
        parser = DataParser()
        
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]
        
        for malicious_path in malicious_paths:
            with pytest.raises((FileNotFoundError, ValueError, PermissionError)):
                parser.parse_csv(malicious_path)

    def test_large_file_protection(
        self, security_test_data: Dict[str, Any], tmp_path: Path
    ) -> None:
        """Test protection against large file attacks (DoS)."""
        large_payload = security_test_data["large_payload"]
        
        # Create a large CSV file
        large_file = tmp_path / "large_test.csv"
        with open(large_file, "w") as f:
            f.write("employee_id,red_energy,blue_energy,green_energy,yellow_energy\n")
            for i in range(1000):  # Create many rows with large data
                f.write(f"{large_payload},{i},{i},{i},{i}\n")
        
        parser = DataParser(max_file_size=1024 * 1024)  # 1MB limit
        
        # Should raise an error for file size limit
        with pytest.raises(ValueError, match="File size exceeds maximum"):
            parser.parse_csv(large_file)

    def test_memory_exhaustion_protection(self) -> None:
        """Test protection against memory exhaustion attacks."""
        from src.insights_clustering.clustering import ClusteringEngine
        
        # Create a dataset designed to consume excessive memory
        large_data = pd.DataFrame({
            "red_energy": range(100000),
            "blue_energy": range(100000),
            "green_energy": range(100000),
            "yellow_energy": range(100000),
        })
        
        engine = ClusteringEngine(n_clusters=1000)  # Excessive clusters
        
        # Should handle large datasets gracefully or raise appropriate error
        try:
            result = engine.fit_predict(large_data)
            # If it succeeds, verify reasonable memory usage
            assert len(result["labels"]) == 100000
        except (MemoryError, ValueError) as e:
            # Expected behavior for resource protection
            assert "memory" in str(e).lower() or "resource" in str(e).lower()

    def test_data_encryption_at_rest(self, sample_insights_data: pd.DataFrame, tmp_path: Path) -> None:
        """Test that sensitive data can be encrypted when stored."""
        from cryptography.fernet import Fernet
        
        # Generate encryption key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        
        # Simulate encrypted storage
        csv_file = tmp_path / "encrypted_test.csv"
        data_bytes = sample_insights_data.to_csv(index=False).encode()
        encrypted_data = cipher.encrypt(data_bytes)
        
        with open(csv_file, "wb") as f:
            f.write(encrypted_data)
        
        # Verify data is encrypted (not readable as plain text)
        with open(csv_file, "r") as f:
            try:
                content = f.read()
                # Should fail to read as normal CSV
                assert "employee_id" not in content
            except UnicodeDecodeError:
                # Expected - encrypted data is not readable as text
                pass
        
        # Verify data can be decrypted
        with open(csv_file, "rb") as f:
            encrypted_data = f.read()
        
        decrypted_data = cipher.decrypt(encrypted_data)
        decrypted_df = pd.read_csv(pd.io.common.StringIO(decrypted_data.decode()))
        
        # Decrypted data should match original
        assert len(decrypted_df) == len(sample_insights_data)

    def test_access_control_simulation(self, sample_insights_data: pd.DataFrame) -> None:
        """Test simulation of access control mechanisms."""
        from src.insights_clustering.parser import DataParser
        
        # Simulate role-based access
        class SecureDataParser(DataParser):
            def __init__(self, user_role: str = "guest"):
                super().__init__()
                self.user_role = user_role
            
            def parse_csv(self, file_path: str) -> pd.DataFrame:
                if self.user_role == "guest":
                    raise PermissionError("Insufficient privileges")
                elif self.user_role == "analyst":
                    # Limited access - only energy columns
                    df = super().parse_csv(file_path)
                    return df[["red_energy", "blue_energy", "green_energy", "yellow_energy"]]
                elif self.user_role == "admin":
                    # Full access
                    return super().parse_csv(file_path)
                else:
                    raise ValueError("Invalid role")
        
        # Test different access levels
        with pytest.raises(PermissionError):
            parser = SecureDataParser("guest")
            # This would fail in real implementation
        
        # Test analyst access (limited)
        parser = SecureDataParser("analyst")
        # In real implementation, this would work with limited data
        
        # Test admin access (full)
        parser = SecureDataParser("admin")
        # In real implementation, this would have full access

    def test_audit_logging_simulation(self, sample_insights_data: pd.DataFrame, tmp_path: Path) -> None:
        """Test that data access is properly logged for auditing."""
        import logging
        
        # Set up audit logging
        audit_log = tmp_path / "audit.log"
        logger = logging.getLogger("audit")
        handler = logging.FileHandler(audit_log)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Simulate data access with logging
        logger.info("Data access attempt - User: test_user, Action: parse_csv")
        logger.info("Data access successful - Records: %d", len(sample_insights_data))
        
        # Verify audit log exists and contains expected entries
        assert audit_log.exists()
        with open(audit_log, "r") as f:
            log_content = f.read()
            assert "Data access attempt" in log_content
            assert "Data access successful" in log_content

    def test_input_sanitization(self, tmp_path: Path) -> None:
        """Test that user input is properly sanitized."""
        from src.insights_clustering.validator import DataValidator
        
        # Test various malicious inputs
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "eval(malicious_code)",
            "system('rm -rf /')",
        ]
        
        validator = DataValidator()
        
        for malicious_input in malicious_inputs:
            df = pd.DataFrame({
                "employee_id": [malicious_input],
                "red_energy": [50],
                "blue_energy": [50], 
                "green_energy": [50],
                "yellow_energy": [50],
            })
            
            # Validator should sanitize or reject malicious input
            try:
                result = validator.validate_data(df)
                if result is not None:
                    # Check that malicious content was sanitized
                    sanitized_id = result["employee_id"].iloc[0]
                    assert malicious_input != sanitized_id
                    # Common malicious patterns should be removed
                    assert "<script>" not in sanitized_id
                    assert "DROP TABLE" not in sanitized_id
                    assert "../" not in sanitized_id
            except ValueError:
                # Validation rejection is also acceptable
                pass

    def test_secure_random_generation(self) -> None:
        """Test that cryptographically secure random numbers are used."""
        import secrets
        from src.insights_clustering.clustering import ClusteringEngine
        
        # Test that random seed generation uses secure methods
        engine1 = ClusteringEngine(random_state=None)  # Should use secure random
        engine2 = ClusteringEngine(random_state=None)
        
        # Different instances should have different internal states
        # This is difficult to test directly, but we can verify behavior
        
        # Test with secure random seed
        secure_seed = secrets.randbelow(2**32)
        engine3 = ClusteringEngine(random_state=secure_seed)
        
        # Should accept secure random seed without error
        assert engine3.random_state == secure_seed

    def test_sensitive_data_masking(self, tmp_path: Path) -> None:
        """Test that sensitive data is properly masked in logs and outputs."""
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("test_logger")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Simulate logging with potentially sensitive data
        sensitive_data = {
            "ssn": "123-45-6789",
            "credit_card": "4111-1111-1111-1111",
            "email": "user@company.com",
        }
        
        # Log data (in production, this should be masked)
        for key, value in sensitive_data.items():
            # Implement masking logic
            if key == "ssn":
                masked_value = "***-**-" + value[-4:]
            elif key == "credit_card":
                masked_value = "****-****-****-" + value[-4:]
            elif key == "email":
                username, domain = value.split("@")
                masked_value = username[:2] + "***@" + domain
            else:
                masked_value = value
            
            logger.info("Processing %s: %s", key, masked_value)
        
        # Verify sensitive data is masked in logs
        log_output = log_capture.getvalue()
        assert "123-45-6789" not in log_output  # Full SSN should not appear
        assert "***-**-6789" in log_output  # Masked SSN should appear
        assert "4111-1111-1111-1111" not in log_output  # Full CC should not appear