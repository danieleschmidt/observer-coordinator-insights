"""
Security utilities for the Observer Coordinator Insights application
Handles data anonymization, input validation, and secure operations
"""

import hashlib
import secrets
import re
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

logger = logging.getLogger(__name__)


class DataAnonymizer:
    """Handles anonymization of sensitive employee data"""
    
    def __init__(self, salt: Optional[bytes] = None):
        self.salt = salt or os.urandom(32)
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption for sensitive data"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(b"insights_discovery_key"))
        self.cipher = Fernet(key)
    
    def anonymize_employee_id(self, employee_id: str) -> str:
        """Create anonymous but consistent employee identifier"""
        if not employee_id or not isinstance(employee_id, str):
            raise ValueError("Employee ID must be a non-empty string")
        
        # Create SHA-256 hash with salt for consistency
        hash_input = f"{employee_id}{self.salt.hex()}".encode()
        hash_result = hashlib.sha256(hash_input).hexdigest()
        return f"EMP_{hash_result[:12].upper()}"
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          pii_columns: List[str] = None) -> pd.DataFrame:
        """Anonymize PII columns in employee dataframe"""
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")
        
        df_anonymized = df.copy()
        pii_columns = pii_columns or ['employee_id', 'name', 'email']
        
        for col in pii_columns:
            if col in df_anonymized.columns:
                if col == 'employee_id':
                    df_anonymized[col] = df_anonymized[col].apply(self.anonymize_employee_id)
                else:
                    # Remove PII columns entirely
                    df_anonymized = df_anonymized.drop(columns=[col])
                    logger.info(f"Removed PII column: {col}")
        
        return df_anonymized
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data for storage"""
        if not data:
            return data
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        return self.cipher.decrypt(encrypted_data.encode()).decode()


class InputValidator:
    """Validates and sanitizes user inputs"""
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        """Validate file path for security"""
        if not file_path or not isinstance(file_path, str):
            return False
        
        # Check for path traversal attempts
        if '..' in file_path or file_path.startswith('/'):
            logger.warning(f"Potentially unsafe file path: {file_path}")
            return False
        
        # Check file extension
        allowed_extensions = ['.csv', '.xlsx', '.json']
        if not any(file_path.lower().endswith(ext) for ext in allowed_extensions):
            logger.warning(f"File extension not allowed: {file_path}")
            return False
        
        return True
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent injection attacks"""
        if not filename:
            return "unnamed_file"
        
        # Remove special characters and spaces
        sanitized = re.sub(r'[^\w\-_\.]', '_', filename)
        # Limit length
        sanitized = sanitized[:100]
        # Ensure it doesn't start with a dot
        if sanitized.startswith('.'):
            sanitized = 'file_' + sanitized[1:]
        
        return sanitized
    
    @staticmethod
    def validate_cluster_count(n_clusters: int, data_size: int) -> bool:
        """Validate cluster count against data size"""
        if not isinstance(n_clusters, int) or n_clusters < 2:
            logger.error("Number of clusters must be an integer >= 2")
            return False
        
        if n_clusters > data_size:
            logger.error(f"Number of clusters ({n_clusters}) cannot exceed data size ({data_size})")
            return False
        
        if n_clusters > data_size / 2:
            logger.warning(f"High cluster count ({n_clusters}) relative to data size ({data_size})")
        
        return True
    
    @staticmethod
    def validate_energy_values(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate energy values in employee data"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        for col in energy_cols:
            if col not in df.columns:
                validation_result['errors'].append(f"Missing required column: {col}")
                validation_result['is_valid'] = False
                continue
            
            # Check data types
            if not pd.api.types.is_numeric_dtype(df[col]):
                validation_result['errors'].append(f"Column {col} must be numeric")
                validation_result['is_valid'] = False
                continue
            
            # Check value ranges
            if (df[col] < 0).any() or (df[col] > 100).any():
                validation_result['warnings'].append(f"Energy values in {col} should be 0-100")
            
            # Check for missing values
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                validation_result['warnings'].append(f"{col} has {null_count} missing values")
        
        # Check if energy values sum to ~100 for each employee
        if all(col in df.columns for col in energy_cols):
            energy_sums = df[energy_cols].sum(axis=1)
            if not energy_sums.between(95, 105).all():
                validation_result['warnings'].append("Energy values should sum to approximately 100% per employee")
        
        return validation_result


class SecurityAuditor:
    """Audits security-related events and data access"""
    
    def __init__(self):
        self.audit_log = []
    
    def log_data_access(self, user_id: str, data_type: str, 
                       record_count: int, operation: str):
        """Log data access event"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'record_count': record_count,
            'operation': operation,
            'session_id': secrets.token_hex(8)
        }
        
        self.audit_log.append(audit_entry)
        logger.info(f"Data access logged: {operation} on {data_type} by {user_id}")
    
    def log_clustering_operation(self, parameters: Dict[str, Any], 
                               result_summary: Dict[str, Any]):
        """Log clustering operation for audit trail"""
        audit_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': 'clustering',
            'parameters': parameters,
            'result_summary': result_summary,
            'session_id': secrets.token_hex(8)
        }
        
        self.audit_log.append(audit_entry)
        logger.info("Clustering operation logged for audit")
    
    def get_audit_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get audit summary for specified time period"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_logs = [
            log for log in self.audit_log 
            if datetime.fromisoformat(log['timestamp']) > cutoff_date
        ]
        
        return {
            'total_operations': len(recent_logs),
            'unique_users': len(set(log.get('user_id') for log in recent_logs if log.get('user_id'))),
            'data_types_accessed': list(set(log.get('data_type') for log in recent_logs if log.get('data_type'))),
            'period_days': days,
            'summary_generated': datetime.utcnow().isoformat()
        }
    
    def check_data_retention_compliance(self, data_timestamp: datetime, 
                                      retention_days: int = 180) -> bool:
        """Check if data exceeds retention policy"""
        retention_cutoff = datetime.utcnow() - timedelta(days=retention_days)
        
        if data_timestamp < retention_cutoff:
            logger.warning(f"Data from {data_timestamp} exceeds retention policy ({retention_days} days)")
            return False
        
        return True


class SecureDataProcessor:
    """Processes data with security best practices"""
    
    def __init__(self):
        self.anonymizer = DataAnonymizer()
        self.validator = InputValidator()
        self.auditor = SecurityAuditor()
    
    def secure_load_data(self, file_path: str, user_id: str = "system") -> pd.DataFrame:
        """Securely load and process employee data"""
        # Validate file path
        if not self.validator.validate_file_path(file_path):
            raise ValueError(f"Invalid or unsafe file path: {file_path}")
        
        try:
            # Load data
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Validate data
            validation_result = self.validator.validate_energy_values(df)
            if not validation_result['is_valid']:
                raise ValueError(f"Data validation failed: {validation_result['errors']}")
            
            # Anonymize data
            df_anonymized = self.anonymizer.anonymize_dataframe(df)
            
            # Log access
            self.auditor.log_data_access(
                user_id=user_id,
                data_type='employee_insights',
                record_count=len(df_anonymized),
                operation='load'
            )
            
            logger.info(f"Securely loaded {len(df_anonymized)} employee records")
            return df_anonymized
            
        except Exception as e:
            logger.error(f"Secure data loading failed: {str(e)}")
            raise
    
    def secure_clustering_pipeline(self, df: pd.DataFrame, n_clusters: int, 
                                 user_id: str = "system") -> Dict[str, Any]:
        """Execute clustering with security controls"""
        # Validate inputs
        if not self.validator.validate_cluster_count(n_clusters, len(df)):
            raise ValueError("Invalid cluster configuration")
        
        try:
            # Import here to avoid circular dependency
            from .insights_clustering import KMeansClusterer
            
            # Extract features (only numeric columns for clustering)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            features = df[numeric_cols]
            
            # Perform clustering
            clusterer = KMeansClusterer(n_clusters=n_clusters)
            clusterer.fit(features)
            
            # Get results
            results = {
                'cluster_assignments': clusterer.get_cluster_assignments(),
                'centroids': clusterer.get_cluster_centroids(),
                'quality_metrics': clusterer.get_cluster_quality_metrics(),
                'data_summary': {
                    'total_employees': len(df),
                    'features_used': list(features.columns),
                    'clusters_created': n_clusters
                }
            }
            
            # Log operation
            self.auditor.log_clustering_operation(
                parameters={'n_clusters': n_clusters, 'data_size': len(df)},
                result_summary=results['data_summary']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Secure clustering failed: {str(e)}")
            raise