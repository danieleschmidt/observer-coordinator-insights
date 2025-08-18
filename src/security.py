"""Enhanced Security utilities for the Observer Coordinator Insights application
Generation 2: Handles data anonymization, input validation, secure operations, 
differential privacy, audit logging, and advanced encryption
"""

import base64
import hashlib
import logging
import os
import re
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events for audit logging"""
    DATA_ACCESS = "data_access"
    DATA_ANONYMIZATION = "data_anonymization"
    ENCRYPTION = "encryption"
    DECRYPTION = "decryption"
    VALIDATION_FAILURE = "validation_failure"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    KEY_GENERATION = "key_generation"
    PRIVACY_PROTECTION = "privacy_protection"
    ANOMALY_DETECTION = "anomaly_detection"


class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class SecurityEvent:
    """Security event for comprehensive audit logging"""
    event_id: str
    event_type: SecurityEventType
    timestamp: datetime
    user_id: str
    resource: str
    action: str
    success: bool
    details: Dict[str, Any]
    correlation_id: str
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    risk_score: float = 0.0

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class DifferentialPrivacyConfig:
    """Configuration for differential privacy protection"""
    epsilon: float = 1.0  # Privacy budget
    delta: float = 1e-5   # Failure probability
    sensitivity: float = 1.0  # Global sensitivity
    mechanism: str = "laplace"  # laplace, gaussian, exponential
    enabled: bool = True


class EnhancedDataAnonymizer:
    """Generation 2: Enhanced anonymization with differential privacy and advanced encryption"""

    def __init__(self, salt: Optional[bytes] = None,
                 privacy_config: Optional[DifferentialPrivacyConfig] = None):
        self.salt = salt or os.urandom(32)
        self.privacy_config = privacy_config or DifferentialPrivacyConfig()
        self._setup_encryption()
        self._anonymization_cache = {}
        self._privacy_budget_tracker = defaultdict(float)
        self._lock = threading.Lock()

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

    def apply_differential_privacy(self, data: Union[np.ndarray, float],
                                 query_type: str = "mean",
                                 correlation_id: str = None) -> Union[np.ndarray, float]:
        """Apply differential privacy to statistical queries"""
        if not self.privacy_config.enabled:
            return data

        correlation_id = correlation_id or str(uuid.uuid4())

        with self._lock:
            # Check privacy budget
            current_budget = self._privacy_budget_tracker[correlation_id]
            if current_budget >= self.privacy_config.epsilon:
                raise ValueError(f"Privacy budget exhausted for {correlation_id}")

            # Add noise based on mechanism
            if isinstance(data, (int, float)):
                noisy_data = self._add_noise_scalar(data)
            else:
                noisy_data = self._add_noise_vector(data)

            # Update privacy budget
            self._privacy_budget_tracker[correlation_id] += 0.1  # Small epsilon per query

            logger.info(f"Applied differential privacy to {query_type} query [correlation_id: {correlation_id}]")
            return noisy_data

    def _add_noise_scalar(self, value: float) -> float:
        """Add noise to scalar value"""
        if self.privacy_config.mechanism == "laplace":
            noise = np.random.laplace(0, self.privacy_config.sensitivity / self.privacy_config.epsilon)
        elif self.privacy_config.mechanism == "gaussian":
            sigma = np.sqrt(2 * np.log(1.25 / self.privacy_config.delta)) * self.privacy_config.sensitivity / self.privacy_config.epsilon
            noise = np.random.normal(0, sigma)
        else:
            # Default to Laplace
            noise = np.random.laplace(0, self.privacy_config.sensitivity / self.privacy_config.epsilon)

        return value + noise

    def _add_noise_vector(self, values: np.ndarray) -> np.ndarray:
        """Add noise to vector of values"""
        if self.privacy_config.mechanism == "laplace":
            noise = np.random.laplace(0, self.privacy_config.sensitivity / self.privacy_config.epsilon, size=values.shape)
        elif self.privacy_config.mechanism == "gaussian":
            sigma = np.sqrt(2 * np.log(1.25 / self.privacy_config.delta)) * self.privacy_config.sensitivity / self.privacy_config.epsilon
            noise = np.random.normal(0, sigma, size=values.shape)
        else:
            # Default to Laplace
            noise = np.random.laplace(0, self.privacy_config.sensitivity / self.privacy_config.epsilon, size=values.shape)

        return values + noise

    def anonymize_clustering_results(self, cluster_labels: np.ndarray,
                                   feature_data: np.ndarray,
                                   correlation_id: str = None) -> Dict[str, Any]:
        """Anonymize clustering results with differential privacy"""
        correlation_id = correlation_id or str(uuid.uuid4())

        # Apply k-anonymity by ensuring minimum cluster sizes
        min_cluster_size = 5
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        # Merge small clusters
        small_clusters = unique_labels[counts < min_cluster_size]
        if len(small_clusters) > 0:
            # Reassign small cluster points to nearest larger cluster
            for small_label in small_clusters:
                small_indices = np.where(cluster_labels == small_label)[0]
                # Find nearest larger cluster centroid (simplified)
                large_clusters = unique_labels[counts >= min_cluster_size]
                if len(large_clusters) > 0:
                    # Assign to first large cluster (could be improved with distance calculation)
                    cluster_labels[small_indices] = large_clusters[0]

        # Apply differential privacy to cluster statistics
        anonymized_stats = {}
        for label in np.unique(cluster_labels):
            cluster_mask = cluster_labels == label
            cluster_size = np.sum(cluster_mask)

            if cluster_size >= min_cluster_size:
                # Add noise to cluster statistics
                cluster_features = feature_data[cluster_mask]
                mean_features = np.mean(cluster_features, axis=0)

                anonymized_stats[int(label)] = {
                    'size': self.apply_differential_privacy(float(cluster_size), "count", correlation_id),
                    'mean_features': self.apply_differential_privacy(mean_features, "mean", correlation_id).tolist()
                }

        return {
            'anonymized_labels': cluster_labels.tolist(),
            'cluster_statistics': anonymized_stats,
            'privacy_parameters': asdict(self.privacy_config),
            'k_anonymity_threshold': min_cluster_size
        }

    def get_privacy_budget_status(self) -> Dict[str, float]:
        """Get current privacy budget usage"""
        with self._lock:
            return dict(self._privacy_budget_tracker)

    def reset_privacy_budget(self, correlation_id: str):
        """Reset privacy budget for a correlation ID"""
        with self._lock:
            if correlation_id in self._privacy_budget_tracker:
                del self._privacy_budget_tracker[correlation_id]
                logger.info(f"Privacy budget reset for {correlation_id}")


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


class EnhancedSecurityAuditor:
    """Generation 2: Advanced security auditing with comprehensive event tracking"""

    def __init__(self, max_log_size: int = 100000):
        self.audit_log: deque = deque(maxlen=max_log_size)
        self.security_events: deque = deque(maxlen=max_log_size)
        self.anomaly_detection_enabled = True
        self.baseline_patterns = defaultdict(list)
        self._lock = threading.Lock()

        # Risk scoring parameters
        self.risk_weights = {
            SecurityEventType.DATA_ACCESS: 0.3,
            SecurityEventType.ENCRYPTION: 0.2,
            SecurityEventType.VALIDATION_FAILURE: 0.8,
            SecurityEventType.AUTHENTICATION: 0.5,
            SecurityEventType.AUTHORIZATION: 0.7
        }

    def log_security_event(self, event_type: SecurityEventType, user_id: str,
                          resource: str, action: str, success: bool,
                          details: Dict[str, Any], correlation_id: str = None,
                          security_level: SecurityLevel = SecurityLevel.INTERNAL) -> str:
        """Log comprehensive security event with risk assessment"""
        correlation_id = correlation_id or str(uuid.uuid4())

        # Calculate risk score
        risk_score = self._calculate_risk_score(event_type, details, success)

        event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            resource=resource,
            action=action,
            success=success,
            details=details,
            correlation_id=correlation_id,
            security_level=security_level,
            risk_score=risk_score
        )

        with self._lock:
            self.security_events.append(event)

            # Legacy audit log entry for backward compatibility
            self.audit_log.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'user_id': user_id,
                'resource': resource,
                'action': action,
                'success': success,
                'risk_score': risk_score,
                'correlation_id': correlation_id
            })

            # Check for anomalies
            if self.anomaly_detection_enabled:
                self._detect_anomalies(event)

        # Log based on risk level
        if risk_score >= 0.8:
            logger.critical(f"HIGH RISK security event: {event.event_type.value} by {user_id} [correlation_id: {correlation_id}]")
        elif risk_score >= 0.6:
            logger.warning(f"MEDIUM RISK security event: {event.event_type.value} by {user_id} [correlation_id: {correlation_id}]")
        else:
            logger.info(f"Security event: {event.event_type.value} by {user_id} [correlation_id: {correlation_id}]")

        return event.event_id

    def _calculate_risk_score(self, event_type: SecurityEventType,
                            details: Dict[str, Any], success: bool) -> float:
        """Calculate risk score for security event"""
        base_risk = self.risk_weights.get(event_type, 0.5)

        # Adjust based on success/failure
        if not success:
            base_risk *= 1.5  # Failed operations are riskier

        # Adjust based on context
        if event_type == SecurityEventType.DATA_ACCESS:
            record_count = details.get('record_count', 0)
            if record_count > 10000:  # Large data access
                base_risk *= 1.3
        elif event_type == SecurityEventType.VALIDATION_FAILURE:
            failure_type = details.get('failure_type', '')
            if 'injection' in failure_type.lower():
                base_risk *= 1.8

        # Cap at 1.0
        return min(1.0, base_risk)

    def _detect_anomalies(self, event: SecurityEvent):
        """Detect anomalous security events"""
        try:
            # Check for unusual patterns
            pattern_key = f"{event.user_id}_{event.event_type.value}"

            # Track baseline patterns
            self.baseline_patterns[pattern_key].append({
                'timestamp': event.timestamp,
                'risk_score': event.risk_score,
                'success': event.success
            })

            # Keep only recent events for baseline (last 30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            self.baseline_patterns[pattern_key] = [
                p for p in self.baseline_patterns[pattern_key]
                if p['timestamp'] > cutoff
            ]

            # Anomaly detection rules
            recent_events = [p for p in self.baseline_patterns[pattern_key]
                           if p['timestamp'] > datetime.utcnow() - timedelta(hours=1)]

            # Rule 1: Too many high-risk events in short time
            high_risk_events = [e for e in recent_events if e['risk_score'] > 0.7]
            if len(high_risk_events) > 5:
                self._trigger_anomaly_alert(event, "High frequency of high-risk events")

            # Rule 2: Unusual failure rate
            if len(recent_events) > 10:
                failure_rate = len([e for e in recent_events if not e['success']]) / len(recent_events)
                if failure_rate > 0.5:
                    self._trigger_anomaly_alert(event, f"High failure rate: {failure_rate:.2%}")

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")

    def _trigger_anomaly_alert(self, event: SecurityEvent, reason: str):
        """Trigger security anomaly alert"""
        alert_event = SecurityEvent(
            event_id=str(uuid.uuid4()),
            event_type=SecurityEventType.ANOMALY_DETECTION,
            timestamp=datetime.utcnow(),
            user_id="system",
            resource="anomaly_detector",
            action="alert",
            success=True,
            details={
                'original_event_id': event.event_id,
                'user_id': event.user_id,
                'reason': reason,
                'risk_score': event.risk_score
            },
            correlation_id=event.correlation_id,
            security_level=SecurityLevel.RESTRICTED,
            risk_score=1.0
        )

        with self._lock:
            self.security_events.append(alert_event)

        logger.critical(f"🚨 SECURITY ANOMALY DETECTED: {reason} for user {event.user_id} [correlation_id: {event.correlation_id}]")

    def log_data_access(self, user_id: str, data_type: str,
                       record_count: int, operation: str, correlation_id: str = None):
        """Enhanced data access logging with security event tracking"""
        self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            resource=data_type,
            action=operation,
            success=True,
            details={
                'record_count': record_count,
                'data_type': data_type,
                'operation': operation
            },
            correlation_id=correlation_id
        )

    def log_clustering_operation(self, parameters: Dict[str, Any],
                               result_summary: Dict[str, Any],
                               user_id: str = "system",
                               correlation_id: str = None):
        """Enhanced clustering operation logging"""
        self.log_security_event(
            event_type=SecurityEventType.DATA_ACCESS,
            user_id=user_id,
            resource="clustering_engine",
            action="cluster_analysis",
            success=result_summary.get('status') == 'success',
            details={
                'parameters': parameters,
                'result_summary': result_summary,
                'clusters_created': result_summary.get('clusters_created', 0)
            },
            correlation_id=correlation_id,
            security_level=SecurityLevel.CONFIDENTIAL
        )

    def log_privacy_operation(self, operation_type: str, privacy_parameters: Dict[str, Any],
                            user_id: str, correlation_id: str = None):
        """Log differential privacy operations"""
        self.log_security_event(
            event_type=SecurityEventType.PRIVACY_PROTECTION,
            user_id=user_id,
            resource="differential_privacy",
            action=operation_type,
            success=True,
            details=privacy_parameters,
            correlation_id=correlation_id,
            security_level=SecurityLevel.RESTRICTED
        )

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
            logger.error(f"Secure data loading failed: {e!s}")
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
            logger.error(f"Secure clustering failed: {e!s}")
            raise


# Global enhanced instances for backward compatibility
enhanced_data_anonymizer = EnhancedDataAnonymizer()
enhanced_security_auditor = EnhancedSecurityAuditor()

# Backward compatibility aliases
data_anonymizer = enhanced_data_anonymizer
security_auditor = enhanced_security_auditor


class SecureDataProcessor:
    """Enhanced secure data processor with Generation 2 features"""

    def __init__(self):
        self.anonymizer = enhanced_data_anonymizer
        self.validator = InputValidator()
        self.auditor = enhanced_security_auditor

    def secure_load_data(self, file_path: str, user_id: str = "system",
                        correlation_id: str = None) -> pd.DataFrame:
        """Securely load and process employee data with enhanced logging"""
        correlation_id = correlation_id or str(uuid.uuid4())

        # Validate file path
        if not self.validator.validate_file_path(file_path):
            self.auditor.log_security_event(
                SecurityEventType.VALIDATION_FAILURE,
                user_id, file_path, "file_validation", False,
                {"error": "Invalid file path", "file_path": file_path},
                correlation_id
            )
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
                self.auditor.log_security_event(
                    SecurityEventType.VALIDATION_FAILURE,
                    user_id, "employee_data", "data_validation", False,
                    {"errors": validation_result['errors']}, correlation_id
                )
                raise ValueError(f"Data validation failed: {validation_result['errors']}")

            # Anonymize data
            df_anonymized = self.anonymizer.anonymize_dataframe(df)

            # Log successful access with privacy protection
            self.auditor.log_data_access(
                user_id=user_id,
                data_type='employee_insights',
                record_count=len(df_anonymized),
                operation='secure_load',
                correlation_id=correlation_id
            )

            # Log anonymization
            self.auditor.log_security_event(
                SecurityEventType.DATA_ANONYMIZATION,
                user_id, "employee_data", "anonymize", True,
                {"original_records": len(df), "anonymized_records": len(df_anonymized)},
                correlation_id
            )

            logger.info(f"Securely loaded {len(df_anonymized)} employee records [correlation_id: {correlation_id}]")
            return df_anonymized

        except Exception as e:
            self.auditor.log_security_event(
                SecurityEventType.DATA_ACCESS,
                user_id, file_path, "secure_load", False,
                {"error": str(e), "file_path": file_path}, correlation_id
            )
            logger.error(f"Secure data loading failed: {e!s} [correlation_id: {correlation_id}]")
            raise

    def secure_clustering_pipeline(self, df: pd.DataFrame, n_clusters: int,
                                 user_id: str = "system", correlation_id: str = None) -> Dict[str, Any]:
        """Execute clustering with enhanced security controls and privacy protection"""
        correlation_id = correlation_id or str(uuid.uuid4())

        # Validate inputs
        if not self.validator.validate_cluster_count(n_clusters, len(df)):
            self.auditor.log_security_event(
                SecurityEventType.VALIDATION_FAILURE,
                user_id, "clustering_params", "validation", False,
                {"n_clusters": n_clusters, "data_size": len(df)}, correlation_id
            )
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

            # Get basic results
            cluster_labels = clusterer.get_cluster_assignments()
            centroids = clusterer.get_cluster_centroids()
            quality_metrics = clusterer.get_cluster_quality_metrics()

            # Apply differential privacy to results
            anonymized_results = self.anonymizer.anonymize_clustering_results(
                cluster_labels, features.values, correlation_id
            )

            results = {
                'cluster_assignments': anonymized_results['anonymized_labels'],
                'centroids': centroids.tolist() if hasattr(centroids, 'tolist') else centroids,
                'quality_metrics': asdict(quality_metrics) if hasattr(quality_metrics, '__dict__') else quality_metrics,
                'privacy_protection': anonymized_results['privacy_parameters'],
                'data_summary': {
                    'total_employees': len(df),
                    'features_used': list(features.columns),
                    'clusters_created': n_clusters,
                    'k_anonymity_applied': True
                },
                'status': 'success'
            }

            # Log clustering operation with comprehensive details
            self.auditor.log_clustering_operation(
                parameters={
                    'n_clusters': n_clusters,
                    'data_size': len(df),
                    'features': list(features.columns)
                },
                result_summary=results['data_summary'],
                user_id=user_id,
                correlation_id=correlation_id
            )

            # Log privacy protection
            self.auditor.log_privacy_operation(
                "clustering_anonymization",
                anonymized_results['privacy_parameters'],
                user_id,
                correlation_id
            )

            return results

        except Exception as e:
            self.auditor.log_security_event(
                SecurityEventType.DATA_ACCESS,
                user_id, "clustering_engine", "clustering", False,
                {"error": str(e), "n_clusters": n_clusters, "data_size": len(df)},
                correlation_id
            )
            logger.error(f"Secure clustering failed: {e!s} [correlation_id: {correlation_id}]")
            raise


class APISecurityManager:
    """Enhanced API security controls for global deployment"""

    def __init__(self):
        self.rate_limiters = {}
        self.api_keys = {}
        self.blocked_ips = set()
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }

    def validate_api_key(self, api_key: str, endpoint: str) -> bool:
        """Validate API key and check permissions"""
        if not api_key or api_key not in self.api_keys:
            return False

        key_info = self.api_keys[api_key]

        # Check if key is expired
        if key_info.get('expires_at') and datetime.utcnow() > key_info['expires_at']:
            return False

        # Check endpoint permissions
        allowed_endpoints = key_info.get('allowed_endpoints', [])
        if allowed_endpoints and endpoint not in allowed_endpoints:
            return False

        return True

    def check_rate_limit(self, client_id: str, endpoint: str, limit: int = 100, window: int = 3600) -> bool:
        """Check if client has exceeded rate limit"""
        now = datetime.utcnow()
        key = f"{client_id}:{endpoint}"

        if key not in self.rate_limiters:
            self.rate_limiters[key] = []

        # Clean old requests outside window
        self.rate_limiters[key] = [
            req_time for req_time in self.rate_limiters[key]
            if (now - req_time).total_seconds() < window
        ]

        # Check if limit exceeded
        if len(self.rate_limiters[key]) >= limit:
            return False

        # Add current request
        self.rate_limiters[key].append(now)
        return True

    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips

    def block_ip(self, ip_address: str, reason: str = ""):
        """Block IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP {ip_address}: {reason}")

    def get_security_headers(self) -> Dict[str, str]:
        """Get security headers for HTTP responses"""
        return self.security_headers.copy()


class EncryptionManager:
    """Advanced encryption management for sensitive data"""

    def __init__(self):
        self.master_key = self._generate_master_key()
        self.key_derivation_salt = os.urandom(32)

    def _generate_master_key(self) -> bytes:
        """Generate or retrieve master encryption key"""
        key_file = Path(os.environ.get('MASTER_KEY_PATH', '.master_key'))

        if key_file.exists():
            return key_file.read_bytes()
        else:
            # Generate new master key
            master_key = os.urandom(32)
            # In production, this should be stored securely (e.g., HSM, key vault)
            key_file.write_bytes(master_key)
            key_file.chmod(0o600)  # Owner read/write only
            return master_key

    def derive_key(self, purpose: str, context: str = "") -> bytes:
        """Derive encryption key for specific purpose"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.key_derivation_salt + purpose.encode() + context.encode(),
            iterations=100000,
        )
        return kdf.derive(self.master_key)

    def encrypt_field(self, data: str, purpose: str, context: str = "") -> str:
        """Encrypt individual field"""
        key = self.derive_key(purpose, context)
        cipher = Fernet(base64.urlsafe_b64encode(key))
        encrypted = cipher.encrypt(data.encode())
        return base64.urlsafe_b64encode(encrypted).decode()

    def decrypt_field(self, encrypted_data: str, purpose: str, context: str = "") -> str:
        """Decrypt individual field"""
        key = self.derive_key(purpose, context)
        cipher = Fernet(base64.urlsafe_b64encode(key))
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
        decrypted = cipher.decrypt(encrypted_bytes)
        return decrypted.decode()

    def encrypt_large_data(self, data: bytes, purpose: str) -> bytes:
        """Encrypt large data using hybrid encryption"""
        # Generate random key for symmetric encryption
        data_key = os.urandom(32)

        # Encrypt data with symmetric key
        cipher = Fernet(base64.urlsafe_b64encode(data_key))
        encrypted_data = cipher.encrypt(data)

        # Encrypt data key with master key
        master_cipher = Fernet(base64.urlsafe_b64encode(self.derive_key(purpose)))
        encrypted_data_key = master_cipher.encrypt(data_key)

        # Combine encrypted key and data
        return encrypted_data_key + b'|' + encrypted_data

    def decrypt_large_data(self, encrypted_data: bytes, purpose: str) -> bytes:
        """Decrypt large data using hybrid encryption"""
        # Split encrypted key and data
        parts = encrypted_data.split(b'|', 1)
        if len(parts) != 2:
            raise ValueError("Invalid encrypted data format")

        encrypted_data_key, encrypted_data = parts

        # Decrypt data key
        master_cipher = Fernet(base64.urlsafe_b64encode(self.derive_key(purpose)))
        data_key = master_cipher.decrypt(encrypted_data_key)

        # Decrypt data
        cipher = Fernet(base64.urlsafe_b64encode(data_key))
        return cipher.decrypt(encrypted_data)


class SecurityMonitor:
    """Real-time security monitoring and alerting"""

    def __init__(self):
        self.threat_patterns = {
            'sql_injection': [
                r"(\s|^)(union|select|insert|delete|update|drop|exec|script)",
                r"'.*?(\s|^)(or|and).*?'",
                r"1\s*=\s*1",
                r";.*?(drop|delete|insert|update)"
            ],
            'xss': [
                r"<script.*?>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe.*?>"
            ],
            'path_traversal': [
                r"\.\./",
                r"\.\.\\",
                r"/etc/passwd",
                r"C:\\Windows\\System32"
            ]
        }
        self.alert_thresholds = {
            'failed_logins': 5,
            'rate_limit_violations': 10,
            'malicious_requests': 3
        }
        self.incident_counter = defaultdict(int)

    def analyze_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze HTTP request for security threats"""
        threats_detected = []
        risk_score = 0.0

        # Check for injection patterns
        for param_value in request_data.get('parameters', {}).values():
            if isinstance(param_value, str):
                for threat_type, patterns in self.threat_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, param_value, re.IGNORECASE):
                            threats_detected.append({
                                'type': threat_type,
                                'pattern': pattern,
                                'value': param_value[:100]  # Truncate for logging
                            })
                            risk_score += 0.3

        # Check suspicious headers
        headers = request_data.get('headers', {})
        if 'X-Forwarded-For' in headers and len(headers['X-Forwarded-For'].split(',')) > 5:
            threats_detected.append({'type': 'proxy_chaining', 'description': 'Suspicious proxy chain'})
            risk_score += 0.2

        # Check file upload attempts
        if request_data.get('method') == 'POST' and 'multipart/form-data' in headers.get('Content-Type', ''):
            threats_detected.append({'type': 'file_upload', 'description': 'File upload detected'})
            risk_score += 0.1

        return {
            'threats_detected': threats_detected,
            'risk_score': min(risk_score, 1.0),
            'blocked': risk_score > 0.7,
            'timestamp': datetime.utcnow().isoformat()
        }

    def log_security_incident(self, incident_type: str, details: Dict[str, Any]):
        """Log security incident and check for patterns"""
        self.incident_counter[incident_type] += 1

        # Check if threshold exceeded
        if self.incident_counter[incident_type] >= self.alert_thresholds.get(incident_type, 10):
            self._trigger_security_alert(incident_type, details)

    def _trigger_security_alert(self, incident_type: str, details: Dict[str, Any]):
        """Trigger security alert"""
        alert = {
            'alert_id': str(uuid.uuid4()),
            'incident_type': incident_type,
            'count': self.incident_counter[incident_type],
            'details': details,
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'HIGH' if self.incident_counter[incident_type] > 20 else 'MEDIUM'
        }

        logger.critical(f"🚨 SECURITY ALERT: {incident_type} - Count: {self.incident_counter[incident_type]}")

        # In production, send to SIEM, notification system, etc.
        return alert


# Enhanced global instances
api_security_manager = APISecurityManager()
encryption_manager = EncryptionManager()
security_monitor = SecurityMonitor()
enhanced_secure_data_processor = SecureDataProcessor()
