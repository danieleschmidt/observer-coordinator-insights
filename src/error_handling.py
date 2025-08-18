"""Centralized error handling and custom exceptions for Observer Coordinator Insights
"""

import json
import logging
import threading
import traceback
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA_VALIDATION = "data_validation"
    CLUSTERING = "clustering"
    NEUROMORPHIC_CLUSTERING = "neuromorphic_clustering"
    TEAM_SIMULATION = "team_simulation"
    FILE_IO = "file_io"
    SECURITY = "security"
    API = "api"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    CIRCUIT_BREAKER = "circuit_breaker"
    TIMEOUT = "timeout"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    FALLBACK = "fallback"


@dataclass
class ErrorDetails:
    """Detailed error information with Generation 2 enhancements"""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    user_message: str
    suggestions: List[str]
    context: Dict[str, Any]
    correlation_id: Optional[str] = None
    recoverable: bool = True
    retry_count: int = 0
    tags: Set[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())


@dataclass
class ErrorPattern:
    """Pattern for error analysis and alerting"""
    category: ErrorCategory
    count: int
    first_occurrence: datetime
    last_occurrence: datetime
    error_ids: List[str]
    recovery_attempts: int = 0


@dataclass
class AlertThreshold:
    """Threshold configuration for error alerting"""
    category: ErrorCategory
    severity: ErrorSeverity
    count_threshold: int
    time_window_minutes: int
    enabled: bool = True


class ObserverCoordinatorError(Exception):
    """Base exception for Observer Coordinator Insights"""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.CONFIGURATION,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 technical_details: str = None, suggestions: List[str] = None,
                 context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.technical_details = technical_details or str(self)
        self.suggestions = suggestions or []
        self.context = context or {}
        self.timestamp = datetime.utcnow()


class DataValidationError(ObserverCoordinatorError):
    """Raised when data validation fails"""

    def __init__(self, message: str, invalid_fields: List[str] = None,
                 data_summary: Dict[str, Any] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_VALIDATION,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.invalid_fields = invalid_fields or []
        self.data_summary = data_summary or {}

        # Add specific suggestions for data validation
        if not self.suggestions:
            self.suggestions = [
                "Check data format and column names",
                "Verify all required fields are present",
                "Ensure numeric fields contain valid numbers",
                "Review data for missing or invalid values"
            ]


class ClusteringError(ObserverCoordinatorError):
    """Raised when clustering operations fail"""

    def __init__(self, message: str, clustering_params: Dict[str, Any] = None,
                 data_shape: tuple = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CLUSTERING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.clustering_params = clustering_params or {}
        self.data_shape = data_shape

        if not self.suggestions:
            self.suggestions = [
                "Check if data has sufficient samples for clustering",
                "Verify all features are numeric",
                "Consider reducing number of clusters",
                "Review data quality and preprocessing"
            ]


class TeamSimulationError(ObserverCoordinatorError):
    """Raised when team simulation fails"""

    def __init__(self, message: str, team_params: Dict[str, Any] = None,
                 employee_count: int = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TEAM_SIMULATION,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.team_params = team_params or {}
        self.employee_count = employee_count

        if not self.suggestions:
            self.suggestions = [
                "Ensure sufficient employees for team generation",
                "Check team size parameters",
                "Verify clustering results are available",
                "Review employee data completeness"
            ]


class SecurityError(ObserverCoordinatorError):
    """Raised for security-related issues"""

    def __init__(self, message: str, security_context: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.SECURITY,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.security_context = security_context

        if not self.suggestions:
            self.suggestions = [
                "Review security configuration",
                "Check data access permissions",
                "Verify input validation",
                "Contact system administrator"
            ]


class FileIOError(ObserverCoordinatorError):
    """Raised for file I/O related issues"""

    def __init__(self, message: str, file_path: str = None,
                 operation: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.FILE_IO,
            severity=ErrorSeverity.MEDIUM,
            **kwargs
        )
        self.file_path = file_path
        self.operation = operation

        if not self.suggestions:
            self.suggestions = [
                "Check file path and permissions",
                "Verify file format is supported",
                "Ensure file is not corrupted",
                "Check available disk space"
            ]


class NeuromorphicClusteringError(ObserverCoordinatorError):
    """Raised when neuromorphic clustering operations fail"""

    def __init__(self, message: str, method: str = None,
                 feature_shape: tuple = None, component: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.NEUROMORPHIC_CLUSTERING,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.method = method
        self.feature_shape = feature_shape
        self.component = component

        if not self.suggestions:
            self.suggestions = [
                "Check data quality and preprocessing",
                "Consider reducing feature dimensions",
                "Enable fallback to K-means clustering",
                "Review neuromorphic component parameters",
                "Verify sufficient memory and computational resources"
            ]


class CircuitBreakerError(ObserverCoordinatorError):
    """Raised when circuit breaker is open"""

    def __init__(self, message: str, failure_count: int = None,
                 recovery_timeout: int = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CIRCUIT_BREAKER,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.failure_count = failure_count
        self.recovery_timeout = recovery_timeout

        if not self.suggestions:
            self.suggestions = [
                f"Wait {recovery_timeout or 'configured'} seconds for circuit breaker reset",
                "Check underlying service health",
                "Review error patterns and root causes",
                "Consider adjusting circuit breaker thresholds"
            ]


class TimeoutError(ObserverCoordinatorError):
    """Raised when operations timeout"""

    def __init__(self, message: str, timeout_seconds: int = None,
                 operation: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.HIGH,
            **kwargs
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation

        if not self.suggestions:
            self.suggestions = [
                "Increase timeout threshold if appropriate",
                "Optimize operation performance",
                "Check system resource availability",
                "Consider breaking operation into smaller chunks"
            ]


class ResourceExhaustionError(ObserverCoordinatorError):
    """Raised when system resources are exhausted"""

    def __init__(self, message: str, resource_type: str = None,
                 current_usage: float = None, threshold: float = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE_EXHAUSTION,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.threshold = threshold

        if not self.suggestions:
            self.suggestions = [
                "Free up system resources",
                "Scale up computational resources",
                "Optimize memory usage patterns",
                "Review and cleanup temporary files",
                "Consider implementing resource pooling"
            ]


class FallbackError(ObserverCoordinatorError):
    """Raised when fallback mechanisms fail"""

    def __init__(self, message: str, primary_error: str = None,
                 fallback_method: str = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.FALLBACK,
            severity=ErrorSeverity.CRITICAL,
            **kwargs
        )
        self.primary_error = primary_error
        self.fallback_method = fallback_method

        if not self.suggestions:
            self.suggestions = [
                "Review primary error cause and fix root issue",
                "Ensure fallback mechanisms are properly configured",
                "Check fallback resource requirements",
                "Consider alternative fallback strategies",
                "Contact system administrator for manual intervention"
            ]


class EnhancedErrorHandler:
    """Enterprise-grade centralized error handling and logging with Generation 2 features"""

    def __init__(self, max_history_size: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.error_history: deque = deque(maxlen=max_history_size)
        self.error_patterns: Dict[str, ErrorPattern] = {}
        self.alert_thresholds: List[AlertThreshold] = self._default_alert_thresholds()
        self.correlation_errors: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

        # Performance tracking
        self.performance_stats = {
            'total_errors': 0,
            'errors_by_category': defaultdict(int),
            'errors_by_severity': defaultdict(int),
            'recovery_success_rate': 0.0,
            'avg_resolution_time': 0.0
        }

    def _default_alert_thresholds(self) -> List[AlertThreshold]:
        """Define default alerting thresholds"""
        return [
            AlertThreshold(ErrorCategory.NEUROMORPHIC_CLUSTERING, ErrorSeverity.HIGH, 5, 15),
            AlertThreshold(ErrorCategory.CIRCUIT_BREAKER, ErrorSeverity.HIGH, 3, 10),
            AlertThreshold(ErrorCategory.RESOURCE_EXHAUSTION, ErrorSeverity.CRITICAL, 1, 5),
            AlertThreshold(ErrorCategory.TIMEOUT, ErrorSeverity.HIGH, 10, 30),
            AlertThreshold(ErrorCategory.FALLBACK, ErrorSeverity.CRITICAL, 2, 10),
            AlertThreshold(ErrorCategory.SECURITY, ErrorSeverity.CRITICAL, 1, 5),
        ]

    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    correlation_id: str = None) -> ErrorDetails:
        """Handle and log errors with enhanced Generation 2 features"""
        with self._lock:
            # Generate unique error ID
            error_id = f"ERR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # Extract correlation ID from context or generate new one
            if correlation_id is None:
                correlation_id = context.get('correlation_id') if context else str(uuid.uuid4())

            # Determine error details
            if isinstance(error, ObserverCoordinatorError):
                category = error.category
                severity = error.severity
                message = error.message
                suggestions = error.suggestions
                technical_details = error.technical_details
                error_context = {**error.context, **(context or {})}
                recoverable = getattr(error, 'recoverable', True)
            else:
                # Handle standard Python exceptions
                category = self._categorize_standard_error(error)
                severity = self._assess_severity(error)
                message = str(error)
                suggestions = self._generate_suggestions(error)
                technical_details = f"{type(error).__name__}: {error!s}\n{traceback.format_exc()}"
                error_context = context or {}
                recoverable = self._is_recoverable_error(error)

            # Create user-friendly message
            user_message = self._create_user_message(category, message)

            # Create enhanced error details
            error_details = ErrorDetails(
                error_id=error_id,
                timestamp=datetime.utcnow(),
                category=category,
                severity=severity,
                message=message,
                technical_details=technical_details,
                user_message=user_message,
                suggestions=suggestions,
                context=error_context,
                correlation_id=correlation_id,
                recoverable=recoverable,
                retry_count=error_context.get('retry_count', 0)
            )

            # Add to correlation tracking
            self.correlation_errors[correlation_id].append(error_id)

            # Update performance statistics
            self._update_performance_stats(error_details)

            # Check for error patterns and alerting
            self._analyze_error_patterns(error_details)

            # Log error with structured logging
            self._log_error_enhanced(error_details)

            # Store in history
            self.error_history.append(error_details)

            return error_details

    def _is_recoverable_error(self, error: Exception) -> bool:
        """Determine if error is recoverable"""
        recoverable_types = (
            ValueError, TypeError, ConnectionError,
            FileNotFoundError, PermissionError
        )
        non_recoverable_types = (
            MemoryError, SystemError, KeyboardInterrupt
        )

        if isinstance(error, non_recoverable_types):
            return False
        elif isinstance(error, recoverable_types):
            return True
        else:
            return True  # Default to recoverable for unknown errors

    def _update_performance_stats(self, error_details: ErrorDetails):
        """Update performance statistics"""
        self.performance_stats['total_errors'] += 1
        self.performance_stats['errors_by_category'][error_details.category.value] += 1
        self.performance_stats['errors_by_severity'][error_details.severity.value] += 1

    def _analyze_error_patterns(self, error_details: ErrorDetails):
        """Analyze error patterns and trigger alerts if thresholds are exceeded"""
        pattern_key = f"{error_details.category.value}_{error_details.severity.value}"

        if pattern_key in self.error_patterns:
            pattern = self.error_patterns[pattern_key]
            pattern.count += 1
            pattern.last_occurrence = error_details.timestamp
            pattern.error_ids.append(error_details.error_id)
        else:
            self.error_patterns[pattern_key] = ErrorPattern(
                category=error_details.category,
                count=1,
                first_occurrence=error_details.timestamp,
                last_occurrence=error_details.timestamp,
                error_ids=[error_details.error_id]
            )

        # Check alert thresholds
        self._check_alert_thresholds(error_details, self.error_patterns[pattern_key])

    def _check_alert_thresholds(self, error_details: ErrorDetails, pattern: ErrorPattern):
        """Check if error patterns exceed alert thresholds"""
        for threshold in self.alert_thresholds:
            if (threshold.category == error_details.category and
                threshold.severity == error_details.severity and
                threshold.enabled):

                # Check if pattern exceeds threshold within time window
                time_window = timedelta(minutes=threshold.time_window_minutes)
                recent_errors = [
                    eid for eid in pattern.error_ids
                    if any(
                        e.error_id == eid and
                        e.timestamp > datetime.utcnow() - time_window
                        for e in self.error_history
                    )
                ]

                if len(recent_errors) >= threshold.count_threshold:
                    self._trigger_alert(error_details, threshold, len(recent_errors))

    def _trigger_alert(self, error_details: ErrorDetails, threshold: AlertThreshold, count: int):
        """Trigger alert for error threshold breach"""
        alert_message = (
            f"ALERT: {threshold.category.value} errors exceeded threshold. "
            f"Count: {count}/{threshold.count_threshold} in {threshold.time_window_minutes} minutes. "
            f"Latest error: {error_details.error_id}"
        )

        self.logger.critical(f"🚨 {alert_message} [correlation_id: {error_details.correlation_id}]")

        # Add alert tag to error
        error_details.tags.add("alert_triggered")
        error_details.tags.add(f"threshold_{threshold.count_threshold}")

    def _log_error_enhanced(self, error_details: ErrorDetails):
        """Enhanced structured logging with correlation ID and tags"""
        log_context = {
            'error_id': error_details.error_id,
            'correlation_id': error_details.correlation_id,
            'category': error_details.category.value,
            'severity': error_details.severity.value,
            'recoverable': error_details.recoverable,
            'retry_count': error_details.retry_count,
            'tags': list(error_details.tags)
        }

        log_message = f"[{error_details.error_id}] {error_details.category.value}: {error_details.message}"

        # Add structured context as extra fields for log aggregation
        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=log_context)
            self.logger.debug(f"Technical details: {error_details.technical_details}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=log_context)
            self.logger.debug(f"Technical details: {error_details.technical_details}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=log_context)
        else:
            self.logger.info(log_message, extra=log_context)

    def _categorize_standard_error(self, error: Exception) -> ErrorCategory:
        """Categorize standard Python exceptions"""
        error_type = type(error).__name__

        if error_type in ['ValueError', 'TypeError']:
            return ErrorCategory.DATA_VALIDATION
        elif error_type in ['FileNotFoundError', 'PermissionError', 'IOError']:
            return ErrorCategory.FILE_IO
        elif error_type in ['ImportError', 'ModuleNotFoundError']:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.CONFIGURATION

    def _assess_severity(self, error: Exception) -> ErrorSeverity:
        """Assess severity of standard exceptions"""
        critical_errors = ['SystemError', 'MemoryError', 'KeyboardInterrupt']
        high_errors = ['ValueError', 'TypeError', 'FileNotFoundError']

        error_type = type(error).__name__

        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.MEDIUM

    def _generate_suggestions(self, error: Exception) -> List[str]:
        """Generate helpful suggestions for standard errors"""
        error_type = type(error).__name__

        suggestions_map = {
            'ValueError': [
                "Check input data format and values",
                "Verify all parameters are within expected ranges",
                "Review data preprocessing steps"
            ],
            'FileNotFoundError': [
                "Check file path is correct",
                "Verify file exists at specified location",
                "Ensure proper file permissions"
            ],
            'TypeError': [
                "Check data types of input parameters",
                "Verify function arguments match expected types",
                "Review data conversion steps"
            ]
        }

        return suggestions_map.get(error_type, ["Contact technical support for assistance"])

    def _create_user_message(self, category: ErrorCategory, message: str) -> str:
        """Create user-friendly error message"""
        category_messages = {
            ErrorCategory.DATA_VALIDATION: "There's an issue with the data format or values.",
            ErrorCategory.CLUSTERING: "The clustering analysis couldn't be completed.",
            ErrorCategory.TEAM_SIMULATION: "Team generation encountered a problem.",
            ErrorCategory.FILE_IO: "There's a problem accessing or reading the file.",
            ErrorCategory.SECURITY: "A security issue was detected.",
            ErrorCategory.API: "An API operation failed.",
            ErrorCategory.DATABASE: "Database operation encountered an error.",
            ErrorCategory.CONFIGURATION: "There's a configuration issue."
        }

        base_message = category_messages.get(category, "An unexpected error occurred.")
        return f"{base_message} {message}"

    def _log_error(self, error_details: ErrorDetails):
        """Log error with appropriate level"""
        log_message = f"[{error_details.error_id}] {error_details.category.value}: {error_details.message}"

        if error_details.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
            self.logger.debug(f"Technical details: {error_details.technical_details}")
        elif error_details.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
            self.logger.debug(f"Technical details: {error_details.technical_details}")
        elif error_details.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def get_comprehensive_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive summary of recent errors with Generation 2 analytics"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            err for err in self.error_history
            if err.timestamp > cutoff_time
        ]

        if not recent_errors:
            return {
                'total_errors': 0,
                'period_hours': hours,
                'summary': 'No errors in specified period',
                'health_score': 1.0
            }

        # Categorize errors
        by_category = defaultdict(int)
        by_severity = defaultdict(int)
        by_correlation = defaultdict(int)
        recovery_attempts = 0
        critical_errors = 0

        for error in recent_errors:
            by_category[error.category.value] += 1
            by_severity[error.severity.value] += 1
            by_correlation[error.correlation_id] += 1

            if error.retry_count > 0:
                recovery_attempts += 1
            if error.severity == ErrorSeverity.CRITICAL:
                critical_errors += 1

        # Calculate health metrics
        total_errors = len(recent_errors)
        error_rate = total_errors / max(hours, 1)  # errors per hour
        recovery_rate = recovery_attempts / total_errors if total_errors > 0 else 0
        critical_rate = critical_errors / total_errors if total_errors > 0 else 0

        # Health score calculation (0-1, higher is better)
        health_score = max(0, 1 - (error_rate * 0.1) - (critical_rate * 0.5))

        # Top error patterns
        top_patterns = sorted(
            self.error_patterns.items(),
            key=lambda x: x[1].count,
            reverse=True
        )[:5]

        # Correlation analysis
        top_correlations = sorted(
            [(corr_id, len(err_ids)) for corr_id, err_ids in by_correlation.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return {
            'total_errors': total_errors,
            'period_hours': hours,
            'error_rate_per_hour': round(error_rate, 2),
            'by_category': dict(by_category),
            'by_severity': dict(by_severity),
            'health_score': round(health_score, 3),
            'recovery_rate': round(recovery_rate, 3),
            'critical_error_rate': round(critical_rate, 3),
            'top_error_patterns': [
                {
                    'pattern': pattern_key,
                    'count': pattern.count,
                    'category': pattern.category.value,
                    'latest_occurrence': pattern.last_occurrence.isoformat()
                }
                for pattern_key, pattern in top_patterns
            ],
            'top_correlations': [
                {
                    'correlation_id': corr_id,
                    'error_count': count
                }
                for corr_id, count in top_correlations
            ],
            'most_recent': recent_errors[-1].error_id if recent_errors else None,
            'performance_stats': dict(self.performance_stats)
        }

    def get_correlation_timeline(self, correlation_id: str) -> Dict[str, Any]:
        """Get timeline of errors for a specific correlation ID"""
        correlated_errors = [
            err for err in self.error_history
            if err.correlation_id == correlation_id
        ]

        if not correlated_errors:
            return {
                'correlation_id': correlation_id,
                'error_count': 0,
                'timeline': []
            }

        # Sort by timestamp
        correlated_errors.sort(key=lambda x: x.timestamp)

        timeline = []
        for error in correlated_errors:
            timeline.append({
                'error_id': error.error_id,
                'timestamp': error.timestamp.isoformat(),
                'category': error.category.value,
                'severity': error.severity.value,
                'message': error.message,
                'recoverable': error.recoverable,
                'retry_count': error.retry_count,
                'tags': list(error.tags)
            })

        return {
            'correlation_id': correlation_id,
            'error_count': len(correlated_errors),
            'first_error': timeline[0]['timestamp'],
            'last_error': timeline[-1]['timestamp'],
            'timeline': timeline
        }

    def suggest_recovery_actions(self, error_id: str) -> List[str]:
        """Suggest recovery actions based on error analysis"""
        error = next((e for e in self.error_history if e.error_id == error_id), None)
        if not error:
            return ["Error not found in history"]

        actions = error.suggestions.copy()

        # Add context-specific suggestions
        if error.category == ErrorCategory.NEUROMORPHIC_CLUSTERING:
            if error.retry_count > 0:
                actions.append("Consider disabling neuromorphic clustering temporarily")
                actions.append("Use K-means fallback for immediate recovery")

        elif error.category == ErrorCategory.CIRCUIT_BREAKER:
            actions.append("Monitor underlying service health before reset")
            actions.append("Check system resource availability")

        elif error.category == ErrorCategory.RESOURCE_EXHAUSTION:
            actions.append("Scale up infrastructure immediately")
            actions.append("Implement resource cleanup procedures")

        # Add correlation-based suggestions
        correlation_errors = self.correlation_errors.get(error.correlation_id, [])
        if len(correlation_errors) > 1:
            actions.append("Review entire request flow for systemic issues")
            actions.append("Consider implementing request throttling")

        return actions

    def export_error_report(self, format_type: str = "json") -> str:
        """Export comprehensive error report"""
        summary = self.get_comprehensive_error_summary(hours=168)  # 1 week

        report_data = {
            'report_timestamp': datetime.utcnow().isoformat(),
            'summary': summary,
            'alert_thresholds': [asdict(t) for t in self.alert_thresholds],
            'error_patterns': {
                k: asdict(v) for k, v in self.error_patterns.items()
            }
        }

        if format_type.lower() == "json":
            return json.dumps(report_data, indent=2, default=str)
        else:
            # Simple text format
            lines = [
                "ERROR ANALYSIS REPORT",
                "=" * 50,
                f"Generated: {report_data['report_timestamp']}",
                "",
                f"Total Errors (7 days): {summary['total_errors']}",
                f"Error Rate: {summary['error_rate_per_hour']} per hour",
                f"System Health Score: {summary['health_score']}",
                "",
                "Top Error Categories:"
            ]

            for category, count in summary['by_category'].items():
                lines.append(f"  - {category}: {count}")

            return "\n".join(lines)


# Global enhanced error handler instance
enhanced_error_handler = EnhancedErrorHandler()

# Backward compatibility alias
error_handler = enhanced_error_handler


def handle_exceptions(correlation_id: str = None):
    """Enhanced decorator for automatic exception handling with correlation ID support"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                error_details = enhanced_error_handler.handle_error(
                    e, context=context, correlation_id=correlation_id
                )

                # Re-raise as ObserverCoordinatorError for consistent handling
                raise ObserverCoordinatorError(
                    message=error_details.user_message,
                    category=error_details.category,
                    severity=error_details.severity,
                    technical_details=error_details.technical_details,
                    suggestions=error_details.suggestions,
                    context=error_details.context
                )
        return wrapper
    return decorator
