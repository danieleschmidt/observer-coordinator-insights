"""
Centralized error handling and custom exceptions for Observer Coordinator Insights
"""

import logging
import traceback
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


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
    TEAM_SIMULATION = "team_simulation"
    FILE_IO = "file_io"
    SECURITY = "security"
    API = "api"
    DATABASE = "database"
    CONFIGURATION = "configuration"


@dataclass
class ErrorDetails:
    """Detailed error information"""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    technical_details: str
    user_message: str
    suggestions: List[str]
    context: Dict[str, Any]


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


class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_history: List[ErrorDetails] = []
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorDetails:
        """Handle and log errors with detailed information"""
        
        # Generate unique error ID
        error_id = f"ERR_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{id(error) % 10000:04d}"
        
        # Determine error details
        if isinstance(error, ObserverCoordinatorError):
            category = error.category
            severity = error.severity
            message = error.message
            suggestions = error.suggestions
            technical_details = error.technical_details
            error_context = {**error.context, **(context or {})}
        else:
            # Handle standard Python exceptions
            category = self._categorize_standard_error(error)
            severity = self._assess_severity(error)
            message = str(error)
            suggestions = self._generate_suggestions(error)
            technical_details = f"{type(error).__name__}: {str(error)}\n{traceback.format_exc()}"
            error_context = context or {}
        
        # Create user-friendly message
        user_message = self._create_user_message(category, message)
        
        # Create error details
        error_details = ErrorDetails(
            error_id=error_id,
            timestamp=datetime.utcnow(),
            category=category,
            severity=severity,
            message=message,
            technical_details=technical_details,
            user_message=user_message,
            suggestions=suggestions,
            context=error_context
        )
        
        # Log error
        self._log_error(error_details)
        
        # Store in history
        self.error_history.append(error_details)
        
        return error_details
    
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
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent errors"""
        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp.timestamp() > cutoff_time
        ]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'period_hours': hours,
                'summary': 'No errors in specified period'
            }
        
        # Categorize errors
        by_category = {}
        by_severity = {}
        
        for error in recent_errors:
            by_category[error.category.value] = by_category.get(error.category.value, 0) + 1
            by_severity[error.severity.value] = by_severity.get(error.severity.value, 0) + 1
        
        return {
            'total_errors': len(recent_errors),
            'period_hours': hours,
            'by_category': by_category,
            'by_severity': by_severity,
            'most_recent': recent_errors[-1].error_id if recent_errors else None
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_exceptions(func):
    """Decorator for automatic exception handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_details = error_handler.handle_error(e, {'function': func.__name__})
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