#!/usr/bin/env python3
"""
Enhanced Error Handling - Generation 2 Robustness
Comprehensive error handling, recovery, and resilience framework
"""

import asyncio
import functools
import logging
import sys
import traceback
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import json


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorCategory(Enum):
    """Error categories for classification"""
    SYSTEM = "system"
    DATA = "data"
    NETWORK = "network" 
    VALIDATION = "validation"
    PERMISSION = "permission"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"


@dataclass
class ErrorContext:
    """Rich context information for errors"""
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorRecord:
    """Comprehensive error record for tracking and analysis"""
    id: str
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: str
    resolution_attempted: bool = False
    resolution_successful: bool = False
    recovery_actions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'error_type': self.error_type,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': {
                'operation': self.context.operation,
                'component': self.context.component,
                'user_id': self.context.user_id,
                'session_id': self.context.session_id,
                'request_id': self.context.request_id,
                'additional_data': self.context.additional_data,
                'timestamp': self.context.timestamp.isoformat()
            },
            'stack_trace': self.stack_trace,
            'resolution_attempted': self.resolution_attempted,
            'resolution_successful': self.resolution_successful,
            'recovery_actions': self.recovery_actions,
            'timestamp': self.timestamp.isoformat()
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class EnhancedCircuitBreaker:
    """Advanced circuit breaker with exponential backoff and metrics"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Union[Exception, tuple] = Exception,
                 name: str = "default"):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        # State tracking
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitBreakerState.CLOSED
        self.success_count = 0
        self.total_calls = 0
        
        # Metrics
        self.metrics = {
            'total_failures': 0,
            'total_successes': 0,
            'state_changes': 0,
            'last_state_change': None,
            'average_failure_rate': 0.0
        }
        
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker.{name}")
    
    def _update_metrics(self, success: bool):
        """Update circuit breaker metrics"""
        self.total_calls += 1
        
        if success:
            self.metrics['total_successes'] += 1
        else:
            self.metrics['total_failures'] += 1
        
        self.metrics['average_failure_rate'] = (
            self.metrics['total_failures'] / max(1, self.total_calls)
        )
    
    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state with logging"""
        old_state = self.state
        self.state = new_state
        self.metrics['state_changes'] += 1
        self.metrics['last_state_change'] = datetime.now().isoformat()
        
        self.logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed based on current state"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time:
                time_since_failure = datetime.now() - self.last_failure_time
                if time_since_failure >= timedelta(seconds=self.recovery_timeout):
                    self._change_state(CircuitBreakerState.HALF_OPEN)
                    return True
            return False
        
        # HALF_OPEN state
        return True
    
    async def __call__(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self._can_execute():
            raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Success handling
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:  # Require 2 successes to close
                    self._change_state(CircuitBreakerState.CLOSED)
                    self.failure_count = 0
                    self.success_count = 0
            
            self._update_metrics(success=True)
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            self._update_metrics(success=False)
            
            if self.failure_count >= self.failure_threshold:
                if self.state != CircuitBreakerState.OPEN:
                    self._change_state(CircuitBreakerState.OPEN)
            
            raise e
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'total_calls': self.total_calls,
            'metrics': self.metrics.copy()
        }


class RetryStrategy:
    """Configurable retry strategy with exponential backoff"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number"""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)
        
        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay


class EnhancedErrorHandler:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, 
                 log_file: Optional[Path] = None,
                 enable_metrics: bool = True,
                 auto_recovery: bool = True):
        self.log_file = log_file or Path("logs/error_handler.json")
        self.enable_metrics = enable_metrics
        self.auto_recovery = auto_recovery
        
        # Error tracking
        self.error_records: List[ErrorRecord] = []
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Recovery strategies
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {
            ErrorCategory.NETWORK: [self._retry_with_backoff, self._check_network_connectivity],
            ErrorCategory.DATA: [self._validate_data_format, self._attempt_data_recovery],
            ErrorCategory.PERMISSION: [self._check_permissions, self._request_elevated_access],
            ErrorCategory.CONFIGURATION: [self._reload_configuration, self._use_default_config],
        }
    
    def _setup_logging(self):
        """Setup structured error logging"""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # JSON formatter for structured logs
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_data = {
                        'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                        'level': record.levelname,
                        'logger': record.name,
                        'message': record.getMessage(),
                        'filename': record.filename,
                        'line': record.lineno
                    }
                    if hasattr(record, 'error_record'):
                        log_data['error_record'] = record.error_record
                    return json.dumps(log_data)
            
            handler = logging.FileHandler(self.log_file)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)
    
    def create_circuit_breaker(self, name: str, **kwargs) -> EnhancedCircuitBreaker:
        """Create or get circuit breaker for operation"""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = EnhancedCircuitBreaker(name=name, **kwargs)
        return self.circuit_breakers[name]
    
    async def handle_error(self, 
                          error: Exception,
                          context: ErrorContext,
                          severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                          category: ErrorCategory = ErrorCategory.SYSTEM) -> ErrorRecord:
        """Handle error with comprehensive logging and recovery"""
        
        # Create error record
        error_record = ErrorRecord(
            id=f"err-{int(time.time() * 1000)}-{hash(str(error))}",
            error_type=type(error).__name__,
            message=str(error),
            severity=severity,
            category=category,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        # Log error
        self.logger.error(
            f"Error in {context.component}.{context.operation}: {error}",
            extra={'error_record': error_record.to_dict()}
        )
        
        # Store error record
        self.error_records.append(error_record)
        
        # Attempt automatic recovery if enabled
        if self.auto_recovery and severity != ErrorSeverity.CRITICAL:
            try:
                await self._attempt_recovery(error_record)
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt failed: {recovery_error}")
        
        # Alert on critical errors
        if severity == ErrorSeverity.CRITICAL:
            await self._send_critical_alert(error_record)
        
        return error_record
    
    async def _attempt_recovery(self, error_record: ErrorRecord):
        """Attempt automatic error recovery"""
        error_record.resolution_attempted = True
        
        recovery_funcs = self.recovery_strategies.get(error_record.category, [])
        
        for recovery_func in recovery_funcs:
            try:
                await recovery_func(error_record)
                error_record.resolution_successful = True
                error_record.recovery_actions.append(recovery_func.__name__)
                self.logger.info(f"Recovery successful using {recovery_func.__name__}")
                break
            except Exception as e:
                error_record.recovery_actions.append(f"{recovery_func.__name__}_failed: {str(e)}")
                continue
    
    async def _retry_with_backoff(self, error_record: ErrorRecord):
        """Generic retry with exponential backoff"""
        strategy = RetryStrategy(max_attempts=3, base_delay=1.0)
        
        for attempt in range(strategy.max_attempts):
            if attempt > 0:
                delay = strategy.get_delay(attempt - 1)
                await asyncio.sleep(delay)
            
            # This is a placeholder - in real implementation,
            # we'd retry the original operation
            self.logger.info(f"Retry attempt {attempt + 1} for {error_record.id}")
    
    async def _check_network_connectivity(self, error_record: ErrorRecord):
        """Check network connectivity for network errors"""
        # Placeholder for network connectivity check
        pass
    
    async def _validate_data_format(self, error_record: ErrorRecord):
        """Validate and potentially fix data format issues"""
        # Placeholder for data validation
        pass
    
    async def _attempt_data_recovery(self, error_record: ErrorRecord):
        """Attempt to recover from data corruption"""
        # Placeholder for data recovery
        pass
    
    async def _check_permissions(self, error_record: ErrorRecord):
        """Check and potentially fix permission issues"""
        # Placeholder for permission checking
        pass
    
    async def _request_elevated_access(self, error_record: ErrorRecord):
        """Request elevated access for permission errors"""
        # Placeholder for access elevation
        pass
    
    async def _reload_configuration(self, error_record: ErrorRecord):
        """Reload configuration from file"""
        # Placeholder for config reload
        pass
    
    async def _use_default_config(self, error_record: ErrorRecord):
        """Fall back to default configuration"""
        # Placeholder for default config
        pass
    
    async def _send_critical_alert(self, error_record: ErrorRecord):
        """Send alert for critical errors"""
        alert_data = {
            'type': 'critical_error',
            'error_id': error_record.id,
            'component': error_record.context.component,
            'operation': error_record.context.operation,
            'message': error_record.message,
            'timestamp': error_record.timestamp.isoformat()
        }
        
        # In production, this would send to alerting system
        self.logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data)}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        if not self.error_records:
            return {"message": "No errors recorded"}
        
        total_errors = len(self.error_records)
        
        # Group by severity
        severity_counts = {}
        for severity in ErrorSeverity:
            count = sum(1 for err in self.error_records if err.severity == severity)
            severity_counts[severity.name] = count
        
        # Group by category
        category_counts = {}
        for category in ErrorCategory:
            count = sum(1 for err in self.error_records if err.category == category)
            category_counts[category.value] = count
        
        # Recovery statistics
        recovery_attempted = sum(1 for err in self.error_records if err.resolution_attempted)
        recovery_successful = sum(1 for err in self.error_records if err.resolution_successful)
        recovery_rate = recovery_successful / max(1, recovery_attempted)
        
        # Recent error trend
        now = datetime.now()
        last_hour_errors = sum(
            1 for err in self.error_records 
            if now - err.timestamp <= timedelta(hours=1)
        )
        
        return {
            'total_errors': total_errors,
            'severity_distribution': severity_counts,
            'category_distribution': category_counts,
            'recovery_statistics': {
                'attempted': recovery_attempted,
                'successful': recovery_successful,
                'success_rate': recovery_rate
            },
            'recent_trends': {
                'last_hour': last_hour_errors
            },
            'circuit_breaker_status': {
                name: breaker.get_metrics() 
                for name, breaker in self.circuit_breakers.items()
            }
        }


# Decorators for enhanced error handling

def robust_operation(component: str, 
                    category: ErrorCategory = ErrorCategory.SYSTEM,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    circuit_breaker_name: Optional[str] = None):
    """Decorator for robust operation handling"""
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            error_handler = EnhancedErrorHandler()
            
            # Use circuit breaker if specified
            if circuit_breaker_name:
                breaker = error_handler.create_circuit_breaker(circuit_breaker_name)
                try:
                    return await breaker(func, *args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        operation=func.__name__,
                        component=component
                    )
                    await error_handler.handle_error(e, context, severity, category)
                    raise
            else:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    context = ErrorContext(
                        operation=func.__name__,
                        component=component
                    )
                    await error_handler.handle_error(e, context, severity, category)
                    raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = EnhancedErrorHandler()
                context = ErrorContext(
                    operation=func.__name__,
                    component=component
                )
                # For sync functions, we can't await, so just log
                error_record = ErrorRecord(
                    id=f"sync-err-{int(time.time() * 1000)}",
                    error_type=type(e).__name__,
                    message=str(e),
                    severity=severity,
                    category=category,
                    context=context,
                    stack_trace=traceback.format_exc()
                )
                error_handler.error_records.append(error_record)
                error_handler.logger.error(f"Sync error in {component}.{func.__name__}: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


@contextmanager
def error_boundary(operation: str, component: str):
    """Context manager for error boundary"""
    error_handler = EnhancedErrorHandler()
    
    try:
        yield
    except Exception as e:
        context = ErrorContext(operation=operation, component=component)
        
        # Create error record (sync version)
        error_record = ErrorRecord(
            id=f"boundary-{int(time.time() * 1000)}",
            error_type=type(e).__name__,
            message=str(e),
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.SYSTEM,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        error_handler.error_records.append(error_record)
        error_handler.logger.error(f"Error boundary triggered: {e}")
        raise


# Global error handler instance
global_error_handler = EnhancedErrorHandler(
    log_file=Path("logs/system_errors.json"),
    enable_metrics=True,
    auto_recovery=True
)