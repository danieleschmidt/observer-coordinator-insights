#!/usr/bin/env python3
"""
Comprehensive Logging System - Generation 2 Robustness
Advanced structured logging, metrics, and observability framework
"""

import asyncio
import json
import logging
import logging.handlers
import os
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
import traceback
import threading
import queue
import uuid
import socket
import psutil


class LogLevel(Enum):
    """Extended log levels"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60
    AUDIT = 70


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: datetime
    level: LogLevel
    logger_name: str
    message: str
    component: str
    operation: str
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    status: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    additional_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'level': self.level.name,
            'logger': self.logger_name,
            'message': self.message,
            'component': self.component,
            'operation': self.operation,
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'duration_ms': self.duration_ms,
            'status': self.status,
            'error_details': self.error_details,
            **self.additional_fields
        }


@dataclass
class MetricEntry:
    """Structured metric entry"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type.value,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags,
            'unit': self.unit
        }


class StructuredLogger:
    """High-performance structured logger"""
    
    def __init__(self, name: str, component: str = "unknown"):
        self.name = name
        self.component = component
        self.logger = logging.getLogger(name)
        self.correlation_id = str(uuid.uuid4())
        
        # Context stack for nested operations
        self._context_stack: List[Dict[str, Any]] = []
        
        # Performance tracking
        self._operation_timers: Dict[str, float] = {}
    
    def _create_log_entry(self,
                         level: LogLevel,
                         message: str,
                         operation: str = "default",
                         **kwargs) -> LogEntry:
        """Create structured log entry"""
        
        # Merge context from stack
        context = {}
        for ctx in self._context_stack:
            context.update(ctx)
        context.update(kwargs)
        
        return LogEntry(
            timestamp=datetime.now(),
            level=level,
            logger_name=self.name,
            message=message,
            component=self.component,
            operation=operation,
            correlation_id=self.correlation_id,
            user_id=context.get('user_id'),
            session_id=context.get('session_id'),
            request_id=context.get('request_id'),
            duration_ms=context.get('duration_ms'),
            status=context.get('status'),
            error_details=context.get('error_details'),
            additional_fields={k: v for k, v in context.items() 
                             if k not in ['user_id', 'session_id', 'request_id', 
                                        'duration_ms', 'status', 'error_details']}
        )
    
    def trace(self, message: str, operation: str = "default", **kwargs):
        """Log trace level message"""
        entry = self._create_log_entry(LogLevel.TRACE, message, operation, **kwargs)
        self._emit_log(entry)
    
    def debug(self, message: str, operation: str = "default", **kwargs):
        """Log debug level message"""
        entry = self._create_log_entry(LogLevel.DEBUG, message, operation, **kwargs)
        self._emit_log(entry)
    
    def info(self, message: str, operation: str = "default", **kwargs):
        """Log info level message"""
        entry = self._create_log_entry(LogLevel.INFO, message, operation, **kwargs)
        self._emit_log(entry)
    
    def warn(self, message: str, operation: str = "default", **kwargs):
        """Log warning level message"""
        entry = self._create_log_entry(LogLevel.WARN, message, operation, **kwargs)
        self._emit_log(entry)
    
    def error(self, message: str, operation: str = "default", 
              error: Optional[Exception] = None, **kwargs):
        """Log error level message"""
        if error:
            kwargs['error_details'] = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            }
        
        entry = self._create_log_entry(LogLevel.ERROR, message, operation, **kwargs)
        self._emit_log(entry)
    
    def critical(self, message: str, operation: str = "default", 
                error: Optional[Exception] = None, **kwargs):
        """Log critical level message"""
        if error:
            kwargs['error_details'] = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc()
            }
        
        entry = self._create_log_entry(LogLevel.CRITICAL, message, operation, **kwargs)
        self._emit_log(entry)
    
    def security(self, message: str, operation: str = "security", **kwargs):
        """Log security event"""
        entry = self._create_log_entry(LogLevel.SECURITY, message, operation, **kwargs)
        self._emit_log(entry)
    
    def audit(self, message: str, operation: str = "audit", **kwargs):
        """Log audit event"""
        entry = self._create_log_entry(LogLevel.AUDIT, message, operation, **kwargs)
        self._emit_log(entry)
    
    def _emit_log(self, entry: LogEntry):
        """Emit log entry to configured handlers"""
        # Convert to standard logging level
        level_mapping = {
            LogLevel.TRACE: 5,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARN: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
            LogLevel.SECURITY: logging.CRITICAL,
            LogLevel.AUDIT: logging.INFO
        }
        
        # Emit to Python logging system
        self.logger.log(level_mapping[entry.level], entry.message, 
                       extra={'structured_data': entry.to_dict()})
    
    @contextmanager
    def operation_context(self, operation: str, **context):
        """Context manager for operation logging"""
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        
        # Push context
        full_context = {'operation': operation, 'operation_id': operation_id, **context}
        self._context_stack.append(full_context)
        
        try:
            self.info(f"Starting operation: {operation}", operation=operation,
                     operation_id=operation_id)
            yield operation_id
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.error(f"Operation failed: {operation}", operation=operation,
                      operation_id=operation_id, duration_ms=duration_ms,
                      status="failed", error=e)
            raise
        else:
            duration_ms = (time.time() - start_time) * 1000
            self.info(f"Operation completed: {operation}", operation=operation,
                     operation_id=operation_id, duration_ms=duration_ms,
                     status="success")
        finally:
            # Pop context
            self._context_stack.pop()
    
    @contextmanager
    def user_context(self, user_id: str, session_id: str = None, **context):
        """Context manager for user-specific logging"""
        full_context = {
            'user_id': user_id,
            'session_id': session_id or str(uuid.uuid4()),
            **context
        }
        self._context_stack.append(full_context)
        
        try:
            yield
        finally:
            self._context_stack.pop()
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{uuid.uuid4()}"
        self._operation_timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str, operation: str = "timed_operation", **kwargs):
        """End timing and log duration"""
        if timer_id in self._operation_timers:
            start_time = self._operation_timers.pop(timer_id)
            duration_ms = (time.time() - start_time) * 1000
            
            self.info(f"Operation timing: {operation}", 
                     operation=operation, duration_ms=duration_ms, **kwargs)
            
            return duration_ms
        return None


class MetricsCollector:
    """High-performance metrics collection system"""
    
    def __init__(self):
        self._metrics: List[MetricEntry] = []
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def counter(self, name: str, value: float = 1, tags: Optional[Dict[str, str]] = None):
        """Record counter metric"""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
            
            metric = MetricEntry(
                name=name,
                value=value,
                metric_type=MetricType.COUNTER,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self._metrics.append(metric)
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record gauge metric"""
        with self._lock:
            self._gauges[name] = value
            
            metric = MetricEntry(
                name=name,
                value=value,
                metric_type=MetricType.GAUGE,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self._metrics.append(metric)
    
    def histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record histogram metric"""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)
            
            metric = MetricEntry(
                name=name,
                value=value,
                metric_type=MetricType.HISTOGRAM,
                timestamp=datetime.now(),
                tags=tags or {}
            )
            self._metrics.append(metric)
    
    def timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        """Record timer metric"""
        with self._lock:
            metric = MetricEntry(
                name=name,
                value=duration_ms,
                metric_type=MetricType.TIMER,
                timestamp=datetime.now(),
                tags=tags or {},
                unit="milliseconds"
            )
            self._metrics.append(metric)
    
    @contextmanager
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.timer(name, duration_ms, tags)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        with self._lock:
            histogram_stats = {}
            for name, values in self._histograms.items():
                if values:
                    histogram_stats[name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values)
                    }
            
            return {
                'counters': self._counters.copy(),
                'gauges': self._gauges.copy(),
                'histograms': histogram_stats,
                'total_metrics': len(self._metrics),
                'generated_at': datetime.now().isoformat()
            }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format"""
        with self._lock:
            if format.lower() == "json":
                return json.dumps([metric.to_dict() for metric in self._metrics], 
                                indent=2)
            elif format.lower() == "prometheus":
                return self._export_prometheus_format()
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format"""
        output = []
        
        for name, value in self._counters.items():
            output.append(f"# TYPE {name} counter")
            output.append(f"{name} {value}")
        
        for name, value in self._gauges.items():
            output.append(f"# TYPE {name} gauge")
            output.append(f"{name} {value}")
        
        return "\n".join(output)


class SystemMetricsCollector:
    """Collect system performance metrics"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self._collection_active = False
        self._collection_thread: Optional[threading.Thread] = None
    
    def start_collection(self, interval_seconds: int = 60):
        """Start automatic system metrics collection"""
        if self._collection_active:
            return
        
        self._collection_active = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._collection_thread.start()
    
    def stop_collection(self):
        """Stop automatic collection"""
        self._collection_active = False
        if self._collection_thread:
            self._collection_thread.join()
    
    def _collection_loop(self, interval_seconds: int):
        """Collection loop for system metrics"""
        while self._collection_active:
            try:
                self.collect_system_metrics()
                time.sleep(interval_seconds)
            except Exception as e:
                # Log error but continue collection
                logger = StructuredLogger("system_metrics", "monitoring")
                logger.error("System metrics collection failed", error=e)
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self):
        """Collect current system metrics"""
        tags = {
            "host": socket.gethostname(),
            "component": "system"
        }
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        self.metrics.gauge("system.cpu.usage_percent", cpu_percent, tags)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.gauge("system.memory.usage_percent", memory.percent, tags)
        self.metrics.gauge("system.memory.available_bytes", memory.available, tags)
        self.metrics.gauge("system.memory.used_bytes", memory.used, tags)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self.metrics.gauge("system.disk.usage_percent", disk_percent, tags)
        self.metrics.gauge("system.disk.free_bytes", disk.free, tags)
        
        # Process metrics
        process = psutil.Process()
        self.metrics.gauge("process.memory.rss_bytes", process.memory_info().rss, tags)
        self.metrics.gauge("process.cpu.usage_percent", process.cpu_percent(), tags)
        
        # File descriptor count (Unix-like systems)
        try:
            self.metrics.gauge("process.open_fds", process.num_fds(), tags)
        except (AttributeError, OSError):
            pass  # Not available on all systems


class LoggingSystem:
    """Comprehensive logging and monitoring system"""
    
    def __init__(self, 
                 log_dir: Path = None,
                 metrics_enabled: bool = True,
                 system_metrics_enabled: bool = True):
        
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize metrics
        self.metrics = MetricsCollector() if metrics_enabled else None
        self.system_metrics = (SystemMetricsCollector(self.metrics) 
                             if system_metrics_enabled and self.metrics else None)
        
        # Configure logging
        self._configure_logging()
        
        # Create main logger
        self.logger = StructuredLogger("system", "logging_system")
        
        # Start system metrics collection
        if self.system_metrics:
            self.system_metrics.start_collection()
    
    def _configure_logging(self):
        """Configure Python logging system"""
        
        class StructuredJSONFormatter(logging.Formatter):
            """JSON formatter for structured logging"""
            
            def format(self, record):
                # Check if this is a structured log entry
                if hasattr(record, 'structured_data'):
                    return json.dumps(record.structured_data)
                
                # Standard log record
                log_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'component': 'standard',
                    'operation': 'log',
                    'correlation_id': 'standard',
                    'filename': record.filename,
                    'line': record.lineno
                }
                
                if record.exc_info:
                    log_data['error_details'] = {
                        'stack_trace': self.formatException(record.exc_info)
                    }
                
                return json.dumps(log_data)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # JSON file handler
        json_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "application.json",
            maxBytes=100*1024*1024,  # 100MB
            backupCount=10
        )
        json_handler.setFormatter(StructuredJSONFormatter())
        root_logger.addHandler(json_handler)
        
        # Console handler (human readable)
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
        
        # Security/Audit handler
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.json",
            maxBytes=50*1024*1024,  # 50MB
            backupCount=20
        )
        security_handler.setFormatter(StructuredJSONFormatter())
        security_handler.setLevel(logging.CRITICAL)  # Only critical and above
        
        # Add security filter
        class SecurityFilter(logging.Filter):
            def filter(self, record):
                structured_data = getattr(record, 'structured_data', {})
                return structured_data.get('level') in ['SECURITY', 'AUDIT']
        
        security_handler.addFilter(SecurityFilter())
        root_logger.addHandler(security_handler)
    
    def get_logger(self, name: str, component: str = "application") -> StructuredLogger:
        """Get structured logger instance"""
        return StructuredLogger(name, component)
    
    def record_metric(self, name: str, value: float, metric_type: str = "counter",
                     tags: Optional[Dict[str, str]] = None):
        """Record metric if metrics are enabled"""
        if self.metrics:
            if metric_type == "counter":
                self.metrics.counter(name, value, tags)
            elif metric_type == "gauge":
                self.metrics.gauge(name, value, tags)
            elif metric_type == "histogram":
                self.metrics.histogram(name, value, tags)
            elif metric_type == "timer":
                self.metrics.timer(name, value, tags)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "logging_system": {
                "log_directory": str(self.log_dir),
                "active": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if self.metrics:
            status["metrics"] = self.metrics.get_metrics_summary()
        
        if self.system_metrics:
            status["system_monitoring"] = {
                "active": self.system_metrics._collection_active,
                "thread_alive": (self.system_metrics._collection_thread and 
                               self.system_metrics._collection_thread.is_alive())
            }
        
        return status
    
    def shutdown(self):
        """Shutdown logging system gracefully"""
        self.logger.info("Shutting down logging system")
        
        if self.system_metrics:
            self.system_metrics.stop_collection()
        
        # Flush all handlers
        for handler in logging.getLogger().handlers:
            handler.flush()


# Global logging system instance
logging_system = LoggingSystem(
    log_dir=Path("logs"),
    metrics_enabled=True,
    system_metrics_enabled=True
)

# Convenience function for getting loggers
def get_logger(name: str, component: str = "application") -> StructuredLogger:
    """Get structured logger instance"""
    return logging_system.get_logger(name, component)