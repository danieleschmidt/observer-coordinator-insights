#!/usr/bin/env python3
"""Generation 2 Robustness Implementation
Comprehensive error handling, validation, logging, monitoring, and health checks
"""

import asyncio
import json
import logging
import queue
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_avg: List[float]
    active_connections: int
    timestamp: datetime


class RobustHealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.checks = {}
        self.check_interval = 30  # seconds
        self.running = False
        self.health_thread = None

    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func

    async def run_check(self, name: str, check_func) -> HealthCheck:
        """Run a single health check with timing"""
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                asyncio.create_task(check_func()),
                timeout=10.0
            )
            response_time = (time.time() - start_time) * 1000

            return HealthCheck(
                component=name,
                status=HealthStatus.HEALTHY,
                message="Check passed",
                timestamp=datetime.now(),
                response_time_ms=response_time,
                details=result if isinstance(result, dict) else {}
            )
        except asyncio.TimeoutError:
            return HealthCheck(
                component=name,
                status=HealthStatus.CRITICAL,
                message="Health check timeout",
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return HealthCheck(
                component=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                timestamp=datetime.now(),
                response_time_ms=(time.time() - start_time) * 1000
            )

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks"""
        results = {}
        tasks = []

        for name, check_func in self.checks.items():
            task = self.run_check(name, check_func)
            tasks.append((name, task))

        for name, task in tasks:
            results[name] = await task

        return results

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        load_avg = list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]

        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            memory_available_mb=memory.available / (1024 * 1024),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024 * 1024 * 1024),
            load_avg=load_avg,
            active_connections=len(psutil.net_connections()),
            timestamp=datetime.now()
        )


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time < self.recovery_timeout:
                    raise Exception("Circuit breaker is OPEN")
                else:
                    self.state = 'HALF_OPEN'

            try:
                result = func(*args, **kwargs)
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                return result
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.warning(f"Circuit breaker OPEN for {func.__name__}")

                raise e


class RetryMechanism:
    """Retry mechanism with exponential backoff"""

    @staticmethod
    def with_retry(func, max_attempts: int = 3, backoff_factor: float = 2.0):
        """Execute function with retry logic"""
        for attempt in range(max_attempts):
            try:
                return func()
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e

                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)


class RobustLoggingManager:
    """Enhanced logging with structured logging and log rotation"""

    def __init__(self, log_level: str = "INFO", log_file: Optional[Path] = None):
        self.log_level = log_level.upper()
        self.log_file = log_file
        self.setup_logging()

    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                *([logging.handlers.RotatingFileHandler(
                    self.log_file,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )] if self.log_file else [])
            ]
        )

        # Structured logging for critical events
        self.audit_logger = logging.getLogger('audit')
        if self.log_file:
            audit_handler = logging.handlers.RotatingFileHandler(
                self.log_file.parent / 'audit.log',
                maxBytes=10*1024*1024,
                backupCount=10
            )
            audit_handler.setFormatter(logging.Formatter(
                '%(asctime)s - AUDIT - %(message)s'
            ))
            self.audit_logger.addHandler(audit_handler)

    def log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log structured audit event"""
        audit_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'details': details
        }
        self.audit_logger.info(json.dumps(audit_data))


class ValidationFramework:
    """Comprehensive data validation framework"""

    @staticmethod
    def validate_insights_data(data) -> Dict[str, Any]:
        """Comprehensive insights data validation"""
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'quality_score': 100.0,
            'validation_details': {}
        }

        required_columns = ['employee_id', 'red_energy', 'blue_energy', 'green_energy', 'yellow_energy']

        # Check required columns
        missing_cols = set(required_columns) - set(data.columns)
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['is_valid'] = False
            results['quality_score'] -= 30

        # Check data quality
        if results['is_valid']:
            # Check for null values
            null_counts = data.isnull().sum()
            if null_counts.any():
                results['warnings'].append(f"Null values found: {null_counts.to_dict()}")
                results['quality_score'] -= min(10, null_counts.sum())

            # Check energy column ranges (should be 0-100)
            energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
            for col in energy_cols:
                if col in data.columns:
                    if (data[col] < 0).any() or (data[col] > 100).any():
                        results['warnings'].append(f"{col} values outside 0-100 range")
                        results['quality_score'] -= 5

            # Check for duplicate employee IDs
            if data['employee_id'].duplicated().any():
                results['errors'].append("Duplicate employee IDs found")
                results['is_valid'] = False
                results['quality_score'] -= 20

        results['validation_details'] = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'null_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict()
        }

        return results


class PerformanceMonitor:
    """Performance monitoring and alerting"""

    def __init__(self):
        self.metrics_queue = queue.Queue(maxsize=1000)
        self.alerts = []
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'response_time_ms': 5000.0
        }

    def record_metric(self, metric_name: str, value: float, timestamp: datetime = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = datetime.now()

        metric = {
            'name': metric_name,
            'value': value,
            'timestamp': timestamp.isoformat()
        }

        try:
            self.metrics_queue.put_nowait(metric)
        except queue.Full:
            # Remove oldest metric if queue is full
            self.metrics_queue.get_nowait()
            self.metrics_queue.put_nowait(metric)

        # Check thresholds
        if metric_name in self.thresholds:
            if value > self.thresholds[metric_name]:
                self.alerts.append({
                    'type': 'threshold_exceeded',
                    'metric': metric_name,
                    'value': value,
                    'threshold': self.thresholds[metric_name],
                    'timestamp': timestamp.isoformat()
                })

    def get_recent_metrics(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get metrics from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_metrics = []

        # Convert queue to list to avoid blocking
        temp_metrics = []
        while not self.metrics_queue.empty():
            temp_metrics.append(self.metrics_queue.get())

        for metric in temp_metrics:
            metric_time = datetime.fromisoformat(metric['timestamp'])
            if metric_time >= cutoff_time:
                recent_metrics.append(metric)
            # Put back in queue
            try:
                self.metrics_queue.put_nowait(metric)
            except queue.Full:
                break

        return recent_metrics


@contextmanager
def robust_operation_context(operation_name: str):
    """Context manager for robust operation execution"""
    start_time = time.time()
    logger.info(f"Starting robust operation: {operation_name}")

    try:
        yield
        duration = time.time() - start_time
        logger.info(f"Robust operation '{operation_name}' completed successfully in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Robust operation '{operation_name}' failed after {duration:.2f}s: {e}")
        raise


# Global instances for Generation 2 robustness
health_checker = RobustHealthChecker()
performance_monitor = PerformanceMonitor()
logging_manager = RobustLoggingManager()


def initialize_gen2_robustness():
    """Initialize Generation 2 robustness features"""
    logger.info("🛡️ Initializing Generation 2 Robustness Features")

    # Register basic health checks
    async def database_health():
        """Mock database health check"""
        return {"status": "connected", "connections": 5}

    async def memory_health():
        """Memory health check"""
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            raise Exception(f"Memory usage too high: {memory.percent}%")
        return {"memory_percent": memory.percent}

    async def disk_health():
        """Disk space health check"""
        disk = psutil.disk_usage('/')
        if disk.percent > 95:
            raise Exception(f"Disk usage critical: {disk.percent}%")
        return {"disk_percent": disk.percent}

    health_checker.register_check('database', database_health)
    health_checker.register_check('memory', memory_health)
    health_checker.register_check('disk', disk_health)

    logger.info("✅ Generation 2 Robustness Features Initialized")


if __name__ == "__main__":
    # Demo the robustness features
    initialize_gen2_robustness()

    # Test health checks
    async def main():
        results = await health_checker.run_all_checks()
        for name, result in results.items():
            print(f"{name}: {result.status.value} - {result.message}")

        # Test system metrics
        metrics = health_checker.get_system_metrics()
        print(f"System Metrics: CPU: {metrics.cpu_percent}%, Memory: {metrics.memory_percent}%")

    asyncio.run(main())
