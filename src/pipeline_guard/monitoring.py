"""Pipeline Monitoring and Health Checking Components
"""

import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


# Optional psutil import for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .models import HealthMetric, PipelineComponent, SystemMetrics


@dataclass
class HealthMetric:
    """Health metric data point"""
    timestamp: float
    value: float
    component: str
    metric_type: str


@dataclass
class SystemMetrics:
    """System-wide metrics"""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    network_io: Dict[str, int]
    open_files: int
    timestamp: float


class PipelineMonitor:
    """Pipeline monitoring system for tracking component health and performance
    """

    def __init__(self, metric_retention_hours: int = 24):
        """Initialize pipeline monitor"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metric_retention_seconds = metric_retention_hours * 3600

        # Health check registry
        self.health_checks: Dict[str, Callable[[], bool]] = {}

        # Metrics storage
        self.health_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics: deque = deque(maxlen=1000)

        # Performance tracking
        self.component_performance: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Metric collection thread
        self.metrics_thread: Optional[threading.Thread] = None
        self.is_collecting = False

        self.logger.info("Pipeline monitor initialized")

    def add_component(self, component_name: str, health_check: Callable[[], bool]) -> None:
        """Add a component for monitoring"""
        self.health_checks[component_name] = health_check
        self.component_performance[component_name] = {
            'success_count': 0,
            'failure_count': 0,
            'avg_response_time': 0.0,
            'last_check': None
        }

        self.logger.info(f"Added component to monitoring: {component_name}")

    def start_metrics_collection(self, interval_seconds: int = 60) -> None:
        """Start background metrics collection"""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.metrics_thread = threading.Thread(
            target=self._collect_system_metrics,
            args=(interval_seconds,),
            daemon=True,
            name="MetricsCollector"
        )
        self.metrics_thread.start()

        self.logger.info("Started metrics collection")

    def stop_metrics_collection(self) -> None:
        """Stop metrics collection"""
        self.is_collecting = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=10)

    def _collect_system_metrics(self, interval_seconds: int) -> None:
        """Background system metrics collection"""
        while self.is_collecting:
            try:
                if PSUTIL_AVAILABLE:
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    disk = psutil.disk_usage('/')
                    network = psutil.net_io_counters()

                    metrics = SystemMetrics(
                        cpu_percent=cpu_percent,
                        memory_percent=memory.percent,
                        disk_usage=(disk.used / disk.total) * 100,
                        network_io={
                            'bytes_sent': network.bytes_sent,
                            'bytes_recv': network.bytes_recv
                        },
                        open_files=len(psutil.Process().open_files()),
                        timestamp=time.time()
                    )
                else:
                    # Fallback metrics when psutil not available
                    metrics = SystemMetrics(
                        cpu_percent=10.0,  # Mock value
                        memory_percent=50.0,  # Mock value
                        disk_usage=30.0,  # Mock value
                        network_io={'bytes_sent': 0, 'bytes_recv': 0},
                        open_files=10,
                        timestamp=time.time()
                    )

                self.system_metrics.append(metrics)

                # Clean old metrics
                self._cleanup_old_metrics()

            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")

            time.sleep(interval_seconds)

    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period"""
        cutoff_time = time.time() - self.metric_retention_seconds

        # Clean health metrics
        for component_name, metrics in self.health_metrics.items():
            while metrics and metrics[0].timestamp < cutoff_time:
                metrics.popleft()

        # Clean system metrics
        while self.system_metrics and self.system_metrics[0].timestamp < cutoff_time:
            self.system_metrics.popleft()

    def record_health_check(self, component_name: str, success: bool, response_time: float) -> None:
        """Record health check result"""
        metric = HealthMetric(
            timestamp=time.time(),
            value=1.0 if success else 0.0,
            component=component_name,
            metric_type="health_check"
        )

        self.health_metrics[component_name].append(metric)

        # Update performance stats
        perf = self.component_performance[component_name]
        if success:
            perf['success_count'] += 1
        else:
            perf['failure_count'] += 1

        # Update average response time
        if perf['avg_response_time'] == 0:
            perf['avg_response_time'] = response_time
        else:
            perf['avg_response_time'] = (perf['avg_response_time'] + response_time) / 2

        perf['last_check'] = time.time()

    def record_system_health(self, health_percentage: float) -> None:
        """Record overall system health percentage"""
        metric = HealthMetric(
            timestamp=time.time(),
            value=health_percentage,
            component="system",
            metric_type="system_health"
        )

        self.health_metrics["system"].append(metric)

    def get_component_health_trend(self, component_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get health trend for a component"""
        if component_name not in self.health_metrics:
            return {'error': 'Component not found'}

        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [
            m for m in self.health_metrics[component_name]
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {'error': 'No recent metrics'}

        values = [m.value for m in recent_metrics]

        return {
            'component': component_name,
            'period_hours': hours,
            'total_checks': len(recent_metrics),
            'success_rate': statistics.mean(values),
            'min_health': min(values),
            'max_health': max(values),
            'current_health': values[-1] if values else 0,
            'trend': 'improving' if len(values) > 1 and values[-1] > values[0] else 'stable'
        }

    def get_system_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive system performance summary"""
        if not self.system_metrics:
            return {'error': 'No system metrics available'}

        recent_metrics = list(self.system_metrics)[-60:]  # Last 60 readings

        cpu_values = [m.cpu_percent for m in recent_metrics]
        memory_values = [m.memory_percent for m in recent_metrics]
        disk_values = [m.disk_usage for m in recent_metrics]

        return {
            'system_health': {
                'cpu': {
                    'current': cpu_values[-1] if cpu_values else 0,
                    'average': statistics.mean(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0
                },
                'memory': {
                    'current': memory_values[-1] if memory_values else 0,
                    'average': statistics.mean(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0
                },
                'disk': {
                    'current': disk_values[-1] if disk_values else 0,
                    'average': statistics.mean(disk_values) if disk_values else 0,
                    'max': max(disk_values) if disk_values else 0
                }
            },
            'components': dict(self.component_performance),
            'metrics_count': len(recent_metrics),
            'collection_active': self.is_collecting
        }


class HealthChecker:
    """Component health checking system with configurable strategies
    """

    def __init__(self):
        """Initialize health checker"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.monitor = None  # Will be set by pipeline guard

        # Health check strategies
        self.strategies = {
            'basic': self._basic_health_check,
            'timeout': self._timeout_health_check,
            'retry': self._retry_health_check
        }

    def set_monitor(self, monitor: PipelineMonitor) -> None:
        """Set the monitor instance for recording metrics"""
        self.monitor = monitor

    def check_component_health(self, component: PipelineComponent, strategy: str = 'basic') -> bool:
        """Check component health using specified strategy"""
        start_time = time.time()

        try:
            if strategy not in self.strategies:
                strategy = 'basic'

            result = self.strategies[strategy](component)

            # Record metrics if monitor available
            if self.monitor:
                response_time = time.time() - start_time
                self.monitor.record_health_check(component.name, result, response_time)

            return result

        except Exception as e:
            self.logger.error(f"Health check failed for {component.name}: {e}")

            if self.monitor:
                response_time = time.time() - start_time
                self.monitor.record_health_check(component.name, False, response_time)

            return False

    def _basic_health_check(self, component: PipelineComponent) -> bool:
        """Basic health check - just call the component's health check"""
        return component.health_check()

    def _timeout_health_check(self, component: PipelineComponent, timeout: int = 30) -> bool:
        """Health check with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            result = component.health_check()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return result
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            self.logger.warning(f"Health check timeout for {component.name}")
            return False
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            raise e

    def _retry_health_check(self, component: PipelineComponent, max_retries: int = 3) -> bool:
        """Health check with retries"""
        for attempt in range(max_retries):
            try:
                result = component.health_check()
                if result:
                    return True

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                self.logger.warning(f"Health check attempt {attempt + 1} failed for {component.name}: {e}")

                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return False

    def batch_health_check(self, components: List[PipelineComponent]) -> Dict[str, bool]:
        """Check health of multiple components"""
        results = {}

        for component in components:
            results[component.name] = self.check_component_health(component)

        return results

    def get_health_summary(self, components: List[PipelineComponent]) -> Dict[str, Any]:
        """Get comprehensive health summary"""
        results = self.batch_health_check(components)

        total = len(results)
        healthy = sum(1 for r in results.values() if r)

        return {
            'total_components': total,
            'healthy_components': healthy,
            'health_percentage': (healthy / total * 100) if total > 0 else 0,
            'component_status': results,
            'timestamp': time.time()
        }
