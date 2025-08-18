"""Resilience and Reliability Module for Neuromorphic Clustering
Provides circuit breakers, fallback mechanisms, resource monitoring, and quality gates
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil


logger = logging.getLogger(__name__)


class ResilienceStatus(Enum):
    """Status levels for resilience components"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CRITICAL = "critical"


class ResourceType(Enum):
    """Types of system resources to monitor"""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class ResourceThresholds:
    """Thresholds for resource monitoring and alerting"""
    warning_threshold: float
    critical_threshold: float
    unit: str
    check_interval_seconds: int = 30


@dataclass
class QualityGate:
    """Quality gate configuration for clustering results"""
    name: str
    metric_type: str  # 'silhouette', 'calinski_harabasz', 'stability', etc.
    min_threshold: float
    max_threshold: Optional[float] = None
    weight: float = 1.0
    enabled: bool = True


@dataclass
class ResilienceMetrics:
    """Metrics for resilience monitoring"""
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    fallback_operations: int = 0
    circuit_breaker_trips: int = 0
    resource_alerts: int = 0
    quality_gate_failures: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_operations == 0:
            return 1.0
        return self.successful_operations / self.total_operations

    @property
    def fallback_rate(self) -> float:
        """Calculate fallback rate"""
        if self.total_operations == 0:
            return 0.0
        return self.fallback_operations / self.total_operations


class ResourceMonitor:
    """Monitors system resources and triggers alerts"""

    def __init__(self, thresholds: Optional[Dict[ResourceType, ResourceThresholds]] = None):
        self.thresholds = thresholds or self._default_thresholds()
        self.current_status = ResilienceStatus.HEALTHY
        self.alerts = []
        self._lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None

    def _default_thresholds(self) -> Dict[ResourceType, ResourceThresholds]:
        """Define default resource thresholds"""
        return {
            ResourceType.MEMORY: ResourceThresholds(
                warning_threshold=80.0,
                critical_threshold=95.0,
                unit="percent"
            ),
            ResourceType.CPU: ResourceThresholds(
                warning_threshold=70.0,
                critical_threshold=90.0,
                unit="percent"
            ),
            ResourceType.DISK: ResourceThresholds(
                warning_threshold=85.0,
                critical_threshold=95.0,
                unit="percent"
            )
        }

    def start_monitoring(self):
        """Start continuous resource monitoring"""
        if self._monitoring:
            return

        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._monitoring:
            try:
                self._check_resources()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_resources(self):
        """Check current resource usage against thresholds"""
        with self._lock:
            # Memory usage
            memory = psutil.virtual_memory()
            self._check_threshold(
                ResourceType.MEMORY,
                memory.percent,
                f"Memory usage: {memory.percent:.1f}%"
            )

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._check_threshold(
                ResourceType.CPU,
                cpu_percent,
                f"CPU usage: {cpu_percent:.1f}%"
            )

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self._check_threshold(
                ResourceType.DISK,
                disk_percent,
                f"Disk usage: {disk_percent:.1f}%"
            )

    def _check_threshold(self, resource_type: ResourceType, value: float, message: str):
        """Check if resource value exceeds thresholds"""
        threshold = self.thresholds.get(resource_type)
        if not threshold:
            return

        alert_level = None
        if value >= threshold.critical_threshold:
            alert_level = "critical"
            self.current_status = ResilienceStatus.CRITICAL
        elif value >= threshold.warning_threshold:
            alert_level = "warning"
            if self.current_status == ResilienceStatus.HEALTHY:
                self.current_status = ResilienceStatus.DEGRADED

        if alert_level:
            alert = {
                'timestamp': datetime.utcnow(),
                'resource_type': resource_type.value,
                'level': alert_level,
                'value': value,
                'threshold': threshold.critical_threshold if alert_level == "critical" else threshold.warning_threshold,
                'message': message
            }
            self.alerts.append(alert)
            logger.warning(f"Resource alert: {message} (threshold: {alert['threshold']}%)")

    def get_current_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        with self._lock:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            return {
                'status': self.current_status.value,
                'timestamp': datetime.utcnow().isoformat(),
                'resources': {
                    'memory': {
                        'usage_percent': memory.percent,
                        'available_gb': memory.available / (1024**3),
                        'total_gb': memory.total / (1024**3)
                    },
                    'cpu': {
                        'usage_percent': cpu_percent,
                        'core_count': psutil.cpu_count()
                    },
                    'disk': {
                        'usage_percent': disk_percent,
                        'free_gb': disk.free / (1024**3),
                        'total_gb': disk.total / (1024**3)
                    }
                },
                'recent_alerts': self.alerts[-10:] if self.alerts else []
            }


class ClusteringQualityGates:
    """Quality gates for validating clustering results"""

    def __init__(self, gates: Optional[List[QualityGate]] = None):
        self.gates = gates or self._default_quality_gates()
        self.validation_history = []

    def _default_quality_gates(self) -> List[QualityGate]:
        """Define default quality gates"""
        return [
            QualityGate("silhouette_score", "silhouette", 0.3, weight=0.4),
            QualityGate("calinski_harabasz", "calinski_harabasz", 50.0, weight=0.3),
            QualityGate("cluster_stability", "stability", 0.7, weight=0.3),
            QualityGate("minimum_cluster_size", "min_cluster_size", 2, weight=0.2)
        ]

    def validate_clustering_results(self, metrics: Dict[str, float],
                                  cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Validate clustering results against quality gates"""
        validation_result = {
            'passed': True,
            'overall_score': 0.0,
            'gate_results': [],
            'timestamp': datetime.utcnow(),
            'cluster_count': len(np.unique(cluster_labels)),
            'sample_count': len(cluster_labels)
        }

        total_weight = 0.0
        weighted_score = 0.0

        for gate in self.gates:
            if not gate.enabled:
                continue

            gate_result = self._evaluate_gate(gate, metrics, cluster_labels)
            validation_result['gate_results'].append(gate_result)

            if not gate_result['passed']:
                validation_result['passed'] = False

            total_weight += gate.weight
            weighted_score += gate_result['score'] * gate.weight

        if total_weight > 0:
            validation_result['overall_score'] = weighted_score / total_weight

        # Store in history
        self.validation_history.append(validation_result)
        if len(self.validation_history) > 1000:  # Keep last 1000 validations
            self.validation_history = self.validation_history[-1000:]

        return validation_result

    def _evaluate_gate(self, gate: QualityGate, metrics: Dict[str, float],
                      cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate a single quality gate"""
        result = {
            'gate_name': gate.name,
            'metric_type': gate.metric_type,
            'passed': False,
            'score': 0.0,
            'actual_value': None,
            'threshold': gate.min_threshold,
            'message': ''
        }

        try:
            if gate.metric_type in metrics:
                actual_value = metrics[gate.metric_type]
            elif gate.metric_type == "min_cluster_size":
                unique_labels, counts = np.unique(cluster_labels, return_counts=True)
                actual_value = np.min(counts)
            else:
                result['message'] = f"Metric {gate.metric_type} not available"
                return result

            result['actual_value'] = actual_value

            # Check thresholds
            if gate.max_threshold is not None:
                # Value should be within range
                if gate.min_threshold <= actual_value <= gate.max_threshold:
                    result['passed'] = True
                    result['score'] = 1.0
                    result['message'] = f"Value {actual_value:.3f} within acceptable range"
                else:
                    result['message'] = f"Value {actual_value:.3f} outside range [{gate.min_threshold}, {gate.max_threshold}]"
            # Value should exceed minimum threshold
            elif actual_value >= gate.min_threshold:
                result['passed'] = True
                result['score'] = min(1.0, actual_value / gate.min_threshold)
                result['message'] = f"Value {actual_value:.3f} meets minimum threshold"
            else:
                result['score'] = actual_value / gate.min_threshold if gate.min_threshold > 0 else 0
                result['message'] = f"Value {actual_value:.3f} below minimum threshold {gate.min_threshold}"

        except Exception as e:
            result['message'] = f"Error evaluating gate: {e!s}"
            logger.error(f"Quality gate evaluation error for {gate.name}: {e}")

        return result

    def get_quality_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get quality trends over time"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_validations = [
            v for v in self.validation_history
            if v['timestamp'] > cutoff_time
        ]

        if not recent_validations:
            return {'message': 'No recent validations', 'trend': 'stable'}

        scores = [v['overall_score'] for v in recent_validations]
        pass_rates = [1 if v['passed'] else 0 for v in recent_validations]

        return {
            'validation_count': len(recent_validations),
            'average_score': np.mean(scores),
            'score_trend': 'improving' if len(scores) > 1 and scores[-1] > scores[0] else 'declining',
            'pass_rate': np.mean(pass_rates),
            'latest_score': scores[-1] if scores else 0,
            'score_range': {
                'min': np.min(scores),
                'max': np.max(scores),
                'std': np.std(scores)
            }
        }


class ResilienceManager:
    """Central manager for all resilience components"""

    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.quality_gates = ClusteringQualityGates()
        self.metrics = ResilienceMetrics()
        self._lock = threading.Lock()

    def start(self):
        """Start all resilience monitoring"""
        self.resource_monitor.start_monitoring()
        logger.info("Resilience manager started")

    def stop(self):
        """Stop all resilience monitoring"""
        self.resource_monitor.stop_monitoring()
        logger.info("Resilience manager stopped")

    @contextmanager
    def operation_context(self, operation_name: str):
        """Context manager for tracking operation resilience"""
        start_time = time.time()
        success = False

        with self._lock:
            self.metrics.total_operations += 1

        try:
            yield
            success = True
            with self._lock:
                self.metrics.successful_operations += 1

        except Exception as e:
            with self._lock:
                self.metrics.failed_operations += 1
            logger.error(f"Operation {operation_name} failed: {e}")
            raise

        finally:
            duration = time.time() - start_time
            logger.info(f"Operation {operation_name} completed in {duration:.2f}s, success: {success}")

    def record_fallback_operation(self, primary_method: str, fallback_method: str):
        """Record when fallback mechanism is used"""
        with self._lock:
            self.metrics.fallback_operations += 1
        logger.info(f"Fallback triggered: {primary_method} -> {fallback_method}")

    def record_circuit_breaker_trip(self, component: str):
        """Record circuit breaker activation"""
        with self._lock:
            self.metrics.circuit_breaker_trips += 1
        logger.warning(f"Circuit breaker tripped for component: {component}")

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive resilience status"""
        resource_status = self.resource_monitor.get_current_status()
        quality_trends = self.quality_gates.get_quality_trends()

        return {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': self._calculate_overall_status(resource_status),
            'metrics': {
                'success_rate': self.metrics.success_rate,
                'fallback_rate': self.metrics.fallback_rate,
                'total_operations': self.metrics.total_operations,
                'circuit_breaker_trips': self.metrics.circuit_breaker_trips
            },
            'resources': resource_status,
            'quality': quality_trends,
            'recommendations': self._generate_recommendations(resource_status, quality_trends)
        }

    def _calculate_overall_status(self, resource_status: Dict[str, Any]) -> str:
        """Calculate overall system status"""
        if resource_status['status'] == ResilienceStatus.CRITICAL.value:
            return 'critical'
        elif self.metrics.success_rate < 0.8 or resource_status['status'] == ResilienceStatus.DEGRADED.value:
            return 'degraded'
        else:
            return 'healthy'

    def _generate_recommendations(self, resource_status: Dict[str, Any],
                                quality_trends: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Resource-based recommendations
        if resource_status['status'] == 'critical':
            recommendations.append("URGENT: Scale up infrastructure immediately")
        elif resource_status['status'] == 'degraded':
            recommendations.append("Consider scaling up resources soon")

        # Performance-based recommendations
        if self.metrics.success_rate < 0.9:
            recommendations.append("Investigate recent failures and improve error handling")

        if self.metrics.fallback_rate > 0.2:
            recommendations.append("High fallback rate detected - review primary method reliability")

        # Quality-based recommendations
        if quality_trends.get('pass_rate', 1.0) < 0.8:
            recommendations.append("Quality gates failing frequently - review clustering parameters")

        if not recommendations:
            recommendations.append("System operating within normal parameters")

        return recommendations


# Global resilience manager instance
resilience_manager = ResilienceManager()


def resilient_operation(operation_name: str):
    """Decorator for resilient operation tracking"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with resilience_manager.operation_context(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
