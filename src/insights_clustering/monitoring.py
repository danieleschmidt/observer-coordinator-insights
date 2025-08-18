"""Clustering-Specific Monitoring and Performance Metrics
Provides detailed monitoring for neuromorphic clustering operations with OpenTelemetry compatibility
"""

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class ClusteringPhase(Enum):
    """Phases of clustering operation for detailed monitoring"""
    INITIALIZATION = "initialization"
    FEATURE_EXTRACTION = "feature_extraction"
    NEUROMORPHIC_PROCESSING = "neuromorphic_processing"
    CLUSTERING = "clustering"
    VALIDATION = "validation"
    FALLBACK = "fallback"


class MetricType(Enum):
    """Types of clustering metrics"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    RESOURCE = "resource"
    BUSINESS = "business"


@dataclass
class ClusteringOperationMetrics:
    """Comprehensive metrics for a single clustering operation"""
    operation_id: str
    correlation_id: str
    method: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"

    # Input characteristics
    dataset_size: int = 0
    feature_dimensions: int = 0
    target_clusters: int = 0

    # Performance metrics
    total_duration_ms: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)
    memory_peak_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    # Quality metrics
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    inertia: Optional[float] = None
    cluster_stability: Optional[float] = None

    # Neuromorphic-specific metrics
    neuromorphic_components_used: List[str] = field(default_factory=list)
    feature_extraction_failures: int = 0
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    circuit_breaker_trips: int = 0

    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_phase_duration(self, phase: ClusteringPhase, duration_ms: float):
        """Add duration for a specific phase"""
        self.phase_durations[phase.value] = duration_ms

    def add_error(self, error_type: str, message: str, recoverable: bool = True):
        """Add error information"""
        self.errors.append({
            'timestamp': datetime.utcnow().isoformat(),
            'type': error_type,
            'message': message,
            'recoverable': recoverable
        })

    def complete_operation(self, status: str = "success"):
        """Mark operation as completed"""
        self.completed_at = datetime.utcnow()
        self.status = status
        if self.started_at:
            self.total_duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000


@dataclass
class ClusteringPerformanceBaseline:
    """Performance baseline for clustering operations"""
    method: str
    dataset_size_bucket: str
    baseline_duration_ms: float
    baseline_quality_score: float
    sample_count: int
    last_updated: datetime
    confidence_interval: float = 0.95


class ClusteringMonitor:
    """Advanced monitoring for clustering operations with detailed analytics"""

    def __init__(self, max_history_size: int = 1000):
        self.active_operations: Dict[str, ClusteringOperationMetrics] = {}
        self.completed_operations: deque = deque(maxlen=max_history_size)
        self.performance_baselines: Dict[str, ClusteringPerformanceBaseline] = {}
        self.alert_thresholds = self._default_alert_thresholds()
        self._lock = threading.Lock()

        # Aggregated statistics
        self.statistics = {
            'total_operations': 0,
            'success_rate': 0.0,
            'average_duration_ms': 0.0,
            'fallback_rate': 0.0,
            'quality_trends': defaultdict(list)
        }

    def _default_alert_thresholds(self) -> Dict[str, Any]:
        """Define default alert thresholds for clustering operations"""
        return {
            'max_duration_ms': 300000,  # 5 minutes
            'min_silhouette_score': 0.3,
            'max_memory_mb': 2048,  # 2GB
            'max_failure_rate': 0.1,  # 10%
            'max_fallback_rate': 0.2   # 20%
        }

    @contextmanager
    def monitor_operation(self, method: str, correlation_id: str = None,
                         dataset_size: int = 0, target_clusters: int = 0):
        """Context manager for monitoring clustering operations"""
        operation_id = str(uuid.uuid4())
        correlation_id = correlation_id or str(uuid.uuid4())

        metrics = ClusteringOperationMetrics(
            operation_id=operation_id,
            correlation_id=correlation_id,
            method=method,
            started_at=datetime.utcnow(),
            dataset_size=dataset_size,
            target_clusters=target_clusters
        )

        with self._lock:
            self.active_operations[operation_id] = metrics
            self.statistics['total_operations'] += 1

        try:
            yield metrics
            metrics.complete_operation("success")

        except Exception as e:
            metrics.complete_operation("failed")
            metrics.add_error(type(e).__name__, str(e), recoverable=False)
            raise

        finally:
            with self._lock:
                # Move to completed operations
                if operation_id in self.active_operations:
                    completed_metrics = self.active_operations.pop(operation_id)
                    self.completed_operations.append(completed_metrics)

                    # Update statistics
                    self._update_statistics(completed_metrics)

                    # Check for alerts
                    self._check_alerts(completed_metrics)

                    logger.info(f"Clustering operation {operation_id} completed: {completed_metrics.status}")

    @contextmanager
    def monitor_phase(self, operation_id: str, phase: ClusteringPhase):
        """Context manager for monitoring specific phases"""
        start_time = time.time()

        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000

            with self._lock:
                if operation_id in self.active_operations:
                    self.active_operations[operation_id].add_phase_duration(phase, duration_ms)
                    logger.debug(f"Phase {phase.value} completed in {duration_ms:.2f}ms")

    def record_neuromorphic_metrics(self, operation_id: str, component: str,
                                  duration_ms: float, success: bool,
                                  feature_count: int = 0):
        """Record neuromorphic-specific metrics"""
        with self._lock:
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]

                if success:
                    operation.neuromorphic_components_used.append(component)
                else:
                    operation.feature_extraction_failures += 1
                    operation.add_error("neuromorphic_failure",
                                      f"Component {component} failed", recoverable=True)

    def record_fallback(self, operation_id: str, reason: str, fallback_method: str):
        """Record fallback operation"""
        with self._lock:
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                operation.fallback_triggered = True
                operation.fallback_reason = reason
                operation.warnings.append(f"Fallback to {fallback_method}: {reason}")

                logger.warning(f"Fallback triggered for operation {operation_id}: {reason}")

    def record_quality_metrics(self, operation_id: str, metrics: Dict[str, float]):
        """Record clustering quality metrics"""
        with self._lock:
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]

                operation.silhouette_score = metrics.get('silhouette_score')
                operation.calinski_harabasz_score = metrics.get('calinski_harabasz_score')
                operation.davies_bouldin_score = metrics.get('davies_bouldin_score')
                operation.inertia = metrics.get('inertia')
                operation.cluster_stability = metrics.get('cluster_stability')

    def record_resource_usage(self, operation_id: str, memory_mb: float, cpu_percent: float):
        """Record resource usage during operation"""
        with self._lock:
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                operation.memory_peak_mb = max(operation.memory_peak_mb, memory_mb)
                operation.cpu_usage_percent = max(operation.cpu_usage_percent, cpu_percent)

    def _update_statistics(self, metrics: ClusteringOperationMetrics):
        """Update aggregated statistics"""
        # Success rate
        successful_ops = sum(1 for op in self.completed_operations if op.status == "success")
        self.statistics['success_rate'] = successful_ops / len(self.completed_operations) if self.completed_operations else 0

        # Average duration
        durations = [op.total_duration_ms for op in self.completed_operations if op.total_duration_ms > 0]
        self.statistics['average_duration_ms'] = np.mean(durations) if durations else 0

        # Fallback rate
        fallback_ops = sum(1 for op in self.completed_operations if op.fallback_triggered)
        self.statistics['fallback_rate'] = fallback_ops / len(self.completed_operations) if self.completed_operations else 0

        # Quality trends
        if metrics.silhouette_score is not None:
            self.statistics['quality_trends']['silhouette'].append(metrics.silhouette_score)
            if len(self.statistics['quality_trends']['silhouette']) > 100:
                self.statistics['quality_trends']['silhouette'] = self.statistics['quality_trends']['silhouette'][-100:]

    def _check_alerts(self, metrics: ClusteringOperationMetrics):
        """Check if metrics exceed alert thresholds"""
        alerts = []

        # Duration alert
        if metrics.total_duration_ms > self.alert_thresholds['max_duration_ms']:
            alerts.append(f"Operation exceeded maximum duration: {metrics.total_duration_ms:.0f}ms")

        # Quality alert
        if (metrics.silhouette_score is not None and
            metrics.silhouette_score < self.alert_thresholds['min_silhouette_score']):
            alerts.append(f"Low clustering quality: silhouette score {metrics.silhouette_score:.3f}")

        # Memory alert
        if metrics.memory_peak_mb > self.alert_thresholds['max_memory_mb']:
            alerts.append(f"High memory usage: {metrics.memory_peak_mb:.0f}MB")

        # Failure rate alert
        if self.statistics['success_rate'] < (1 - self.alert_thresholds['max_failure_rate']):
            alerts.append(f"High failure rate: {(1-self.statistics['success_rate'])*100:.1f}%")

        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT [{metrics.operation_id}]: {alert}")

    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific operation"""
        with self._lock:
            if operation_id in self.active_operations:
                return asdict(self.active_operations[operation_id])

        # Check completed operations
        for op in self.completed_operations:
            if op.operation_id == operation_id:
                return asdict(op)

        return None

    def get_performance_dashboard(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance dashboard"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        with self._lock:
            recent_operations = [
                op for op in self.completed_operations
                if op.completed_at and op.completed_at > cutoff_time
            ]

            if not recent_operations:
                return {
                    'period_hours': hours,
                    'message': 'No operations in specified period'
                }

            # Calculate metrics
            successful_ops = [op for op in recent_operations if op.status == "success"]
            failed_ops = [op for op in recent_operations if op.status == "failed"]
            fallback_ops = [op for op in recent_operations if op.fallback_triggered]

            # Performance metrics
            durations = [op.total_duration_ms for op in successful_ops if op.total_duration_ms > 0]
            memory_usage = [op.memory_peak_mb for op in recent_operations if op.memory_peak_mb > 0]

            # Quality metrics
            quality_scores = {
                'silhouette': [op.silhouette_score for op in successful_ops if op.silhouette_score is not None],
                'calinski_harabasz': [op.calinski_harabasz_score for op in successful_ops if op.calinski_harabasz_score is not None]
            }

            # Method breakdown
            method_stats = defaultdict(lambda: {'count': 0, 'success': 0, 'avg_duration': 0})
            for op in recent_operations:
                method_stats[op.method]['count'] += 1
                if op.status == "success":
                    method_stats[op.method]['success'] += 1
                if op.total_duration_ms > 0:
                    method_stats[op.method]['avg_duration'] += op.total_duration_ms

            # Calculate averages
            for method, stats in method_stats.items():
                if stats['count'] > 0:
                    stats['success_rate'] = stats['success'] / stats['count']
                    stats['avg_duration'] = stats['avg_duration'] / stats['count']

            return {
                'period_hours': hours,
                'summary': {
                    'total_operations': len(recent_operations),
                    'success_rate': len(successful_ops) / len(recent_operations),
                    'failure_rate': len(failed_ops) / len(recent_operations),
                    'fallback_rate': len(fallback_ops) / len(recent_operations),
                    'avg_duration_ms': np.mean(durations) if durations else 0,
                    'p95_duration_ms': np.percentile(durations, 95) if durations else 0,
                    'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0
                },
                'quality_metrics': {
                    'avg_silhouette_score': np.mean(quality_scores['silhouette']) if quality_scores['silhouette'] else None,
                    'avg_calinski_harabasz': np.mean(quality_scores['calinski_harabasz']) if quality_scores['calinski_harabasz'] else None
                },
                'method_breakdown': dict(method_stats),
                'trends': {
                    'duration_trend': self._calculate_trend([op.total_duration_ms for op in successful_ops[-10:]] if len(successful_ops) >= 10 else []),
                    'quality_trend': self._calculate_trend(quality_scores['silhouette'][-10:] if len(quality_scores['silhouette']) >= 10 else [])
                },
                'alerts': {
                    'high_duration_operations': len([op for op in recent_operations if op.total_duration_ms > self.alert_thresholds['max_duration_ms']]),
                    'low_quality_operations': len([op for op in recent_operations if op.silhouette_score is not None and op.silhouette_score < self.alert_thresholds['min_silhouette_score']]),
                    'high_memory_operations': len([op for op in recent_operations if op.memory_peak_mb > self.alert_thresholds['max_memory_mb']])
                }
            }

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values"""
        if len(values) < 2:
            return "stable"

        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0

        if slope > 0.1:
            return "improving"
        elif slope < -0.1:
            return "declining"
        else:
            return "stable"

    def export_metrics(self, format_type: str = "json", hours: int = 24) -> str:
        """Export metrics for external analysis"""
        dashboard = self.get_performance_dashboard(hours)

        if format_type.lower() == "json":
            return json.dumps(dashboard, indent=2, default=str)
        elif format_type.lower() == "csv":
            # Simple CSV export of recent operations
            with self._lock:
                cutoff_time = datetime.utcnow() - timedelta(hours=hours)
                recent_ops = [op for op in self.completed_operations if op.completed_at and op.completed_at > cutoff_time]

                if not recent_ops:
                    return "No data available for export"

                csv_lines = ["operation_id,method,status,duration_ms,dataset_size,silhouette_score,fallback_triggered"]
                for op in recent_ops:
                    csv_lines.append(f"{op.operation_id},{op.method},{op.status},{op.total_duration_ms},"
                                   f"{op.dataset_size},{op.silhouette_score or ''},"
                                   f"{op.fallback_triggered}")
                return "\n".join(csv_lines)
        else:
            return f"Unsupported format: {format_type}"


# Global clustering monitor instance
clustering_monitor = ClusteringMonitor()


def monitor_clustering_operation(method: str, **kwargs):
    """Decorator for automatic clustering operation monitoring"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **func_kwargs):
            correlation_id = func_kwargs.pop('correlation_id', None)
            dataset_size = func_kwargs.pop('dataset_size', 0)
            target_clusters = func_kwargs.pop('target_clusters', 0)

            with clustering_monitor.monitor_operation(
                method=method,
                correlation_id=correlation_id,
                dataset_size=dataset_size,
                target_clusters=target_clusters
            ) as metrics:
                result = func(*args, **func_kwargs)

                # Extract quality metrics if available
                if hasattr(result, 'get_clustering_metrics'):
                    quality_metrics = result.get_clustering_metrics()
                    clustering_monitor.record_quality_metrics(
                        metrics.operation_id,
                        asdict(quality_metrics)
                    )

                return result
        return wrapper
    return decorator
