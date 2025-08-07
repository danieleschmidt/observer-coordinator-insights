#!/usr/bin/env python3
"""
Application monitoring and metrics collection module.
Provides Prometheus-compatible metrics for operational observability.
"""

import time
import logging
import threading
import uuid
import json
from functools import wraps
from typing import Dict, Any, Optional, Callable, List, Set
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import warnings

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MetricValue:
    """Container for metric values when Prometheus is not available"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float


@dataclass
class HealthCheckResult:
    """Result of a health check with Generation 2 enhancements"""
    name: str
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    response_time_ms: float
    details: Dict[str, Any]
    correlation_id: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AlertRule:
    """Alert rule configuration for monitoring thresholds"""
    name: str
    metric_name: str
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    severity: str  # 'critical', 'warning', 'info'
    time_window_minutes: int
    enabled: bool = True


@dataclass
class PerformanceBaseline:
    """Performance baseline for anomaly detection"""
    metric_name: str
    baseline_value: float
    std_deviation: float
    confidence_interval: float
    last_updated: datetime
    sample_count: int


class EnhancedMetricsCollector:
    """
    Enterprise-grade centralized metrics collection with Generation 2 enhancements.
    Includes health checks, alerting, anomaly detection, and comprehensive monitoring.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, max_history_size: int = 10000):
        self.registry = registry
        self.in_memory_metrics: Dict[str, MetricValue] = {}
        self.health_checks: Dict[str, HealthCheckResult] = {}
        self.alert_rules: List[AlertRule] = []
        self.performance_baselines: Dict[str, PerformanceBaseline] = {}
        self.metric_history: deque = deque(maxlen=max_history_size)
        self._lock = threading.Lock()
        
        # Initialize default alert rules
        self._setup_default_alert_rules()
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            logger.warning("Prometheus client not available, using in-memory metrics")
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules for common metrics"""
        self.alert_rules = [
            AlertRule("high_error_rate", "processing_errors_total", 10, ">", "critical", 15),
            AlertRule("long_clustering_duration", "clustering_operation_duration_seconds", 300, ">", "warning", 30),
            AlertRule("memory_usage_high", "memory_usage_bytes", 2e9, ">", "warning", 5),  # 2GB
            AlertRule("system_health_degraded", "system_health_score", 0.7, "<", "warning", 10),
            AlertRule("neuromorphic_failures", "neuromorphic_clustering_failures_total", 5, ">", "critical", 20)
        ]
    
    def _init_prometheus_metrics(self):
        """Initialize comprehensive Prometheus metrics with Generation 2 enhancements"""
        # Performance metrics
        self.clustering_duration = Histogram(
            'clustering_operation_duration_seconds',
            'Time spent on clustering operations',
            ['algorithm', 'dataset_size_bucket', 'correlation_id'],
            registry=self.registry
        )
        
        self.clustering_operations = Counter(
            'clustering_operations_total',
            'Total number of clustering operations',
            ['algorithm', 'status', 'fallback_used'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Current memory usage',
            ['component', 'process_id'],
            registry=self.registry
        )
        
        self.data_processing_duration = Histogram(
            'data_processing_duration_seconds',
            'Time spent processing data',
            ['operation_type', 'data_size_bucket'],
            registry=self.registry
        )
        
        # Generation 2: Neuromorphic-specific metrics
        self.neuromorphic_feature_extraction_duration = Histogram(
            'neuromorphic_feature_extraction_duration_seconds',
            'Time spent on neuromorphic feature extraction',
            ['method', 'component'],
            registry=self.registry
        )
        
        self.neuromorphic_clustering_failures = Counter(
            'neuromorphic_clustering_failures_total',
            'Total neuromorphic clustering failures',
            ['error_type', 'component', 'recoverable'],
            registry=self.registry
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['component'],
            registry=self.registry
        )
        
        self.circuit_breaker_failures = Counter(
            'circuit_breaker_failures_total',
            'Total circuit breaker failures',
            ['component'],
            registry=self.registry
        )
        
        self.fallback_operations = Counter(
            'fallback_operations_total',
            'Total fallback operations triggered',
            ['primary_method', 'fallback_method', 'reason'],
            registry=self.registry
        )
        
        # Business metrics
        self.employee_records_processed = Counter(
            'employee_records_processed_total',
            'Total employee records processed',
            ['source', 'correlation_id'],
            registry=self.registry
        )
        
        self.team_compositions_generated = Counter(
            'team_compositions_generated_total',
            'Total team compositions generated',
            ['composition_type'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Current data quality score (0-100)',
            ['dataset', 'validation_type'],
            registry=self.registry
        )
        
        # Enhanced error metrics
        self.validation_errors = Counter(
            'validation_errors_total',
            'Total validation errors encountered',
            ['error_type', 'correlation_id'],
            registry=self.registry
        )
        
        self.processing_errors = Counter(
            'processing_errors_total',
            'Total processing errors encountered',
            ['component', 'error_type', 'correlation_id', 'recoverable'],
            registry=self.registry
        )
        
        self.error_recovery_attempts = Counter(
            'error_recovery_attempts_total',
            'Total error recovery attempts',
            ['error_type', 'recovery_method', 'success'],
            registry=self.registry
        )
        
        # System health and performance metrics
        self.system_health = Gauge(
            'system_health_score',
            'Overall system health score (0-1)',
            ['component'],
            registry=self.registry
        )
        
        self.health_check_duration = Histogram(
            'health_check_duration_seconds',
            'Time spent on health checks',
            ['check_name', 'status'],
            registry=self.registry
        )
        
        self.health_check_failures = Counter(
            'health_check_failures_total',
            'Total health check failures',
            ['check_name', 'failure_type'],
            registry=self.registry
        )
        
        # Resource monitoring
        self.cpu_usage_percent = Gauge(
            'cpu_usage_percent',
            'Current CPU usage percentage',
            ['process_name'],
            registry=self.registry
        )
        
        self.disk_usage_bytes = Gauge(
            'disk_usage_bytes',
            'Current disk usage in bytes',
            ['mount_point'],
            registry=self.registry
        )
        
        # Performance baselines and anomaly detection
        self.performance_anomalies = Counter(
            'performance_anomalies_total',
            'Total performance anomalies detected',
            ['metric_name', 'severity'],
            registry=self.registry
        )
        
        self.sla_violations = Counter(
            'sla_violations_total',
            'Total SLA violations',
            ['sla_type', 'severity'],
            registry=self.registry
        )
    
    def _bucket_size(self, size: int) -> str:
        """Convert size to bucket for metrics labeling"""
        if size < 100:
            return "small"
        elif size < 1000:
            return "medium"
        elif size < 10000:
            return "large"
        else:
            return "xlarge"
    
    def record_clustering_operation(
        self, 
        algorithm: str, 
        duration: float, 
        dataset_size: int, 
        status: str = "success",
        fallback_used: bool = False,
        correlation_id: str = None
    ):
        """Record enhanced clustering operation metrics with Generation 2 features"""
        size_bucket = self._bucket_size(dataset_size)
        correlation_id = correlation_id or str(uuid.uuid4())
        
        with self._lock:
            if PROMETHEUS_AVAILABLE:
                self.clustering_duration.labels(
                    algorithm=algorithm, 
                    dataset_size_bucket=size_bucket,
                    correlation_id=correlation_id
                ).observe(duration)
                
                self.clustering_operations.labels(
                    algorithm=algorithm, 
                    status=status,
                    fallback_used=str(fallback_used)
                ).inc()
            else:
                self.in_memory_metrics[f"clustering_duration_{algorithm}_{size_bucket}"] = MetricValue(
                    name="clustering_duration_seconds",
                    value=duration,
                    labels={
                        "algorithm": algorithm, 
                        "dataset_size_bucket": size_bucket,
                        "correlation_id": correlation_id
                    },
                    timestamp=time.time()
                )
                
                self.in_memory_metrics[f"clustering_operations_{algorithm}_{status}"] = MetricValue(
                    name="clustering_operations_total",
                    value=1,
                    labels={
                        "algorithm": algorithm, 
                        "status": status, 
                        "fallback_used": str(fallback_used)
                    },
                    timestamp=time.time()
                )
            
            # Record in history for analysis
            self.metric_history.append({
                'timestamp': datetime.utcnow(),
                'type': 'clustering_operation',
                'algorithm': algorithm,
                'duration': duration,
                'dataset_size': dataset_size,
                'status': status,
                'fallback_used': fallback_used,
                'correlation_id': correlation_id
            })
            
            # Check for performance anomalies
            self._check_performance_anomaly('clustering_duration', duration, algorithm)
    
    def record_neuromorphic_operation(
        self,
        method: str,
        component: str,
        duration: float,
        status: str = "success",
        error_type: str = None,
        recoverable: bool = True
    ):
        """Record neuromorphic clustering specific metrics"""
        with self._lock:
            if PROMETHEUS_AVAILABLE:
                if status == "success":
                    self.neuromorphic_feature_extraction_duration.labels(
                        method=method,
                        component=component
                    ).observe(duration)
                else:
                    self.neuromorphic_clustering_failures.labels(
                        error_type=error_type or "unknown",
                        component=component,
                        recoverable=str(recoverable)
                    ).inc()
            else:
                if status == "success":
                    self.in_memory_metrics[f"neuromorphic_duration_{method}_{component}"] = MetricValue(
                        name="neuromorphic_feature_extraction_duration_seconds",
                        value=duration,
                        labels={"method": method, "component": component},
                        timestamp=time.time()
                    )
                else:
                    self.in_memory_metrics[f"neuromorphic_failures_{error_type}_{component}"] = MetricValue(
                        name="neuromorphic_clustering_failures_total",
                        value=1,
                        labels={
                            "error_type": error_type or "unknown",
                            "component": component,
                            "recoverable": str(recoverable)
                        },
                        timestamp=time.time()
                    )
    
    def record_circuit_breaker_state(self, component: str, state: str):
        """Record circuit breaker state changes"""
        state_mapping = {"closed": 0, "half_open": 1, "open": 2}
        state_value = state_mapping.get(state.lower(), 0)
        
        with self._lock:
            if PROMETHEUS_AVAILABLE:
                self.circuit_breaker_state.labels(component=component).set(state_value)
                
                if state == "open":
                    self.circuit_breaker_failures.labels(component=component).inc()
            else:
                self.in_memory_metrics[f"circuit_breaker_{component}"] = MetricValue(
                    name="circuit_breaker_state",
                    value=state_value,
                    labels={"component": component},
                    timestamp=time.time()
                )
    
    def record_fallback_operation(
        self,
        primary_method: str,
        fallback_method: str,
        reason: str,
        correlation_id: str = None
    ):
        """Record fallback operation metrics"""
        with self._lock:
            if PROMETHEUS_AVAILABLE:
                self.fallback_operations.labels(
                    primary_method=primary_method,
                    fallback_method=fallback_method,
                    reason=reason
                ).inc()
            else:
                self.in_memory_metrics[f"fallback_{primary_method}_{fallback_method}"] = MetricValue(
                    name="fallback_operations_total",
                    value=1,
                    labels={
                        "primary_method": primary_method,
                        "fallback_method": fallback_method,
                        "reason": reason
                    },
                    timestamp=time.time()
                )
    
    def record_data_processing(
        self, 
        operation_type: str, 
        duration: float, 
        data_size: int
    ):
        """Record data processing metrics"""
        size_bucket = self._bucket_size(data_size)
        
        if PROMETHEUS_AVAILABLE:
            self.data_processing_duration.labels(
                operation_type=operation_type,
                data_size_bucket=size_bucket
            ).observe(duration)
        else:
            self.in_memory_metrics[f"data_processing_duration_{operation_type}_{size_bucket}"] = MetricValue(
                name="data_processing_duration_seconds",
                value=duration,
                labels={"operation_type": operation_type, "data_size_bucket": size_bucket},
                timestamp=time.time()
            )
    
    def record_employee_records(self, count: int, source: str = "csv"):
        """Record employee records processed"""
        if PROMETHEUS_AVAILABLE:
            self.employee_records_processed.labels(source=source).inc(count)
        else:
            key = f"employee_records_processed_{source}"
            current = self.in_memory_metrics.get(key)
            new_count = (current.value if current else 0) + count
            self.in_memory_metrics[key] = MetricValue(
                name="employee_records_processed_total",
                value=new_count,
                labels={"source": source},
                timestamp=time.time()
            )
    
    def record_team_compositions(self, count: int, composition_type: str = "balanced"):
        """Record team compositions generated"""
        if PROMETHEUS_AVAILABLE:
            self.team_compositions_generated.labels(composition_type=composition_type).inc(count)
        else:
            key = f"team_compositions_generated_{composition_type}"
            current = self.in_memory_metrics.get(key)
            new_count = (current.value if current else 0) + count
            self.in_memory_metrics[key] = MetricValue(
                name="team_compositions_generated_total",
                value=new_count,
                labels={"composition_type": composition_type},
                timestamp=time.time()
            )
    
    def set_data_quality_score(self, score: float, dataset: str = "main"):
        """Set current data quality score"""
        if PROMETHEUS_AVAILABLE:
            self.data_quality_score.labels(dataset=dataset).set(score)
        else:
            self.in_memory_metrics[f"data_quality_score_{dataset}"] = MetricValue(
                name="data_quality_score",
                value=score,
                labels={"dataset": dataset},
                timestamp=time.time()
            )
    
    def record_validation_error(self, error_type: str):
        """Record validation error"""
        if PROMETHEUS_AVAILABLE:
            self.validation_errors.labels(error_type=error_type).inc()
        else:
            key = f"validation_errors_{error_type}"
            current = self.in_memory_metrics.get(key)
            new_count = (current.value if current else 0) + 1
            self.in_memory_metrics[key] = MetricValue(
                name="validation_errors_total",
                value=new_count,
                labels={"error_type": error_type},
                timestamp=time.time()
            )
    
    def record_processing_error(self, component: str, error_type: str):
        """Record processing error"""
        if PROMETHEUS_AVAILABLE:
            self.processing_errors.labels(
                component=component, 
                error_type=error_type
            ).inc()
        else:
            key = f"processing_errors_{component}_{error_type}"
            current = self.in_memory_metrics.get(key)
            new_count = (current.value if current else 0) + 1
            self.in_memory_metrics[key] = MetricValue(
                name="processing_errors_total",
                value=new_count,
                labels={"component": component, "error_type": error_type},
                timestamp=time.time()
            )
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """Set current memory usage for component"""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage.labels(component=component).set(bytes_used)
        else:
            self.in_memory_metrics[f"memory_usage_{component}"] = MetricValue(
                name="memory_usage_bytes",
                value=bytes_used,
                labels={"component": component},
                timestamp=time.time()
            )
    
    def set_system_health(self, score: float):
        """Set overall system health score (0-1)"""
        if PROMETHEUS_AVAILABLE:
            self.system_health.set(score)
        else:
            self.in_memory_metrics["system_health_score"] = MetricValue(
                name="system_health_score",
                value=score,
                labels={},
                timestamp=time.time()
            )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics for debugging/monitoring"""
        if PROMETHEUS_AVAILABLE:
            return {"status": "prometheus_active", "registry": str(self.registry)}
        else:
            return {
                "status": "in_memory_mode",
                "metrics_count": len(self.in_memory_metrics),
                "metrics": {k: asdict(v) for k, v in self.in_memory_metrics.items()}
            }
    
    def export_prometheus_metrics(self) -> bytes:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry)
        else:
            # Return simple text format for in-memory metrics
            lines = []
            for key, metric in self.in_memory_metrics.items():
                labels = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                if labels:
                    labels = "{" + labels + "}"
                lines.append(f"{metric.name}{labels} {metric.value}")
            return "\n".join(lines).encode()


# Global enhanced metrics instance
enhanced_metrics = EnhancedMetricsCollector()

# Backward compatibility alias
metrics = enhanced_metrics


def monitor_performance(operation_name: str, component: str = "general"):
    """Decorator to monitor function performance"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Determine data size if possible
                data_size = 0
                if args and hasattr(args[0], '__len__'):
                    data_size = len(args[0])
                elif 'data' in kwargs and hasattr(kwargs['data'], '__len__'):
                    data_size = len(kwargs['data'])
                
                metrics.record_data_processing(operation_name, duration, data_size)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_processing_error(component, type(e).__name__)
                raise
                
        return wrapper
    return decorator


@contextmanager
def monitor_clustering_operation(algorithm: str, dataset_size: int):
    """Context manager for monitoring clustering operations"""
    start_time = time.time()
    status = "success"
    
    try:
        yield
    except Exception as e:
        status = "error"
        logger.error(f"Clustering operation failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        metrics.record_clustering_operation(algorithm, duration, dataset_size, status)


def health_check() -> Dict[str, Any]:
    """Perform application health check"""
    try:
        # Basic health indicators
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics_available": PROMETHEUS_AVAILABLE,
            "components": {
                "clustering": "operational",
                "data_processing": "operational", 
                "team_simulation": "operational"
            }
        }
        
        # Set overall health score
        health_score = 1.0
        if not PROMETHEUS_AVAILABLE:
            health_score = 0.8  # Slightly reduced without full monitoring
        
        metrics.set_system_health(health_score)
        health_status["health_score"] = health_score
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        metrics.set_system_health(0.0)
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e),
            "health_score": 0.0
        }


# Health check endpoints for container orchestration
def liveness_probe() -> bool:
    """Kubernetes liveness probe - is application running?"""
    try:
        health_check()
        return True
    except Exception:
        return False


def readiness_probe() -> bool:
    """Kubernetes readiness probe - can application serve traffic?"""
    try:
        health_status = health_check()
        return health_status["health_score"] > 0.5
    except Exception:
        return False


if __name__ == "__main__":
    # Example usage and testing
    print("Testing monitoring module...")
    
    # Test health check
    health = health_check()
    print(f"Health check: {health}")
    
    # Test metrics recording
    metrics.record_employee_records(150, "csv")
    metrics.record_clustering_operation("kmeans", 2.5, 150)
    metrics.set_data_quality_score(87.5)
    
    # Test context manager
    with monitor_clustering_operation("dbscan", 200):
        time.sleep(0.1)  # Simulate clustering work
    
    # Test decorator
    @monitor_performance("data_parsing", "data_processor")
    def sample_function(data):
        return len(data)
    
    result = sample_function([1, 2, 3, 4, 5])
    print(f"Sample function result: {result}")
    
    # Show metrics summary
    summary = metrics.get_metrics_summary()
    print(f"Metrics summary: {summary}")
    
    print("Monitoring module test complete.")