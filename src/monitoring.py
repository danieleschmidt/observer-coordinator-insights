#!/usr/bin/env python3
"""
Application monitoring and metrics collection module.
Provides Prometheus-compatible metrics for operational observability.
"""

import time
import logging
from functools import wraps
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

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


class MetricsCollector:
    """
    Centralized metrics collection for application monitoring.
    Falls back to in-memory storage when Prometheus client is not available.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry
        self.in_memory_metrics: Dict[str, MetricValue] = {}
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            logger.warning("Prometheus client not available, using in-memory metrics")
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Performance metrics
        self.clustering_duration = Histogram(
            'clustering_operation_duration_seconds',
            'Time spent on clustering operations',
            ['algorithm', 'dataset_size_bucket'],
            registry=self.registry
        )
        
        self.clustering_operations = Counter(
            'clustering_operations_total',
            'Total number of clustering operations',
            ['algorithm', 'status'],
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Current memory usage',
            ['component'],
            registry=self.registry
        )
        
        self.data_processing_duration = Histogram(
            'data_processing_duration_seconds',
            'Time spent processing data',
            ['operation_type', 'data_size_bucket'],
            registry=self.registry
        )
        
        # Business metrics
        self.employee_records_processed = Counter(
            'employee_records_processed_total',
            'Total employee records processed',
            ['source'],
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
            ['dataset'],
            registry=self.registry
        )
        
        # Error metrics
        self.validation_errors = Counter(
            'validation_errors_total',
            'Total validation errors encountered',
            ['error_type'],
            registry=self.registry
        )
        
        self.processing_errors = Counter(
            'processing_errors_total',
            'Total processing errors encountered',
            ['component', 'error_type'],
            registry=self.registry
        )
        
        # System health metrics
        self.system_health = Gauge(
            'system_health_score',
            'Overall system health score (0-1)',
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
        status: str = "success"
    ):
        """Record clustering operation metrics"""
        size_bucket = self._bucket_size(dataset_size)
        
        if PROMETHEUS_AVAILABLE:
            self.clustering_duration.labels(
                algorithm=algorithm, 
                dataset_size_bucket=size_bucket
            ).observe(duration)
            
            self.clustering_operations.labels(
                algorithm=algorithm, 
                status=status
            ).inc()
        else:
            self.in_memory_metrics[f"clustering_duration_{algorithm}_{size_bucket}"] = MetricValue(
                name="clustering_duration_seconds",
                value=duration,
                labels={"algorithm": algorithm, "dataset_size_bucket": size_bucket},
                timestamp=time.time()
            )
            
            self.in_memory_metrics[f"clustering_operations_{algorithm}_{status}"] = MetricValue(
                name="clustering_operations_total",
                value=1,
                labels={"algorithm": algorithm, "status": status},
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


# Global metrics instance
metrics = MetricsCollector()


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