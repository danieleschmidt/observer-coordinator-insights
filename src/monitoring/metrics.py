"""
Metrics collection and Prometheus integration for Observer Coordinator Insights.
"""

import time
import psutil
from typing import Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.analysis_count = 0
        self.error_count = 0
        
    def increment_requests(self):
        """Increment request counter."""
        self.request_count += 1
        
    def increment_analyses(self):
        """Increment analysis counter."""
        self.analysis_count += 1
        
    def increment_errors(self):
        """Increment error counter."""
        self.error_count += 1
        
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        uptime = time.time() - self.start_time
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent()
        
        metrics = f"""# HELP app_uptime_seconds Application uptime
# TYPE app_uptime_seconds counter
app_uptime_seconds {uptime}

# HELP app_requests_total Total number of requests
# TYPE app_requests_total counter
app_requests_total {self.request_count}

# HELP app_analyses_total Total number of analyses completed
# TYPE app_analyses_total counter
app_analyses_total {self.analysis_count}

# HELP app_errors_total Total number of errors
# TYPE app_errors_total counter
app_errors_total {self.error_count}

# HELP system_memory_usage_percent Memory usage percentage
# TYPE system_memory_usage_percent gauge
system_memory_usage_percent {memory.percent}

# HELP system_cpu_usage_percent CPU usage percentage
# TYPE system_cpu_usage_percent gauge
system_cpu_usage_percent {cpu}
"""
        return metrics


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics_collector


# Flask blueprint for metrics endpoint
try:
    from flask import Blueprint, Response
    
    metrics_blueprint = Blueprint('metrics', __name__)
    
    @metrics_blueprint.route('/metrics')
    def metrics_endpoint():
        """Prometheus metrics endpoint."""
        metrics_data = _metrics_collector.get_metrics()
        return Response(metrics_data, mimetype='text/plain')

except ImportError:
    metrics_blueprint = None
    logger.info("Flask not available, metrics endpoint disabled")