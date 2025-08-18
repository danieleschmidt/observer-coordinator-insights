"""Health check endpoints and monitoring."""

import sys
import time
from typing import Any, Dict

import psutil


class HealthChecker:
    """Application health monitoring."""

    def __init__(self):
        self.start_time = time.time()

    def check_health(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "version": "0.1.0",
            "python_version": sys.version,
            "system": self._system_health(),
            "dependencies": self._dependency_health(),
        }

    def _system_health(self) -> Dict[str, Any]:
        """System resource health check."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
        }

    def _dependency_health(self) -> Dict[str, str]:
        """Check critical dependencies."""
        checks = {}

        try:
            import numpy
            checks["numpy"] = "ok"
        except ImportError:
            checks["numpy"] = "error"

        try:
            import pandas
            checks["pandas"] = "ok"
        except ImportError:
            checks["pandas"] = "error"

        try:
            import sklearn
            checks["sklearn"] = "ok"
        except ImportError:
            checks["sklearn"] = "error"

        return checks
