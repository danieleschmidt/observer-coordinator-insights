"""Metrics collection and monitoring."""

import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict


class MetricsCollector:
    """Collect and expose application metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics = defaultdict(lambda: deque(maxlen=max_history))
        self.counters = defaultdict(int)
        self.gauges = {}
        self.lock = threading.Lock()

    def increment_counter(self, name: str, value: int = 1, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            key = self._make_key(name, tags)
            self.counters[key] += value

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        with self.lock:
            key = self._make_key(name, tags)
            self.gauges[key] = {
                "value": value,
                "timestamp": time.time()
            }

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        with self.lock:
            key = self._make_key(name, tags)
            self.metrics[key].append({
                "value": value,
                "timestamp": time.time()
            })

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {
                    name: list(values) for name, values in self.metrics.items()
                },
                "timestamp": time.time()
            }

    def _make_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """Create a unique key for metric storage."""
        if not tags:
            return name

        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
