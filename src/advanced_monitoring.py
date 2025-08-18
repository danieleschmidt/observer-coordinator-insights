#!/usr/bin/env python3
"""Advanced Monitoring and Observability System
Real-time metrics, alerting, and system observability
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List

import psutil


logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Alert configuration"""
    name: str
    metric_name: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    duration_seconds: int
    severity: str
    message_template: str
    actions: List[str] = field(default_factory=list)
    enabled: bool = True


class AdvancedMetricsCollector:
    """Advanced metrics collection and monitoring system"""

    def __init__(self, retention_hours: int = 24):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alert_rules: List[AlertRule] = []
        self.alert_history: deque = deque(maxlen=10000)
        self.retention_hours = retention_hours
        self.collection_interval = 5  # seconds
        self.running = False
        self.collection_thread = None
        self.callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Initialize default alert rules
        self._setup_default_alerts()

    def _setup_default_alerts(self):
        """Setup default monitoring alerts"""
        default_alerts = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu_percent",
                threshold=80.0,
                comparison="gt",
                duration_seconds=60,
                severity="warning",
                message_template="High CPU usage: {value}% for {duration}s"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory_percent",
                threshold=85.0,
                comparison="gt",
                duration_seconds=30,
                severity="critical",
                message_template="High memory usage: {value}% for {duration}s"
            ),
            AlertRule(
                name="clustering_performance_degradation",
                metric_name="clustering.response_time_ms",
                threshold=5000.0,
                comparison="gt",
                duration_seconds=0,
                severity="warning",
                message_template="Clustering response time degraded: {value}ms"
            ),
            AlertRule(
                name="low_silhouette_score",
                metric_name="clustering.silhouette_score",
                threshold=0.3,
                comparison="lt",
                duration_seconds=0,
                severity="info",
                message_template="Low clustering quality: silhouette score {value}"
            )
        ]

        self.alert_rules.extend(default_alerts)

    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric value"""
        if labels is None:
            labels = {}

        metric_point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            labels=labels
        )

        self.metrics[name].append(metric_point)
        self._check_alerts(name, metric_point)

        # Trigger callbacks
        for callback in self.callbacks.get(name, []):
            try:
                callback(name, metric_point)
            except Exception as e:
                logger.warning(f"Metric callback error for {name}: {e}")

    def _check_alerts(self, metric_name: str, metric_point: MetricPoint):
        """Check if metric triggers any alerts"""
        for rule in self.alert_rules:
            if not rule.enabled or rule.metric_name != metric_name:
                continue

            triggered = self._evaluate_alert_condition(rule, metric_point)
            if triggered:
                self._fire_alert(rule, metric_point)

    def _evaluate_alert_condition(self, rule: AlertRule, metric_point: MetricPoint) -> bool:
        """Evaluate if alert condition is met"""
        value = metric_point.value
        threshold = rule.threshold

        if rule.comparison == "gt":
            condition_met = value > threshold
        elif rule.comparison == "lt":
            condition_met = value < threshold
        elif rule.comparison == "eq":
            condition_met = abs(value - threshold) < 0.001
        else:
            return False

        if not condition_met:
            return False

        # Check duration requirement
        if rule.duration_seconds == 0:
            return True

        # Look back through recent metrics to check duration
        cutoff_time = datetime.now() - timedelta(seconds=rule.duration_seconds)
        recent_metrics = [
            m for m in self.metrics[rule.metric_name]
            if m.timestamp >= cutoff_time
        ]

        if len(recent_metrics) < 2:
            return False

        # Check if condition has been met for the entire duration
        for metric in recent_metrics:
            if (rule.comparison == "gt" and metric.value <= threshold) or (rule.comparison == "lt" and metric.value >= threshold):
                return False

        return True

    def _fire_alert(self, rule: AlertRule, metric_point: MetricPoint):
        """Fire an alert"""
        alert_data = {
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "value": metric_point.value,
            "threshold": rule.threshold,
            "severity": rule.severity,
            "timestamp": metric_point.timestamp.isoformat(),
            "message": rule.message_template.format(
                value=metric_point.value,
                threshold=rule.threshold,
                duration=rule.duration_seconds
            )
        }

        self.alert_history.append(alert_data)
        logger.warning(f"ALERT: {alert_data['message']}")

        # Execute alert actions
        for action in rule.actions:
            self._execute_alert_action(action, alert_data)

    def _execute_alert_action(self, action: str, alert_data: Dict[str, Any]):
        """Execute alert action"""
        try:
            if action == "log":
                logger.warning(f"Alert: {alert_data['message']}")
            elif action == "email":
                # Placeholder for email notification
                logger.info(f"Email alert would be sent: {alert_data['message']}")
            elif action.startswith("webhook:"):
                webhook_url = action.split(":", 1)[1]
                logger.info(f"Webhook alert would be sent to {webhook_url}")
        except Exception as e:
            logger.error(f"Failed to execute alert action {action}: {e}")

    def get_metric_statistics(self, metric_name: str, hours: int = 1) -> Dict[str, float]:
        """Get statistical summary of a metric"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m.value for m in self.metrics[metric_name]
            if m.timestamp >= cutoff_time
        ]

        if not recent_metrics:
            return {}

        return {
            "count": len(recent_metrics),
            "min": min(recent_metrics),
            "max": max(recent_metrics),
            "avg": sum(recent_metrics) / len(recent_metrics),
            "latest": recent_metrics[-1] if recent_metrics else 0
        }

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        try:
            # Get recent system metrics
            cpu_stats = self.get_metric_statistics("system.cpu_percent", hours=0.1)
            memory_stats = self.get_metric_statistics("system.memory_percent", hours=0.1)

            if not cpu_stats or not memory_stats:
                return 75.0  # Default moderate score

            # Calculate health components
            cpu_health = max(0, 100 - cpu_stats.get("avg", 50))
            memory_health = max(0, 100 - memory_stats.get("avg", 50))

            # Check recent alerts
            recent_alerts = [
                alert for alert in self.alert_history
                if datetime.fromisoformat(alert["timestamp"]) >
                   datetime.now() - timedelta(minutes=15)
            ]

            alert_penalty = min(50, len(recent_alerts) * 10)

            # Overall health score
            health_score = ((cpu_health + memory_health) / 2) - alert_penalty
            return max(0, min(100, health_score))

        except Exception as e:
            logger.warning(f"Failed to calculate health score: {e}")
            return 50.0

    def start_collection(self):
        """Start background metrics collection"""
        if self.running:
            return

        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started advanced metrics collection")

    def stop_collection(self):
        """Stop background metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Stopped advanced metrics collection")

    def _collection_loop(self):
        """Background metrics collection loop"""
        while self.running:
            try:
                # Collect system metrics
                self.record_metric("system.cpu_percent", psutil.cpu_percent())

                memory = psutil.virtual_memory()
                self.record_metric("system.memory_percent", memory.percent)
                self.record_metric("system.memory_used_mb", memory.used / (1024*1024))
                self.record_metric("system.memory_available_mb", memory.available / (1024*1024))

                disk = psutil.disk_usage('/')
                self.record_metric("system.disk_percent", disk.percent)
                self.record_metric("system.disk_free_gb", disk.free / (1024*1024*1024))

                # Load average (Unix-like systems only)
                try:
                    load_avg = psutil.getloadavg()
                    self.record_metric("system.load_avg_1m", load_avg[0])
                    self.record_metric("system.load_avg_5m", load_avg[1])
                    self.record_metric("system.load_avg_15m", load_avg[2])
                except (AttributeError, OSError):
                    # getloadavg not available on Windows
                    pass

            except Exception as e:
                logger.warning(f"Error collecting system metrics: {e}")

            time.sleep(self.collection_interval)

    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format"""
        output = []

        for metric_name, metric_points in self.metrics.items():
            if not metric_points:
                continue

            # Convert metric name to Prometheus format
            prom_name = metric_name.replace(".", "_").replace("-", "_")

            # Add help and type
            output.append(f"# HELP {prom_name} Generated by Observer Coordinator Insights")
            output.append(f"# TYPE {prom_name} gauge")

            # Export latest value for each unique label combination
            latest_point = metric_points[-1]
            label_str = ""
            if latest_point.labels:
                labels = [f'{k}="{v}"' for k, v in latest_point.labels.items()]
                label_str = "{" + ",".join(labels) + "}"

            output.append(f"{prom_name}{label_str} {latest_point.value}")

        return "\n".join(output)

    def save_metrics_to_file(self, file_path: Path):
        """Save current metrics to JSON file"""
        try:
            metrics_data = {}
            for metric_name, points in self.metrics.items():
                metrics_data[metric_name] = [
                    {
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "labels": point.labels
                    }
                    for point in points
                ]

            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metrics to {file_path}: {e}")


# Global instance
advanced_monitor = AdvancedMetricsCollector()


def initialize_advanced_monitoring():
    """Initialize the advanced monitoring system"""
    try:
        advanced_monitor.start_collection()
        logger.info("🔍 Advanced monitoring system initialized")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize advanced monitoring: {e}")
        return False


def shutdown_advanced_monitoring():
    """Shutdown the advanced monitoring system"""
    try:
        advanced_monitor.stop_collection()
        logger.info("Advanced monitoring system shutdown")
    except Exception as e:
        logger.warning(f"Error shutting down advanced monitoring: {e}")


# Convenience functions
def record_metric(name: str, value: float, labels: Dict[str, str] = None):
    """Record a metric (convenience function)"""
    advanced_monitor.record_metric(name, value, labels)


def get_system_health() -> float:
    """Get current system health score"""
    return advanced_monitor.get_system_health_score()


def get_recent_alerts(minutes: int = 60) -> List[Dict[str, Any]]:
    """Get recent alerts"""
    cutoff_time = datetime.now() - timedelta(minutes=minutes)
    return [
        alert for alert in advanced_monitor.alert_history
        if datetime.fromisoformat(alert["timestamp"]) >= cutoff_time
    ]
