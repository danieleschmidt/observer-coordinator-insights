#!/usr/bin/env python3
"""Intelligent Auto-Scaling System
Dynamic resource allocation and performance optimization based on workload patterns
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


class ScalingTrigger(Enum):
    """Auto-scaling trigger types"""
    CPU_THRESHOLD = "cpu_threshold"
    MEMORY_THRESHOLD = "memory_threshold"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    CUSTOM_METRIC = "custom_metric"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration"""
    name: str
    trigger: ScalingTrigger
    metric_name: str
    threshold_up: float
    threshold_down: float
    scale_up_by: int
    scale_down_by: int
    min_instances: int
    max_instances: int
    cooldown_seconds: int
    evaluation_periods: int = 3
    enabled: bool = True


@dataclass
class ResourceInstance:
    """Scalable resource instance"""
    instance_id: str
    instance_type: str
    created_at: datetime
    status: str  # "starting", "running", "stopping", "stopped"
    metrics: Dict[str, float] = field(default_factory=dict)
    load_factor: float = 0.0


class IntelligentAutoScaler:
    """Intelligent auto-scaling system with predictive capabilities"""

    def __init__(self):
        self.scaling_rules: List[ScalingRule] = []
        self.instances: Dict[str, ResourceInstance] = {}
        self.metrics_history: Dict[str, deque] = {}
        self.scaling_history: deque = deque(maxlen=1000)
        self.running = False
        self.scaling_thread = None
        self._lock = threading.Lock()

        # Predictive scaling parameters
        self.prediction_window_minutes = 15
        self.learning_enabled = True
        self.workload_patterns = {}

        # Performance tracking
        self.performance_metrics = {
            "scaling_decisions": 0,
            "scale_up_events": 0,
            "scale_down_events": 0,
            "prediction_accuracy": 0.0
        }

        # Initialize default scaling rules
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default auto-scaling rules"""
        default_rules = [
            ScalingRule(
                name="cpu_based_scaling",
                trigger=ScalingTrigger.CPU_THRESHOLD,
                metric_name="system.cpu_percent",
                threshold_up=70.0,
                threshold_down=30.0,
                scale_up_by=2,
                scale_down_by=1,
                min_instances=1,
                max_instances=16,
                cooldown_seconds=300
            ),
            ScalingRule(
                name="memory_based_scaling",
                trigger=ScalingTrigger.MEMORY_THRESHOLD,
                metric_name="system.memory_percent",
                threshold_up=80.0,
                threshold_down=40.0,
                scale_up_by=1,
                scale_down_by=1,
                min_instances=1,
                max_instances=8,
                cooldown_seconds=180
            ),
            ScalingRule(
                name="queue_based_scaling",
                trigger=ScalingTrigger.QUEUE_LENGTH,
                metric_name="processing.queue_size",
                threshold_up=50.0,
                threshold_down=10.0,
                scale_up_by=2,
                scale_down_by=1,
                min_instances=1,
                max_instances=32,
                cooldown_seconds=120
            ),
            ScalingRule(
                name="response_time_scaling",
                trigger=ScalingTrigger.RESPONSE_TIME,
                metric_name="clustering.response_time_ms",
                threshold_up=3000.0,
                threshold_down=1000.0,
                scale_up_by=1,
                scale_down_by=1,
                min_instances=1,
                max_instances=10,
                cooldown_seconds=240
            )
        ]

        self.scaling_rules.extend(default_rules)

    def start_scaling_engine(self):
        """Start the auto-scaling engine"""
        if self.running:
            return

        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()

        logger.info("🎯 Intelligent auto-scaling engine started")

    def stop_scaling_engine(self):
        """Stop the auto-scaling engine"""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5)

        logger.info("Auto-scaling engine stopped")

    def record_metric(self, metric_name: str, value: float):
        """Record metric for scaling decisions"""
        with self._lock:
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = deque(maxlen=1000)

            self.metrics_history[metric_name].append({
                "timestamp": datetime.now(),
                "value": value
            })

    def _scaling_loop(self):
        """Main auto-scaling evaluation loop"""
        while self.running:
            try:
                self._evaluate_scaling_rules()
                self._update_workload_patterns()
                self._cleanup_old_metrics()

                time.sleep(30)  # Evaluate every 30 seconds

            except Exception as e:
                logger.error(f"Error in scaling loop: {e}")
                time.sleep(60)  # Longer delay on error

    def _evaluate_scaling_rules(self):
        """Evaluate all scaling rules"""
        for rule in self.scaling_rules:
            if not rule.enabled:
                continue

            try:
                self._evaluate_single_rule(rule)
            except Exception as e:
                logger.error(f"Error evaluating scaling rule {rule.name}: {e}")

    def _evaluate_single_rule(self, rule: ScalingRule):
        """Evaluate a single scaling rule"""
        # Get recent metric values
        metric_values = self._get_recent_metric_values(
            rule.metric_name,
            periods=rule.evaluation_periods
        )

        if len(metric_values) < rule.evaluation_periods:
            return  # Not enough data

        avg_value = np.mean(metric_values)
        current_instances = len([i for i in self.instances.values() if i.status == "running"])

        # Check scaling triggers
        should_scale_up = avg_value > rule.threshold_up and current_instances < rule.max_instances
        should_scale_down = avg_value < rule.threshold_down and current_instances > rule.min_instances

        # Check cooldown period
        last_scaling = self._get_last_scaling_time(rule.name)
        if last_scaling and (datetime.now() - last_scaling).total_seconds() < rule.cooldown_seconds:
            return

        # Apply predictive scaling if learning is enabled
        if self.learning_enabled:
            prediction = self._predict_future_load(rule.metric_name)
            if prediction:
                should_scale_up = should_scale_up or (prediction > rule.threshold_up)

        # Execute scaling decision
        if should_scale_up:
            self._scale_up(rule)
        elif should_scale_down:
            self._scale_down(rule)

    def _get_recent_metric_values(self, metric_name: str, periods: int) -> List[float]:
        """Get recent metric values for evaluation"""
        with self._lock:
            if metric_name not in self.metrics_history:
                return []

            recent_metrics = list(self.metrics_history[metric_name])[-periods:]
            return [m["value"] for m in recent_metrics]

    def _get_last_scaling_time(self, rule_name: str) -> Optional[datetime]:
        """Get last scaling time for a rule"""
        for event in reversed(self.scaling_history):
            if event.get("rule_name") == rule_name:
                return event.get("timestamp")
        return None

    def _predict_future_load(self, metric_name: str) -> Optional[float]:
        """Predict future load using historical patterns"""
        try:
            with self._lock:
                if metric_name not in self.metrics_history:
                    return None

                recent_values = [
                    m["value"] for m in
                    list(self.metrics_history[metric_name])[-60:]  # Last 60 data points
                ]

                if len(recent_values) < 10:
                    return None

                # Simple linear trend prediction
                x = np.arange(len(recent_values))
                y = np.array(recent_values)

                # Fit linear trend
                coeffs = np.polyfit(x, y, 1)

                # Predict next few minutes
                future_x = len(recent_values) + (self.prediction_window_minutes / 2)  # 30-second intervals
                prediction = coeffs[0] * future_x + coeffs[1]

                return max(0, prediction)  # Ensure non-negative

        except Exception as e:
            logger.warning(f"Prediction failed for {metric_name}: {e}")
            return None

    def _scale_up(self, rule: ScalingRule):
        """Execute scale-up decision"""
        with self._lock:
            current_instances = len([i for i in self.instances.values() if i.status == "running"])
            target_instances = min(current_instances + rule.scale_up_by, rule.max_instances)
            instances_to_add = target_instances - current_instances

            if instances_to_add <= 0:
                return

            logger.info(f"Scaling up by {instances_to_add} instances due to rule: {rule.name}")

            # Create new instances
            for _ in range(instances_to_add):
                self._create_instance(rule)

            # Record scaling event
            self.scaling_history.append({
                "timestamp": datetime.now(),
                "rule_name": rule.name,
                "action": "scale_up",
                "instances_added": instances_to_add,
                "total_instances": target_instances
            })

            self.performance_metrics["scaling_decisions"] += 1
            self.performance_metrics["scale_up_events"] += 1

    def _scale_down(self, rule: ScalingRule):
        """Execute scale-down decision"""
        with self._lock:
            running_instances = [i for i in self.instances.values() if i.status == "running"]
            current_count = len(running_instances)
            target_count = max(current_count - rule.scale_down_by, rule.min_instances)
            instances_to_remove = current_count - target_count

            if instances_to_remove <= 0:
                return

            logger.info(f"Scaling down by {instances_to_remove} instances due to rule: {rule.name}")

            # Remove instances with lowest load
            instances_to_stop = sorted(running_instances, key=lambda x: x.load_factor)[:instances_to_remove]

            for instance in instances_to_stop:
                self._stop_instance(instance.instance_id)

            # Record scaling event
            self.scaling_history.append({
                "timestamp": datetime.now(),
                "rule_name": rule.name,
                "action": "scale_down",
                "instances_removed": instances_to_remove,
                "total_instances": target_count
            })

            self.performance_metrics["scaling_decisions"] += 1
            self.performance_metrics["scale_down_events"] += 1

    def _create_instance(self, rule: ScalingRule):
        """Create a new resource instance"""
        instance_id = f"instance_{int(time.time() * 1000)}"

        instance = ResourceInstance(
            instance_id=instance_id,
            instance_type=rule.trigger.value,
            created_at=datetime.now(),
            status="starting"
        )

        self.instances[instance_id] = instance

        # Simulate instance startup (in real implementation, this would create actual resources)
        def start_instance():
            time.sleep(2)  # Simulate startup time
            with self._lock:
                if instance_id in self.instances:
                    self.instances[instance_id].status = "running"

        threading.Thread(target=start_instance, daemon=True).start()

    def _stop_instance(self, instance_id: str):
        """Stop a resource instance"""
        if instance_id in self.instances:
            self.instances[instance_id].status = "stopping"

            def stop_instance():
                time.sleep(1)  # Simulate shutdown time
                with self._lock:
                    if instance_id in self.instances:
                        self.instances[instance_id].status = "stopped"
                        # Keep for metrics history, don't delete immediately

            threading.Thread(target=stop_instance, daemon=True).start()

    def _update_workload_patterns(self):
        """Update workload pattern learning"""
        # Simple pattern recognition based on time of day and day of week
        current_time = datetime.now()
        hour = current_time.hour
        day_of_week = current_time.weekday()

        pattern_key = f"{day_of_week}_{hour}"

        # Calculate current load
        current_load = 0.0
        for metric_name in ["system.cpu_percent", "system.memory_percent"]:
            recent_values = self._get_recent_metric_values(metric_name, 5)
            if recent_values:
                current_load += np.mean(recent_values)

        current_load = current_load / 2  # Average of CPU and memory

        # Update pattern
        if pattern_key not in self.workload_patterns:
            self.workload_patterns[pattern_key] = deque(maxlen=30)  # Keep 30 days

        self.workload_patterns[pattern_key].append(current_load)

    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=24)

        with self._lock:
            for metric_name, metrics in self.metrics_history.items():
                while metrics and metrics[0]["timestamp"] < cutoff_time:
                    metrics.popleft()

            # Clean up stopped instances
            stopped_instances = [
                instance_id for instance_id, instance in self.instances.items()
                if instance.status == "stopped" and
                   (datetime.now() - instance.created_at).total_seconds() > 3600
            ]

            for instance_id in stopped_instances:
                del self.instances[instance_id]

    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        with self._lock:
            running_instances = len([i for i in self.instances.values() if i.status == "running"])
            total_instances = len(self.instances)

            stats = {
                "running_instances": running_instances,
                "total_instances": total_instances,
                "scaling_rules_count": len([r for r in self.scaling_rules if r.enabled]),
                "recent_scaling_events": len([
                    e for e in self.scaling_history
                    if (datetime.now() - e["timestamp"]).total_seconds() < 3600
                ]),
                **self.performance_metrics
            }

            return stats

    def get_instance_status(self) -> List[Dict[str, Any]]:
        """Get status of all instances"""
        with self._lock:
            return [
                {
                    "instance_id": instance.instance_id,
                    "type": instance.instance_type,
                    "status": instance.status,
                    "created_at": instance.created_at.isoformat(),
                    "load_factor": instance.load_factor
                }
                for instance in self.instances.values()
            ]


# Global auto-scaler instance
intelligent_scaler = IntelligentAutoScaler()


def initialize_intelligent_scaling():
    """Initialize intelligent auto-scaling"""
    try:
        intelligent_scaler.start_scaling_engine()
        logger.info("🎯 Intelligent auto-scaling initialized")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize intelligent scaling: {e}")
        return False


def shutdown_intelligent_scaling():
    """Shutdown intelligent auto-scaling"""
    try:
        intelligent_scaler.stop_scaling_engine()
        logger.info("Intelligent auto-scaling shutdown")
    except Exception as e:
        logger.warning(f"Error shutting down intelligent scaling: {e}")


# Convenience functions
def record_scaling_metric(metric_name: str, value: float):
    """Record metric for auto-scaling decisions"""
    intelligent_scaler.record_metric(metric_name, value)


def get_scaling_status() -> Dict[str, Any]:
    """Get current auto-scaling status"""
    return intelligent_scaler.get_scaling_statistics()
