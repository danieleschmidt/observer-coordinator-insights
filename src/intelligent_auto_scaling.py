#!/usr/bin/env python3
"""
Intelligent Auto-Scaling System - Generation 3 Scalability
Advanced resource management and predictive scaling
"""

import asyncio
import json
import math
import statistics
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Callable
import psutil
import logging


class ScalingDirection(Enum):
    """Scaling direction options"""
    UP = "up"
    DOWN = "down"
    STEADY = "steady"


class ScalingStrategy(Enum):
    """Auto-scaling strategies"""
    REACTIVE = "reactive"          # React to current metrics
    PREDICTIVE = "predictive"      # Predict future load
    HYBRID = "hybrid"              # Combination of both
    MACHINE_LEARNING = "ml"        # ML-based scaling


class ResourceType(Enum):
    """Types of scalable resources"""
    CPU_CORES = "cpu_cores"
    MEMORY_MB = "memory_mb"
    WORKER_THREADS = "worker_threads"
    CACHE_SIZE = "cache_size"
    CONNECTION_POOL = "connection_pool"


@dataclass
class ScalingMetric:
    """Metric for scaling decisions"""
    name: str
    current_value: float
    target_value: float
    weight: float  # Importance in scaling decision (0.0-1.0)
    threshold_up: float    # Scale up when above this
    threshold_down: float  # Scale down when below this
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_scaling_pressure(self) -> float:
        """Get scaling pressure (-1.0 to 1.0)"""
        if self.current_value > self.threshold_up:
            # Need to scale up
            pressure = min(1.0, (self.current_value - self.threshold_up) / 
                          (self.target_value - self.threshold_up) if self.target_value > self.threshold_up else 1.0)
        elif self.current_value < self.threshold_down:
            # Can scale down
            pressure = max(-1.0, (self.current_value - self.threshold_down) / 
                          (self.target_value - self.threshold_down) if self.target_value < self.threshold_down else -1.0)
        else:
            # In acceptable range
            pressure = 0.0
        
        return pressure * self.weight


@dataclass
class ScalingAction:
    """Represents a scaling action"""
    resource_type: ResourceType
    direction: ScalingDirection
    magnitude: float  # How much to scale (percentage or absolute)
    confidence: float  # Confidence in this decision (0.0-1.0)
    reason: str
    triggered_by_metrics: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'resource_type': self.resource_type.value,
            'direction': self.direction.value,
            'magnitude': self.magnitude,
            'confidence': self.confidence,
            'reason': self.reason,
            'triggered_by': self.triggered_by_metrics,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ResourceLimit:
    """Resource scaling limits"""
    min_value: float
    max_value: float
    step_size: float
    cooldown_seconds: int  # Minimum time between scaling actions


class PredictiveAnalyzer:
    """Analyzes metrics to predict future resource needs"""
    
    def __init__(self, history_window_minutes: int = 60):
        self.history_window = timedelta(minutes=history_window_minutes)
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.predictions: Dict[str, float] = {}
        
    def add_metric_sample(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """Add metric sample to history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
        
        self.metric_history[metric_name].append((timestamp, value))
        
        # Trim old data
        cutoff_time = datetime.now() - self.history_window
        self.metric_history[metric_name] = [
            (ts, val) for ts, val in self.metric_history[metric_name]
            if ts > cutoff_time
        ]
    
    def predict_future_value(self, metric_name: str, minutes_ahead: int = 10) -> Optional[float]:
        """Predict metric value in the future"""
        if metric_name not in self.metric_history:
            return None
        
        history = self.metric_history[metric_name]
        if len(history) < 3:
            return None
        
        # Simple linear regression for trend prediction
        timestamps = [(ts - history[0][0]).total_seconds() for ts, _ in history]
        values = [val for _, val in history]
        
        # Calculate linear trend
        n = len(timestamps)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)
        
        # Linear regression coefficients
        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            # No clear trend, return current average
            return statistics.mean(values[-5:])  # Average of last 5 values
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict future value
        future_timestamp = timestamps[-1] + (minutes_ahead * 60)  # Convert to seconds
        predicted_value = slope * future_timestamp + intercept
        
        # Apply seasonal adjustments if we have enough data
        if len(history) > 20:
            seasonal_adjustment = self._calculate_seasonal_factor(values, minutes_ahead)
            predicted_value *= seasonal_adjustment
        
        # Store prediction
        self.predictions[metric_name] = predicted_value
        
        return predicted_value
    
    def _calculate_seasonal_factor(self, values: List[float], minutes_ahead: int) -> float:
        """Calculate seasonal adjustment factor"""
        # Simple seasonality based on recent patterns
        if len(values) < 10:
            return 1.0
        
        # Look for patterns in recent data
        recent_values = values[-10:]
        avg_recent = statistics.mean(recent_values)
        overall_avg = statistics.mean(values)
        
        if overall_avg == 0:
            return 1.0
        
        # Seasonal factor based on recent trend
        return avg_recent / overall_avg
    
    def get_trend_analysis(self, metric_name: str) -> Dict[str, Any]:
        """Get comprehensive trend analysis"""
        if metric_name not in self.metric_history:
            return {"error": "No data available"}
        
        history = self.metric_history[metric_name]
        if len(history) < 2:
            return {"error": "Insufficient data"}
        
        values = [val for _, val in history]
        
        return {
            "metric_name": metric_name,
            "data_points": len(values),
            "current_value": values[-1],
            "average": statistics.mean(values),
            "min_value": min(values),
            "max_value": max(values),
            "trend_direction": "increasing" if values[-1] > values[0] else "decreasing",
            "volatility": statistics.stdev(values) if len(values) > 1 else 0,
            "predicted_10min": self.predictions.get(metric_name),
            "confidence": self._calculate_prediction_confidence(values)
        }
    
    def _calculate_prediction_confidence(self, values: List[float]) -> float:
        """Calculate confidence in predictions based on data quality"""
        if len(values) < 3:
            return 0.0
        
        # Base confidence on data amount and consistency
        data_confidence = min(1.0, len(values) / 20.0)  # More data = higher confidence
        
        # Consistency confidence (lower volatility = higher confidence)
        if len(values) > 1:
            volatility = statistics.stdev(values)
            mean_val = statistics.mean(values)
            if mean_val != 0:
                cv = volatility / abs(mean_val)  # Coefficient of variation
                consistency_confidence = max(0.0, 1.0 - cv)
            else:
                consistency_confidence = 0.5
        else:
            consistency_confidence = 0.5
        
        return (data_confidence + consistency_confidence) / 2.0


class AutoScalingEngine:
    """Main auto-scaling engine with intelligent decision making"""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.predictor = PredictiveAnalyzer()
        
        # Configuration
        self.scaling_metrics: Dict[str, ScalingMetric] = {}
        self.resource_limits: Dict[ResourceType, ResourceLimit] = {}
        self.last_scaling_actions: Dict[ResourceType, datetime] = {}
        
        # Current resource allocations
        self.current_resources: Dict[ResourceType, float] = {}
        
        # Scaling history
        self.scaling_history: List[ScalingAction] = []
        
        # Control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Initialize default configuration
        self._initialize_default_config()
    
    def _initialize_default_config(self):
        """Initialize default scaling configuration"""
        # Default metrics
        self.scaling_metrics.update({
            "cpu_usage": ScalingMetric(
                name="cpu_usage",
                current_value=0.0,
                target_value=70.0,
                weight=0.8,
                threshold_up=80.0,
                threshold_down=30.0
            ),
            "memory_usage": ScalingMetric(
                name="memory_usage",
                current_value=0.0,
                target_value=75.0,
                weight=0.7,
                threshold_up=85.0,
                threshold_down=40.0
            ),
            "response_time": ScalingMetric(
                name="response_time",
                current_value=0.0,
                target_value=200.0,
                weight=0.9,
                threshold_up=500.0,
                threshold_down=100.0
            ),
            "queue_length": ScalingMetric(
                name="queue_length",
                current_value=0.0,
                target_value=5.0,
                weight=0.6,
                threshold_up=10.0,
                threshold_down=2.0
            )
        })
        
        # Default resource limits
        self.resource_limits.update({
            ResourceType.WORKER_THREADS: ResourceLimit(
                min_value=1,
                max_value=50,
                step_size=2,
                cooldown_seconds=30
            ),
            ResourceType.CACHE_SIZE: ResourceLimit(
                min_value=100,
                max_value=10000,
                step_size=500,
                cooldown_seconds=60
            ),
            ResourceType.CONNECTION_POOL: ResourceLimit(
                min_value=5,
                max_value=100,
                step_size=5,
                cooldown_seconds=45
            )
        })
        
        # Initialize current resources
        self.current_resources.update({
            ResourceType.WORKER_THREADS: 4,
            ResourceType.CACHE_SIZE: 1000,
            ResourceType.CONNECTION_POOL: 10
        })
    
    def update_metric(self, metric_name: str, value: float):
        """Update metric value and add to prediction history"""
        if metric_name in self.scaling_metrics:
            self.scaling_metrics[metric_name].current_value = value
            self.scaling_metrics[metric_name].timestamp = datetime.now()
            
            # Add to predictor
            self.predictor.add_metric_sample(metric_name, value)
    
    def start_monitoring(self, interval_seconds: int = 30):
        """Start automatic scaling monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Auto-scaling monitoring started")
    
    def stop_monitoring(self):
        """Stop automatic scaling monitoring"""
        self._monitoring_active = False
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join()
        self.logger.info("Auto-scaling monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring and scaling loop"""
        while not self._stop_monitoring.wait(interval_seconds):
            try:
                # Collect current system metrics
                self._collect_system_metrics()
                
                # Make scaling decisions
                actions = self._make_scaling_decisions()
                
                # Execute scaling actions
                for action in actions:
                    self._execute_scaling_action(action)
            
            except Exception as e:
                self.logger.error(f"Error in scaling monitoring loop: {e}")
    
    def _collect_system_metrics(self):
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.update_metric("cpu_usage", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.update_metric("memory_usage", memory.percent)
            
            # Simulated response time and queue length
            # In a real implementation, these would come from application metrics
            import random
            base_response = 100 + (cpu_percent * 2)  # Response time increases with CPU
            response_time = base_response + random.uniform(-20, 20)
            self.update_metric("response_time", response_time)
            
            queue_length = max(0, (cpu_percent - 50) / 10)  # Queue builds up at high CPU
            self.update_metric("queue_length", queue_length)
            
        except Exception as e:
            self.logger.warning(f"Failed to collect system metrics: {e}")
    
    def _make_scaling_decisions(self) -> List[ScalingAction]:
        """Make intelligent scaling decisions"""
        actions = []
        
        # Calculate overall scaling pressure
        total_pressure = 0.0
        total_weight = 0.0
        triggered_metrics = []
        
        for metric_name, metric in self.scaling_metrics.items():
            pressure = metric.get_scaling_pressure()
            if abs(pressure) > 0.1:  # Significant pressure
                total_pressure += pressure
                total_weight += metric.weight
                triggered_metrics.append(metric_name)
        
        if total_weight == 0:
            return actions
        
        avg_pressure = total_pressure / total_weight
        
        # Determine scaling direction and confidence
        if avg_pressure > 0.3:  # Scale up threshold
            direction = ScalingDirection.UP
            confidence = min(1.0, avg_pressure)
        elif avg_pressure < -0.3:  # Scale down threshold
            direction = ScalingDirection.DOWN
            confidence = min(1.0, abs(avg_pressure))
        else:
            return actions  # No significant scaling needed
        
        # Enhance decision with predictive analysis if using hybrid/ML strategy
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID, ScalingStrategy.MACHINE_LEARNING]:
            predictive_adjustment = self._get_predictive_adjustment(triggered_metrics)
            confidence *= predictive_adjustment
        
        # Generate scaling actions for appropriate resources
        if confidence > 0.5:  # Only act if confident enough
            for resource_type in self.current_resources.keys():
                if self._can_scale_resource(resource_type):
                    magnitude = self._calculate_scaling_magnitude(avg_pressure, confidence)
                    
                    action = ScalingAction(
                        resource_type=resource_type,
                        direction=direction,
                        magnitude=magnitude,
                        confidence=confidence,
                        reason=f"Pressure: {avg_pressure:.2f}, Confidence: {confidence:.2f}",
                        triggered_by_metrics=triggered_metrics
                    )
                    actions.append(action)
        
        return actions
    
    def _get_predictive_adjustment(self, metrics: List[str]) -> float:
        """Get predictive adjustment factor for scaling decision"""
        if self.strategy == ScalingStrategy.REACTIVE:
            return 1.0
        
        adjustment_factor = 1.0
        predictions_made = 0
        
        for metric_name in metrics:
            predicted_value = self.predictor.predict_future_value(metric_name, 10)  # 10 minutes ahead
            if predicted_value is not None:
                current_metric = self.scaling_metrics.get(metric_name)
                if current_metric:
                    trend_analysis = self.predictor.get_trend_analysis(metric_name)
                    prediction_confidence = trend_analysis.get("confidence", 0.5)
                    
                    # If prediction shows worsening conditions, increase confidence
                    if predicted_value > current_metric.threshold_up:
                        adjustment_factor += 0.2 * prediction_confidence
                    elif predicted_value < current_metric.threshold_down:
                        adjustment_factor += 0.1 * prediction_confidence
                    
                    predictions_made += 1
        
        # Average the adjustment if multiple predictions were made
        if predictions_made > 1:
            adjustment_factor = 1.0 + (adjustment_factor - 1.0) / predictions_made
        
        return min(1.5, adjustment_factor)  # Cap the adjustment
    
    def _can_scale_resource(self, resource_type: ResourceType) -> bool:
        """Check if resource can be scaled (cooldown period)"""
        if resource_type not in self.resource_limits:
            return False
        
        cooldown = self.resource_limits[resource_type].cooldown_seconds
        last_action = self.last_scaling_actions.get(resource_type)
        
        if last_action is None:
            return True
        
        return datetime.now() - last_action >= timedelta(seconds=cooldown)
    
    def _calculate_scaling_magnitude(self, pressure: float, confidence: float) -> float:
        """Calculate how much to scale based on pressure and confidence"""
        # Base magnitude on pressure intensity
        base_magnitude = min(50.0, abs(pressure) * 100)  # Cap at 50%
        
        # Adjust based on confidence
        adjusted_magnitude = base_magnitude * confidence
        
        # Ensure minimum meaningful change
        return max(10.0, adjusted_magnitude)
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action"""
        resource_type = action.resource_type
        limits = self.resource_limits.get(resource_type)
        
        if not limits:
            self.logger.warning(f"No limits defined for resource: {resource_type}")
            return
        
        current_value = self.current_resources.get(resource_type, limits.min_value)
        
        # Calculate new value
        if action.direction == ScalingDirection.UP:
            change = max(limits.step_size, current_value * action.magnitude / 100)
            new_value = min(limits.max_value, current_value + change)
        else:  # ScalingDirection.DOWN
            change = max(limits.step_size, current_value * action.magnitude / 100)
            new_value = max(limits.min_value, current_value - change)
        
        # Apply the scaling
        if new_value != current_value:
            self.current_resources[resource_type] = new_value
            self.last_scaling_actions[resource_type] = datetime.now()
            
            # Record the action
            action.magnitude = abs(new_value - current_value)  # Actual magnitude
            self.scaling_history.append(action)
            
            # Trim history
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-1000:]
            
            self.logger.info(f"Scaled {resource_type.value}: {current_value} -> {new_value} "
                           f"(reason: {action.reason})")
            
            # Apply the actual scaling (this would integrate with actual resources)
            self._apply_resource_scaling(resource_type, new_value)
    
    def _apply_resource_scaling(self, resource_type: ResourceType, new_value: float):
        """Apply actual resource scaling (placeholder for real implementation)"""
        # In a real implementation, this would:
        # - Adjust thread pool sizes
        # - Resize caches
        # - Scale connection pools
        # - Modify Kubernetes resources
        # etc.
        
        self.logger.info(f"Applied scaling: {resource_type.value} = {new_value}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status"""
        current_metrics = {
            name: {
                'current': metric.current_value,
                'target': metric.target_value,
                'pressure': metric.get_scaling_pressure(),
                'weight': metric.weight
            }
            for name, metric in self.scaling_metrics.items()
        }
        
        recent_actions = [
            action.to_dict() 
            for action in self.scaling_history[-10:]  # Last 10 actions
        ]
        
        predictions = {}
        for metric_name in self.scaling_metrics.keys():
            trend = self.predictor.get_trend_analysis(metric_name)
            if "error" not in trend:
                predictions[metric_name] = trend
        
        return {
            "strategy": self.strategy.value,
            "monitoring_active": self._monitoring_active,
            "current_metrics": current_metrics,
            "current_resources": {rt.value: val for rt, val in self.current_resources.items()},
            "resource_limits": {
                rt.value: {
                    'min': limits.min_value,
                    'max': limits.max_value,
                    'step': limits.step_size,
                    'cooldown': limits.cooldown_seconds
                }
                for rt, limits in self.resource_limits.items()
            },
            "recent_actions": recent_actions,
            "predictions": predictions,
            "total_scaling_actions": len(self.scaling_history),
            "last_update": datetime.now().isoformat()
        }


# Global auto-scaling instance
auto_scaler = AutoScalingEngine(strategy=ScalingStrategy.HYBRID)