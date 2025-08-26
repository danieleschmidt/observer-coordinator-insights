#!/usr/bin/env python3
"""Autonomous Self-Healing System
Advanced autonomous monitoring, anomaly detection, and self-healing capabilities
with predictive maintenance and intelligent incident response
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import psutil


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class IncidentSeverity(Enum):
    """Incident severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetric:
    """System performance metric"""
    name: str
    value: float
    timestamp: datetime
    threshold_warning: float
    threshold_critical: float
    unit: str = ""


@dataclass
class HealthIncident:
    """System health incident"""
    incident_id: str
    timestamp: datetime
    severity: IncidentSeverity
    component: str
    description: str
    metrics_snapshot: Dict[str, float]
    auto_resolved: bool = False
    resolution_actions: List[str] = None
    resolution_time: Optional[datetime] = None


@dataclass
class HealingAction:
    """Self-healing action"""
    action_id: str
    action_type: str
    target_component: str
    parameters: Dict[str, Any]
    expected_impact: str
    risk_level: str
    auto_execute: bool


class MetricsCollector:
    """Advanced system metrics collection"""
    
    def __init__(self, collection_interval: float = 5.0):
        self.collection_interval = collection_interval
        self.metrics_history = deque(maxlen=1000)
        self.custom_metrics = {}
        self.is_collecting = False
        self.collection_thread = None
        
    def start_collection(self):
        """Start metrics collection thread"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(target=self._collect_metrics_loop, daemon=True)
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        logger.info("Metrics collection stopped")
    
    def _collect_metrics_loop(self):
        """Continuous metrics collection loop"""
        while self.is_collecting:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append({
                    'timestamp': datetime.now(),
                    'metrics': metrics
                })
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> Dict[str, SystemMetric]:
        """Collect comprehensive system metrics"""
        now = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        
        # Network metrics (simplified)
        network = psutil.net_io_counters()
        
        # Process metrics
        process_count = len(psutil.pids())
        
        metrics = {
            'cpu_percent': SystemMetric(
                name='cpu_percent',
                value=cpu_percent,
                timestamp=now,
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit='%'
            ),
            'cpu_load_avg': SystemMetric(
                name='cpu_load_avg',
                value=load_avg,
                timestamp=now,
                threshold_warning=cpu_count * 0.8,
                threshold_critical=cpu_count * 1.2,
                unit=''
            ),
            'memory_percent': SystemMetric(
                name='memory_percent',
                value=memory.percent,
                timestamp=now,
                threshold_warning=85.0,
                threshold_critical=95.0,
                unit='%'
            ),
            'memory_available_gb': SystemMetric(
                name='memory_available_gb',
                value=memory.available / (1024**3),
                timestamp=now,
                threshold_warning=2.0,
                threshold_critical=1.0,
                unit='GB'
            ),
            'disk_percent': SystemMetric(
                name='disk_percent',
                value=disk.percent,
                timestamp=now,
                threshold_warning=85.0,
                threshold_critical=95.0,
                unit='%'
            ),
            'disk_free_gb': SystemMetric(
                name='disk_free_gb',
                value=disk.free / (1024**3),
                timestamp=now,
                threshold_warning=10.0,
                threshold_critical=5.0,
                unit='GB'
            ),
            'swap_percent': SystemMetric(
                name='swap_percent',
                value=swap.percent,
                timestamp=now,
                threshold_warning=50.0,
                threshold_critical=80.0,
                unit='%'
            ),
            'process_count': SystemMetric(
                name='process_count',
                value=float(process_count),
                timestamp=now,
                threshold_warning=500.0,
                threshold_critical=1000.0,
                unit=''
            ),
            'network_bytes_sent_mb': SystemMetric(
                name='network_bytes_sent_mb',
                value=network.bytes_sent / (1024**2),
                timestamp=now,
                threshold_warning=10000.0,
                threshold_critical=50000.0,
                unit='MB'
            ),
            'network_bytes_recv_mb': SystemMetric(
                name='network_bytes_recv_mb',
                value=network.bytes_recv / (1024**2),
                timestamp=now,
                threshold_warning=10000.0,
                threshold_critical=50000.0,
                unit='MB'
            )
        }
        
        # Add custom metrics
        metrics.update(self.custom_metrics)
        
        return metrics
    
    def add_custom_metric(self, name: str, value: float, threshold_warning: float, 
                         threshold_critical: float, unit: str = ""):
        """Add custom application metric"""
        self.custom_metrics[name] = SystemMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            threshold_warning=threshold_warning,
            threshold_critical=threshold_critical,
            unit=unit
        )
    
    def get_recent_metrics(self, minutes: int = 10) -> List[Dict]:
        """Get recent metrics within specified time window"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            entry for entry in self.metrics_history
            if entry['timestamp'] >= cutoff_time
        ]
    
    def get_metric_trends(self, metric_name: str, minutes: int = 30) -> Dict:
        """Analyze metric trends"""
        recent_data = self.get_recent_metrics(minutes)
        values = []
        
        for entry in recent_data:
            if metric_name in entry['metrics']:
                values.append(entry['metrics'][metric_name].value)
        
        if len(values) < 2:
            return {'trend': 'insufficient_data', 'values': values}
        
        # Simple trend analysis
        values_array = np.array(values)
        trend_slope = np.polyfit(range(len(values)), values_array, 1)[0]
        
        if trend_slope > 0.1:
            trend = 'increasing'
        elif trend_slope < -0.1:
            trend = 'decreasing' 
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'slope': trend_slope,
            'values': values,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }


class AnomalyDetector:
    """Advanced anomaly detection with machine learning"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_windows = {}
        self.anomaly_history = []
        
    def detect_anomalies(self, metrics: Dict[str, SystemMetric]) -> List[Dict]:
        """Detect anomalies in current metrics"""
        anomalies = []
        
        for metric_name, metric in metrics.items():
            # Statistical anomaly detection
            if self._is_statistical_anomaly(metric_name, metric.value):
                anomalies.append({
                    'type': 'statistical_anomaly',
                    'metric': metric_name,
                    'current_value': metric.value,
                    'expected_range': self._get_expected_range(metric_name),
                    'severity': self._calculate_anomaly_severity(metric),
                    'timestamp': metric.timestamp
                })
            
            # Threshold-based detection
            if metric.value >= metric.threshold_critical:
                anomalies.append({
                    'type': 'critical_threshold',
                    'metric': metric_name,
                    'current_value': metric.value,
                    'threshold': metric.threshold_critical,
                    'severity': 'critical',
                    'timestamp': metric.timestamp
                })
            elif metric.value >= metric.threshold_warning:
                anomalies.append({
                    'type': 'warning_threshold',
                    'metric': metric_name,
                    'current_value': metric.value,
                    'threshold': metric.threshold_warning,
                    'severity': 'warning',
                    'timestamp': metric.timestamp
                })
        
        if anomalies:
            self.anomaly_history.extend(anomalies)
            
        return anomalies
    
    def _is_statistical_anomaly(self, metric_name: str, current_value: float) -> bool:
        """Check if current value is statistical anomaly"""
        if metric_name not in self.baseline_windows:
            return False
        
        baseline = self.baseline_windows[metric_name]
        if len(baseline) < 10:  # Need minimum data points
            return False
        
        mean = np.mean(baseline)
        std = np.std(baseline)
        
        if std == 0:  # No variation
            return abs(current_value - mean) > 0.1
        
        z_score = abs(current_value - mean) / std
        return z_score > self.sensitivity
    
    def update_baselines(self, metrics: Dict[str, SystemMetric], window_size: int = 100):
        """Update baseline windows for metrics"""
        for metric_name, metric in metrics.items():
            if metric_name not in self.baseline_windows:
                self.baseline_windows[metric_name] = deque(maxlen=window_size)
            
            self.baseline_windows[metric_name].append(metric.value)
    
    def _get_expected_range(self, metric_name: str) -> Tuple[float, float]:
        """Get expected range for metric"""
        if metric_name not in self.baseline_windows:
            return (0.0, 0.0)
        
        baseline = list(self.baseline_windows[metric_name])
        mean = np.mean(baseline)
        std = np.std(baseline)
        
        return (mean - self.sensitivity * std, mean + self.sensitivity * std)
    
    def _calculate_anomaly_severity(self, metric: SystemMetric) -> str:
        """Calculate anomaly severity"""
        if metric.value >= metric.threshold_critical:
            return 'critical'
        elif metric.value >= metric.threshold_warning:
            return 'high'
        else:
            return 'medium'


class SelfHealingEngine:
    """Autonomous self-healing and recovery engine"""
    
    def __init__(self):
        self.healing_actions = {}
        self.healing_history = []
        self.recovery_strategies = {}
        self.auto_healing_enabled = True
        
    def register_healing_action(self, component: str, action: HealingAction):
        """Register a healing action for a component"""
        if component not in self.healing_actions:
            self.healing_actions[component] = []
        self.healing_actions[component].append(action)
        logger.info(f"Registered healing action {action.action_id} for component {component}")
    
    async def execute_healing(self, incident: HealthIncident) -> List[Dict]:
        """Execute appropriate healing actions for an incident"""
        if not self.auto_healing_enabled:
            logger.info(f"Auto-healing disabled, skipping incident {incident.incident_id}")
            return []
        
        logger.info(f"Starting healing process for incident {incident.incident_id}")
        
        applicable_actions = self._find_applicable_actions(incident)
        executed_actions = []
        
        for action in applicable_actions:
            if action.auto_execute and action.risk_level in ['low', 'medium']:
                try:
                    result = await self._execute_action(action, incident)
                    executed_actions.append(result)
                    logger.info(f"Executed healing action {action.action_id}")
                except Exception as e:
                    logger.error(f"Failed to execute healing action {action.action_id}: {e}")
                    executed_actions.append({
                        'action_id': action.action_id,
                        'status': 'failed',
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
            else:
                logger.info(f"Healing action {action.action_id} queued for manual approval (risk: {action.risk_level})")
                executed_actions.append({
                    'action_id': action.action_id,
                    'status': 'queued_for_approval',
                    'reason': f"Risk level {action.risk_level} requires manual approval"
                })
        
        self.healing_history.append({
            'incident_id': incident.incident_id,
            'actions_executed': executed_actions,
            'timestamp': datetime.now()
        })
        
        return executed_actions
    
    def _find_applicable_actions(self, incident: HealthIncident) -> List[HealingAction]:
        """Find healing actions applicable to the incident"""
        applicable = []
        
        if incident.component in self.healing_actions:
            for action in self.healing_actions[incident.component]:
                if self._is_action_applicable(action, incident):
                    applicable.append(action)
        
        # Generic system-wide actions
        if 'system' in self.healing_actions:
            for action in self.healing_actions['system']:
                if self._is_action_applicable(action, incident):
                    applicable.append(action)
        
        # Sort by risk level (lower risk first)
        risk_order = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        applicable.sort(key=lambda a: risk_order.get(a.risk_level, 5))
        
        return applicable
    
    def _is_action_applicable(self, action: HealingAction, incident: HealthIncident) -> bool:
        """Check if healing action is applicable to the incident"""
        # Basic applicability check based on component and severity
        if action.target_component not in [incident.component, 'system', 'any']:
            return False
        
        # Check if incident severity warrants the action
        severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        incident_level = severity_levels.get(incident.severity.value, 0)
        
        # Only execute actions for appropriate severity levels
        if action.action_type == 'restart' and incident_level < 3:
            return False
        if action.action_type == 'resource_cleanup' and incident_level < 2:
            return False
        
        return True
    
    async def _execute_action(self, action: HealingAction, incident: HealthIncident) -> Dict:
        """Execute a specific healing action"""
        start_time = time.time()
        
        try:
            if action.action_type == 'resource_cleanup':
                await self._cleanup_resources(action.parameters)
            elif action.action_type == 'restart_component':
                await self._restart_component(action.parameters)
            elif action.action_type == 'scale_resources':
                await self._scale_resources(action.parameters)
            elif action.action_type == 'clear_cache':
                await self._clear_cache(action.parameters)
            elif action.action_type == 'optimize_memory':
                await self._optimize_memory(action.parameters)
            else:
                raise ValueError(f"Unknown action type: {action.action_type}")
            
            execution_time = time.time() - start_time
            
            return {
                'action_id': action.action_id,
                'status': 'completed',
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now(),
                'expected_impact': action.expected_impact
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'action_id': action.action_id,
                'status': 'failed',
                'error': str(e),
                'execution_time_seconds': execution_time,
                'timestamp': datetime.now()
            }
    
    async def _cleanup_resources(self, parameters: Dict):
        """Clean up system resources"""
        # Simulate resource cleanup
        cleanup_type = parameters.get('type', 'memory')
        logger.info(f"Performing {cleanup_type} cleanup")
        await asyncio.sleep(1)  # Simulate cleanup work
    
    async def _restart_component(self, parameters: Dict):
        """Restart system component"""
        component = parameters.get('component', 'unknown')
        logger.info(f"Restarting component: {component}")
        await asyncio.sleep(2)  # Simulate restart
    
    async def _scale_resources(self, parameters: Dict):
        """Scale system resources"""
        resource_type = parameters.get('resource_type', 'cpu')
        scale_factor = parameters.get('scale_factor', 1.5)
        logger.info(f"Scaling {resource_type} by factor {scale_factor}")
        await asyncio.sleep(1)
    
    async def _clear_cache(self, parameters: Dict):
        """Clear system caches"""
        cache_type = parameters.get('cache_type', 'all')
        logger.info(f"Clearing {cache_type} cache")
        await asyncio.sleep(0.5)
    
    async def _optimize_memory(self, parameters: Dict):
        """Optimize memory usage"""
        optimization_level = parameters.get('level', 'standard')
        logger.info(f"Optimizing memory (level: {optimization_level})")
        await asyncio.sleep(1.5)


class AutonomousSelfHealingSystem:
    """Main autonomous self-healing system orchestrator"""
    
    def __init__(self, metrics_interval: float = 5.0, healing_enabled: bool = True):
        self.metrics_collector = MetricsCollector(metrics_interval)
        self.anomaly_detector = AnomalyDetector()
        self.healing_engine = SelfHealingEngine()
        
        self.system_health = HealthStatus.HEALTHY
        self.active_incidents = {}
        self.incident_history = []
        self.healing_enabled = healing_enabled
        
        self.is_running = False
        self.monitoring_task = None
        
        # Register default healing actions
        self._register_default_healing_actions()
    
    def _register_default_healing_actions(self):
        """Register default system healing actions"""
        
        # Memory cleanup action
        memory_cleanup = HealingAction(
            action_id='memory_cleanup_001',
            action_type='optimize_memory',
            target_component='system',
            parameters={'level': 'aggressive'},
            expected_impact='Reduce memory usage by 10-20%',
            risk_level='low',
            auto_execute=True
        )
        self.healing_engine.register_healing_action('system', memory_cleanup)
        
        # CPU optimization action
        cpu_optimization = HealingAction(
            action_id='cpu_optimization_001',
            action_type='resource_cleanup',
            target_component='system',
            parameters={'type': 'cpu', 'priority': 'low_priority_processes'},
            expected_impact='Reduce CPU load by 5-15%',
            risk_level='low',
            auto_execute=True
        )
        self.healing_engine.register_healing_action('system', cpu_optimization)
        
        # Disk cleanup action
        disk_cleanup = HealingAction(
            action_id='disk_cleanup_001',
            action_type='resource_cleanup',
            target_component='system',
            parameters={'type': 'disk', 'target_paths': ['/tmp', '/var/log']},
            expected_impact='Free 1-5GB disk space',
            risk_level='low',
            auto_execute=True
        )
        self.healing_engine.register_healing_action('system', disk_cleanup)
        
        # Cache clearing action
        cache_clear = HealingAction(
            action_id='cache_clear_001',
            action_type='clear_cache',
            target_component='application',
            parameters={'cache_type': 'application'},
            expected_impact='Reduce memory usage and refresh stale data',
            risk_level='low',
            auto_execute=True
        )
        self.healing_engine.register_healing_action('application', cache_clear)
    
    async def start_monitoring(self):
        """Start the autonomous monitoring and healing system"""
        if self.is_running:
            logger.warning("Monitoring system already running")
            return
        
        logger.info("🚀 Starting Autonomous Self-Healing System")
        self.is_running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start monitoring loop
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Autonomous Self-Healing System started successfully")
    
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_running:
            return
        
        logger.info("Stopping Autonomous Self-Healing System")
        self.is_running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Stop monitoring task
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Autonomous Self-Healing System stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring and healing loop"""
        logger.info("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Get current metrics
                recent_metrics = self.metrics_collector.get_recent_metrics(1)
                if not recent_metrics:
                    await asyncio.sleep(5)
                    continue
                
                current_metrics = recent_metrics[-1]['metrics']
                
                # Update anomaly detection baselines
                self.anomaly_detector.update_baselines(current_metrics)
                
                # Detect anomalies
                anomalies = self.anomaly_detector.detect_anomalies(current_metrics)
                
                # Process anomalies and create incidents
                for anomaly in anomalies:
                    await self._process_anomaly(anomaly, current_metrics)
                
                # Update system health status
                self._update_system_health(current_metrics, anomalies)
                
                # Check for incident resolution
                await self._check_incident_resolution(current_metrics)
                
                # Add custom monitoring metrics
                self._update_custom_metrics()
                
                await asyncio.sleep(5)  # Monitoring interval
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _process_anomaly(self, anomaly: Dict, current_metrics: Dict[str, SystemMetric]):
        """Process detected anomaly and potentially create incident"""
        
        # Check if similar incident already exists
        incident_key = f"{anomaly['metric']}_{anomaly['type']}"
        
        if incident_key in self.active_incidents:
            # Update existing incident
            incident = self.active_incidents[incident_key]
            incident.metrics_snapshot.update({
                m.name: m.value for m in current_metrics.values()
            })
            logger.debug(f"Updated existing incident {incident.incident_id}")
        else:
            # Create new incident
            incident = HealthIncident(
                incident_id=f"incident_{int(time.time())}_{anomaly['metric']}",
                timestamp=datetime.now(),
                severity=IncidentSeverity(anomaly['severity']),
                component='system',
                description=f"Anomaly detected in {anomaly['metric']}: {anomaly.get('current_value', 'N/A')}",
                metrics_snapshot={m.name: m.value for m in current_metrics.values()},
                resolution_actions=[]
            )
            
            self.active_incidents[incident_key] = incident
            self.incident_history.append(incident)
            
            logger.warning(f"Created new incident {incident.incident_id}: {incident.description}")
            
            # Execute healing actions if enabled
            if self.healing_enabled:
                healing_results = await self.healing_engine.execute_healing(incident)
                incident.resolution_actions = [r.get('action_id', 'unknown') for r in healing_results]
                
                # Check if any actions were executed successfully
                successful_actions = [r for r in healing_results if r.get('status') == 'completed']
                if successful_actions:
                    logger.info(f"Executed {len(successful_actions)} healing actions for incident {incident.incident_id}")
    
    def _update_system_health(self, metrics: Dict[str, SystemMetric], anomalies: List[Dict]):
        """Update overall system health status"""
        critical_anomalies = [a for a in anomalies if a.get('severity') == 'critical']
        warning_anomalies = [a for a in anomalies if a.get('severity') in ['warning', 'high']]
        
        if critical_anomalies:
            self.system_health = HealthStatus.CRITICAL
        elif warning_anomalies:
            self.system_health = HealthStatus.WARNING
        elif len(self.active_incidents) > 0:
            self.system_health = HealthStatus.RECOVERING
        else:
            self.system_health = HealthStatus.HEALTHY
    
    async def _check_incident_resolution(self, current_metrics: Dict[str, SystemMetric]):
        """Check if active incidents have been resolved"""
        resolved_incidents = []
        
        for incident_key, incident in self.active_incidents.items():
            # Check if the problematic metric has returned to normal
            metric_name = incident_key.split('_')[0]
            
            if metric_name in current_metrics:
                metric = current_metrics[metric_name]
                
                # Consider incident resolved if metric is below warning threshold
                if metric.value < metric.threshold_warning:
                    incident.auto_resolved = True
                    incident.resolution_time = datetime.now()
                    resolved_incidents.append(incident_key)
                    
                    logger.info(f"Incident {incident.incident_id} auto-resolved: {metric_name} = {metric.value}")
        
        # Remove resolved incidents from active list
        for incident_key in resolved_incidents:
            del self.active_incidents[incident_key]
    
    def _update_custom_metrics(self):
        """Update custom application metrics"""
        # Add custom metrics for the insights clustering system
        try:
            # Simulate application-specific metrics
            active_incidents_count = len(self.active_incidents)
            healing_actions_count = len(self.healing_engine.healing_history)
            
            self.metrics_collector.add_custom_metric(
                'active_incidents_count',
                float(active_incidents_count),
                5.0,
                10.0,
                'count'
            )
            
            self.metrics_collector.add_custom_metric(
                'healing_actions_executed',
                float(healing_actions_count),
                100.0,
                200.0,
                'count'
            )
            
            # Simulate clustering performance metric
            clustering_performance = 0.85 + np.random.normal(0, 0.05)  # Simulate variation
            self.metrics_collector.add_custom_metric(
                'clustering_performance_score',
                max(0.0, min(1.0, clustering_performance)),
                0.7,
                0.5,
                'score'
            )
            
        except Exception as e:
            logger.debug(f"Error updating custom metrics: {e}")
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status report"""
        recent_metrics = self.metrics_collector.get_recent_metrics(5)
        current_metrics = recent_metrics[-1]['metrics'] if recent_metrics else {}
        
        return {
            'system_health': self.system_health.value,
            'timestamp': datetime.now().isoformat(),
            'active_incidents': len(self.active_incidents),
            'total_incidents': len(self.incident_history),
            'healing_actions_executed': len(self.healing_engine.healing_history),
            'monitoring_active': self.is_running,
            'healing_enabled': self.healing_enabled,
            'current_metrics': {
                name: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'status': self._get_metric_status(metric)
                }
                for name, metric in current_metrics.items()
            },
            'recent_incidents': [
                {
                    'incident_id': incident.incident_id,
                    'severity': incident.severity.value,
                    'description': incident.description,
                    'timestamp': incident.timestamp.isoformat(),
                    'auto_resolved': incident.auto_resolved,
                    'resolution_time': incident.resolution_time.isoformat() if incident.resolution_time else None
                }
                for incident in self.incident_history[-5:]  # Last 5 incidents
            ]
        }
    
    def _get_metric_status(self, metric: SystemMetric) -> str:
        """Get status of a metric based on thresholds"""
        if metric.value >= metric.threshold_critical:
            return 'critical'
        elif metric.value >= metric.threshold_warning:
            return 'warning'
        else:
            return 'healthy'
    
    def get_healing_history(self) -> List[Dict]:
        """Get complete healing action history"""
        return self.healing_engine.healing_history
    
    def enable_healing(self):
        """Enable autonomous healing"""
        self.healing_enabled = True
        self.healing_engine.auto_healing_enabled = True
        logger.info("Autonomous healing enabled")
    
    def disable_healing(self):
        """Disable autonomous healing"""
        self.healing_enabled = False
        self.healing_engine.auto_healing_enabled = False
        logger.info("Autonomous healing disabled")


# Global self-healing system instance
autonomous_healing_system = AutonomousSelfHealingSystem()


async def start_self_healing_system():
    """Start the autonomous self-healing system"""
    await autonomous_healing_system.start_monitoring()


async def stop_self_healing_system():
    """Stop the autonomous self-healing system"""
    await autonomous_healing_system.stop_monitoring()


def get_system_health_status() -> Dict:
    """Get current system health status"""
    return autonomous_healing_system.get_system_status()


if __name__ == "__main__":
    # Demo execution
    async def demo():
        logger.info("Starting Autonomous Self-Healing System Demo")
        
        # Start the system
        await start_self_healing_system()
        
        # Run for demo period
        demo_duration = 30  # seconds
        logger.info(f"Running demo for {demo_duration} seconds...")
        
        for i in range(demo_duration):
            await asyncio.sleep(1)
            if i % 10 == 0:
                status = get_system_health_status()
                logger.info(f"System Health: {status['system_health']}, Active Incidents: {status['active_incidents']}")
        
        # Stop the system
        await stop_self_healing_system()
        
        # Print final status
        final_status = get_system_health_status()
        print(json.dumps(final_status, indent=2, default=str))
    
    # Setup logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal")
        asyncio.create_task(stop_self_healing_system())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run demo
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")