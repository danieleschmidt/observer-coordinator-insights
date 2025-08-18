"""Core Self-Healing Pipeline Guard Implementation
Autonomous system for pipeline protection and recovery
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import PipelineComponent, PipelineState
from .monitoring import HealthChecker, PipelineMonitor
from .predictor import FailurePredictor
from .recovery import FailureAnalyzer, RecoveryEngine


class SelfHealingPipelineGuard:
    """Core Self-Healing Pipeline Guard System
    
    Provides autonomous monitoring, failure detection, and recovery
    for complex data processing pipelines.
    """

    def __init__(self,
                 monitoring_interval: int = 30,
                 recovery_timeout: int = 300,
                 max_concurrent_recoveries: int = 3):
        """Initialize the pipeline guard system"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Core configuration
        self.monitoring_interval = monitoring_interval
        self.recovery_timeout = recovery_timeout
        self.max_concurrent_recoveries = max_concurrent_recoveries

        # Component registry
        self.components: Dict[str, PipelineComponent] = {}
        self.component_graph: Dict[str, List[str]] = {}

        # Core systems
        self.monitor = PipelineMonitor()
        self.health_checker = HealthChecker()
        self.recovery_engine = RecoveryEngine()
        self.failure_analyzer = FailureAnalyzer()
        self.failure_predictor = FailurePredictor()

        # Runtime state
        self.is_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.recovery_executor = ThreadPoolExecutor(max_workers=max_concurrent_recoveries)

        # Statistics
        self.stats = {
            'total_failures': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'uptime_start': time.time(),
            'last_health_check': None
        }

        self.logger.info("Self-Healing Pipeline Guard initialized")

    def register_component(self, component: PipelineComponent) -> None:
        """Register a component for monitoring"""
        self.components[component.name] = component
        self.component_graph[component.name] = component.dependencies.copy()

        self.logger.info(f"Registered component: {component.name} ({component.component_type})")

        # Initialize monitoring for this component
        self.monitor.add_component(component.name, component.health_check)

    def start_monitoring(self) -> None:
        """Start the autonomous monitoring system"""
        if self.is_running:
            self.logger.warning("Pipeline guard already running")
            return

        self.is_running = True
        self.stats['uptime_start'] = time.time()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PipelineGuardMonitor"
        )
        self.monitor_thread.start()

        self.logger.info("Self-healing pipeline monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the monitoring system"""
        self.is_running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        self.recovery_executor.shutdown(wait=True)
        self.logger.info("Pipeline guard monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_running:
            try:
                self._check_all_components()
                self._analyze_system_health()
                self._predict_failures()

                self.stats['last_health_check'] = time.time()

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.monitoring_interval)

    def _check_all_components(self) -> None:
        """Check health of all registered components"""
        for component_name, component in self.components.items():
            try:
                is_healthy = self.health_checker.check_component_health(component)

                if not is_healthy:
                    self._handle_component_failure(component)
                else:
                    self._handle_component_recovery(component)

            except Exception as e:
                self.logger.error(f"Error checking component {component_name}: {e}")

    def _handle_component_failure(self, component: PipelineComponent) -> None:
        """Handle component failure"""
        component.failure_count += 1
        component.last_failure = time.time()
        self.stats['total_failures'] += 1

        # Update component state based on failure count
        if component.failure_count >= component.max_failures:
            component.state = PipelineState.CRITICAL if component.critical else PipelineState.FAILING
        else:
            component.state = PipelineState.DEGRADED

        self.logger.warning(
            f"Component {component.name} failure #{component.failure_count} "
            f"(state: {component.state.value})"
        )

        # Trigger recovery if needed
        if component.state in [PipelineState.FAILING, PipelineState.CRITICAL]:
            self._trigger_recovery(component)

    def _handle_component_recovery(self, component: PipelineComponent) -> None:
        """Handle component recovery"""
        if component.state != PipelineState.HEALTHY:
            component.state = PipelineState.HEALTHY
            component.failure_count = 0
            self.logger.info(f"Component {component.name} recovered")

    def _trigger_recovery(self, component: PipelineComponent) -> None:
        """Trigger autonomous recovery for a component"""
        if component.state == PipelineState.RECOVERING:
            return  # Already recovering

        component.state = PipelineState.RECOVERING

        # Submit recovery task
        future = self.recovery_executor.submit(self._execute_recovery, component)

        def recovery_callback(fut):
            try:
                success = fut.result()
                if success:
                    self.stats['successful_recoveries'] += 1
                    self.logger.info(f"Recovery successful for {component.name}")
                else:
                    self.stats['failed_recoveries'] += 1
                    self.logger.error(f"Recovery failed for {component.name}")
            except Exception as e:
                self.stats['failed_recoveries'] += 1
                self.logger.error(f"Recovery error for {component.name}: {e}")

        future.add_done_callback(recovery_callback)

    def _execute_recovery(self, component: PipelineComponent) -> bool:
        """Execute recovery actions for a component"""
        self.logger.info(f"Starting recovery for component: {component.name}")

        # Analyze failure before recovery
        failure_analysis = self.failure_analyzer.analyze_failure(component)
        self.logger.info(f"Failure analysis: {failure_analysis}")

        # Execute recovery actions
        for i, recovery_action in enumerate(component.recovery_actions):
            try:
                self.logger.info(f"Executing recovery action {i+1}/{len(component.recovery_actions)}")
                success = recovery_action()

                if success:
                    # Test if component is now healthy
                    if component.health_check():
                        component.state = PipelineState.HEALTHY
                        component.failure_count = 0
                        return True

            except Exception as e:
                self.logger.error(f"Recovery action {i+1} failed: {e}")

        # All recovery actions failed
        component.state = PipelineState.OFFLINE if component.critical else PipelineState.FAILING
        return False

    def _analyze_system_health(self) -> None:
        """Analyze overall system health"""
        total_components = len(self.components)
        if total_components == 0:
            return

        healthy_count = sum(1 for c in self.components.values() if c.state == PipelineState.HEALTHY)
        health_percentage = (healthy_count / total_components) * 100

        self.monitor.record_system_health(health_percentage)

        # Log system status periodically
        if int(time.time()) % 300 == 0:  # Every 5 minutes
            self.logger.info(f"System health: {health_percentage:.1f}% ({healthy_count}/{total_components} components healthy)")

    def _predict_failures(self) -> None:
        """Run predictive failure analysis"""
        try:
            # Collect current metrics for all components
            component_metrics = {}
            for name, component in self.components.items():
                component_metrics[name] = {
                    'failure_count': component.failure_count,
                    'state': component.state.value,
                    'last_failure': component.last_failure,
                    'critical': component.critical
                }

            # Run prediction analysis
            predictions = self.failure_predictor.predict_component_failures(component_metrics)

            # Act on high-risk predictions
            for component_name, risk_score in predictions.items():
                if risk_score > 0.8:  # High risk threshold
                    self.logger.warning(f"High failure risk predicted for {component_name}: {risk_score:.2f}")

        except Exception as e:
            self.logger.error(f"Error in failure prediction: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = time.time() - self.stats['uptime_start']

        component_status = {}
        for name, component in self.components.items():
            component_status[name] = {
                'state': component.state.value,
                'failure_count': component.failure_count,
                'last_failure': component.last_failure,
                'critical': component.critical
            }

        return {
            'system': {
                'is_running': self.is_running,
                'uptime_seconds': uptime,
                'total_components': len(self.components),
                'healthy_components': sum(1 for c in self.components.values() if c.state == PipelineState.HEALTHY),
                'last_health_check': self.stats['last_health_check']
            },
            'components': component_status,
            'statistics': self.stats.copy(),
            'monitoring': {
                'interval_seconds': self.monitoring_interval,
                'recovery_timeout': self.recovery_timeout,
                'max_concurrent_recoveries': self.max_concurrent_recoveries
            }
        }

    def save_status_report(self, output_path: Path) -> None:
        """Save comprehensive status report"""
        status = self.get_system_status()

        with open(output_path, 'w') as f:
            json.dump(status, f, indent=2, default=str)

        self.logger.info(f"Status report saved to {output_path}")

    def force_recovery(self, component_name: str) -> bool:
        """Manually trigger recovery for a specific component"""
        if component_name not in self.components:
            self.logger.error(f"Component {component_name} not found")
            return False

        component = self.components[component_name]
        self.logger.info(f"Forcing recovery for component: {component_name}")

        return self._execute_recovery(component)

    def get_component_dependencies(self, component_name: str) -> List[str]:
        """Get dependency chain for a component"""
        if component_name not in self.component_graph:
            return []

        visited = set()
        dependencies = []

        def collect_deps(name):
            if name in visited:
                return
            visited.add(name)

            for dep in self.component_graph.get(name, []):
                collect_deps(dep)
                if dep not in dependencies:
                    dependencies.append(dep)

        collect_deps(component_name)
        return dependencies

    def __enter__(self):
        """Context manager entry"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_monitoring()
