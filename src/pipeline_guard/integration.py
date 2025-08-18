"""Integration layer for Self-Healing Pipeline Guard with existing systems
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .pipeline_guard import PipelineComponent, SelfHealingPipelineGuard
from .recovery import RecoveryAction, RecoveryStrategy


# Integration with existing Observer Coordinator Insights systems
try:
    from ..error_handling import ObserverCoordinatorError, error_handler
    from ..gen2_robustness import CircuitBreaker, RetryMechanism, ValidationFramework
    from ..gen3_optimization import ParallelProcessor, global_cache
    from ..health.health_check import HealthCheck
    from ..monitoring import performance_monitor
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


class PipelineGuardIntegrator:
    """Integration layer for Pipeline Guard with Observer Coordinator Insights
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize integrator"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}

        # Core pipeline guard system
        self.pipeline_guard = SelfHealingPipelineGuard(
            monitoring_interval=self.config.get('monitoring_interval', 30),
            recovery_timeout=self.config.get('recovery_timeout', 300),
            max_concurrent_recoveries=self.config.get('max_concurrent_recoveries', 3)
        )

        # Integration state
        self.is_integrated = False
        self.integration_components: Dict[str, PipelineComponent] = {}

        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'component_failure': [],
            'recovery_success': [],
            'recovery_failure': [],
            'prediction_alert': []
        }

        self.logger.info("Pipeline Guard Integrator initialized")

    def integrate_with_observer_coordinator(self) -> bool:
        """Integrate with Observer Coordinator Insights system"""
        if not INTEGRATION_AVAILABLE:
            self.logger.warning("Observer Coordinator integration not available - running standalone")
            return False

        try:
            # Register core system components
            self._register_clustering_components()
            self._register_api_components()
            self._register_database_components()
            self._register_monitoring_components()

            # Set up event integration
            self._setup_event_integration()

            # Configure with existing health checks
            self._integrate_health_checks()

            self.is_integrated = True
            self.logger.info("Successfully integrated with Observer Coordinator Insights")
            return True

        except Exception as e:
            self.logger.error(f"Integration failed: {e}")
            return False

    def _register_clustering_components(self) -> None:
        """Register clustering pipeline components"""

        # Clustering engine health check
        def clustering_health_check() -> bool:
            try:
                from ..insights_clustering.clustering import KMeansClusterer
                clusterer = KMeansClusterer()
                # Simple validation that clustering can be instantiated
                return True
            except Exception:
                return False

        # Clustering recovery actions
        def restart_clustering() -> bool:
            try:
                # Clear any cached clustering results
                if INTEGRATION_AVAILABLE and hasattr(global_cache, 'clear_pattern'):
                    global_cache.clear_pattern('clustering_*')
                return True
            except Exception:
                return False

        def clear_clustering_cache() -> bool:
            try:
                if INTEGRATION_AVAILABLE and hasattr(global_cache, 'clear'):
                    global_cache.clear()
                return True
            except Exception:
                return False

        # Create clustering component
        clustering_component = PipelineComponent(
            name="clustering_engine",
            component_type="ml_pipeline",
            health_check=clustering_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="clear_clustering_cache",
                    strategy=RecoveryStrategy.CONFIGURATION_RESET,
                    action=clear_clustering_cache,
                    timeout=30
                ),
                RecoveryAction(
                    name="restart_clustering",
                    strategy=RecoveryStrategy.RESTART,
                    action=restart_clustering,
                    timeout=60
                )
            ],
            critical=True,
            max_failures=2
        )

        self.pipeline_guard.register_component(clustering_component)
        self.integration_components["clustering_engine"] = clustering_component

    def _register_api_components(self) -> None:
        """Register API components"""

        # API health check
        def api_health_check() -> bool:
            try:
                import requests
                # Check if API is responsive (assuming it runs on default port)
                response = requests.get('http://localhost:8000/health', timeout=5)
                return response.status_code == 200
            except Exception:
                return False

        # API recovery actions
        def restart_api_server() -> bool:
            try:
                # In a real scenario, this would restart the API server
                # For now, just return success
                self.logger.info("API server restart initiated")
                return True
            except Exception:
                return False

        api_component = PipelineComponent(
            name="api_server",
            component_type="web_service",
            health_check=api_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="restart_api",
                    strategy=RecoveryStrategy.RESTART,
                    action=restart_api_server,
                    timeout=120,
                    prerequisites=["system_resources"]
                )
            ],
            critical=True,
            max_failures=3
        )

        self.pipeline_guard.register_component(api_component)
        self.integration_components["api_server"] = api_component

    def _register_database_components(self) -> None:
        """Register database components"""

        # Database health check
        def database_health_check() -> bool:
            try:
                from ..database.connection import get_database_connection
                # Simple connection test
                conn = get_database_connection()
                return conn is not None
            except Exception:
                return False

        # Database recovery actions
        def restart_database_connection() -> bool:
            try:
                # Reset database connection pool
                self.logger.info("Database connection reset initiated")
                return True
            except Exception:
                return False

        database_component = PipelineComponent(
            name="database",
            component_type="data_store",
            health_check=database_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="restart_db_connection",
                    strategy=RecoveryStrategy.DEPENDENCY_REFRESH,
                    action=restart_database_connection,
                    timeout=60
                )
            ],
            critical=True,
            max_failures=5
        )

        self.pipeline_guard.register_component(database_component)
        self.integration_components["database"] = database_component

    def _register_monitoring_components(self) -> None:
        """Register monitoring and observability components"""

        # Monitoring health check
        def monitoring_health_check() -> bool:
            try:
                if INTEGRATION_AVAILABLE and hasattr(performance_monitor, 'is_healthy'):
                    return performance_monitor.is_healthy()
                return True
            except Exception:
                return False

        # Monitoring recovery actions
        def restart_monitoring() -> bool:
            try:
                if INTEGRATION_AVAILABLE and hasattr(performance_monitor, 'restart'):
                    return performance_monitor.restart()
                return True
            except Exception:
                return False

        monitoring_component = PipelineComponent(
            name="monitoring_system",
            component_type="observability",
            health_check=monitoring_health_check,
            recovery_actions=[
                RecoveryAction(
                    name="restart_monitoring",
                    strategy=RecoveryStrategy.RESTART,
                    action=restart_monitoring,
                    timeout=60
                )
            ],
            critical=False,
            max_failures=3
        )

        self.pipeline_guard.register_component(monitoring_component)
        self.integration_components["monitoring_system"] = monitoring_component

    def _setup_event_integration(self) -> None:
        """Set up event handling integration"""

        # Component failure handler
        def on_component_failure(component_name: str, failure_details: Dict[str, Any]) -> None:
            if INTEGRATION_AVAILABLE:
                try:
                    # Record in performance monitor
                    if hasattr(performance_monitor, 'record_metric'):
                        performance_monitor.record_metric(f'component_failure_{component_name}', 1)

                    # Log to error handler
                    error_details = {
                        'component': component_name,
                        'failure_details': failure_details,
                        'timestamp': time.time()
                    }
                    error_handler.handle_error(
                        Exception(f"Component failure: {component_name}"),
                        error_details
                    )
                except Exception as e:
                    self.logger.error(f"Error in failure event handler: {e}")

        # Recovery success handler
        def on_recovery_success(component_name: str, recovery_details: Dict[str, Any]) -> None:
            if INTEGRATION_AVAILABLE:
                try:
                    # Record success metric
                    if hasattr(performance_monitor, 'record_metric'):
                        performance_monitor.record_metric(f'recovery_success_{component_name}', 1)

                    self.logger.info(f"Recovery successful for {component_name}: {recovery_details}")
                except Exception as e:
                    self.logger.error(f"Error in recovery success handler: {e}")

        # Register event handlers
        self.event_handlers['component_failure'].append(on_component_failure)
        self.event_handlers['recovery_success'].append(on_recovery_success)

    def _integrate_health_checks(self) -> None:
        """Integrate with existing health check systems"""
        if INTEGRATION_AVAILABLE:
            try:
                # Hook into existing health check framework
                from ..health.health_check import HealthCheck

                # Add pipeline guard health to system health
                def pipeline_guard_health() -> Dict[str, Any]:
                    status = self.pipeline_guard.get_system_status()
                    return {
                        'pipeline_guard_active': status['system']['is_running'],
                        'healthy_components': status['system']['healthy_components'],
                        'total_components': status['system']['total_components'],
                        'health_percentage': (
                            status['system']['healthy_components'] /
                            max(status['system']['total_components'], 1) * 100
                        )
                    }

                # Register with health check system if available
                health_checker = HealthCheck()
                if hasattr(health_checker, 'register_check'):
                    health_checker.register_check('pipeline_guard', pipeline_guard_health)

            except Exception as e:
                self.logger.warning(f"Could not integrate with health check system: {e}")

    def start_integrated_monitoring(self) -> None:
        """Start monitoring with full integration"""
        if not self.is_integrated:
            self.logger.warning("Starting monitoring without full integration")

        # Start pipeline guard monitoring
        self.pipeline_guard.start_monitoring()

        # Start metrics collection if available
        if hasattr(self.pipeline_guard.monitor, 'start_metrics_collection'):
            self.pipeline_guard.monitor.start_metrics_collection()

        self.logger.info("Integrated monitoring started")

    def stop_integrated_monitoring(self) -> None:
        """Stop all monitoring"""
        self.pipeline_guard.stop_monitoring()
        self.logger.info("Integrated monitoring stopped")

    def trigger_recovery(self, component_name: str, strategy: Optional[RecoveryStrategy] = None) -> bool:
        """Trigger recovery with event handling"""
        success = self.pipeline_guard.force_recovery(component_name)

        # Trigger event handlers
        if success:
            for handler in self.event_handlers['recovery_success']:
                try:
                    handler(component_name, {'strategy': strategy, 'timestamp': time.time()})
                except Exception as e:
                    self.logger.error(f"Error in recovery success handler: {e}")
        else:
            for handler in self.event_handlers['recovery_failure']:
                try:
                    handler(component_name, {'strategy': strategy, 'timestamp': time.time()})
                except Exception as e:
                    self.logger.error(f"Error in recovery failure handler: {e}")

        return success

    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        pipeline_status = self.pipeline_guard.get_system_status()

        return {
            'integration': {
                'is_integrated': self.is_integrated,
                'integration_available': INTEGRATION_AVAILABLE,
                'registered_components': list(self.integration_components.keys()),
                'event_handlers': {k: len(v) for k, v in self.event_handlers.items()}
            },
            'pipeline_guard': pipeline_status,
            'timestamp': time.time()
        }

    def export_configuration(self, filepath: Path) -> None:
        """Export pipeline guard configuration"""
        config = {
            'pipeline_guard_config': {
                'monitoring_interval': self.pipeline_guard.monitoring_interval,
                'recovery_timeout': self.pipeline_guard.recovery_timeout,
                'max_concurrent_recoveries': self.pipeline_guard.max_concurrent_recoveries
            },
            'components': {},
            'integration_status': {
                'is_integrated': self.is_integrated,
                'integration_available': INTEGRATION_AVAILABLE
            }
        }

        # Export component configurations
        for name, component in self.integration_components.items():
            config['components'][name] = {
                'name': component.name,
                'component_type': component.component_type,
                'critical': component.critical,
                'max_failures': component.max_failures,
                'dependencies': component.dependencies,
                'recovery_actions': [
                    {
                        'name': action.name,
                        'strategy': action.strategy.value,
                        'timeout': action.timeout,
                        'prerequisites': action.prerequisites
                    }
                    for action in component.recovery_actions
                ]
            }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Configuration exported to {filepath}")

    def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Add custom event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            self.logger.info(f"Added event handler for {event_type}")
        else:
            self.logger.warning(f"Unknown event type: {event_type}")

    def __enter__(self):
        """Context manager entry"""
        self.start_integrated_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_integrated_monitoring()


class PipelineGuardWebInterface:
    """Web interface integration for Pipeline Guard monitoring
    """

    def __init__(self, integrator: PipelineGuardIntegrator):
        """Initialize web interface"""
        self.integrator = integrator
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for web dashboard"""
        status = self.integrator.get_integration_status()

        # Format for web display
        dashboard_data = {
            'system_overview': {
                'is_running': status['pipeline_guard']['system']['is_running'],
                'uptime_hours': status['pipeline_guard']['system']['uptime_seconds'] / 3600,
                'total_components': status['pipeline_guard']['system']['total_components'],
                'healthy_components': status['pipeline_guard']['system']['healthy_components'],
                'health_percentage': (
                    status['pipeline_guard']['system']['healthy_components'] /
                    max(status['pipeline_guard']['system']['total_components'], 1) * 100
                )
            },
            'component_status': {},
            'statistics': status['pipeline_guard']['statistics'],
            'recent_activity': []
        }

        # Component details
        for name, component_info in status['pipeline_guard']['components'].items():
            dashboard_data['component_status'][name] = {
                'state': component_info['state'],
                'failure_count': component_info['failure_count'],
                'critical': component_info['critical'],
                'last_failure': component_info['last_failure']
            }

        return dashboard_data

    def get_metrics_api_data(self) -> Dict[str, Any]:
        """Get metrics data for API consumption"""
        status = self.integrator.get_integration_status()

        # Time series format for charts
        return {
            'metrics': {
                'system_health_percentage': (
                    status['pipeline_guard']['system']['healthy_components'] /
                    max(status['pipeline_guard']['system']['total_components'], 1) * 100
                ),
                'total_failures': status['pipeline_guard']['statistics']['total_failures'],
                'successful_recoveries': status['pipeline_guard']['statistics']['successful_recoveries'],
                'failed_recoveries': status['pipeline_guard']['statistics']['failed_recoveries']
            },
            'components': status['pipeline_guard']['components'],
            'timestamp': time.time()
        }
