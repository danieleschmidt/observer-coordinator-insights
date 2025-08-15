#!/usr/bin/env python3
"""
Self-Healing Pipeline Guard - Main Entry Point
Production-ready autonomous pipeline monitoring and recovery system
"""

import sys
import argparse
import logging
import asyncio
import signal
import time
from pathlib import Path
from typing import Optional, Dict, Any
import json
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Pipeline Guard imports
from pipeline_guard import (
    SelfHealingPipelineGuard,
    PipelineComponent,
    PipelineState,
    RecoveryAction,
    RecoveryStrategy
)
from pipeline_guard.integration import PipelineGuardIntegrator
from pipeline_guard.distributed import DistributedPipelineGuard
from pipeline_guard.web_interface import WebDashboard, APIServer

# Observer Coordinator Insights integration
try:
    from insights_clustering import InsightsDataParser, KMeansClusterer
    from team_simulator import TeamCompositionSimulator
    from gen2_robustness import CircuitBreaker, ValidationFramework
    from gen3_optimization import global_cache, parallel_processor
    OBSERVER_COORDINATOR_AVAILABLE = True
except ImportError:
    OBSERVER_COORDINATOR_AVAILABLE = False


class PipelineGuardRunner:
    """
    Main runner for Self-Healing Pipeline Guard system
    """
    
    def __init__(self):
        """Initialize pipeline guard runner"""
        self.logger = self._setup_logging()
        
        # Core components
        self.integrator: Optional[PipelineGuardIntegrator] = None
        self.distributed_guard: Optional[DistributedPipelineGuard] = None
        self.web_dashboard: Optional[WebDashboard] = None
        self.api_server: Optional[APIServer] = None
        
        # Runtime state
        self.is_running = False
        self.shutdown_requested = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("Pipeline Guard Runner initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('pipeline_guard.log')
            ]
        )
        
        return logging.getLogger(__name__)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
    
    def load_configuration(self, config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'pipeline_guard': {
                'monitoring_interval': 30,
                'recovery_timeout': 300,
                'max_concurrent_recoveries': 3
            },
            'distributed': {
                'enabled': False,
                'node_id': 'node-1',
                'redis': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0
                }
            },
            'web_interface': {
                'enabled': True,
                'dashboard_port': 8080,
                'api_port': 8000,
                'host': '0.0.0.0'
            },
            'integration': {
                'observer_coordinator': OBSERVER_COORDINATOR_AVAILABLE
            }
        }
        
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    self._deep_merge(default_config, loaded_config)
                    self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                self.logger.info("Using default configuration")
        
        return default_config
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Deep merge source into target dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def initialize_components(self, config: Dict[str, Any]) -> None:
        """Initialize pipeline guard components"""
        try:
            # Initialize integrator
            self.integrator = PipelineGuardIntegrator(config['pipeline_guard'])
            
            # Integrate with Observer Coordinator if available
            if config['integration']['observer_coordinator']:
                integration_success = self.integrator.integrate_with_observer_coordinator()
                if integration_success:
                    self.logger.info("Successfully integrated with Observer Coordinator Insights")
                else:
                    self.logger.warning("Observer Coordinator integration failed - running standalone")
            
            # Initialize distributed mode if enabled
            if config['distributed']['enabled']:
                self.distributed_guard = DistributedPipelineGuard(
                    node_id=config['distributed']['node_id'],
                    redis_config=config['distributed']['redis'],
                    **config['pipeline_guard']
                )
                self.logger.info(f"Distributed mode enabled: {config['distributed']['node_id']}")
            
            # Initialize web interface if enabled
            if config['web_interface']['enabled']:
                self.web_dashboard = WebDashboard(
                    integrator=self.integrator,
                    distributed_guard=self.distributed_guard
                )
                
                self.api_server = APIServer(self.integrator)
                self.logger.info("Web interface initialized")
            
            # Register example components for demonstration
            self._register_example_components()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _register_example_components(self) -> None:
        """Register example components for demonstration"""
        try:
            # Example: Database component
            def database_health_check() -> bool:
                # Simulate database connectivity check
                import random
                return random.random() > 0.1  # 90% success rate
            
            def restart_database() -> bool:
                self.logger.info("Simulating database restart...")
                time.sleep(2)
                return True
            
            database_component = PipelineComponent(
                name="database",
                component_type="data_store",
                health_check=database_health_check,
                recovery_actions=[
                    RecoveryAction(
                        name="restart_database",
                        strategy=RecoveryStrategy.RESTART,
                        action=restart_database,
                        timeout=60
                    )
                ],
                critical=True,
                max_failures=3
            )
            
            # Example: API component
            def api_health_check() -> bool:
                import random
                return random.random() > 0.05  # 95% success rate
            
            def restart_api() -> bool:
                self.logger.info("Simulating API restart...")
                time.sleep(1)
                return True
            
            api_component = PipelineComponent(
                name="api_server",
                component_type="web_service",
                health_check=api_health_check,
                recovery_actions=[
                    RecoveryAction(
                        name="restart_api",
                        strategy=RecoveryStrategy.RESTART,
                        action=restart_api,
                        timeout=30
                    )
                ],
                critical=True,
                max_failures=2
            )
            
            # Example: Cache component
            def cache_health_check() -> bool:
                import random
                return random.random() > 0.15  # 85% success rate
            
            def clear_cache() -> bool:
                self.logger.info("Simulating cache clear...")
                if hasattr(global_cache, 'clear'):
                    global_cache.clear()
                return True
            
            cache_component = PipelineComponent(
                name="cache_system",
                component_type="cache",
                health_check=cache_health_check,
                recovery_actions=[
                    RecoveryAction(
                        name="clear_cache",
                        strategy=RecoveryStrategy.CONFIGURATION_RESET,
                        action=clear_cache,
                        timeout=15
                    )
                ],
                critical=False,
                max_failures=5
            )
            
            # Register components
            if self.distributed_guard:
                self.distributed_guard.register_component(database_component)
                self.distributed_guard.register_component(api_component)
                self.distributed_guard.register_component(cache_component)
            else:
                self.integrator.pipeline_guard.register_component(database_component)
                self.integrator.pipeline_guard.register_component(api_component)
                self.integrator.pipeline_guard.register_component(cache_component)
            
            self.logger.info("Registered example components: database, api_server, cache_system")
            
        except Exception as e:
            self.logger.error(f"Failed to register example components: {e}")
    
    def start_monitoring(self) -> None:
        """Start all monitoring systems"""
        try:
            # Start core monitoring
            if self.distributed_guard:
                self.distributed_guard.start_monitoring()
            else:
                self.integrator.start_integrated_monitoring()
            
            self.is_running = True
            self.logger.info("Pipeline Guard monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            raise
    
    def start_web_services(self, config: Dict[str, Any]) -> None:
        """Start web dashboard and API server"""
        if not config['web_interface']['enabled']:
            return
        
        try:
            import threading
            
            # Start API server in background
            if self.api_server:
                api_thread = threading.Thread(
                    target=lambda: self.api_server.run(
                        host=config['web_interface']['host'],
                        port=config['web_interface']['api_port'],
                        log_level="warning"
                    ),
                    daemon=True,
                    name="APIServer"
                )
                api_thread.start()
                self.logger.info(f"API server started on port {config['web_interface']['api_port']}")
            
            # Start web dashboard (this will block)
            if self.web_dashboard:
                self.logger.info(f"Starting web dashboard on port {config['web_interface']['dashboard_port']}")
                self.web_dashboard.run(
                    host=config['web_interface']['host'],
                    port=config['web_interface']['dashboard_port'],
                    log_level="warning"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to start web services: {e}")
    
    def stop_all_services(self) -> None:
        """Stop all services gracefully"""
        try:
            self.logger.info("Stopping all services...")
            
            # Stop monitoring
            if self.distributed_guard:
                self.distributed_guard.stop_monitoring()
            elif self.integrator:
                self.integrator.stop_integrated_monitoring()
            
            # Stop web services
            if self.web_dashboard:
                self.web_dashboard.stop_realtime_broadcasting()
            
            self.is_running = False
            self.logger.info("All services stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def run_monitoring_loop(self) -> None:
        """Run main monitoring loop"""
        self.logger.info("Starting main monitoring loop...")
        
        last_status_report = time.time()
        status_report_interval = 300  # 5 minutes
        
        try:
            while self.is_running and not self.shutdown_requested:
                # Periodic status reporting
                if time.time() - last_status_report >= status_report_interval:
                    self._log_status_report()
                    last_status_report = time.time()
                
                # Sleep briefly to avoid busy waiting
                time.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
        finally:
            self.stop_all_services()
    
    def _log_status_report(self) -> None:
        """Log periodic status report"""
        try:
            if self.integrator:
                status = self.integrator.get_integration_status()
                
                system_info = status['pipeline_guard']['system']
                stats = status['pipeline_guard']['statistics']
                
                self.logger.info(
                    f"Status Report - "
                    f"Uptime: {system_info['uptime_seconds']/3600:.1f}h, "
                    f"Components: {system_info['healthy_components']}/{system_info['total_components']}, "
                    f"Failures: {stats['total_failures']}, "
                    f"Recoveries: {stats['successful_recoveries']}/{stats['successful_recoveries'] + stats['failed_recoveries']}"
                )
                
                # Save status to file
                status_file = Path("pipeline_guard_status.json")
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to generate status report: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Self-Healing Pipeline Guard - Autonomous monitoring and recovery system"
    )
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file path (JSON format)'
    )
    parser.add_argument(
        '--distributed',
        action='store_true',
        help='Enable distributed mode'
    )
    parser.add_argument(
        '--node-id',
        type=str,
        default='node-1',
        help='Node ID for distributed mode'
    )
    parser.add_argument(
        '--web-only',
        action='store_true',
        help='Run only web interface (no monitoring)'
    )
    parser.add_argument(
        '--dashboard-port',
        type=int,
        default=8080,
        help='Web dashboard port'
    )
    parser.add_argument(
        '--api-port',
        type=int,
        default=8000,
        help='API server port'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = PipelineGuardRunner()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configuration
        config = runner.load_configuration(args.config)
        
        # Override config with command line arguments
        if args.distributed:
            config['distributed']['enabled'] = True
            config['distributed']['node_id'] = args.node_id
        
        config['web_interface']['dashboard_port'] = args.dashboard_port
        config['web_interface']['api_port'] = args.api_port
        
        # Initialize components
        runner.initialize_components(config)
        
        if args.web_only:
            # Run only web services
            runner.logger.info("Running in web-only mode")
            runner.start_web_services(config)
        else:
            # Start monitoring
            runner.start_monitoring()
            
            # Start web services in background and run monitoring loop
            if config['web_interface']['enabled']:
                import threading
                
                web_thread = threading.Thread(
                    target=lambda: runner.start_web_services(config),
                    daemon=True,
                    name="WebServices"
                )
                web_thread.start()
                
                # Give web services time to start
                time.sleep(2)
            
            # Run main monitoring loop
            runner.run_monitoring_loop()
    
    except Exception as e:
        runner.logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)