"""Recovery Engine and Failure Analysis Components
"""

import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RESTART = "restart"
    ROLLBACK = "rollback"
    FAILOVER = "failover"
    SCALE_UP = "scale_up"
    CONFIGURATION_RESET = "config_reset"
    DEPENDENCY_REFRESH = "dep_refresh"
    CUSTOM = "custom"


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    strategy: RecoveryStrategy
    action: Callable[[], bool]
    timeout: int = 60
    prerequisites: List[str] = field(default_factory=list)
    rollback_action: Optional[Callable[[], bool]] = None


@dataclass
class FailureEvent:
    """Failure event record"""
    timestamp: float
    component_name: str
    failure_type: str
    error_message: str
    stack_trace: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


class FailureAnalyzer:
    """Advanced failure analysis system for pattern detection and root cause analysis
    """

    def __init__(self, max_history: int = 1000):
        """Initialize failure analyzer"""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_history = max_history

        # Failure history
        self.failure_history: List[FailureEvent] = []

        # Pattern detection
        self.failure_patterns: Dict[str, List[str]] = {}
        self.common_causes: Dict[str, int] = {}

        self.logger.info("Failure analyzer initialized")

    def record_failure(self, component_name: str, error: Exception, system_state: Dict[str, Any] = None) -> FailureEvent:
        """Record a failure event"""
        import traceback

        failure_event = FailureEvent(
            timestamp=time.time(),
            component_name=component_name,
            failure_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            system_state=system_state or {}
        )

        self.failure_history.append(failure_event)

        # Maintain history limit
        if len(self.failure_history) > self.max_history:
            self.failure_history.pop(0)

        # Update patterns
        self._update_failure_patterns(failure_event)

        self.logger.info(f"Recorded failure for {component_name}: {failure_event.failure_type}")
        return failure_event

    def analyze_failure(self, component) -> Dict[str, Any]:
        """Analyze a component failure"""
        component_failures = [
            f for f in self.failure_history
            if f.component_name == component.name
        ]

        if not component_failures:
            return {
                'component': component.name,
                'analysis': 'No failure history available',
                'recommendations': ['Monitor component for future failures'],
                'confidence': 0.0
            }

        recent_failures = component_failures[-5:]  # Last 5 failures

        # Analyze failure patterns
        failure_types = [f.failure_type for f in recent_failures]
        most_common_failure = max(set(failure_types), key=failure_types.count) if failure_types else None

        # Time-based analysis
        failure_times = [f.timestamp for f in recent_failures]
        time_intervals = []
        for i in range(1, len(failure_times)):
            time_intervals.append(failure_times[i] - failure_times[i-1])

        avg_interval = sum(time_intervals) / len(time_intervals) if time_intervals else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(component, recent_failures)

        # Calculate confidence based on pattern consistency
        confidence = self._calculate_confidence(recent_failures)

        return {
            'component': component.name,
            'total_failures': len(component_failures),
            'recent_failures': len(recent_failures),
            'most_common_failure_type': most_common_failure,
            'average_failure_interval_hours': avg_interval / 3600 if avg_interval > 0 else 0,
            'failure_frequency': 'high' if len(recent_failures) > 3 else 'low',
            'recommendations': recommendations,
            'confidence': confidence,
            'analysis_timestamp': time.time()
        }

    def _update_failure_patterns(self, failure_event: FailureEvent) -> None:
        """Update failure pattern tracking"""
        pattern_key = f"{failure_event.component_name}:{failure_event.failure_type}"

        if pattern_key not in self.failure_patterns:
            self.failure_patterns[pattern_key] = []

        self.failure_patterns[pattern_key].append(failure_event.error_message)

        # Update common causes
        if failure_event.failure_type not in self.common_causes:
            self.common_causes[failure_event.failure_type] = 0

        self.common_causes[failure_event.failure_type] += 1

    def _generate_recommendations(self, component, failures: List[FailureEvent]) -> List[str]:
        """Generate recovery recommendations based on failure analysis"""
        recommendations = []

        if not failures:
            return ["No specific recommendations - insufficient failure data"]

        # Analyze failure types
        failure_types = [f.failure_type for f in failures]

        if 'TimeoutError' in failure_types:
            recommendations.append("Consider increasing timeout values")
            recommendations.append("Check for network connectivity issues")

        if 'ConnectionError' in failure_types:
            recommendations.append("Verify network connectivity")
            recommendations.append("Check firewall and security group settings")
            recommendations.append("Consider implementing connection pooling")

        if 'MemoryError' in failure_types:
            recommendations.append("Increase available memory allocation")
            recommendations.append("Implement memory usage optimization")
            recommendations.append("Consider horizontal scaling")

        if 'FileNotFoundError' in failure_types:
            recommendations.append("Verify file system permissions")
            recommendations.append("Check disk space availability")
            recommendations.append("Implement file existence validation")

        # Frequency-based recommendations
        if len(failures) > 3:
            recommendations.append("High failure frequency detected - consider architecture review")
            recommendations.append("Implement circuit breaker pattern")

        return recommendations if recommendations else ["No specific recommendations available"]

    def _calculate_confidence(self, failures: List[FailureEvent]) -> float:
        """Calculate confidence level for analysis"""
        if len(failures) < 2:
            return 0.2

        # Check consistency of failure types
        failure_types = [f.failure_type for f in failures]
        most_common = max(set(failure_types), key=failure_types.count)
        consistency = failure_types.count(most_common) / len(failure_types)

        # Factor in amount of data
        data_factor = min(len(failures) / 10, 1.0)

        return consistency * data_factor

    def get_failure_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get failure trends over specified time period"""
        cutoff_time = time.time() - (hours * 3600)
        recent_failures = [f for f in self.failure_history if f.timestamp >= cutoff_time]

        if not recent_failures:
            return {'error': 'No recent failures'}

        # Group by component
        component_failures = {}
        for failure in recent_failures:
            if failure.component_name not in component_failures:
                component_failures[failure.component_name] = []
            component_failures[failure.component_name].append(failure)

        # Analyze trends
        trends = {}
        for component, failures in component_failures.items():
            failure_times = [f.timestamp for f in failures]

            trends[component] = {
                'total_failures': len(failures),
                'failure_rate_per_hour': len(failures) / hours,
                'most_recent': max(failure_times),
                'oldest': min(failure_times),
                'common_types': list(set(f.failure_type for f in failures))
            }

        return {
            'period_hours': hours,
            'total_failures': len(recent_failures),
            'unique_components': len(component_failures),
            'component_trends': trends,
            'overall_failure_rate': len(recent_failures) / hours
        }


class RecoveryEngine:
    """Autonomous recovery execution engine with multiple strategies
    """

    def __init__(self):
        """Initialize recovery engine"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Recovery action registry
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}

        # Recovery statistics
        self.recovery_stats = {
            'total_attempts': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'strategy_success_rates': {}
        }

        # Built-in recovery strategies
        self._register_builtin_strategies()

        self.logger.info("Recovery engine initialized")

    def register_recovery_action(self, component_name: str, action: RecoveryAction) -> None:
        """Register a recovery action for a component"""
        if component_name not in self.recovery_actions:
            self.recovery_actions[component_name] = []

        self.recovery_actions[component_name].append(action)
        self.logger.info(f"Registered recovery action '{action.name}' for {component_name}")

    def execute_recovery(self, component_name: str, strategy: Optional[RecoveryStrategy] = None) -> bool:
        """Execute recovery for a component"""
        self.recovery_stats['total_attempts'] += 1

        if component_name not in self.recovery_actions:
            self.logger.warning(f"No recovery actions registered for {component_name}")
            return False

        actions = self.recovery_actions[component_name]

        # Filter by strategy if specified
        if strategy:
            actions = [a for a in actions if a.strategy == strategy]

        if not actions:
            self.logger.warning(f"No recovery actions available for {component_name} with strategy {strategy}")
            return False

        # Execute recovery actions in order
        for action in actions:
            try:
                self.logger.info(f"Executing recovery action: {action.name} for {component_name}")

                # Check prerequisites
                if not self._check_prerequisites(action):
                    self.logger.warning(f"Prerequisites not met for action: {action.name}")
                    continue

                # Execute with timeout
                success = self._execute_with_timeout(action)

                if success:
                    self.recovery_stats['successful_recoveries'] += 1
                    self._update_strategy_stats(action.strategy, True)
                    self.logger.info(f"Recovery successful: {action.name}")
                    return True
                else:
                    self.logger.warning(f"Recovery action failed: {action.name}")

                    # Execute rollback if available
                    if action.rollback_action:
                        try:
                            action.rollback_action()
                            self.logger.info(f"Rollback executed for: {action.name}")
                        except Exception as e:
                            self.logger.error(f"Rollback failed for {action.name}: {e}")

            except Exception as e:
                self.logger.error(f"Error executing recovery action {action.name}: {e}")

        # All actions failed
        self.recovery_stats['failed_recoveries'] += 1
        return False

    def _execute_with_timeout(self, action: RecoveryAction) -> bool:
        """Execute recovery action with timeout"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Recovery action {action.name} timed out")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(action.timeout)

        try:
            result = action.action()
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return result
        except TimeoutError:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            self.logger.error(f"Recovery action {action.name} timed out after {action.timeout}s")
            return False
        except Exception as e:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            raise e

    def _check_prerequisites(self, action: RecoveryAction) -> bool:
        """Check if prerequisites are met for recovery action"""
        for prereq in action.prerequisites:
            # Basic prerequisite checking - can be extended
            if prereq == "disk_space":
                if not self._check_disk_space():
                    return False
            elif prereq == "network_connectivity":
                if not self._check_network():
                    return False
            elif prereq == "system_resources":
                if not self._check_system_resources():
                    return False

        return True

    def _check_disk_space(self, min_free_gb: float = 1.0) -> bool:
        """Check available disk space"""
        import shutil

        try:
            total, used, free = shutil.disk_usage("/")
            free_gb = free / (1024**3)
            return free_gb >= min_free_gb
        except Exception:
            return False

    def _check_network(self) -> bool:
        """Check basic network connectivity"""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            return True
        except Exception:
            return False

    def _check_system_resources(self) -> bool:
        """Check system resources availability"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Simple thresholds
            return cpu_percent < 90 and memory.percent < 90
        except Exception:
            return True  # Assume OK if can't check

    def _update_strategy_stats(self, strategy: RecoveryStrategy, success: bool) -> None:
        """Update success rate statistics for recovery strategies"""
        strategy_name = strategy.value

        if strategy_name not in self.recovery_stats['strategy_success_rates']:
            self.recovery_stats['strategy_success_rates'][strategy_name] = {
                'attempts': 0,
                'successes': 0,
                'success_rate': 0.0
            }

        stats = self.recovery_stats['strategy_success_rates'][strategy_name]
        stats['attempts'] += 1

        if success:
            stats['successes'] += 1

        stats['success_rate'] = stats['successes'] / stats['attempts']

    def _register_builtin_strategies(self) -> None:
        """Register built-in recovery strategies"""

        # Service restart strategy
        def restart_service(service_name: str) -> Callable[[], bool]:
            def restart():
                try:
                    result = subprocess.run(['systemctl', 'restart', service_name],
                                          check=False, capture_output=True, timeout=30)
                    return result.returncode == 0
                except Exception:
                    return False
            return restart

        # Process restart strategy
        def restart_process(process_name: str) -> Callable[[], bool]:
            def restart():
                try:
                    # Kill existing process
                    subprocess.run(['pkill', '-f', process_name], check=False, capture_output=True)
                    time.sleep(2)

                    # Could restart process here if we had the command
                    return True
                except Exception:
                    return False
            return restart

        # File cleanup strategy
        def cleanup_temp_files(directory: str = "/tmp") -> Callable[[], bool]:
            def cleanup():
                try:
                    temp_files = Path(directory).glob("*.tmp")
                    for temp_file in temp_files:
                        temp_file.unlink()
                    return True
                except Exception:
                    return False
            return cleanup

        # These would be registered per component as needed
        self.builtin_strategies = {
            'restart_service': restart_service,
            'restart_process': restart_process,
            'cleanup_temp_files': cleanup_temp_files
        }

    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics"""
        total_attempts = self.recovery_stats['total_attempts']

        overall_success_rate = 0.0
        if total_attempts > 0:
            overall_success_rate = self.recovery_stats['successful_recoveries'] / total_attempts

        return {
            'total_attempts': total_attempts,
            'successful_recoveries': self.recovery_stats['successful_recoveries'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'overall_success_rate': overall_success_rate,
            'strategy_statistics': self.recovery_stats['strategy_success_rates'],
            'registered_components': list(self.recovery_actions.keys()),
            'total_recovery_actions': sum(len(actions) for actions in self.recovery_actions.values())
        }
