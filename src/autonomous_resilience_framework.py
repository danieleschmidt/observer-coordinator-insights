#!/usr/bin/env python3
"""
Autonomous Resilience Framework
Advanced error handling, monitoring, and self-healing capabilities for SDLC operations
"""

import asyncio
import logging
import traceback
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib
import psutil
import signal
import sys


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FailureEvent:
    """Represents a system failure event"""
    id: str
    timestamp: str
    component: str
    error_type: str
    error_message: str
    stack_trace: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    recovery_attempted: bool
    recovery_successful: bool
    context: Dict[str, Any]
    resolution_time: Optional[float] = None


@dataclass
class HealthMetric:
    """Represents a system health metric"""
    name: str
    value: float
    threshold: float
    status: str  # 'healthy', 'warning', 'critical'
    timestamp: str
    trend: str  # 'stable', 'improving', 'degrading'


class CircuitBreakerState:
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open" 
    HALF_OPEN = "half_open"


class AdvancedCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and recovery strategies"""
    
    def __init__(self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 half_open_max_calls: int = 3):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.half_open_calls = 0
        
        # Adaptive thresholds
        self.success_count = 0
        self.total_calls = 0
        self.adaptive_threshold_enabled = True
        
        # Metrics
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'circuit_opens': 0,
            'recovery_successes': 0
        }
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker"""
        with self._lock:
            self.metrics['total_calls'] += 1
            
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker {self.name} transitioning to half-open")
                else:
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    self.state = CircuitBreakerState.OPEN
                    self.last_failure_time = time.time()
                    raise CircuitBreakerOpenError(f"Circuit breaker {self.name} exceeded half-open limit")
                self.half_open_calls += 1
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        with self._lock:
            self.success_count += 1
            self.metrics['successful_calls'] += 1
            
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.metrics['recovery_successes'] += 1
                logger.info(f"Circuit breaker {self.name} recovered and closed")
            
            # Adaptive threshold adjustment
            if self.adaptive_threshold_enabled:
                self._adjust_adaptive_threshold()
    
    def _on_failure(self):
        """Handle failed call"""
        with self._lock:
            self.failure_count += 1
            self.metrics['failed_calls'] += 1
            self.last_failure_time = time.time()
            
            current_threshold = self._get_adaptive_threshold()
            
            if self.failure_count >= current_threshold:
                if self.state != CircuitBreakerState.OPEN:
                    self.metrics['circuit_opens'] += 1
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker {self.name} opened due to {self.failure_count} failures")
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _get_adaptive_threshold(self) -> int:
        """Get adaptive failure threshold based on success rate"""
        if not self.adaptive_threshold_enabled or self.total_calls < 10:
            return self.failure_threshold
        
        success_rate = self.success_count / max(self.total_calls, 1)
        
        # Adjust threshold based on historical success rate
        if success_rate > 0.95:
            return max(self.failure_threshold + 2, 3)  # More tolerant for reliable services
        elif success_rate < 0.8:
            return max(self.failure_threshold - 1, 2)  # Less tolerant for unreliable services
        
        return self.failure_threshold
    
    def _adjust_adaptive_threshold(self):
        """Adjust adaptive threshold based on recent performance"""
        self.total_calls += 1
        
        # Reset counters periodically to adapt to changing conditions
        if self.total_calls > 1000:
            self.success_count = int(self.success_count * 0.8)  # Weighted recent history
            self.total_calls = int(self.total_calls * 0.8)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics"""
        with self._lock:
            return {
                **self.metrics,
                'state': self.state,
                'failure_count': self.failure_count,
                'current_threshold': self._get_adaptive_threshold(),
                'success_rate': self.success_count / max(self.total_calls, 1)
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class AutoRecoveryManager:
    """Manages automatic recovery strategies for various failure types"""
    
    def __init__(self):
        self.recovery_strategies = {}
        self.failure_history = []
        self.recovery_attempts = {}
        
    def register_strategy(self, error_type: str, strategy: Callable[[Exception, Dict], bool]):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type}")
    
    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery based on error type"""
        error_type = type(error).__name__
        
        # Record failure
        failure_event = FailureEvent(
            id=hashlib.md5(f"{error_type}{str(error)}{time.time()}".encode()).hexdigest()[:16],
            timestamp=datetime.now().isoformat(),
            component=context.get('component', 'unknown'),
            error_type=error_type,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            severity=self._classify_severity(error),
            recovery_attempted=False,
            recovery_successful=False,
            context=context
        )
        
        self.failure_history.append(failure_event)
        
        # Attempt recovery if strategy exists
        if error_type in self.recovery_strategies:
            try:
                failure_event.recovery_attempted = True
                start_time = time.time()
                
                recovery_success = await self._execute_recovery(
                    self.recovery_strategies[error_type], error, context
                )
                
                failure_event.recovery_successful = recovery_success
                failure_event.resolution_time = time.time() - start_time
                
                if recovery_success:
                    logger.info(f"Successfully recovered from {error_type}")
                else:
                    logger.warning(f"Recovery failed for {error_type}")
                
                return recovery_success
                
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                failure_event.recovery_successful = False
                return False
        
        logger.warning(f"No recovery strategy available for {error_type}")
        return False
    
    async def _execute_recovery(self, strategy: Callable, error: Exception, context: Dict) -> bool:
        """Execute recovery strategy with timeout"""
        try:
            if asyncio.iscoroutinefunction(strategy):
                return await asyncio.wait_for(strategy(error, context), timeout=30.0)
            else:
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, strategy, error, context),
                    timeout=30.0
                )
        except asyncio.TimeoutError:
            logger.error("Recovery strategy timed out")
            return False
    
    def _classify_severity(self, error: Exception) -> str:
        """Classify error severity"""
        critical_errors = [
            'SystemExit', 'KeyboardInterrupt', 'MemoryError', 'SystemError'
        ]
        high_errors = [
            'ConnectionError', 'TimeoutError', 'PermissionError', 'FileNotFoundError'
        ]
        medium_errors = [
            'ValueError', 'TypeError', 'KeyError', 'AttributeError'
        ]
        
        error_type = type(error).__name__
        
        if error_type in critical_errors:
            return 'critical'
        elif error_type in high_errors:
            return 'high'
        elif error_type in medium_errors:
            return 'medium'
        else:
            return 'low'
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get failure and recovery statistics"""
        if not self.failure_history:
            return {"no_failures": True}
        
        recent_failures = [
            f for f in self.failure_history 
            if (datetime.now() - datetime.fromisoformat(f.timestamp)).total_seconds() < 3600
        ]
        
        error_types = {}
        recovery_success_rate = 0
        total_recovery_attempts = 0
        
        for failure in self.failure_history:
            error_types[failure.error_type] = error_types.get(failure.error_type, 0) + 1
            
            if failure.recovery_attempted:
                total_recovery_attempts += 1
                if failure.recovery_successful:
                    recovery_success_rate += 1
        
        return {
            'total_failures': len(self.failure_history),
            'recent_failures': len(recent_failures),
            'error_types': error_types,
            'recovery_success_rate': recovery_success_rate / max(total_recovery_attempts, 1),
            'total_recovery_attempts': total_recovery_attempts
        }


class SystemHealthMonitor:
    """Monitors system health and triggers alerts/recovery"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.metrics = {}
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'response_time_ms': 5000.0,
            'error_rate': 0.05
        }
        self.alert_callbacks = []
        self.monitoring_active = False
        self.monitoring_task = None
        
    def add_alert_callback(self, callback: Callable[[HealthMetric], None]):
        """Add callback for health alerts"""
        self.alert_callbacks.append(callback)
    
    async def start_monitoring(self):
        """Start continuous health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("System health monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("System health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self.check_system_health()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_system_health(self) -> Dict[str, HealthMetric]:
        """Check all system health metrics"""
        current_metrics = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        current_metrics['cpu_percent'] = HealthMetric(
            name='cpu_percent',
            value=cpu_percent,
            threshold=self.thresholds['cpu_percent'],
            status=self._get_status(cpu_percent, self.thresholds['cpu_percent']),
            timestamp=datetime.now().isoformat(),
            trend=self._calculate_trend('cpu_percent', cpu_percent)
        )
        
        # Memory usage
        memory = psutil.virtual_memory()
        current_metrics['memory_percent'] = HealthMetric(
            name='memory_percent',
            value=memory.percent,
            threshold=self.thresholds['memory_percent'],
            status=self._get_status(memory.percent, self.thresholds['memory_percent']),
            timestamp=datetime.now().isoformat(),
            trend=self._calculate_trend('memory_percent', memory.percent)
        )
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        current_metrics['disk_percent'] = HealthMetric(
            name='disk_percent',
            value=disk_percent,
            threshold=self.thresholds['disk_percent'],
            status=self._get_status(disk_percent, self.thresholds['disk_percent']),
            timestamp=datetime.now().isoformat(),
            trend=self._calculate_trend('disk_percent', disk_percent)
        )
        
        # Check for unhealthy metrics and trigger alerts
        for metric in current_metrics.values():
            if metric.status in ['warning', 'critical']:
                await self._trigger_alerts(metric)
        
        self.metrics = current_metrics
        return current_metrics
    
    def _get_status(self, value: float, threshold: float) -> str:
        """Determine health status based on threshold"""
        if value >= threshold:
            return 'critical'
        elif value >= threshold * 0.8:
            return 'warning'
        else:
            return 'healthy'
    
    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend based on historical data"""
        if metric_name not in self.metrics:
            return 'stable'
        
        previous_value = self.metrics[metric_name].value
        diff_percent = abs(current_value - previous_value) / max(previous_value, 1)
        
        if diff_percent < 0.05:
            return 'stable'
        elif current_value > previous_value:
            return 'degrading'
        else:
            return 'improving'
    
    async def _trigger_alerts(self, metric: HealthMetric):
        """Trigger alerts for unhealthy metrics"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(metric)
                else:
                    callback(metric)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")


class ResilientOperationContext:
    """Context manager for resilient operations with automatic retry and recovery"""
    
    def __init__(self, name: str, max_retries: int = 3, backoff_factor: float = 2.0,
                 circuit_breaker: Optional[AdvancedCircuitBreaker] = None,
                 recovery_manager: Optional[AutoRecoveryManager] = None):
        self.name = name
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.circuit_breaker = circuit_breaker
        self.recovery_manager = recovery_manager
        self.start_time = None
        self.context = {'operation': name}
    
    async def __aenter__(self):
        self.start_time = time.time()
        logger.debug(f"Starting resilient operation: {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time if self.start_time else 0
        
        if exc_type is None:
            logger.debug(f"Operation {self.name} completed successfully in {duration:.2f}s")
            return False
        
        logger.warning(f"Operation {self.name} failed after {duration:.2f}s: {exc_val}")
        
        # Attempt recovery if recovery manager is available
        if self.recovery_manager:
            try:
                recovery_success = await self.recovery_manager.attempt_recovery(
                    exc_val, self.context
                )
                if recovery_success:
                    logger.info(f"Recovery successful for operation {self.name}")
                    return True  # Suppress the exception
            except Exception as recovery_error:
                logger.error(f"Recovery failed for operation {self.name}: {recovery_error}")
        
        return False  # Re-raise the original exception
    
    async def execute(self, func: Callable, *args, **kwargs):
        """Execute function with resilience patterns"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.circuit_breaker:
                    return self.circuit_breaker.call(func, *args, **kwargs)
                else:
                    return await self._safe_execute(func, *args, **kwargs)
                    
            except CircuitBreakerOpenError:
                raise  # Don't retry if circuit breaker is open
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    backoff_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {self.name}, "
                        f"retrying in {backoff_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(backoff_time)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed for {self.name}")
                    break
        
        raise last_exception
    
    async def _safe_execute(self, func: Callable, *args, **kwargs):
        """Safely execute function, handling both sync and async"""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args, **kwargs)


class AutonomousResilienceOrchestrator:
    """Main orchestrator for resilience and self-healing capabilities"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.recovery_manager = AutoRecoveryManager()
        self.health_monitor = SystemHealthMonitor()
        self.active_operations = {}
        self.resilience_metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'recovered_operations': 0,
            'circuit_breaker_activations': 0
        }
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        # Set up health monitoring alerts
        self.health_monitor.add_alert_callback(self._handle_health_alert)
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common errors"""
        
        async def retry_file_operation(error: Exception, context: Dict) -> bool:
            """Recovery strategy for file operation errors"""
            if isinstance(error, (FileNotFoundError, PermissionError)):
                file_path = context.get('file_path')
                if file_path:
                    # Attempt to create directory if missing
                    try:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        return True
                    except Exception:
                        return False
            return False
        
        async def restart_service(error: Exception, context: Dict) -> bool:
            """Recovery strategy for service connection errors"""
            if isinstance(error, ConnectionError):
                service_name = context.get('service_name')
                if service_name:
                    logger.info(f"Attempting to restart service: {service_name}")
                    # Simulate service restart logic
                    await asyncio.sleep(2)  # Wait for restart
                    return True
            return False
        
        async def clear_cache_on_memory_error(error: Exception, context: Dict) -> bool:
            """Recovery strategy for memory errors"""
            if isinstance(error, MemoryError):
                logger.info("Attempting memory recovery by clearing caches")
                # Clear various caches
                import gc
                gc.collect()
                return True
            return False
        
        self.recovery_manager.register_strategy('FileNotFoundError', retry_file_operation)
        self.recovery_manager.register_strategy('PermissionError', retry_file_operation)
        self.recovery_manager.register_strategy('ConnectionError', restart_service)
        self.recovery_manager.register_strategy('MemoryError', clear_cache_on_memory_error)
    
    def create_circuit_breaker(self, name: str, **kwargs) -> AdvancedCircuitBreaker:
        """Create and register a circuit breaker"""
        circuit_breaker = AdvancedCircuitBreaker(name, **kwargs)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    @asynccontextmanager
    async def resilient_operation(self, name: str, **kwargs):
        """Create resilient operation context"""
        circuit_breaker = self.circuit_breakers.get(name)
        
        context = ResilientOperationContext(
            name=name,
            circuit_breaker=circuit_breaker,
            recovery_manager=self.recovery_manager,
            **kwargs
        )
        
        operation_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:8]
        self.active_operations[operation_id] = context
        self.resilience_metrics['total_operations'] += 1
        
        try:
            async with context as ctx:
                yield ctx
            self.resilience_metrics['successful_operations'] += 1
        except Exception as e:
            self.resilience_metrics['failed_operations'] += 1
            raise
        finally:
            del self.active_operations[operation_id]
    
    async def _handle_health_alert(self, metric: HealthMetric):
        """Handle health monitoring alerts"""
        logger.warning(
            f"Health alert: {metric.name} = {metric.value:.1f} "
            f"(threshold: {metric.threshold}, status: {metric.status})"
        )
        
        # Implement automatic responses based on metric
        if metric.name == 'memory_percent' and metric.status == 'critical':
            # Trigger memory cleanup
            import gc
            gc.collect()
            logger.info("Triggered garbage collection due to high memory usage")
        
        elif metric.name == 'cpu_percent' and metric.status == 'critical':
            # Reduce concurrent operations
            if len(self.active_operations) > 1:
                logger.info("High CPU detected, operations may be throttled")
        
        # Record health event
        await self._record_health_event(metric)
    
    async def _record_health_event(self, metric: HealthMetric):
        """Record health monitoring events"""
        event = {
            'timestamp': metric.timestamp,
            'metric_name': metric.name,
            'value': metric.value,
            'status': metric.status,
            'threshold': metric.threshold,
            'trend': metric.trend
        }
        
        # Save to health log
        health_log_path = Path('.terragon') / 'health_events.jsonl'
        health_log_path.parent.mkdir(exist_ok=True)
        
        with open(health_log_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    async def start_health_monitoring(self):
        """Start system health monitoring"""
        await self.health_monitor.start_monitoring()
    
    async def stop_health_monitoring(self):
        """Stop system health monitoring"""
        await self.health_monitor.stop_monitoring()
    
    def get_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report"""
        circuit_breaker_stats = {
            name: cb.get_metrics() 
            for name, cb in self.circuit_breakers.items()
        }
        
        failure_stats = self.recovery_manager.get_failure_statistics()
        
        current_health = self.health_monitor.metrics
        health_summary = {}
        for name, metric in current_health.items():
            health_summary[name] = {
                'value': metric.value,
                'status': metric.status,
                'trend': metric.trend
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'resilience_metrics': self.resilience_metrics,
            'circuit_breakers': circuit_breaker_stats,
            'failure_statistics': failure_stats,
            'health_summary': health_summary,
            'active_operations': len(self.active_operations),
            'system_status': self._calculate_system_status(health_summary)
        }
    
    def _calculate_system_status(self, health_summary: Dict) -> str:
        """Calculate overall system status"""
        if not health_summary:
            return 'unknown'
        
        status_scores = {'healthy': 0, 'warning': 1, 'critical': 2}
        total_score = sum(status_scores.get(metric['status'], 0) for metric in health_summary.values())
        avg_score = total_score / len(health_summary)
        
        if avg_score < 0.5:
            return 'healthy'
        elif avg_score < 1.5:
            return 'warning'
        else:
            return 'critical'
    
    async def save_resilience_report(self, output_path: str = '.terragon'):
        """Save resilience report to file"""
        report = self.get_resilience_report()
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"resilience_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Resilience report saved to {report_file}")
        return report_file


# Global resilience orchestrator instance
resilience_orchestrator = AutonomousResilienceOrchestrator()


# Convenience decorators and functions
def resilient_operation(name: str, **kwargs):
    """Decorator for resilient operations"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **func_kwargs):
            async with resilience_orchestrator.resilient_operation(name, **kwargs) as ctx:
                return await ctx.execute(func, *args, **func_kwargs)
        return wrapper
    return decorator


async def with_resilience(name: str, func: Callable, *args, **kwargs):
    """Execute function with resilience patterns"""
    async with resilience_orchestrator.resilient_operation(name) as ctx:
        return await ctx.execute(func, *args, **kwargs)


# Initialize resilience system
async def initialize_resilience_framework():
    """Initialize the resilience framework"""
    logger.info("🛡️  Initializing Autonomous Resilience Framework...")
    
    # Create default circuit breakers for common operations
    resilience_orchestrator.create_circuit_breaker('file_operations', failure_threshold=3)
    resilience_orchestrator.create_circuit_breaker('network_operations', failure_threshold=5)
    resilience_orchestrator.create_circuit_breaker('database_operations', failure_threshold=4)
    resilience_orchestrator.create_circuit_breaker('computation_operations', failure_threshold=10)
    
    # Start health monitoring
    await resilience_orchestrator.start_health_monitoring()
    
    logger.info("✅ Resilience framework initialized successfully")


# Shutdown handler
async def shutdown_resilience_framework():
    """Shutdown the resilience framework gracefully"""
    logger.info("🛑 Shutting down Resilience Framework...")
    
    await resilience_orchestrator.stop_health_monitoring()
    await resilience_orchestrator.save_resilience_report()
    
    logger.info("✅ Resilience framework shutdown complete")


if __name__ == "__main__":
    async def demo_resilience():
        """Demonstrate resilience framework capabilities"""
        await initialize_resilience_framework()
        
        # Demonstrate resilient operation
        @resilient_operation("demo_operation", max_retries=2)
        async def unstable_operation():
            import random
            if random.random() < 0.7:  # 70% failure rate
                raise ConnectionError("Simulated connection failure")
            return "Success!"
        
        # Test resilient execution
        try:
            for i in range(5):
                result = await unstable_operation()
                print(f"Attempt {i+1}: {result}")
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Operation failed: {e}")
        
        # Generate report
        report = resilience_orchestrator.get_resilience_report()
        print(f"\n📊 Resilience Report:")
        print(f"Total Operations: {report['resilience_metrics']['total_operations']}")
        print(f"Success Rate: {report['resilience_metrics']['successful_operations']/report['resilience_metrics']['total_operations']*100:.1f}%")
        print(f"System Status: {report['system_status']}")
        
        await shutdown_resilience_framework()
    
    asyncio.run(demo_resilience())