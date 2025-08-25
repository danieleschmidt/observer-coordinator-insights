#!/usr/bin/env python3
"""
Adaptive Resilience System
Self-healing infrastructure with intelligent failure recovery
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import psutil
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


@dataclass
class FailurePattern:
    """Failure pattern for learning and prediction"""
    failure_type: str
    component: str
    frequency: int
    severity: str
    recovery_strategy: str
    success_rate: float
    last_occurrence: str


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    command: str
    timeout: int
    retry_count: int
    success_criteria: Callable
    rollback_action: Optional[str] = None


class CircuitBreakerAdvanced:
    """Advanced circuit breaker with learning capabilities"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, 
                 learning_enabled: bool = True):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.learning_enabled = learning_enabled
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
        self.success_count = 0
        
        # Learning components
        self.failure_patterns = deque(maxlen=100)
        self.success_patterns = deque(maxlen=100)
        self.adaptation_history = []
    
    async def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info("🔄 Circuit breaker transitioning to half-open")
            else:
                raise Exception("Circuit breaker is open - calls blocked")
        
        try:
            start_time = time.time()
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            self._record_success(execution_time)
            
            if self.state == "half_open":
                self.state = "closed"
                logger.info("✅ Circuit breaker reset to closed")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._record_failure(str(e), execution_time)
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                self.last_failure_time = time.time()
                logger.error(f"🚨 Circuit breaker opened after {self.failure_count} failures")
                
                # Learn from failure pattern
                if self.learning_enabled:
                    await self._learn_from_failure(str(e))
            
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        if self.last_failure_time is None:
            return True
        
        return time.time() - self.last_failure_time > self.recovery_timeout
    
    def _record_success(self, execution_time: float):
        """Record successful execution"""
        self.failure_count = max(0, self.failure_count - 1)
        self.success_count += 1
        
        self.success_patterns.append({
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time
        })
    
    def _record_failure(self, error: str, execution_time: float):
        """Record failure"""
        self.failure_count += 1
        
        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "execution_time": execution_time,
            "failure_count": self.failure_count
        }
        
        self.failure_patterns.append(failure_record)
    
    async def _learn_from_failure(self, error: str):
        """Learn from failure patterns and adapt"""
        if not self.learning_enabled:
            return
        
        # Analyze recent failure patterns
        recent_failures = list(self.failure_patterns)[-10:]  # Last 10 failures
        
        # Check for recurring patterns
        error_types = defaultdict(int)
        for failure in recent_failures:
            error_type = self._classify_error(failure['error'])
            error_types[error_type] += 1
        
        # Adapt thresholds if patterns detected
        most_common_error = max(error_types.items(), key=lambda x: x[1]) if error_types else None
        
        if most_common_error and most_common_error[1] >= 3:  # 3+ occurrences
            # Increase recovery timeout for this error type
            adaptation = {
                "timestamp": datetime.now().isoformat(),
                "error_type": most_common_error[0],
                "frequency": most_common_error[1],
                "adaptation": "increased_recovery_timeout",
                "old_timeout": self.recovery_timeout,
                "new_timeout": min(self.recovery_timeout * 1.5, 300)  # Cap at 5 minutes
            }
            
            self.recovery_timeout = adaptation["new_timeout"]
            self.adaptation_history.append(adaptation)
            
            logger.info(f"🧠 Circuit breaker adapted: {most_common_error[0]} -> timeout {self.recovery_timeout}s")
    
    def _classify_error(self, error: str) -> str:
        """Classify error type for pattern analysis"""
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "out of memory" in error_lower:
            return "memory"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "permission" in error_lower or "access" in error_lower:
            return "permission"
        elif "file" in error_lower and ("not found" in error_lower or "missing" in error_lower):
            return "file_missing"
        else:
            return "unknown"


class AdaptiveResilienceSystem:
    """Adaptive resilience system with intelligent recovery"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.recovery_actions = self._initialize_recovery_actions()
        self.failure_history = deque(maxlen=1000)
        self.health_monitors = {}
        self.auto_recovery_enabled = True
        self.learning_enabled = True
        
        # Performance tracking
        self.system_baseline = {}
        self.anomaly_detection = True
        self.predictive_maintenance = True
        
        # Threading for background operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_active = False
        
    def _initialize_recovery_actions(self) -> Dict[str, RecoveryAction]:
        """Initialize recovery action definitions"""
        actions = {}
        
        actions["restart_process"] = RecoveryAction(
            name="Restart Process",
            command="pkill -f {process_name} && sleep 2",
            timeout=30,
            retry_count=3,
            success_criteria=lambda: psutil.cpu_percent() < 90
        )
        
        actions["clear_memory"] = RecoveryAction(
            name="Clear Memory",
            command="python3 -c \"import gc; gc.collect()\"",
            timeout=10,
            retry_count=1,
            success_criteria=lambda: psutil.virtual_memory().percent < 85
        )
        
        actions["restart_dependencies"] = RecoveryAction(
            name="Restart Dependencies",
            command="source venv/bin/activate && pip install --force-reinstall -e .",
            timeout=120,
            retry_count=2,
            success_criteria=lambda: True  # Always consider successful if completes
        )
        
        actions["cleanup_temp"] = RecoveryAction(
            name="Cleanup Temporary Files",
            command="find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true",
            timeout=30,
            retry_count=1,
            success_criteria=lambda: True
        )
        
        actions["reset_configuration"] = RecoveryAction(
            name="Reset Configuration",
            command="git checkout HEAD -- *.yml *.yaml *.json 2>/dev/null || true",
            timeout=15,
            retry_count=1,
            success_criteria=lambda: True,
            rollback_action="git stash"
        )
        
        return actions
    
    def get_circuit_breaker(self, component: str) -> CircuitBreakerAdvanced:
        """Get or create circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerAdvanced(
                failure_threshold=5,
                recovery_timeout=60,
                learning_enabled=self.learning_enabled
            )
        
        return self.circuit_breakers[component]
    
    async def execute_with_resilience(self, component: str, func: Callable, *args, **kwargs):
        """Execute function with full resilience protection"""
        circuit_breaker = self.get_circuit_breaker(component)
        
        try:
            # Execute with circuit breaker protection
            result = await circuit_breaker.call(func, *args, **kwargs)
            
            # Record successful operation
            await self._record_operation_success(component, func.__name__)
            
            return result
            
        except Exception as e:
            # Record failure
            await self._record_operation_failure(component, func.__name__, str(e))
            
            # Attempt automatic recovery if enabled
            if self.auto_recovery_enabled:
                recovery_success = await self._attempt_auto_recovery(component, str(e))
                
                if recovery_success:
                    logger.info(f"🔄 Auto-recovery successful for {component}")
                    # Retry original operation
                    try:
                        return await func(*args, **kwargs)
                    except Exception as retry_e:
                        logger.error(f"❌ Retry after recovery failed: {retry_e}")
                        raise
                else:
                    logger.error(f"💥 Auto-recovery failed for {component}")
            
            raise
    
    async def _record_operation_success(self, component: str, operation: str):
        """Record successful operation for learning"""
        success_record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "result": "success",
            "system_metrics": self._collect_system_metrics()
        }
        
        # Store for pattern analysis
        await self._store_operation_record(success_record)
    
    async def _record_operation_failure(self, component: str, operation: str, error: str):
        """Record operation failure for learning"""
        failure_record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "operation": operation,
            "result": "failure",
            "error": error,
            "system_metrics": self._collect_system_metrics()
        }
        
        self.failure_history.append(failure_record)
        await self._store_operation_record(failure_record)
        
        # Trigger anomaly detection
        if self.anomaly_detection:
            await self._detect_anomalies(failure_record)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0],
            "process_count": len(psutil.pids()),
            "timestamp": datetime.now().isoformat()
        }
    
    async def _store_operation_record(self, record: Dict[str, Any]):
        """Store operation record for analysis"""
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = Path(f'.terragon/logs/operations_{timestamp}.jsonl')
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Append to JSONL log
        with open(log_file, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')
    
    async def _detect_anomalies(self, failure_record: Dict[str, Any]):
        """Detect system anomalies from failure patterns"""
        if len(self.failure_history) < 5:
            return
        
        # Check for failure rate anomalies
        recent_failures = [f for f in self.failure_history if 
                          datetime.fromisoformat(f['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        if len(recent_failures) >= 5:  # 5+ failures in 1 hour
            logger.warning(f"🚨 Anomaly detected: {len(recent_failures)} failures in last hour")
            
            # Trigger predictive maintenance
            if self.predictive_maintenance:
                await self._trigger_predictive_maintenance()
    
    async def _trigger_predictive_maintenance(self):
        """Trigger predictive maintenance procedures"""
        logger.info("🔮 Triggering predictive maintenance...")
        
        maintenance_actions = [
            "cleanup_temp",
            "clear_memory"
        ]
        
        for action_name in maintenance_actions:
            try:
                await self._execute_recovery_action(action_name)
                logger.info(f"✅ Predictive maintenance: {action_name} completed")
            except Exception as e:
                logger.error(f"❌ Predictive maintenance: {action_name} failed - {e}")
    
    async def _attempt_auto_recovery(self, component: str, error: str) -> bool:
        """Attempt automatic recovery based on error pattern"""
        logger.info(f"🔄 Attempting auto-recovery for {component}: {error}")
        
        # Classify error and select recovery strategy
        error_type = self._classify_error(error)
        recovery_strategies = self._get_recovery_strategies(error_type)
        
        for strategy in recovery_strategies:
            try:
                logger.info(f"🛠️  Executing recovery strategy: {strategy}")
                success = await self._execute_recovery_action(strategy)
                
                if success:
                    # Learn successful recovery pattern
                    await self._learn_recovery_pattern(component, error_type, strategy, True)
                    return True
                else:
                    # Learn failed recovery pattern
                    await self._learn_recovery_pattern(component, error_type, strategy, False)
                    
            except Exception as e:
                logger.error(f"❌ Recovery strategy {strategy} failed: {e}")
                await self._learn_recovery_pattern(component, error_type, strategy, False)
        
        return False
    
    def _classify_error(self, error: str) -> str:
        """Classify error for targeted recovery"""
        error_lower = error.lower()
        
        if "memory" in error_lower or "out of memory" in error_lower:
            return "memory_pressure"
        elif "timeout" in error_lower:
            return "timeout"
        elif "connection" in error_lower or "network" in error_lower:
            return "network"
        elif "permission" in error_lower or "access denied" in error_lower:
            return "permission"
        elif "file not found" in error_lower or "no such file" in error_lower:
            return "missing_file"
        elif "import" in error_lower or "module" in error_lower:
            return "dependency"
        else:
            return "unknown"
    
    def _get_recovery_strategies(self, error_type: str) -> List[str]:
        """Get recovery strategies for error type"""
        strategies = {
            "memory_pressure": ["clear_memory", "cleanup_temp"],
            "timeout": ["restart_process"],
            "network": ["restart_process"],
            "permission": ["reset_configuration"],
            "missing_file": ["reset_configuration"],
            "dependency": ["restart_dependencies"],
            "unknown": ["cleanup_temp", "clear_memory"]
        }
        
        return strategies.get(error_type, ["cleanup_temp"])
    
    async def _execute_recovery_action(self, action_name: str) -> bool:
        """Execute recovery action"""
        if action_name not in self.recovery_actions:
            logger.error(f"Unknown recovery action: {action_name}")
            return False
        
        action = self.recovery_actions[action_name]
        
        for attempt in range(action.retry_count):
            try:
                logger.info(f"🔧 Executing recovery action: {action.name} (attempt {attempt + 1})")
                
                # Execute command
                proc = await asyncio.create_subprocess_shell(
                    action.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/root/repo"
                )
                
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=action.timeout
                )
                
                # Check success criteria
                if proc.returncode == 0 and action.success_criteria():
                    logger.info(f"✅ Recovery action {action.name} succeeded")
                    return True
                else:
                    logger.warning(f"⚠️  Recovery action {action.name} completed but criteria not met")
                    
            except asyncio.TimeoutError:
                logger.error(f"⏰ Recovery action {action.name} timed out")
            except Exception as e:
                logger.error(f"❌ Recovery action {action.name} failed: {e}")
        
        return False
    
    async def _learn_recovery_pattern(self, component: str, error_type: str, strategy: str, success: bool):
        """Learn from recovery attempt"""
        pattern_record = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "error_type": error_type,
            "recovery_strategy": strategy,
            "success": success
        }
        
        # Store learning record
        learning_file = Path('.terragon/learning/recovery_patterns.jsonl')
        learning_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(learning_file, 'a') as f:
            f.write(json.dumps(pattern_record) + '\n')
    
    async def start_health_monitoring(self, interval_seconds: int = 30):
        """Start continuous health monitoring"""
        logger.info(f"💓 Starting health monitoring (every {interval_seconds}s)")
        
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval_seconds)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        metrics = self._collect_system_metrics()
        
        # Check for health issues
        issues = []
        
        if metrics["cpu_percent"] > 90:
            issues.append(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
            
        if metrics["memory_percent"] > 85:
            issues.append(f"High memory usage: {metrics['memory_percent']:.1f}%")
            
        if metrics["disk_percent"] > 90:
            issues.append(f"High disk usage: {metrics['disk_percent']:.1f}%")
        
        # Store metrics
        timestamp = datetime.now().isoformat()
        self.system_metrics[timestamp] = metrics
        
        # Trigger recovery if issues detected
        if issues and self.auto_recovery_enabled:
            logger.warning(f"🚨 Health issues detected: {', '.join(issues)}")
            await self._trigger_automatic_recovery(issues)
    
    async def _trigger_automatic_recovery(self, issues: List[str]):
        """Trigger automatic recovery for health issues"""
        recovery_map = {
            "High CPU usage": ["restart_process"],
            "High memory usage": ["clear_memory", "cleanup_temp"],
            "High disk usage": ["cleanup_temp"]
        }
        
        recovery_actions = set()
        
        for issue in issues:
            for issue_type, actions in recovery_map.items():
                if issue_type in issue:
                    recovery_actions.update(actions)
        
        # Execute recovery actions
        for action_name in recovery_actions:
            try:
                await self._execute_recovery_action(action_name)
            except Exception as e:
                logger.error(f"Auto-recovery action {action_name} failed: {e}")
    
    async def validate_system_resilience(self) -> Dict[str, Any]:
        """Validate system resilience capabilities"""
        logger.info("🧪 Validating system resilience...")
        
        validation_start = time.time()
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "overall_score": 0.0
        }
        
        # Test 1: Circuit breaker functionality
        logger.info("Testing circuit breaker functionality...")
        cb_test = await self._test_circuit_breaker()
        results["tests"]["circuit_breaker"] = cb_test
        
        # Test 2: Recovery action execution
        logger.info("Testing recovery actions...")
        recovery_test = await self._test_recovery_actions()
        results["tests"]["recovery_actions"] = recovery_test
        
        # Test 3: Health monitoring
        logger.info("Testing health monitoring...")
        health_test = await self._test_health_monitoring()
        results["tests"]["health_monitoring"] = health_test
        
        # Test 4: Failure pattern learning
        logger.info("Testing failure pattern learning...")
        learning_test = await self._test_learning_system()
        results["tests"]["learning_system"] = learning_test
        
        # Calculate overall score
        test_scores = [test["score"] for test in results["tests"].values()]
        results["overall_score"] = sum(test_scores) / len(test_scores) if test_scores else 0
        
        results["validation_duration"] = time.time() - validation_start
        results["resilience_level"] = self._assess_resilience_level(results["overall_score"])
        
        # Save validation report
        await self._save_resilience_report(results)
        
        logger.info(f"🏁 Resilience validation complete: {results['overall_score']:.1f}/100")
        
        return results
    
    async def _test_circuit_breaker(self) -> Dict[str, Any]:
        """Test circuit breaker functionality"""
        test_component = "test_component"
        
        async def failing_function():
            raise Exception("Simulated failure")
        
        async def succeeding_function():
            return "success"
        
        cb = self.get_circuit_breaker(test_component)
        
        # Test failure threshold
        failure_count = 0
        for i in range(7):  # Exceed threshold of 5
            try:
                await cb.call(failing_function)
            except:
                failure_count += 1
        
        # Circuit should be open now
        circuit_open = cb.state == "open"
        
        # Wait for recovery timeout (reduced for testing)
        cb.recovery_timeout = 1  # 1 second for testing
        await asyncio.sleep(2)
        
        # Test recovery
        try:
            await cb.call(succeeding_function)
            recovery_successful = True
        except:
            recovery_successful = False
        
        score = 0
        if failure_count >= 5:
            score += 30  # Failures detected
        if circuit_open:
            score += 40  # Circuit opened properly  
        if recovery_successful:
            score += 30  # Recovery worked
        
        return {
            "score": score,
            "failures_detected": failure_count,
            "circuit_opened": circuit_open,
            "recovery_successful": recovery_successful
        }
    
    async def _test_recovery_actions(self) -> Dict[str, Any]:
        """Test recovery action execution"""
        test_actions = ["cleanup_temp", "clear_memory"]
        
        successful_actions = 0
        total_actions = len(test_actions)
        
        for action_name in test_actions:
            try:
                success = await self._execute_recovery_action(action_name)
                if success:
                    successful_actions += 1
            except Exception as e:
                logger.warning(f"Recovery action test failed: {action_name} - {e}")
        
        score = (successful_actions / total_actions) * 100
        
        return {
            "score": score,
            "successful_actions": successful_actions,
            "total_actions": total_actions
        }
    
    async def _test_health_monitoring(self) -> Dict[str, Any]:
        """Test health monitoring capabilities"""
        # Start monitoring briefly
        monitoring_task = asyncio.create_task(self.start_health_monitoring(5))
        
        await asyncio.sleep(10)  # Monitor for 10 seconds
        
        self.monitoring_active = False
        
        try:
            await asyncio.wait_for(monitoring_task, timeout=1)
        except:
            pass  # Expected timeout
        
        # Check if metrics were collected
        metrics_collected = len(self.system_metrics) > 0
        
        score = 100 if metrics_collected else 0
        
        return {
            "score": score,
            "metrics_collected": metrics_collected,
            "metric_points": len(self.system_metrics)
        }
    
    async def _test_learning_system(self) -> Dict[str, Any]:
        """Test failure pattern learning"""
        # Check if learning files exist
        learning_files = [
            '.terragon/learning/recovery_patterns.jsonl',
            '.terragon/adaptation_log.json'
        ]
        
        existing_files = [f for f in learning_files if Path(f).exists()]
        
        score = (len(existing_files) / len(learning_files)) * 100
        
        return {
            "score": score,
            "learning_files_present": len(existing_files),
            "total_learning_files": len(learning_files)
        }
    
    def _assess_resilience_level(self, score: float) -> str:
        """Assess overall resilience level"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "adequate"
        elif score >= 40:
            return "poor"
        else:
            return "critical"
    
    async def _save_resilience_report(self, results: Dict[str, Any]):
        """Save resilience validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = Path(f'.terragon/reports/resilience_validation_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate markdown report
        md_file = json_file.with_suffix('.md')
        await self._generate_resilience_markdown_report(results, md_file)
        
        logger.info(f"📊 Resilience report saved: {json_file}")
    
    async def _generate_resilience_markdown_report(self, results: Dict[str, Any], output_file: Path):
        """Generate markdown resilience report"""
        content = f"""# 🛡️ System Resilience Validation Report

**Timestamp:** {results['timestamp']}
**Overall Score:** {results['overall_score']:.1f}/100
**Resilience Level:** {results['resilience_level'].upper()}
**Validation Duration:** {results['validation_duration']:.1f} seconds

## 🧪 Test Results

"""
        
        for test_name, test_data in results["tests"].items():
            score = test_data["score"]
            status_emoji = "✅" if score >= 80 else "⚠️" if score >= 60 else "❌"
            
            content += f"### {status_emoji} {test_name.replace('_', ' ').title()}\n"
            content += f"- **Score:** {score:.1f}/100\n"
            
            # Test-specific details
            for key, value in test_data.items():
                if key != "score":
                    content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            
            content += "\n"
        
        # Circuit breaker status
        content += "## 🔄 Circuit Breaker Status\n\n"
        for component, cb in self.circuit_breakers.items():
            content += f"### {component}\n"
            content += f"- **State:** {cb.state.upper()}\n"
            content += f"- **Failure Count:** {cb.failure_count}\n"
            content += f"- **Success Count:** {cb.success_count}\n"
            content += f"- **Learning Enabled:** {'✅' if cb.learning_enabled else '❌'}\n\n"
        
        # Recent failures analysis
        if self.failure_history:
            content += "## 📉 Recent Failures Analysis\n\n"
            recent_failures = list(self.failure_history)[-5:]  # Last 5 failures
            
            for failure in recent_failures:
                content += f"- **{failure['timestamp']}:** {failure['component']} - {failure['error'][:100]}...\n"
        
        content += f"""

## 📊 System Health Metrics

- **Total Circuit Breakers:** {len(self.circuit_breakers)}
- **Recovery Actions Available:** {len(self.recovery_actions)}
- **Historical Failures:** {len(self.failure_history)}
- **Auto-Recovery Enabled:** {'✅' if self.auto_recovery_enabled else '❌'}
- **Learning Enabled:** {'✅' if self.learning_enabled else '❌'}

---
*Generated by Adaptive Resilience System*
*Report Time: {datetime.now().isoformat()}*
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Health monitoring stopped")
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get current resilience system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "learning_enabled": self.learning_enabled,
            "monitoring_active": self.monitoring_active,
            "circuit_breakers": len(self.circuit_breakers),
            "recent_failures": len(self.failure_history),
            "recovery_actions": len(self.recovery_actions),
            "system_health": self._assess_current_health()
        }
    
    def _assess_current_health(self) -> str:
        """Assess current system health"""
        metrics = self._collect_system_metrics()
        
        if (metrics["cpu_percent"] < 70 and 
            metrics["memory_percent"] < 80 and 
            metrics["disk_percent"] < 85):
            return "healthy"
        elif (metrics["cpu_percent"] < 85 and 
              metrics["memory_percent"] < 90 and 
              metrics["disk_percent"] < 95):
            return "warning"
        else:
            return "critical"


# Global resilience system instance
resilience_system = AdaptiveResilienceSystem()


async def execute_with_full_resilience(component: str, func: Callable, *args, **kwargs):
    """Execute function with full resilience protection"""
    return await resilience_system.execute_with_resilience(component, func, *args, **kwargs)


async def validate_resilience_capabilities() -> Dict[str, Any]:
    """Validate system resilience capabilities"""
    return await resilience_system.validate_system_resilience()


async def main():
    """Main execution for resilience system testing"""
    print("🛡️ Adaptive Resilience System - Validation")
    print("="*60)
    
    # Start health monitoring
    monitoring_task = asyncio.create_task(resilience_system.start_health_monitoring(10))
    
    # Run resilience validation
    validation_result = await resilience_system.validate_system_resilience()
    
    print(f"\n🏁 Resilience Validation Complete!")
    print(f"   Overall Score: {validation_result['overall_score']:.1f}/100")
    print(f"   Resilience Level: {validation_result['resilience_level'].upper()}")
    print(f"   Duration: {validation_result['validation_duration']:.1f}s")
    
    # Display test results
    print("\n🧪 Test Results:")
    for test_name, test_data in validation_result["tests"].items():
        score = test_data["score"]
        status = "PASS" if score >= 80 else "WARN" if score >= 60 else "FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status} ({score:.1f}%)")
    
    # Stop monitoring
    resilience_system.stop_monitoring()
    
    try:
        await asyncio.wait_for(monitoring_task, timeout=1)
    except:
        pass
    
    print(f"\n📁 Reports saved in .terragon/reports/")
    print("✨ Resilience system validation complete!")


if __name__ == "__main__":
    asyncio.run(main())