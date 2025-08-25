#!/usr/bin/env python3
"""
Progressive Quality Gates System
Autonomous quality validation with evolving standards
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import subprocess
import sys


logger = logging.getLogger(__name__)


@dataclass
class QualityGate:
    """Individual quality gate definition"""
    name: str
    command: str
    threshold: float
    weight: float
    timeout: int = 300
    retry_count: int = 3
    generation: int = 1
    required: bool = True


@dataclass 
class QualityResult:
    """Quality gate execution result"""
    gate_name: str
    status: str  # passed, failed, skipped, error
    score: float
    message: str
    execution_time: float
    generation: int
    details: Dict[str, Any]


class ProgressiveQualityGates:
    """Progressive quality gates that evolve with each generation"""
    
    def __init__(self):
        self.gates = self._initialize_quality_gates()
        self.execution_history = []
        self.current_generation = 1
        self.adaptation_enabled = True
        
    def _initialize_quality_gates(self) -> Dict[str, QualityGate]:
        """Initialize quality gates for all generations"""
        gates = {}
        
        # Generation 1: Basic functionality gates
        gates["unit_tests"] = QualityGate(
            name="Unit Tests",
            command="source venv/bin/activate && python -m pytest tests/unit/ -v --tb=short",
            threshold=80.0,
            weight=20.0,
            generation=1
        )
        
        gates["code_syntax"] = QualityGate(
            name="Code Syntax Check", 
            command="source venv/bin/activate && python -m py_compile src/*.py",
            threshold=100.0,
            weight=15.0,
            generation=1
        )
        
        gates["basic_security"] = QualityGate(
            name="Basic Security Scan",
            command="source venv/bin/activate && python -m bandit -r src/ -f json",
            threshold=70.0,
            weight=15.0,
            generation=1
        )
        
        # Generation 2: Robustness gates
        gates["integration_tests"] = QualityGate(
            name="Integration Tests",
            command="source venv/bin/activate && python -m pytest tests/integration/ -v",
            threshold=85.0,
            weight=25.0,
            generation=2
        )
        
        gates["type_checking"] = QualityGate(
            name="Type Checking",
            command="source venv/bin/activate && python -m mypy src/",
            threshold=90.0,
            weight=20.0,
            generation=2
        )
        
        gates["code_quality"] = QualityGate(
            name="Code Quality Analysis",
            command="source venv/bin/activate && python -m ruff check src/",
            threshold=85.0,
            weight=20.0,
            generation=2
        )
        
        gates["security_comprehensive"] = QualityGate(
            name="Comprehensive Security",
            command="source venv/bin/activate && python scripts/security_scan.py",
            threshold=85.0,
            weight=25.0,
            generation=2
        )
        
        # Generation 3: Performance and scale gates
        gates["performance_tests"] = QualityGate(
            name="Performance Benchmarks",
            command="source venv/bin/activate && python -m pytest tests/performance/ --benchmark-only",
            threshold=90.0,
            weight=30.0,
            generation=3
        )
        
        gates["load_tests"] = QualityGate(
            name="Load Testing",
            command="source venv/bin/activate && python -m pytest tests/performance/test_load_scenarios.py",
            threshold=85.0,
            weight=25.0,
            generation=3
        )
        
        gates["memory_profiling"] = QualityGate(
            name="Memory Profiling",
            command="source venv/bin/activate && python -m memory_profiler src/main.py --quick-demo",
            threshold=80.0,
            weight=20.0,
            generation=3,
            required=False
        )
        
        gates["e2e_tests"] = QualityGate(
            name="End-to-End Tests",
            command="source venv/bin/activate && python -m pytest tests/e2e/ -v",
            threshold=90.0,
            weight=35.0,
            generation=3
        )
        
        return gates
    
    async def execute_generation_gates(self, generation: int = None) -> Dict[str, Any]:
        """Execute quality gates for specified generation"""
        if generation is None:
            generation = self.current_generation
            
        logger.info(f"🛡️ Executing Generation {generation} Quality Gates...")
        
        # Filter gates by generation
        applicable_gates = {
            name: gate for name, gate in self.gates.items() 
            if gate.generation <= generation
        }
        
        results = []
        start_time = time.time()
        total_weight = sum(gate.weight for gate in applicable_gates.values())
        weighted_score = 0.0
        
        # Execute gates in parallel where possible
        execution_tasks = []
        for gate_name, gate in applicable_gates.items():
            task = self._execute_single_gate(gate)
            execution_tasks.append((gate_name, task))
        
        # Wait for all gates to complete
        for gate_name, task in execution_tasks:
            try:
                result = await task
                results.append(result)
                
                # Calculate weighted contribution
                gate = applicable_gates[gate_name]
                if result.status == "passed":
                    weighted_score += gate.weight
                elif result.status == "failed":
                    # Partial credit based on score
                    weighted_score += gate.weight * (result.score / 100.0)
                    
                logger.info(f"  {'✅' if result.status == 'passed' else '❌'} {result.gate_name}: {result.status}")
                
            except Exception as e:
                logger.error(f"  ❌ {gate_name}: Error executing gate - {e}")
                results.append(QualityResult(
                    gate_name=gate_name,
                    status="error",
                    score=0.0,
                    message=str(e),
                    execution_time=0.0,
                    generation=generation,
                    details={"error": str(e)}
                ))
        
        execution_time = time.time() - start_time
        overall_score = (weighted_score / total_weight) * 100 if total_weight > 0 else 0
        
        # Determine overall status
        passed_gates = len([r for r in results if r.status == "passed"])
        failed_gates = len([r for r in results if r.status == "failed"])
        error_gates = len([r for r in results if r.status == "error"])
        
        if overall_score >= 85 and error_gates == 0:
            overall_status = "passed"
        elif overall_score >= 70:
            overall_status = "warning"
        else:
            overall_status = "failed"
        
        execution_report = {
            "timestamp": datetime.now().isoformat(),
            "generation": generation,
            "overall_status": overall_status,
            "overall_score": overall_score,
            "execution_time": execution_time,
            "gates_summary": {
                "total": len(applicable_gates),
                "passed": passed_gates,
                "failed": failed_gates,
                "errors": error_gates
            },
            "gate_results": [asdict(result) for result in results],
            "recommendations": self._generate_recommendations(results, overall_score)
        }
        
        # Store execution history
        self.execution_history.append(execution_report)
        
        # Save detailed report
        await self._save_execution_report(execution_report)
        
        # Auto-adapt gates if enabled
        if self.adaptation_enabled:
            await self._adapt_quality_gates(execution_report)
        
        logger.info(f"🏁 Generation {generation} Quality Gates Complete: {overall_status.upper()} ({overall_score:.1f}%)")
        
        return execution_report
    
    async def _execute_single_gate(self, gate: QualityGate) -> QualityResult:
        """Execute a single quality gate"""
        start_time = time.time()
        
        for attempt in range(gate.retry_count):
            try:
                # Execute command with timeout
                proc = await asyncio.create_subprocess_shell(
                    gate.command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd="/root/repo"
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        proc.communicate(), timeout=gate.timeout
                    )
                    
                    execution_time = time.time() - start_time
                    return_code = proc.returncode
                    
                    # Parse result based on gate type
                    score, status, message, details = self._parse_gate_result(
                        gate, return_code, stdout.decode(), stderr.decode()
                    )
                    
                    # Check if gate passed threshold
                    if score >= gate.threshold:
                        final_status = "passed"
                    elif not gate.required:
                        final_status = "skipped"
                    else:
                        final_status = "failed"
                    
                    return QualityResult(
                        gate_name=gate.name,
                        status=final_status,
                        score=score,
                        message=message,
                        execution_time=execution_time,
                        generation=gate.generation,
                        details=details
                    )
                    
                except asyncio.TimeoutError:
                    if attempt < gate.retry_count - 1:
                        logger.warning(f"Gate {gate.name} timed out, retrying...")
                        continue
                    else:
                        return QualityResult(
                            gate_name=gate.name,
                            status="error",
                            score=0.0,
                            message=f"Execution timed out after {gate.timeout}s",
                            execution_time=time.time() - start_time,
                            generation=gate.generation,
                            details={"timeout": gate.timeout}
                        )
                        
            except Exception as e:
                if attempt < gate.retry_count - 1:
                    logger.warning(f"Gate {gate.name} failed, retrying: {e}")
                    continue
                else:
                    return QualityResult(
                        gate_name=gate.name,
                        status="error", 
                        score=0.0,
                        message=f"Execution error: {str(e)}",
                        execution_time=time.time() - start_time,
                        generation=gate.generation,
                        details={"error": str(e)}
                    )
    
    def _parse_gate_result(self, gate: QualityGate, return_code: int, stdout: str, stderr: str) -> tuple[float, str, str, Dict[str, Any]]:
        """Parse gate execution result into score and status"""
        details = {
            "return_code": return_code,
            "stdout_length": len(stdout),
            "stderr_length": len(stderr)
        }
        
        # Basic success/failure based on return code
        if return_code == 0:
            base_score = 100.0
            status = "passed"
            message = f"{gate.name} completed successfully"
        else:
            base_score = 0.0
            status = "failed"
            message = f"{gate.name} failed with return code {return_code}"
        
        # Gate-specific parsing
        if "pytest" in gate.command:
            score, message, parse_details = self._parse_pytest_output(stdout, stderr)
            details.update(parse_details)
        elif "bandit" in gate.command:
            score, message, parse_details = self._parse_bandit_output(stdout, stderr)
            details.update(parse_details)
        elif "mypy" in gate.command:
            score, message, parse_details = self._parse_mypy_output(stdout, stderr)
            details.update(parse_details)
        elif "ruff" in gate.command:
            score, message, parse_details = self._parse_ruff_output(stdout, stderr, return_code)
            details.update(parse_details)
        else:
            score = base_score
        
        return score, status, message, details
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> tuple[float, str, Dict[str, Any]]:
        """Parse pytest output for detailed scoring"""
        details = {}
        
        # Look for test results summary
        lines = stdout.split('\n')
        passed = failed = skipped = 0
        
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "5 passed, 2 failed, 1 skipped"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed" and i > 0:
                        passed = int(parts[i-1])
                    elif part == "failed" and i > 0:
                        failed = int(parts[i-1])
                    elif part == "skipped" and i > 0:
                        skipped = int(parts[i-1])
        
        total_tests = passed + failed + skipped
        if total_tests > 0:
            score = (passed / total_tests) * 100
            message = f"Tests: {passed} passed, {failed} failed, {skipped} skipped"
            details = {
                "passed": passed,
                "failed": failed, 
                "skipped": skipped,
                "total": total_tests
            }
        else:
            score = 0.0
            message = "No tests found or executed"
            details = {"no_tests": True}
        
        return score, message, details
    
    def _parse_bandit_output(self, stdout: str, stderr: str) -> tuple[float, str, Dict[str, Any]]:
        """Parse bandit security scan output"""
        details = {}
        
        try:
            # Try to parse JSON output
            bandit_result = json.loads(stdout)
            
            high_issues = len([issue for issue in bandit_result.get('results', []) 
                             if issue.get('issue_severity') == 'HIGH'])
            medium_issues = len([issue for issue in bandit_result.get('results', []) 
                               if issue.get('issue_severity') == 'MEDIUM'])
            low_issues = len([issue for issue in bandit_result.get('results', []) 
                            if issue.get('issue_severity') == 'LOW'])
            
            total_issues = high_issues + medium_issues + low_issues
            
            # Scoring based on severity weighted issues
            severity_score = max(0, 100 - (high_issues * 20 + medium_issues * 10 + low_issues * 5))
            
            details = {
                "high_issues": high_issues,
                "medium_issues": medium_issues,
                "low_issues": low_issues,
                "total_issues": total_issues
            }
            
            message = f"Security: {high_issues} high, {medium_issues} medium, {low_issues} low issues"
            
        except json.JSONDecodeError:
            # Fallback to simple parsing
            if "No issues identified" in stdout:
                severity_score = 100.0
                message = "No security issues identified"
                details = {"clean": True}
            else:
                severity_score = 50.0
                message = "Security scan completed with potential issues"
                details = {"unparseable_output": True}
        
        return severity_score, message, details
    
    def _parse_mypy_output(self, stdout: str, stderr: str) -> tuple[float, str, Dict[str, Any]]:
        """Parse mypy type checking output"""
        lines = stderr.split('\n') if stderr else stdout.split('\n')
        
        error_count = 0
        warning_count = 0
        
        for line in lines:
            if "error:" in line.lower():
                error_count += 1
            elif "warning:" in line.lower():
                warning_count += 1
        
        # Success message parsing
        if "Success: no issues found" in stdout or error_count == 0:
            score = 100.0
            message = "Type checking passed"
        else:
            # Penalize based on errors
            score = max(0, 100 - (error_count * 10 + warning_count * 5))
            message = f"Type checking: {error_count} errors, {warning_count} warnings"
        
        details = {
            "errors": error_count,
            "warnings": warning_count
        }
        
        return score, message, details
    
    def _parse_ruff_output(self, stdout: str, stderr: str, return_code: int) -> tuple[float, str, Dict[str, Any]]:
        """Parse ruff linting output"""
        if return_code == 0:
            return 100.0, "Code style perfect", {"violations": 0}
        
        # Count violations
        lines = stdout.split('\n') + stderr.split('\n')
        violations = len([line for line in lines if line.strip() and not line.startswith('Found')])
        
        # Score based on violations
        score = max(0, 100 - (violations * 2))
        message = f"Code style: {violations} violations found"
        
        return score, message, {"violations": violations}
    
    def _generate_recommendations(self, results: List[QualityResult], overall_score: float) -> List[str]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        if overall_score < 70:
            recommendations.append("🚨 Critical: Overall quality below acceptable threshold - immediate attention required")
        elif overall_score < 85:
            recommendations.append("⚠️  Warning: Quality gates need improvement for production readiness")
        
        failed_gates = [r for r in results if r.status == "failed"]
        error_gates = [r for r in results if r.status == "error"]
        
        if failed_gates:
            recommendations.append(f"🔧 Fix {len(failed_gates)} failing quality gates: {', '.join([g.gate_name for g in failed_gates])}")
        
        if error_gates:
            recommendations.append(f"🛠️  Resolve {len(error_gates)} gate execution errors: {', '.join([g.gate_name for g in error_gates])}")
        
        # Generation-specific recommendations
        gen1_gates = [r for r in results if r.generation == 1]
        gen2_gates = [r for r in results if r.generation == 2]
        gen3_gates = [r for r in results if r.generation == 3]
        
        if gen1_gates and all(r.status == "passed" for r in gen1_gates):
            if not gen2_gates:
                recommendations.append("🎯 Ready for Generation 2: Enable robustness quality gates")
        
        if gen2_gates and all(r.status == "passed" for r in gen2_gates):
            if not gen3_gates:
                recommendations.append("⚡ Ready for Generation 3: Enable performance quality gates")
        
        if not recommendations:
            recommendations.append("✅ All quality gates passing - system ready for deployment")
        
        return recommendations
    
    async def _save_execution_report(self, report: Dict[str, Any]):
        """Save detailed execution report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure reports directory exists
        Path('.terragon/reports').mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        json_file = Path(f'.terragon/reports/quality_gates_gen{report["generation"]}_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save markdown summary
        md_file = json_file.with_suffix('.md')
        await self._generate_markdown_report(report, md_file)
        
        logger.info(f"📊 Quality gates report saved: {json_file}")
    
    async def _generate_markdown_report(self, report: Dict[str, Any], output_file: Path):
        """Generate markdown quality gates report"""
        content = f"""# 🛡️ Quality Gates Report - Generation {report['generation']}

**Timestamp:** {report['timestamp']}
**Overall Status:** {report['overall_status'].upper()}
**Overall Score:** {report['overall_score']:.1f}/100
**Execution Time:** {report['execution_time']:.1f} seconds

## 📊 Gates Summary

- **Total Gates:** {report['gates_summary']['total']}
- **Passed:** {report['gates_summary']['passed']} ✅
- **Failed:** {report['gates_summary']['failed']} ❌
- **Errors:** {report['gates_summary']['errors']} 🔥

## 🔍 Detailed Results

"""
        
        for gate_result in report['gate_results']:
            status_emoji = {
                'passed': '✅',
                'failed': '❌', 
                'error': '🔥',
                'skipped': '⏭️'
            }.get(gate_result['status'], '❓')
            
            content += f"### {status_emoji} {gate_result['gate_name']}\n"
            content += f"- **Status:** {gate_result['status'].upper()}\n"
            content += f"- **Score:** {gate_result['score']:.1f}/100\n"
            content += f"- **Generation:** {gate_result['generation']}\n"
            content += f"- **Execution Time:** {gate_result['execution_time']:.1f}s\n"
            content += f"- **Message:** {gate_result['message']}\n"
            
            if gate_result.get('details'):
                content += "- **Details:**\n"
                for key, value in gate_result['details'].items():
                    content += f"  - {key}: {value}\n"
            
            content += "\n"
        
        # Recommendations section
        recommendations = report.get('recommendations', [])
        if recommendations:
            content += "## 💡 Recommendations\n\n"
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
        
        content += f"""

## 📈 Trend Analysis

- **Generation:** {report['generation']}
- **Historical Executions:** {len(self.execution_history)}
- **Score Trend:** {'📈 Improving' if self._calculate_score_trend() > 0 else '📉 Declining' if self._calculate_score_trend() < 0 else '📊 Stable'}

---
*Generated by Progressive Quality Gates System*
*Report Time: {datetime.now().isoformat()}*
"""
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    def _calculate_score_trend(self) -> float:
        """Calculate score trend over recent executions"""
        if len(self.execution_history) < 3:
            return 0.0
        
        recent_scores = [h['overall_score'] for h in self.execution_history[-5:]]
        
        # Simple linear trend calculation
        if len(recent_scores) >= 2:
            return recent_scores[-1] - recent_scores[0]
        
        return 0.0
    
    async def _adapt_quality_gates(self, report: Dict[str, Any]):
        """Adapt quality gate thresholds based on performance"""
        if not self.adaptation_enabled:
            return
        
        # Increase generation if current gates consistently pass
        if (report['overall_score'] >= 90 and 
            len(self.execution_history) >= 3 and
            all(h['overall_score'] >= 85 for h in self.execution_history[-3:])):
            
            if self.current_generation < 3:
                old_gen = self.current_generation
                self.current_generation += 1
                logger.info(f"🎯 Auto-advancing to Generation {self.current_generation} quality gates")
                
                # Log adaptation event
                adaptation_log = {
                    "timestamp": datetime.now().isoformat(),
                    "action": "generation_advancement",
                    "old_generation": old_gen,
                    "new_generation": self.current_generation,
                    "trigger_score": report['overall_score'],
                    "historical_scores": [h['overall_score'] for h in self.execution_history[-3:]]
                }
                
                # Save adaptation log
                with open('.terragon/adaptation_log.json', 'a') as f:
                    f.write(json.dumps(adaptation_log) + '\n')
    
    async def run_continuous_quality_monitoring(self, interval_minutes: int = 30):
        """Run continuous quality monitoring"""
        logger.info(f"🔄 Starting continuous quality monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                await self.execute_generation_gates()
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Continuous monitoring error: {e}")
                await asyncio.sleep(60)  # Shorter retry interval
    
    def get_current_quality_status(self) -> Dict[str, Any]:
        """Get current quality status summary"""
        if not self.execution_history:
            return {
                "status": "unknown",
                "message": "No quality gates executed yet"
            }
        
        latest = self.execution_history[-1]
        
        return {
            "timestamp": latest['timestamp'],
            "current_generation": self.current_generation,
            "overall_status": latest['overall_status'],
            "overall_score": latest['overall_score'],
            "trend": self._calculate_score_trend(),
            "gates_summary": latest['gates_summary'],
            "ready_for_next_generation": latest['overall_score'] >= 90
        }


# Global instance
progressive_gates = ProgressiveQualityGates()


async def execute_progressive_quality_validation(generation: int = None) -> Dict[str, Any]:
    """Execute progressive quality validation"""
    return await progressive_gates.execute_generation_gates(generation)


async def main():
    """Main execution for testing"""
    print("🛡️ Progressive Quality Gates System - Autonomous Execution")
    print("="*60)
    
    # Execute all generations progressively
    for gen in range(1, 4):
        print(f"\n🎯 Executing Generation {gen} Quality Gates...")
        result = await progressive_gates.execute_generation_gates(gen)
        
        print(f"   Status: {result['overall_status'].upper()}")
        print(f"   Score: {result['overall_score']:.1f}/100")
        print(f"   Gates: {result['gates_summary']['passed']}/{result['gates_summary']['total']} passed")
        
        if result['overall_status'] == 'failed':
            print(f"   ❌ Generation {gen} failed - stopping progression")
            break
        elif result['overall_status'] == 'warning':
            print(f"   ⚠️  Generation {gen} has warnings but continuing")
    
    # Display final status
    status = progressive_gates.get_current_quality_status()
    print(f"\n🏁 Final Quality Status:")
    print(f"   Generation: {status['current_generation']}")
    print(f"   Status: {status['overall_status'].upper()}")
    print(f"   Score: {status['overall_score']:.1f}/100")
    print(f"   Ready for Next Gen: {'✅' if status['ready_for_next_generation'] else '❌'}")


if __name__ == "__main__":
    asyncio.run(main())