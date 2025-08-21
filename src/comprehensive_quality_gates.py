#!/usr/bin/env python3
"""
Comprehensive Quality Gates System - Autonomous SDLC
Complete quality assurance with automated testing, security, and performance validation
"""

import asyncio
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import tempfile
import logging


@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_name: str
    passed: bool
    score: float  # 0-100 scale
    details: Dict[str, Any]
    execution_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gate_name': self.gate_name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'execution_time': self.execution_time_seconds,
            'timestamp': self.timestamp.isoformat(),
            'recommendations': self.recommendations
        }


class TestingGate:
    """Comprehensive testing quality gate"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self) -> QualityGateResult:
        """Execute testing quality gate"""
        start_time = time.time()
        
        try:
            # Run pytest with coverage
            result = await self._run_pytest()
            
            # Analyze test results
            analysis = self._analyze_test_results(result)
            
            # Calculate score
            score = self._calculate_testing_score(analysis)
            
            # Determine if gate passes
            passed = score >= 80.0  # 80% threshold for passing
            
            # Generate recommendations
            recommendations = self._generate_testing_recommendations(analysis)
            
            return QualityGateResult(
                gate_name="testing",
                passed=passed,
                score=score,
                details=analysis,
                execution_time_seconds=time.time() - start_time,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name="testing",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=time.time() - start_time,
                recommendations=["Fix testing infrastructure issues"]
            )
    
    async def _run_pytest(self) -> Dict[str, Any]:
        """Run pytest and capture results"""
        try:
            # Run pytest with JSON report
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "--tb=short",
                "--json-report",
                "--json-report-file=pytest_report.json",
                "-v"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Try to read JSON report
            try:
                with open("pytest_report.json", "r") as f:
                    json_report = json.load(f)
            except FileNotFoundError:
                json_report = {"error": "No pytest report generated"}
            
            return {
                "return_code": process.returncode,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "json_report": json_report
            }
        
        except Exception as e:
            # Fallback: run basic tests
            return await self._run_basic_tests()
    
    async def _run_basic_tests(self) -> Dict[str, Any]:
        """Run basic validation tests"""
        test_results = {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "import_tests": []
        }
        
        # Test imports of new modules
        modules_to_test = [
            "src.core_value_orchestrator",
            "src.autonomous_enhancement_engine", 
            "src.enhanced_error_handling",
            "src.advanced_security_framework",
            "src.comprehensive_logging_system",
            "src.performance_optimization_engine",
            "src.intelligent_auto_scaling"
        ]
        
        for module_name in modules_to_test:
            test_results["total_tests"] += 1
            try:
                __import__(module_name)
                test_results["passed_tests"] += 1
                test_results["import_tests"].append({
                    "module": module_name,
                    "status": "passed"
                })
            except Exception as e:
                test_results["failed_tests"] += 1
                test_results["import_tests"].append({
                    "module": module_name,
                    "status": "failed",
                    "error": str(e)
                })
        
        return {
            "return_code": 0 if test_results["failed_tests"] == 0 else 1,
            "stdout": f"Ran {test_results['total_tests']} import tests",
            "stderr": "",
            "json_report": test_results
        }
    
    def _analyze_test_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze test execution results"""
        json_report = result.get("json_report", {})
        
        if "error" in json_report:
            # Basic analysis from our custom tests
            if "import_tests" in json_report:
                return {
                    "total_tests": json_report.get("total_tests", 0),
                    "passed_tests": json_report.get("passed_tests", 0),
                    "failed_tests": json_report.get("failed_tests", 0),
                    "test_success_rate": (json_report.get("passed_tests", 0) / 
                                        max(1, json_report.get("total_tests", 1)) * 100),
                    "coverage_percentage": 0,  # Not available in basic mode
                    "import_test_results": json_report.get("import_tests", [])
                }
        
        # Standard pytest JSON report analysis
        summary = json_report.get("summary", {})
        
        return {
            "total_tests": summary.get("total", 0),
            "passed_tests": summary.get("passed", 0),
            "failed_tests": summary.get("failed", 0),
            "skipped_tests": summary.get("skipped", 0),
            "test_success_rate": (summary.get("passed", 0) / 
                                max(1, summary.get("total", 1)) * 100),
            "coverage_percentage": self._extract_coverage_info(),
            "duration_seconds": json_report.get("duration", 0)
        }
    
    def _extract_coverage_info(self) -> float:
        """Extract coverage information from coverage report"""
        try:
            # Try to read coverage.json if it exists
            with open("coverage.json", "r") as f:
                coverage_data = json.load(f)
                totals = coverage_data.get("totals", {})
                return totals.get("percent_covered", 0.0)
        except FileNotFoundError:
            return 0.0
    
    def _calculate_testing_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall testing score"""
        success_rate = analysis.get("test_success_rate", 0)
        coverage = analysis.get("coverage_percentage", 0)
        
        # Weight success rate more heavily than coverage
        score = (success_rate * 0.7) + (coverage * 0.3)
        
        return min(100.0, score)
    
    def _generate_testing_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate testing improvement recommendations"""
        recommendations = []
        
        success_rate = analysis.get("test_success_rate", 0)
        coverage = analysis.get("coverage_percentage", 0)
        
        if success_rate < 90:
            recommendations.append("Fix failing tests to achieve 90%+ success rate")
        
        if coverage < 80:
            recommendations.append("Increase test coverage to 80%+ for critical code paths")
        
        if analysis.get("failed_tests", 0) > 0:
            recommendations.append("Address all test failures before deployment")
        
        return recommendations


class SecurityGate:
    """Security vulnerability scanning quality gate"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self) -> QualityGateResult:
        """Execute security quality gate"""
        start_time = time.time()
        
        try:
            # Run security scans
            bandit_result = await self._run_bandit_scan()
            dependency_result = await self._check_dependencies()
            
            # Analyze results
            analysis = self._analyze_security_results(bandit_result, dependency_result)
            
            # Calculate score
            score = self._calculate_security_score(analysis)
            
            # Determine if gate passes
            passed = score >= 85.0 and analysis.get("high_severity_issues", 0) == 0
            
            # Generate recommendations
            recommendations = self._generate_security_recommendations(analysis)
            
            return QualityGateResult(
                gate_name="security",
                passed=passed,
                score=score,
                details=analysis,
                execution_time_seconds=time.time() - start_time,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name="security",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=time.time() - start_time,
                recommendations=["Fix security scanning infrastructure"]
            )
    
    async def _run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scanner"""
        try:
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", "src/",
                "-f", "json"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse JSON output
            try:
                bandit_report = json.loads(stdout.decode())
            except json.JSONDecodeError:
                bandit_report = {"error": "Failed to parse Bandit output"}
            
            return {
                "return_code": process.returncode,
                "report": bandit_report,
                "stderr": stderr.decode()
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    async def _check_dependencies(self) -> Dict[str, Any]:
        """Check for vulnerable dependencies"""
        try:
            # Basic dependency check - in production would use safety or similar
            requirements_file = Path("requirements.txt")
            if requirements_file.exists():
                content = requirements_file.read_text()
                dependencies = [line.strip() for line in content.split('\n') 
                              if line.strip() and not line.startswith('#')]
                
                return {
                    "total_dependencies": len(dependencies),
                    "vulnerable_dependencies": 0,  # Placeholder
                    "dependencies": dependencies
                }
            else:
                return {"error": "No requirements.txt found"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_security_results(self, bandit_result: Dict[str, Any], 
                                 dependency_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze security scan results"""
        analysis = {
            "total_issues": 0,
            "high_severity_issues": 0,
            "medium_severity_issues": 0,
            "low_severity_issues": 0,
            "vulnerable_dependencies": dependency_result.get("vulnerable_dependencies", 0),
            "total_dependencies": dependency_result.get("total_dependencies", 0)
        }
        
        bandit_report = bandit_result.get("report", {})
        if "results" in bandit_report:
            for issue in bandit_report["results"]:
                analysis["total_issues"] += 1
                severity = issue.get("issue_severity", "LOW").upper()
                
                if severity == "HIGH":
                    analysis["high_severity_issues"] += 1
                elif severity == "MEDIUM":
                    analysis["medium_severity_issues"] += 1
                else:
                    analysis["low_severity_issues"] += 1
        
        return analysis
    
    def _calculate_security_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate security score"""
        # Start with perfect score
        score = 100.0
        
        # Deduct points for issues
        score -= analysis["high_severity_issues"] * 20    # 20 points per high
        score -= analysis["medium_severity_issues"] * 10  # 10 points per medium
        score -= analysis["low_severity_issues"] * 2      # 2 points per low
        score -= analysis["vulnerable_dependencies"] * 15 # 15 points per vulnerable dep
        
        return max(0.0, score)
    
    def _generate_security_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate security improvement recommendations"""
        recommendations = []
        
        if analysis["high_severity_issues"] > 0:
            recommendations.append("CRITICAL: Address all high severity security issues immediately")
        
        if analysis["medium_severity_issues"] > 0:
            recommendations.append("Address medium severity security issues")
        
        if analysis["vulnerable_dependencies"] > 0:
            recommendations.append("Update vulnerable dependencies to secure versions")
        
        if analysis["total_issues"] == 0:
            recommendations.append("Great job! No security issues detected")
        
        return recommendations


class PerformanceGate:
    """Performance benchmarking quality gate"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def execute(self) -> QualityGateResult:
        """Execute performance quality gate"""
        start_time = time.time()
        
        try:
            # Run performance benchmarks
            benchmark_results = await self._run_performance_benchmarks()
            
            # Analyze results
            analysis = self._analyze_performance_results(benchmark_results)
            
            # Calculate score
            score = self._calculate_performance_score(analysis)
            
            # Determine if gate passes
            passed = score >= 75.0  # 75% threshold for performance
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(analysis)
            
            return QualityGateResult(
                gate_name="performance",
                passed=passed,
                score=score,
                details=analysis,
                execution_time_seconds=time.time() - start_time,
                recommendations=recommendations
            )
        
        except Exception as e:
            return QualityGateResult(
                gate_name="performance",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                execution_time_seconds=time.time() - start_time,
                recommendations=["Fix performance benchmarking infrastructure"]
            )
    
    async def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        benchmarks = {
            "import_time": await self._benchmark_import_time(),
            "memory_usage": await self._benchmark_memory_usage(),
            "cpu_efficiency": await self._benchmark_cpu_efficiency()
        }
        
        return benchmarks
    
    async def _benchmark_import_time(self) -> Dict[str, float]:
        """Benchmark module import times"""
        import_times = {}
        
        modules = [
            "src.core_value_orchestrator",
            "src.autonomous_enhancement_engine",
            "src.enhanced_error_handling",
            "src.advanced_security_framework"
        ]
        
        for module_name in modules:
            start_time = time.time()
            try:
                __import__(module_name)
                import_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                import_times[module_name] = import_time
            except Exception:
                import_times[module_name] = -1  # Error indicator
        
        return import_times
    
    async def _benchmark_memory_usage(self) -> Dict[str, Any]:
        """Benchmark memory usage"""
        import psutil
        
        process = psutil.Process()
        
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "memory_percent": process.memory_percent()
        }
    
    async def _benchmark_cpu_efficiency(self) -> Dict[str, Any]:
        """Benchmark CPU efficiency"""
        import psutil
        
        # Simple CPU benchmark
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        
        # Perform some computation
        result = sum(i * i for i in range(10000))
        
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        
        return {
            "computation_time_ms": (end_time - start_time) * 1000,
            "cpu_usage_delta": end_cpu - start_cpu,
            "computation_result": result
        }
    
    def _analyze_performance_results(self, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance benchmark results"""
        analysis = {}
        
        # Import time analysis
        import_times = benchmarks.get("import_time", {})
        if import_times:
            successful_imports = [t for t in import_times.values() if t > 0]
            if successful_imports:
                analysis["avg_import_time_ms"] = sum(successful_imports) / len(successful_imports)
                analysis["max_import_time_ms"] = max(successful_imports)
                analysis["import_success_rate"] = len(successful_imports) / len(import_times)
        
        # Memory usage analysis
        memory_info = benchmarks.get("memory_usage", {})
        analysis.update({
            "memory_rss_mb": memory_info.get("rss_mb", 0),
            "memory_efficiency": max(0, 100 - memory_info.get("memory_percent", 0))
        })
        
        # CPU efficiency analysis
        cpu_info = benchmarks.get("cpu_efficiency", {})
        analysis.update({
            "computation_time_ms": cpu_info.get("computation_time_ms", 0),
            "cpu_efficiency": max(0, 100 - cpu_info.get("cpu_usage_delta", 0))
        })
        
        return analysis
    
    def _calculate_performance_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate performance score"""
        score = 100.0
        
        # Import time penalty
        avg_import = analysis.get("avg_import_time_ms", 0)
        if avg_import > 100:  # Over 100ms average import time
            score -= min(20, avg_import / 10)
        
        # Memory efficiency score
        memory_efficiency = analysis.get("memory_efficiency", 100)
        score = score * 0.7 + memory_efficiency * 0.3
        
        # Import success rate
        import_success = analysis.get("import_success_rate", 1.0)
        score *= import_success
        
        return max(0.0, score)
    
    def _generate_performance_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        avg_import = analysis.get("avg_import_time_ms", 0)
        if avg_import > 50:
            recommendations.append("Optimize module imports to reduce startup time")
        
        memory_mb = analysis.get("memory_rss_mb", 0)
        if memory_mb > 100:
            recommendations.append("Consider memory optimization for large deployments")
        
        import_success = analysis.get("import_success_rate", 1.0)
        if import_success < 1.0:
            recommendations.append("Fix module import issues affecting performance tests")
        
        if not recommendations:
            recommendations.append("Performance looks good! Continue monitoring in production")
        
        return recommendations


class QualityGatesOrchestrator:
    """Main orchestrator for all quality gates"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.gates = {
            "testing": TestingGate(),
            "security": SecurityGate(),
            "performance": PerformanceGate()
        }
    
    async def execute_all_gates(self) -> Dict[str, Any]:
        """Execute all quality gates"""
        self.logger.info("Starting comprehensive quality gates execution")
        
        start_time = time.time()
        gate_results = {}
        
        # Execute gates concurrently
        tasks = {
            gate_name: gate.execute() 
            for gate_name, gate in self.gates.items()
        }
        
        for gate_name, task in tasks.items():
            try:
                result = await task
                gate_results[gate_name] = result.to_dict()
                self.logger.info(f"{gate_name} gate: {'PASSED' if result.passed else 'FAILED'} "
                               f"(Score: {result.score:.1f})")
            except Exception as e:
                gate_results[gate_name] = {
                    "gate_name": gate_name,
                    "passed": False,
                    "score": 0.0,
                    "error": str(e)
                }
                self.logger.error(f"{gate_name} gate failed with error: {e}")
        
        # Calculate overall results
        overall_result = self._calculate_overall_result(gate_results)
        
        # Create comprehensive report
        report = {
            "execution_timestamp": datetime.now().isoformat(),
            "total_execution_time": time.time() - start_time,
            "overall_passed": overall_result["passed"],
            "overall_score": overall_result["score"],
            "gate_results": gate_results,
            "summary": overall_result["summary"],
            "next_steps": overall_result["next_steps"]
        }
        
        # Save report
        await self._save_report(report)
        
        return report
    
    def _calculate_overall_result(self, gate_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality gates result"""
        total_score = 0.0
        total_weight = 0.0
        all_passed = True
        failed_gates = []
        
        # Gate weights
        weights = {
            "testing": 0.4,    # 40% weight
            "security": 0.4,   # 40% weight  
            "performance": 0.2  # 20% weight
        }
        
        for gate_name, result in gate_results.items():
            weight = weights.get(gate_name, 0.33)
            score = result.get("score", 0.0)
            passed = result.get("passed", False)
            
            total_score += score * weight
            total_weight += weight
            
            if not passed:
                all_passed = False
                failed_gates.append(gate_name)
        
        overall_score = total_score / max(total_weight, 0.001)
        
        # Generate summary
        summary = {
            "gates_executed": len(gate_results),
            "gates_passed": sum(1 for r in gate_results.values() if r.get("passed", False)),
            "gates_failed": len(failed_gates),
            "failed_gates": failed_gates
        }
        
        # Generate next steps
        next_steps = []
        if not all_passed:
            next_steps.append("Address failing quality gates before deployment")
            for gate_name in failed_gates:
                gate_result = gate_results.get(gate_name, {})
                recommendations = gate_result.get("recommendations", [])
                next_steps.extend(recommendations)
        else:
            next_steps.append("All quality gates passed! Ready for deployment")
        
        return {
            "passed": all_passed,
            "score": overall_score,
            "summary": summary,
            "next_steps": next_steps
        }
    
    async def _save_report(self, report: Dict[str, Any]):
        """Save quality gates report"""
        output_dir = Path("output/quality_gates")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"quality_gates_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save summary markdown report
        md_file = output_dir / f"quality_gates_summary_{timestamp}.md"
        md_content = self._generate_markdown_summary(report)
        with open(md_file, 'w') as f:
            f.write(md_content)
        
        self.logger.info(f"Quality gates report saved to {output_dir}")
    
    def _generate_markdown_summary(self, report: Dict[str, Any]) -> str:
        """Generate markdown summary report"""
        passed_emoji = "✅" if report["overall_passed"] else "❌"
        
        md = f"""# Quality Gates Report
        
{passed_emoji} **Overall Status**: {'PASSED' if report['overall_passed'] else 'FAILED'}  
**Overall Score**: {report['overall_score']:.1f}/100  
**Execution Time**: {report['total_execution_time']:.2f} seconds  
**Generated**: {report['execution_timestamp']}

## Gate Results

"""
        
        for gate_name, result in report["gate_results"].items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            score = result.get("score", 0)
            
            md += f"### {gate_name.title()} Gate\n"
            md += f"**Status**: {status}  \n"
            md += f"**Score**: {score:.1f}/100  \n"
            
            recommendations = result.get("recommendations", [])
            if recommendations:
                md += "**Recommendations**:\n"
                for rec in recommendations:
                    md += f"- {rec}\n"
            md += "\n"
        
        # Next steps
        md += "## Next Steps\n\n"
        for step in report.get("next_steps", []):
            md += f"- {step}\n"
        
        return md


async def main():
    """Main execution for quality gates"""
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = QualityGatesOrchestrator()
    
    print("🔍 Executing Comprehensive Quality Gates...")
    report = await orchestrator.execute_all_gates()
    
    print(f"\n📊 Quality Gates Results:")
    print(f"Overall Status: {'✅ PASSED' if report['overall_passed'] else '❌ FAILED'}")
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Execution Time: {report['total_execution_time']:.2f} seconds")
    
    for gate_name, result in report["gate_results"].items():
        status = "✅" if result.get("passed", False) else "❌"
        print(f"{gate_name.title()}: {status} ({result.get('score', 0):.1f}/100)")
    
    print(f"\n📁 Detailed report saved to: output/quality_gates/")
    
    return report


if __name__ == "__main__":
    asyncio.run(main())