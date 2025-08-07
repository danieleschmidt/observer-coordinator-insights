#!/usr/bin/env python3
"""
Comprehensive Quality Gates Runner
Orchestrates all quality checks including tests, security, performance, and compliance
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

@dataclass 
class QualityGateResult:
    """Result of a single quality gate"""
    gate_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    score: float  # 0-100
    execution_time_seconds: float
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]

@dataclass
class QualityGatesReport:
    """Complete quality gates execution report"""
    execution_timestamp: str
    total_execution_time_seconds: float
    overall_status: str
    overall_score: float
    gates_executed: int
    gates_passed: int
    gates_failed: int
    gates_skipped: int
    gate_results: List[QualityGateResult]
    environment_info: Dict[str, Any]
    quality_thresholds: Dict[str, float]
    production_ready: bool
    recommendations: List[str]

class QualityGatesRunner:
    """Orchestrates execution of all quality gates"""
    
    def __init__(self, project_root: Path, config_file: Optional[Path] = None):
        self.project_root = Path(project_root)
        self.config = self._load_configuration(config_file)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Output directory for reports
        self.output_dir = self.project_root / "quality_reports"
        self.output_dir.mkdir(exist_ok=True)
        
        # Quality gates configuration
        self.quality_gates = self._configure_quality_gates()
        
    def _load_configuration(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load quality gates configuration"""
        default_config = {
            'thresholds': {
                'test_coverage': 95.0,
                'security_scan': 85.0,
                'performance_benchmarks': 80.0,
                'code_quality': 85.0,
                'integration_tests': 90.0,
                'e2e_tests': 85.0,
                'overall_quality': 85.0
            },
            'timeouts': {
                'unit_tests': 600,      # 10 minutes
                'integration_tests': 1200,  # 20 minutes
                'e2e_tests': 1800,      # 30 minutes
                'performance_tests': 1800,  # 30 minutes
                'security_scan': 600    # 10 minutes
            },
            'parallel_execution': True,
            'fail_fast': False,
            'generate_reports': True,
            'production_deployment_gate': True
        }
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge configurations
                    for key, value in user_config.items():
                        if isinstance(value, dict) and key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                self.logger.warning(f"Could not load config from {config_file}: {e}")
        
        return default_config
    
    def _configure_quality_gates(self) -> List[Dict[str, Any]]:
        """Configure the quality gates to execute"""
        return [
            {
                'name': 'unit_tests',
                'description': 'Run comprehensive unit tests with coverage',
                'command': self._run_unit_tests,
                'timeout': self.config['timeouts']['unit_tests'],
                'required': True,
                'weight': 25
            },
            {
                'name': 'integration_tests',
                'description': 'Run integration tests for clustering pipelines',
                'command': self._run_integration_tests,
                'timeout': self.config['timeouts']['integration_tests'],
                'required': True,
                'weight': 20
            },
            {
                'name': 'security_scan',
                'description': 'Comprehensive security vulnerability scanning',
                'command': self._run_security_scan,
                'timeout': self.config['timeouts']['security_scan'],
                'required': True,
                'weight': 20
            },
            {
                'name': 'performance_benchmarks',
                'description': 'Performance benchmarking and regression detection',
                'command': self._run_performance_benchmarks,
                'timeout': self.config['timeouts']['performance_tests'],
                'required': True,
                'weight': 15
            },
            {
                'name': 'e2e_tests',
                'description': 'End-to-end organizational scenario tests',
                'command': self._run_e2e_tests,
                'timeout': self.config['timeouts']['e2e_tests'],
                'required': True,
                'weight': 15
            },
            {
                'name': 'code_quality',
                'description': 'Static code analysis and quality metrics',
                'command': self._run_code_quality_checks,
                'timeout': 300,
                'required': True,
                'weight': 5
            }
        ]
    
    def run_all_quality_gates(self, selected_gates: Optional[List[str]] = None) -> QualityGatesReport:
        """Execute all quality gates and generate comprehensive report"""
        self.logger.info("Starting comprehensive quality gates execution...")
        start_time = time.time()
        
        # Filter gates if specific ones are selected
        gates_to_run = self.quality_gates
        if selected_gates:
            gates_to_run = [gate for gate in self.quality_gates if gate['name'] in selected_gates]
        
        # Execute gates
        gate_results = []
        
        if self.config.get('parallel_execution', True) and len(gates_to_run) > 1:
            gate_results = self._run_gates_parallel(gates_to_run)
        else:
            gate_results = self._run_gates_sequential(gates_to_run)
        
        # Calculate overall metrics
        total_time = time.time() - start_time
        overall_status, overall_score = self._calculate_overall_quality(gate_results)
        
        # Determine production readiness
        production_ready = self._assess_production_readiness(gate_results, overall_score)
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations(gate_results, overall_score)
        
        # Create comprehensive report
        report = QualityGatesReport(
            execution_timestamp=datetime.utcnow().isoformat(),
            total_execution_time_seconds=total_time,
            overall_status=overall_status,
            overall_score=overall_score,
            gates_executed=len([r for r in gate_results if r.status != 'SKIP']),
            gates_passed=len([r for r in gate_results if r.status == 'PASS']),
            gates_failed=len([r for r in gate_results if r.status == 'FAIL']),
            gates_skipped=len([r for r in gate_results if r.status == 'SKIP']),
            gate_results=gate_results,
            environment_info=self._collect_environment_info(),
            quality_thresholds=self.config['thresholds'],
            production_ready=production_ready,
            recommendations=recommendations
        )
        
        # Save comprehensive report
        self._save_quality_report(report)
        
        # Log summary
        self._log_execution_summary(report)
        
        return report
    
    def _run_gates_parallel(self, gates_to_run: List[Dict[str, Any]]) -> List[QualityGateResult]:
        """Run quality gates in parallel"""
        results = []
        
        # Separate gates that can run in parallel vs sequentially
        parallel_gates = [gate for gate in gates_to_run if gate['name'] not in ['e2e_tests']]
        sequential_gates = [gate for gate in gates_to_run if gate['name'] in ['e2e_tests']]
        
        # Run parallel gates
        if parallel_gates:
            with ThreadPoolExecutor(max_workers=min(4, len(parallel_gates))) as executor:
                future_to_gate = {
                    executor.submit(self._execute_quality_gate, gate): gate 
                    for gate in parallel_gates
                }
                
                for future in as_completed(future_to_gate):
                    gate = future_to_gate[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        if self.config.get('fail_fast') and result.status == 'FAIL':
                            self.logger.warning(f"Fail-fast enabled: {gate['name']} failed")
                            # Cancel remaining futures
                            for f in future_to_gate:
                                f.cancel()
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Quality gate {gate['name']} failed with exception: {e}")
                        results.append(QualityGateResult(
                            gate_name=gate['name'],
                            status='ERROR',
                            score=0.0,
                            execution_time_seconds=0.0,
                            details={'error': str(e)},
                            errors=[str(e)],
                            warnings=[],
                            recommendations=['Fix execution error and retry']
                        ))
        
        # Run sequential gates
        for gate in sequential_gates:
            result = self._execute_quality_gate(gate)
            results.append(result)
            
            if self.config.get('fail_fast') and result.status == 'FAIL':
                break
        
        return results
    
    def _run_gates_sequential(self, gates_to_run: List[Dict[str, Any]]) -> List[QualityGateResult]:
        """Run quality gates sequentially"""
        results = []
        
        for gate in gates_to_run:
            result = self._execute_quality_gate(gate)
            results.append(result)
            
            if self.config.get('fail_fast') and result.status == 'FAIL':
                self.logger.warning(f"Fail-fast enabled: stopping after {gate['name']} failure")
                break
        
        return results
    
    def _execute_quality_gate(self, gate: Dict[str, Any]) -> QualityGateResult:
        """Execute a single quality gate"""
        gate_name = gate['name']
        self.logger.info(f"Executing quality gate: {gate_name}")
        
        start_time = time.time()
        
        try:
            # Execute the gate command
            result = gate['command']()
            execution_time = time.time() - start_time
            
            # Enhance result with timing
            result.execution_time_seconds = execution_time
            
            self.logger.info(f"Quality gate {gate_name} completed: {result.status} (score: {result.score:.1f})")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Quality gate {gate_name} failed with exception: {e}")
            
            return QualityGateResult(
                gate_name=gate_name,
                status='ERROR',
                score=0.0,
                execution_time_seconds=execution_time,
                details={'error': str(e), 'gate_description': gate.get('description', '')},
                errors=[str(e)],
                warnings=[],
                recommendations=['Fix execution error and retry quality gate']
            )
    
    def _run_unit_tests(self) -> QualityGateResult:
        """Execute comprehensive unit tests with coverage"""
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest',
                str(self.project_root / 'tests' / 'unit'),
                '--cov=src',
                '--cov-report=json',
                f'--cov-report=html:{self.output_dir}/coverage_html',
                '--cov-report=term-missing',
                '-v',
                '--tb=short',
                '--durations=10'
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True, 
                text=True, 
                timeout=self.config['timeouts']['unit_tests']
            )
            
            # Parse coverage results
            coverage_file = self.project_root / 'coverage.json'
            coverage_data = {}
            coverage_percentage = 0.0
            
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                    coverage_percentage = coverage_data.get('totals', {}).get('percent_covered', 0.0)
            
            # Determine status
            threshold = self.config['thresholds']['test_coverage']
            status = 'PASS' if result.returncode == 0 and coverage_percentage >= threshold else 'FAIL'
            
            # Calculate score
            score = min(100.0, (coverage_percentage / threshold) * 100) if threshold > 0 else 0.0
            
            details = {
                'tests_run': self._extract_tests_run(result.stdout),
                'coverage_percentage': coverage_percentage,
                'coverage_threshold': threshold,
                'coverage_details': coverage_data.get('files', {}),
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            errors = []
            warnings = []
            recommendations = []
            
            if result.returncode != 0:
                errors.append(f"Unit tests failed with return code {result.returncode}")
                
            if coverage_percentage < threshold:
                warnings.append(f"Coverage {coverage_percentage:.1f}% below threshold {threshold:.1f}%")
                recommendations.append(f"Increase test coverage to at least {threshold:.1f}%")
            
            return QualityGateResult(
                gate_name='unit_tests',
                status=status,
                score=score,
                execution_time_seconds=0,  # Will be set by caller
                details=details,
                errors=errors,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name='unit_tests',
                status='FAIL',
                score=0.0,
                execution_time_seconds=0,
                details={'error': 'Test execution timed out'},
                errors=['Unit tests timed out'],
                warnings=[],
                recommendations=['Optimize slow tests or increase timeout']
            )
    
    def _run_integration_tests(self) -> QualityGateResult:
        """Execute integration tests"""
        try:
            cmd = [
                'python', '-m', 'pytest',
                str(self.project_root / 'tests' / 'integration'),
                '-v',
                '--tb=short',
                '--durations=10'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['timeouts']['integration_tests']
            )
            
            # Parse results
            tests_run = self._extract_tests_run(result.stdout)
            threshold = self.config['thresholds']['integration_tests']
            
            # Determine success rate
            success_rate = 100.0 if result.returncode == 0 else 0.0
            status = 'PASS' if success_rate >= threshold else 'FAIL'
            score = min(100.0, (success_rate / threshold) * 100) if threshold > 0 else 0.0
            
            return QualityGateResult(
                gate_name='integration_tests',
                status=status,
                score=score,
                execution_time_seconds=0,
                details={
                    'tests_run': tests_run,
                    'success_rate': success_rate,
                    'threshold': threshold,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                },
                errors=['Integration tests failed'] if result.returncode != 0 else [],
                warnings=[],
                recommendations=['Fix failing integration tests'] if result.returncode != 0 else []
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name='integration_tests',
                status='FAIL',
                score=0.0,
                execution_time_seconds=0,
                details={'error': 'Integration tests timed out'},
                errors=['Integration tests timed out'],
                warnings=[],
                recommendations=['Optimize integration tests or increase timeout']
            )
    
    def _run_security_scan(self) -> QualityGateResult:
        """Execute security scanning"""
        try:
            # Import and run security scanner
            from security_scan import SecurityScanner
            
            scanner = SecurityScanner(self.project_root, self.output_dir / 'security')
            scan_result = scanner.run_comprehensive_scan()
            
            # Determine status based on security findings
            threshold = self.config['thresholds']['security_scan']
            score = scan_result.code_quality_score
            status = 'PASS' if score >= threshold else 'FAIL'
            
            # Extract key metrics
            details = {
                'overall_risk_level': scan_result.overall_risk_level,
                'total_findings': scan_result.total_findings,
                'findings_by_severity': scan_result.findings_by_severity,
                'dependency_vulnerabilities': scan_result.dependency_vulnerabilities,
                'code_quality_score': scan_result.code_quality_score,
                'scan_duration': scan_result.scan_duration_seconds
            }
            
            errors = []
            warnings = []
            
            if scan_result.findings_by_severity.get('CRITICAL', 0) > 0:
                errors.append(f"Critical security vulnerabilities found: {scan_result.findings_by_severity['CRITICAL']}")
                
            if scan_result.findings_by_severity.get('HIGH', 0) > 0:
                warnings.append(f"High severity security issues found: {scan_result.findings_by_severity['HIGH']}")
            
            return QualityGateResult(
                gate_name='security_scan',
                status=status,
                score=score,
                execution_time_seconds=0,
                details=details,
                errors=errors,
                warnings=warnings,
                recommendations=scan_result.recommendations
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name='security_scan',
                status='ERROR',
                score=0.0,
                execution_time_seconds=0,
                details={'error': str(e)},
                errors=[f'Security scan failed: {e}'],
                warnings=[],
                recommendations=['Fix security scanner configuration and retry']
            )
    
    def _run_performance_benchmarks(self) -> QualityGateResult:
        """Execute performance benchmarks"""
        try:
            cmd = [
                'python', '-m', 'pytest',
                str(self.project_root / 'tests' / 'performance'),
                '-v',
                '--tb=short',
                '-m', 'not slow',  # Skip very slow tests in regular runs
                '--durations=10'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['timeouts']['performance_tests']
            )
            
            # Parse performance results
            threshold = self.config['thresholds']['performance_benchmarks']
            success_rate = 100.0 if result.returncode == 0 else 0.0
            status = 'PASS' if success_rate >= threshold else 'FAIL'
            score = min(100.0, (success_rate / threshold) * 100) if threshold > 0 else 0.0
            
            details = {
                'benchmark_success_rate': success_rate,
                'threshold': threshold,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
            # Look for performance regression warnings in output
            warnings = []
            if "regression" in result.stdout.lower() or "performance" in result.stderr.lower():
                warnings.append("Potential performance regression detected")
            
            return QualityGateResult(
                gate_name='performance_benchmarks',
                status=status,
                score=score,
                execution_time_seconds=0,
                details=details,
                errors=['Performance benchmarks failed'] if result.returncode != 0 else [],
                warnings=warnings,
                recommendations=['Investigate performance issues'] if result.returncode != 0 else []
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name='performance_benchmarks',
                status='FAIL',
                score=0.0,
                execution_time_seconds=0,
                details={'error': 'Performance tests timed out'},
                errors=['Performance benchmarks timed out'],
                warnings=[],
                recommendations=['Optimize performance tests or increase timeout']
            )
    
    def _run_e2e_tests(self) -> QualityGateResult:
        """Execute end-to-end organizational scenario tests"""
        try:
            cmd = [
                'python', '-m', 'pytest',
                str(self.project_root / 'tests' / 'e2e'),
                '-v',
                '--tb=short',
                '--durations=10'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.config['timeouts']['e2e_tests']
            )
            
            threshold = self.config['thresholds']['e2e_tests']
            success_rate = 100.0 if result.returncode == 0 else 0.0
            status = 'PASS' if success_rate >= threshold else 'FAIL'
            score = min(100.0, (success_rate / threshold) * 100) if threshold > 0 else 0.0
            
            return QualityGateResult(
                gate_name='e2e_tests',
                status=status,
                score=score,
                execution_time_seconds=0,
                details={
                    'success_rate': success_rate,
                    'threshold': threshold,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                },
                errors=['End-to-end tests failed'] if result.returncode != 0 else [],
                warnings=[],
                recommendations=['Fix failing E2E scenarios'] if result.returncode != 0 else []
            )
            
        except subprocess.TimeoutExpired:
            return QualityGateResult(
                gate_name='e2e_tests',
                status='FAIL',
                score=0.0,
                execution_time_seconds=0,
                details={'error': 'E2E tests timed out'},
                errors=['End-to-end tests timed out'],
                warnings=[],
                recommendations=['Optimize E2E tests or increase timeout']
            )
    
    def _run_code_quality_checks(self) -> QualityGateResult:
        """Execute static code analysis and quality checks"""
        try:
            quality_score = 100.0
            details = {}
            warnings = []
            recommendations = []
            
            # Run flake8 for style checking
            try:
                flake8_result = subprocess.run(
                    ['flake8', 'src/', '--max-line-length=100', '--extend-ignore=E203,W503'],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                flake8_issues = len(flake8_result.stdout.splitlines()) if flake8_result.stdout else 0
                details['flake8_issues'] = flake8_issues
                
                if flake8_issues > 0:
                    quality_score -= min(20, flake8_issues * 0.5)  # Penalize style issues
                    warnings.append(f"Found {flake8_issues} code style issues")
                    recommendations.append("Fix code style issues identified by flake8")
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                warnings.append("Could not run flake8 code style checks")
            
            # Check for basic code quality metrics
            python_files = list((self.project_root / 'src').rglob('*.py'))
            total_lines = 0
            total_functions = 0
            
            for py_file in python_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')
                        total_lines += len(lines)
                        total_functions += content.count('def ')
                except Exception:
                    continue
            
            details.update({
                'total_python_files': len(python_files),
                'total_lines_of_code': total_lines,
                'total_functions': total_functions,
                'avg_lines_per_file': total_lines / max(len(python_files), 1)
            })
            
            # Quality thresholds
            threshold = self.config['thresholds']['code_quality']
            status = 'PASS' if quality_score >= threshold else 'FAIL'
            
            return QualityGateResult(
                gate_name='code_quality',
                status=status,
                score=quality_score,
                execution_time_seconds=0,
                details=details,
                errors=[],
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            return QualityGateResult(
                gate_name='code_quality',
                status='ERROR',
                score=0.0,
                execution_time_seconds=0,
                details={'error': str(e)},
                errors=[f'Code quality checks failed: {e}'],
                warnings=[],
                recommendations=['Fix code quality checker configuration']
            )
    
    def _extract_tests_run(self, pytest_output: str) -> int:
        """Extract number of tests run from pytest output"""
        import re
        
        # Look for patterns like "collected 42 items" or "42 passed"
        collected_match = re.search(r'collected (\d+) items', pytest_output)
        if collected_match:
            return int(collected_match.group(1))
        
        passed_match = re.search(r'(\d+) passed', pytest_output)
        if passed_match:
            return int(passed_match.group(1))
            
        return 0
    
    def _calculate_overall_quality(self, gate_results: List[QualityGateResult]) -> Tuple[str, float]:
        """Calculate overall quality status and score"""
        if not gate_results:
            return 'FAIL', 0.0
        
        # Weight the scores by gate importance
        total_weighted_score = 0.0
        total_weight = 0.0
        failed_gates = 0
        error_gates = 0
        
        gate_weights = {gate['name']: gate['weight'] for gate in self.quality_gates}
        
        for result in gate_results:
            weight = gate_weights.get(result.gate_name, 10)  # Default weight
            total_weighted_score += result.score * weight
            total_weight += weight
            
            if result.status == 'FAIL':
                failed_gates += 1
            elif result.status == 'ERROR':
                error_gates += 1
        
        # Calculate weighted average score
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine overall status
        if error_gates > 0:
            overall_status = 'ERROR'
        elif failed_gates > 0:
            overall_status = 'FAIL'
        elif overall_score >= self.config['thresholds']['overall_quality']:
            overall_status = 'PASS'
        else:
            overall_status = 'FAIL'
        
        return overall_status, overall_score
    
    def _assess_production_readiness(self, gate_results: List[QualityGateResult], 
                                   overall_score: float) -> bool:
        """Assess if the system is ready for production deployment"""
        if not self.config.get('production_deployment_gate', True):
            return True  # Production gate disabled
        
        # Critical requirements for production
        critical_gates = ['unit_tests', 'security_scan']
        
        for result in gate_results:
            if result.gate_name in critical_gates and result.status in ['FAIL', 'ERROR']:
                return False
        
        # Overall score threshold
        if overall_score < self.config['thresholds']['overall_quality']:
            return False
        
        # No critical security vulnerabilities
        security_results = [r for r in gate_results if r.gate_name == 'security_scan']
        if security_results:
            security_details = security_results[0].details
            if security_details.get('findings_by_severity', {}).get('CRITICAL', 0) > 0:
                return False
        
        return True
    
    def _generate_overall_recommendations(self, gate_results: List[QualityGateResult],
                                        overall_score: float) -> List[str]:
        """Generate overall recommendations based on all gate results"""
        recommendations = []
        
        # Collect recommendations from individual gates
        for result in gate_results:
            recommendations.extend(result.recommendations)
        
        # Add overall recommendations
        if overall_score < 60:
            recommendations.append("URGENT: Overall quality is critically low - address major issues immediately")
        elif overall_score < 80:
            recommendations.append("Overall quality needs improvement - focus on failing quality gates")
        
        # Specific gate-based recommendations
        failed_gates = [r.gate_name for r in gate_results if r.status == 'FAIL']
        if failed_gates:
            recommendations.append(f"Priority: Fix failing quality gates: {', '.join(failed_gates)}")
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        for rec in recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations
    
    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect environment information for the report"""
        import platform
        
        return {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'hostname': platform.node(),
            'timestamp': datetime.utcnow().isoformat(),
            'project_root': str(self.project_root),
            'working_directory': os.getcwd()
        }
    
    def _save_quality_report(self, report: QualityGatesReport):
        """Save comprehensive quality report"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"quality_gates_report_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Save human-readable report
        markdown_file = self.output_dir / f"quality_gates_report_{timestamp}.md"
        self._generate_markdown_report(report, markdown_file)
        
        # Save summary for CI/CD
        summary_file = self.output_dir / "quality_gates_summary.json"
        summary = {
            'timestamp': report.execution_timestamp,
            'overall_status': report.overall_status,
            'overall_score': report.overall_score,
            'production_ready': report.production_ready,
            'gates_passed': report.gates_passed,
            'gates_failed': report.gates_failed,
            'total_execution_time': report.total_execution_time_seconds
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Update latest report symlink
        latest_report = self.output_dir / "latest_quality_report.json"
        if latest_report.is_symlink():
            latest_report.unlink()
        latest_report.symlink_to(json_file.name)
        
        self.logger.info(f"Quality reports saved to {self.output_dir}")
    
    def _generate_markdown_report(self, report: QualityGatesReport, output_file: Path):
        """Generate human-readable markdown report"""
        with open(output_file, 'w') as f:
            f.write("# Quality Gates Execution Report\n\n")
            f.write(f"**Execution Date:** {report.execution_timestamp}\n")
            f.write(f"**Total Execution Time:** {report.total_execution_time_seconds:.2f} seconds\n")
            f.write(f"**Overall Status:** {report.overall_status}\n")
            f.write(f"**Overall Score:** {report.overall_score:.1f}/100\n")
            f.write(f"**Production Ready:** {'✅ Yes' if report.production_ready else '❌ No'}\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Gates Executed:** {report.gates_executed}\n")
            f.write(f"- **Gates Passed:** {report.gates_passed}\n")
            f.write(f"- **Gates Failed:** {report.gates_failed}\n")
            f.write(f"- **Gates Skipped:** {report.gates_skipped}\n\n")
            
            # Quality Gates Results
            f.write("## Quality Gates Results\n\n")
            
            for result in report.gate_results:
                status_emoji = {
                    'PASS': '✅',
                    'FAIL': '❌', 
                    'ERROR': '💥',
                    'SKIP': '⏭️'
                }.get(result.status, '❓')
                
                f.write(f"### {status_emoji} {result.gate_name.replace('_', ' ').title()}\n")
                f.write(f"- **Status:** {result.status}\n")
                f.write(f"- **Score:** {result.score:.1f}/100\n")
                f.write(f"- **Execution Time:** {result.execution_time_seconds:.2f} seconds\n")
                
                if result.errors:
                    f.write(f"- **Errors:** {len(result.errors)}\n")
                    for error in result.errors:
                        f.write(f"  - {error}\n")
                
                if result.warnings:
                    f.write(f"- **Warnings:** {len(result.warnings)}\n")
                    for warning in result.warnings:
                        f.write(f"  - {warning}\n")
                
                if result.recommendations:
                    f.write(f"- **Recommendations:**\n")
                    for rec in result.recommendations:
                        f.write(f"  - {rec}\n")
                
                f.write("\n")
            
            # Overall Recommendations
            if report.recommendations:
                f.write("## Overall Recommendations\n\n")
                for rec in report.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Environment Info
            f.write("## Environment Information\n\n")
            for key, value in report.environment_info.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
    
    def _log_execution_summary(self, report: QualityGatesReport):
        """Log execution summary to console"""
        self.logger.info("\n" + "="*80)
        self.logger.info("QUALITY GATES EXECUTION SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Overall Status: {report.overall_status}")
        self.logger.info(f"Overall Score: {report.overall_score:.1f}/100")
        self.logger.info(f"Production Ready: {'Yes' if report.production_ready else 'No'}")
        self.logger.info(f"Execution Time: {report.total_execution_time_seconds:.2f} seconds")
        self.logger.info("")
        
        self.logger.info("Gate Results:")
        for result in report.gate_results:
            status_symbol = {'PASS': '✓', 'FAIL': '✗', 'ERROR': '!', 'SKIP': '-'}.get(result.status, '?')
            self.logger.info(f"  {status_symbol} {result.gate_name}: {result.status} ({result.score:.1f}/100)")
        
        if report.recommendations:
            self.logger.info("\nKey Recommendations:")
            for rec in report.recommendations[:5]:  # Show top 5
                self.logger.info(f"  - {rec}")


def main():
    """Main function for quality gates runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive quality gates')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Root directory of the project')
    parser.add_argument('--config', type=Path,
                       help='Configuration file for quality gates')
    parser.add_argument('--gates', nargs='+',
                       choices=['unit_tests', 'integration_tests', 'security_scan', 
                              'performance_benchmarks', 'e2e_tests', 'code_quality'],
                       help='Specific gates to run (default: all)')
    parser.add_argument('--fail-on-error', action='store_true',
                       help='Exit with non-zero code if any gate fails')
    parser.add_argument('--production-gate', action='store_true',
                       help='Exit with non-zero code if not production ready')
    
    args = parser.parse_args()
    
    # Run quality gates
    runner = QualityGatesRunner(args.project_root, args.config)
    report = runner.run_all_quality_gates(args.gates)
    
    # Exit codes based on results
    if args.production_gate and not report.production_ready:
        print("FAILED: System not ready for production deployment")
        sys.exit(1)
    elif args.fail_on_error and report.overall_status in ['FAIL', 'ERROR']:
        print(f"FAILED: Quality gates status: {report.overall_status}")
        sys.exit(1)
    else:
        print("Quality gates completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()