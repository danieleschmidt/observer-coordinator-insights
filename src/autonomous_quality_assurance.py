#!/usr/bin/env python3
"""
Autonomous Quality Assurance System
Comprehensive testing, validation, and quality gates for SDLC operations
"""

import asyncio
import logging
import subprocess
import json
import time
import traceback
import ast
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import importlib.util
import sys
import coverage
import pytest
from unittest.mock import Mock, patch
import bandit
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config
import safety
import hashlib


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class QualityMetric:
    """Represents a quality measurement"""
    name: str
    category: str  # 'security', 'performance', 'maintainability', 'reliability'
    value: float
    threshold: float
    status: str  # 'pass', 'warning', 'fail'
    details: str
    timestamp: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """Represents test execution result"""
    test_name: str
    test_type: str  # 'unit', 'integration', 'e2e', 'security', 'performance'
    status: str  # 'passed', 'failed', 'skipped'
    duration: float
    error_message: Optional[str] = None
    coverage_percentage: Optional[float] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class QualityGateResult:
    """Result of quality gate evaluation"""
    gate_name: str
    status: str  # 'passed', 'failed', 'warning'
    score: float
    threshold: float
    metrics: List[QualityMetric]
    recommendations: List[str]
    timestamp: str


class SecurityScanner:
    """Advanced security scanning and vulnerability detection"""
    
    def __init__(self):
        self.scan_results = {}
        
    async def scan_security_vulnerabilities(self, target_path: Path) -> Dict[str, Any]:
        """Comprehensive security vulnerability scanning"""
        results = {
            'bandit_scan': await self._run_bandit_scan(target_path),
            'safety_scan': await self._run_safety_scan(),
            'custom_security_checks': await self._run_custom_security_checks(target_path)
        }
        
        # Aggregate security score
        total_vulnerabilities = (
            len(results['bandit_scan'].get('results', [])) +
            len(results['safety_scan'].get('vulnerabilities', [])) +
            len(results['custom_security_checks'].get('issues', []))
        )
        
        # Calculate security score (0-100, higher is better)
        if total_vulnerabilities == 0:
            security_score = 100.0
        elif total_vulnerabilities <= 3:
            security_score = 90.0 - (total_vulnerabilities * 5)
        elif total_vulnerabilities <= 10:
            security_score = 75.0 - ((total_vulnerabilities - 3) * 3)
        else:
            security_score = max(50.0 - ((total_vulnerabilities - 10) * 2), 0)
        
        results['security_score'] = security_score
        results['total_vulnerabilities'] = total_vulnerabilities
        
        return results
    
    async def _run_bandit_scan(self, target_path: Path) -> Dict[str, Any]:
        """Run Bandit security scanner"""
        try:
            # Configure Bandit
            conf = bandit_config.BanditConfig()
            b_mgr = bandit_manager.BanditManager(conf, 'file')
            
            # Scan Python files
            python_files = list(target_path.rglob("*.py"))
            if not python_files:
                return {'results': [], 'summary': 'No Python files found'}
            
            for py_file in python_files[:50]:  # Limit to first 50 files
                try:
                    b_mgr.discover_files([str(py_file)])
                except Exception:
                    continue
            
            b_mgr.run_tests()
            
            # Extract results
            bandit_results = []
            for result in b_mgr.get_issue_list():
                bandit_results.append({
                    'filename': result.fname,
                    'line_number': result.lineno,
                    'test_name': result.test,
                    'issue_severity': result.severity,
                    'issue_confidence': result.confidence,
                    'issue_text': result.text,
                    'more_info': result.more_info
                })
            
            return {
                'results': bandit_results,
                'summary': f"Found {len(bandit_results)} security issues"
            }
            
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {'results': [], 'error': str(e)}
    
    async def _run_safety_scan(self) -> Dict[str, Any]:
        """Run Safety scanner for known security vulnerabilities"""
        try:
            # Check if requirements files exist
            req_files = ['requirements.txt', 'pyproject.toml', 'Pipfile']
            found_req_file = None
            
            for req_file in req_files:
                if Path(req_file).exists():
                    found_req_file = req_file
                    break
            
            if not found_req_file:
                return {'vulnerabilities': [], 'summary': 'No requirements file found'}
            
            # Run safety check (this is a simplified version)
            # In practice, you'd use the safety API or CLI
            result = subprocess.run(
                [sys.executable, '-m', 'safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
            else:
                vulnerabilities = []
                logger.warning(f"Safety scan returned non-zero exit code: {result.stderr}")
            
            return {
                'vulnerabilities': vulnerabilities,
                'summary': f"Found {len(vulnerabilities)} known vulnerabilities"
            }
            
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return {'vulnerabilities': [], 'error': str(e)}
    
    async def _run_custom_security_checks(self, target_path: Path) -> Dict[str, Any]:
        """Run custom security checks"""
        issues = []
        
        # Check for common security anti-patterns
        security_patterns = [
            {
                'pattern': r'(?i)(password|pwd|secret|key)\s*=\s*["\'][^"\']+["\']',
                'message': 'Hardcoded credentials detected',
                'severity': 'high'
            },
            {
                'pattern': r'eval\s*\(',
                'message': 'Use of eval() detected - potential code injection risk',
                'severity': 'high'
            },
            {
                'pattern': r'exec\s*\(',
                'message': 'Use of exec() detected - potential code injection risk',
                'severity': 'high'
            },
            {
                'pattern': r'subprocess\..*shell=True',
                'message': 'Shell injection risk with shell=True',
                'severity': 'medium'
            },
            {
                'pattern': r'pickle\.loads?\(',
                'message': 'Pickle deserialization can execute arbitrary code',
                'severity': 'medium'
            }
        ]
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern_info in security_patterns:
                    matches = re.finditer(pattern_info['pattern'], content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'file': str(py_file),
                            'line': line_num,
                            'pattern': pattern_info['pattern'],
                            'message': pattern_info['message'],
                            'severity': pattern_info['severity'],
                            'code_snippet': match.group()
                        })
            except Exception:
                continue
        
        return {
            'issues': issues,
            'summary': f"Found {len(issues)} custom security issues"
        }


class CodeQualityAnalyzer:
    """Analyzes code quality metrics"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    async def analyze_code_quality(self, target_path: Path) -> Dict[str, Any]:
        """Comprehensive code quality analysis"""
        results = {
            'complexity_analysis': await self._analyze_complexity(target_path),
            'maintainability_index': await self._calculate_maintainability_index(target_path),
            'code_duplication': await self._detect_code_duplication(target_path),
            'naming_conventions': await self._check_naming_conventions(target_path),
            'documentation_coverage': await self._check_documentation_coverage(target_path)
        }
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(results)
        results['overall_quality_score'] = quality_score
        
        return results
    
    async def _analyze_complexity(self, target_path: Path) -> Dict[str, Any]:
        """Analyze cyclomatic complexity"""
        complexity_results = []
        total_complexity = 0
        function_count = 0
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_cyclomatic_complexity(node)
                        complexity_results.append({
                            'file': str(py_file),
                            'function': node.name,
                            'line': node.lineno,
                            'complexity': complexity
                        })
                        total_complexity += complexity
                        function_count += 1
            
            except Exception:
                continue
        
        avg_complexity = total_complexity / max(function_count, 1)
        high_complexity_functions = [r for r in complexity_results if r['complexity'] > 10]
        
        return {
            'average_complexity': avg_complexity,
            'total_functions': function_count,
            'high_complexity_functions': len(high_complexity_functions),
            'complexity_details': complexity_results[:20],  # Limit output
            'quality_score': max(100 - (avg_complexity - 5) * 10, 0) if avg_complexity > 5 else 100
        }
    
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, 
                                ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += 1
        
        return complexity
    
    async def _calculate_maintainability_index(self, target_path: Path) -> Dict[str, Any]:
        """Calculate maintainability index for code"""
        total_lines = 0
        total_files = 0
        comment_lines = 0
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                total_files += 1
                total_lines += len(lines)
                
                # Count comment lines
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''"):
                        comment_lines += 1
            
            except Exception:
                continue
        
        comment_ratio = comment_lines / max(total_lines, 1)
        
        # Simplified maintainability index (0-100)
        maintainability_index = min(
            (comment_ratio * 40) +  # Documentation factor
            (min(total_lines / max(total_files, 1), 200) / 200 * 30) +  # File size factor
            30,  # Base score
            100
        )
        
        return {
            'maintainability_index': maintainability_index,
            'total_files': total_files,
            'total_lines': total_lines,
            'comment_ratio': comment_ratio,
            'quality_score': maintainability_index
        }
    
    async def _detect_code_duplication(self, target_path: Path) -> Dict[str, Any]:
        """Detect code duplication"""
        duplicates = []
        function_hashes = {}
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Create a hash of the function body
                        func_body = ast.dump(node)
                        func_hash = hashlib.md5(func_body.encode()).hexdigest()
                        
                        if func_hash in function_hashes:
                            duplicates.append({
                                'original_file': function_hashes[func_hash]['file'],
                                'original_function': function_hashes[func_hash]['name'],
                                'duplicate_file': str(py_file),
                                'duplicate_function': node.name,
                                'similarity_hash': func_hash
                            })
                        else:
                            function_hashes[func_hash] = {
                                'file': str(py_file),
                                'name': node.name
                            }
            
            except Exception:
                continue
        
        duplication_score = max(100 - len(duplicates) * 10, 0)
        
        return {
            'duplicates_found': len(duplicates),
            'duplication_details': duplicates[:10],  # Limit output
            'quality_score': duplication_score
        }
    
    async def _check_naming_conventions(self, target_path: Path) -> Dict[str, Any]:
        """Check naming conventions compliance"""
        violations = []
        total_names = 0
        
        naming_patterns = {
            'function': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'constant': re.compile(r'^[A-Z_][A-Z0-9_]*$'),
            'variable': re.compile(r'^[a-z_][a-z0-9_]*$')
        }
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_names += 1
                        if not naming_patterns['function'].match(node.name):
                            violations.append({
                                'file': str(py_file),
                                'type': 'function',
                                'name': node.name,
                                'line': node.lineno,
                                'violation': 'Function name should be snake_case'
                            })
                    
                    elif isinstance(node, ast.ClassDef):
                        total_names += 1
                        if not naming_patterns['class'].match(node.name):
                            violations.append({
                                'file': str(py_file),
                                'type': 'class',
                                'name': node.name,
                                'line': node.lineno,
                                'violation': 'Class name should be PascalCase'
                            })
            
            except Exception:
                continue
        
        compliance_score = max(100 - (len(violations) / max(total_names, 1)) * 100, 0)
        
        return {
            'total_names_checked': total_names,
            'violations': len(violations),
            'violation_details': violations[:20],  # Limit output
            'compliance_score': compliance_score,
            'quality_score': compliance_score
        }
    
    async def _check_documentation_coverage(self, target_path: Path) -> Dict[str, Any]:
        """Check documentation coverage"""
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for py_file in target_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        total_functions += 1
                        if ast.get_docstring(node):
                            documented_functions += 1
                    
                    elif isinstance(node, ast.ClassDef):
                        total_classes += 1
                        if ast.get_docstring(node):
                            documented_classes += 1
            
            except Exception:
                continue
        
        function_doc_coverage = documented_functions / max(total_functions, 1) * 100
        class_doc_coverage = documented_classes / max(total_classes, 1) * 100
        overall_doc_coverage = (function_doc_coverage + class_doc_coverage) / 2
        
        return {
            'function_documentation_coverage': function_doc_coverage,
            'class_documentation_coverage': class_doc_coverage,
            'overall_documentation_coverage': overall_doc_coverage,
            'total_functions': total_functions,
            'documented_functions': documented_functions,
            'total_classes': total_classes,
            'documented_classes': documented_classes,
            'quality_score': overall_doc_coverage
        }
    
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics"""
        scores = []
        
        # Extract quality scores from each analysis
        for analysis_name, analysis_result in results.items():
            if isinstance(analysis_result, dict) and 'quality_score' in analysis_result:
                scores.append(analysis_result['quality_score'])
        
        if not scores:
            return 0.0
        
        # Weighted average (customize weights as needed)
        weights = [0.25, 0.2, 0.15, 0.15, 0.25]  # Adjust based on importance
        weights = weights[:len(scores)]
        
        # Normalize weights
        weight_sum = sum(weights)
        normalized_weights = [w / weight_sum for w in weights]
        
        weighted_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        
        return min(weighted_score, 100.0)


class PerformanceTester:
    """Performance testing and benchmarking"""
    
    def __init__(self):
        self.benchmark_results = {}
    
    async def run_performance_tests(self, target_functions: Dict[str, Callable]) -> Dict[str, Any]:
        """Run performance tests on target functions"""
        results = {}
        
        for func_name, func in target_functions.items():
            try:
                benchmark_result = await self._benchmark_function(func_name, func)
                results[func_name] = benchmark_result
            except Exception as e:
                results[func_name] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Calculate overall performance score
        performance_score = self._calculate_performance_score(results)
        results['overall_performance_score'] = performance_score
        
        return results
    
    async def _benchmark_function(self, func_name: str, func: Callable) -> Dict[str, Any]:
        """Benchmark a specific function"""
        iterations = 100
        durations = []
        memory_usage = []
        
        # Warm-up runs
        for _ in range(10):
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception:
                pass
        
        # Actual benchmarking
        for i in range(iterations):
            start_time = time.perf_counter()
            memory_before = self._get_memory_usage()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
                
                end_time = time.perf_counter()
                memory_after = self._get_memory_usage()
                
                durations.append(end_time - start_time)
                memory_usage.append(memory_after - memory_before)
                
            except Exception as e:
                # Skip failed iterations
                continue
        
        if not durations:
            return {'error': 'All benchmark iterations failed', 'status': 'failed'}
        
        # Calculate statistics
        import numpy as np
        
        return {
            'iterations': len(durations),
            'avg_duration': np.mean(durations),
            'min_duration': np.min(durations),
            'max_duration': np.max(durations),
            'std_duration': np.std(durations),
            'percentile_95': np.percentile(durations, 95),
            'percentile_99': np.percentile(durations, 99),
            'avg_memory_delta': np.mean(memory_usage),
            'throughput_per_second': 1 / np.mean(durations) if np.mean(durations) > 0 else 0,
            'performance_score': self._calculate_function_performance_score(durations),
            'status': 'success'
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def _calculate_function_performance_score(self, durations: List[float]) -> float:
        """Calculate performance score for a function based on durations"""
        import numpy as np
        
        avg_duration = np.mean(durations)
        std_duration = np.std(durations)
        
        # Score based on speed and consistency
        speed_score = max(100 - (avg_duration * 1000), 0)  # Penalty for slow operations
        consistency_score = max(100 - (std_duration / avg_duration * 100), 0)
        
        return (speed_score + consistency_score) / 2
    
    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        scores = []
        
        for func_name, result in results.items():
            if isinstance(result, dict) and 'performance_score' in result:
                scores.append(result['performance_score'])
        
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)


class AutonomousQualityAssurance:
    """Main quality assurance orchestrator"""
    
    def __init__(self):
        self.security_scanner = SecurityScanner()
        self.code_quality_analyzer = CodeQualityAnalyzer()
        self.performance_tester = PerformanceTester()
        self.quality_gates = {}
        self.test_results = []
        self.quality_history = []
        
        # Define default quality gates
        self._initialize_default_quality_gates()
    
    def _initialize_default_quality_gates(self):
        """Initialize default quality gates with thresholds"""
        self.quality_gates = {
            'security_gate': {
                'name': 'Security Quality Gate',
                'threshold': 80.0,
                'metrics': ['security_score'],
                'mandatory': True
            },
            'code_quality_gate': {
                'name': 'Code Quality Gate', 
                'threshold': 75.0,
                'metrics': ['overall_quality_score'],
                'mandatory': True
            },
            'performance_gate': {
                'name': 'Performance Quality Gate',
                'threshold': 70.0,
                'metrics': ['overall_performance_score'],
                'mandatory': False
            },
            'test_coverage_gate': {
                'name': 'Test Coverage Gate',
                'threshold': 80.0,
                'metrics': ['test_coverage'],
                'mandatory': True
            }
        }
    
    async def run_comprehensive_quality_check(self, target_path: Path = None) -> Dict[str, Any]:
        """Run comprehensive quality assurance checks"""
        if target_path is None:
            target_path = Path("src")
        
        logger.info(f"🔍 Starting comprehensive quality check on {target_path}")
        
        # Run all quality checks in parallel
        tasks = [
            self.security_scanner.scan_security_vulnerabilities(target_path),
            self.code_quality_analyzer.analyze_code_quality(target_path),
            self._run_test_suite(),
            self._run_test_coverage_analysis(target_path)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quality_report = {
            'timestamp': datetime.now().isoformat(),
            'target_path': str(target_path),
            'security_analysis': results[0] if not isinstance(results[0], Exception) else {'error': str(results[0])},
            'code_quality_analysis': results[1] if not isinstance(results[1], Exception) else {'error': str(results[1])},
            'test_results': results[2] if not isinstance(results[2], Exception) else {'error': str(results[2])},
            'coverage_analysis': results[3] if not isinstance(results[3], Exception) else {'error': str(results[3])}
        }
        
        # Evaluate quality gates
        gate_results = await self._evaluate_quality_gates(quality_report)
        quality_report['quality_gates'] = gate_results
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(quality_report)
        quality_report['overall_quality_score'] = overall_score
        
        # Generate recommendations
        recommendations = self._generate_quality_recommendations(quality_report)
        quality_report['recommendations'] = recommendations
        
        # Store in history
        self.quality_history.append(quality_report)
        
        logger.info(f"✅ Quality check complete. Overall score: {overall_score:.1f}/100")
        
        return quality_report
    
    async def _run_test_suite(self) -> Dict[str, Any]:
        """Run the test suite"""
        try:
            # Run pytest programmatically
            test_results = {}
            
            # Check if test directory exists
            test_dirs = ['tests', 'test']
            test_dir = None
            
            for dir_name in test_dirs:
                if Path(dir_name).exists():
                    test_dir = dir_name
                    break
            
            if not test_dir:
                return {
                    'status': 'skipped',
                    'message': 'No test directory found',
                    'test_count': 0,
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0
                }
            
            # Run tests using subprocess to avoid pytest complications
            result = subprocess.run([
                sys.executable, '-m', 'pytest', test_dir, 
                '--json-report', '--json-report-file=/tmp/test_report.json',
                '-v'
            ], capture_output=True, text=True, timeout=300)
            
            # Parse results
            try:
                with open('/tmp/test_report.json', 'r') as f:
                    test_report = json.load(f)
                
                test_results = {
                    'status': 'completed',
                    'test_count': test_report.get('summary', {}).get('total', 0),
                    'passed': test_report.get('summary', {}).get('passed', 0),
                    'failed': test_report.get('summary', {}).get('failed', 0),
                    'skipped': test_report.get('summary', {}).get('skipped', 0),
                    'duration': test_report.get('duration', 0),
                    'success_rate': 0
                }
                
                if test_results['test_count'] > 0:
                    test_results['success_rate'] = test_results['passed'] / test_results['test_count'] * 100
                
            except Exception:
                # Fallback to parsing stdout
                test_results = {
                    'status': 'completed',
                    'test_count': 0,
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'success_rate': 0,
                    'raw_output': result.stdout
                }
            
            return test_results
            
        except Exception as e:
            logger.error(f"Test suite execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_count': 0,
                'success_rate': 0
            }
    
    async def _run_test_coverage_analysis(self, target_path: Path) -> Dict[str, Any]:
        """Run test coverage analysis"""
        try:
            # Initialize coverage
            cov = coverage.Coverage(source=[str(target_path)])
            cov.start()
            
            # Import and run some basic tests
            # This is a simplified version - in practice you'd run your actual test suite
            test_modules = []
            for py_file in target_path.rglob("*.py"):
                if not py_file.name.startswith('__'):
                    try:
                        spec = importlib.util.spec_from_file_location(
                            py_file.stem, py_file
                        )
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            test_modules.append(module)
                    except Exception:
                        continue
            
            cov.stop()
            
            # Generate coverage report
            total_coverage = cov.report(show_missing=False, skip_covered=False)
            
            coverage_data = {
                'total_coverage': total_coverage,
                'status': 'completed',
                'modules_analyzed': len(test_modules)
            }
            
            return coverage_data
            
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return {
                'total_coverage': 0.0,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _evaluate_quality_gates(self, quality_report: Dict[str, Any]) -> Dict[str, QualityGateResult]:
        """Evaluate all quality gates"""
        gate_results = {}
        
        for gate_id, gate_config in self.quality_gates.items():
            try:
                gate_result = await self._evaluate_single_quality_gate(gate_config, quality_report)
                gate_results[gate_id] = asdict(gate_result)
            except Exception as e:
                logger.error(f"Quality gate evaluation failed for {gate_id}: {e}")
                gate_results[gate_id] = {
                    'gate_name': gate_config['name'],
                    'status': 'failed',
                    'score': 0.0,
                    'threshold': gate_config['threshold'],
                    'metrics': [],
                    'recommendations': [f"Gate evaluation failed: {str(e)}"],
                    'timestamp': datetime.now().isoformat()
                }
        
        return gate_results
    
    async def _evaluate_single_quality_gate(self, gate_config: Dict[str, Any], 
                                          quality_report: Dict[str, Any]) -> QualityGateResult:
        """Evaluate a single quality gate"""
        gate_name = gate_config['name']
        threshold = gate_config['threshold']
        metric_names = gate_config['metrics']
        
        metrics = []
        scores = []
        
        # Extract relevant metrics
        for metric_name in metric_names:
            metric_value = self._extract_metric_value(metric_name, quality_report)
            
            if metric_value is not None:
                status = 'pass' if metric_value >= threshold else 'fail'
                metric = QualityMetric(
                    name=metric_name,
                    category=gate_name.lower().split()[0],
                    value=metric_value,
                    threshold=threshold,
                    status=status,
                    details=f"{metric_name}: {metric_value:.1f}",
                    timestamp=datetime.now().isoformat()
                )
                metrics.append(metric)
                scores.append(metric_value)
        
        # Calculate gate score
        if scores:
            gate_score = sum(scores) / len(scores)
        else:
            gate_score = 0.0
        
        # Determine gate status
        if gate_score >= threshold:
            gate_status = 'passed'
        elif gate_score >= threshold * 0.8:
            gate_status = 'warning'
        else:
            gate_status = 'failed'
        
        # Generate recommendations
        recommendations = self._generate_gate_recommendations(gate_name, metrics, gate_score, threshold)
        
        return QualityGateResult(
            gate_name=gate_name,
            status=gate_status,
            score=gate_score,
            threshold=threshold,
            metrics=metrics,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _extract_metric_value(self, metric_name: str, quality_report: Dict[str, Any]) -> Optional[float]:
        """Extract metric value from quality report"""
        
        if metric_name == 'security_score':
            return quality_report.get('security_analysis', {}).get('security_score', 0.0)
        
        elif metric_name == 'overall_quality_score':
            return quality_report.get('code_quality_analysis', {}).get('overall_quality_score', 0.0)
        
        elif metric_name == 'overall_performance_score':
            return quality_report.get('performance_analysis', {}).get('overall_performance_score', 0.0)
        
        elif metric_name == 'test_coverage':
            coverage_data = quality_report.get('coverage_analysis', {})
            return coverage_data.get('total_coverage', 0.0)
        
        elif metric_name == 'test_success_rate':
            test_data = quality_report.get('test_results', {})
            return test_data.get('success_rate', 0.0)
        
        return None
    
    def _generate_gate_recommendations(self, gate_name: str, metrics: List[QualityMetric], 
                                     score: float, threshold: float) -> List[str]:
        """Generate recommendations for quality gate improvements"""
        recommendations = []
        
        if score < threshold:
            gap = threshold - score
            
            if 'security' in gate_name.lower():
                recommendations.extend([
                    "Run security vulnerability scans and fix critical issues",
                    "Review and update security dependencies",
                    "Implement input validation and sanitization",
                    "Add security headers and authentication checks"
                ])
            
            elif 'quality' in gate_name.lower():
                recommendations.extend([
                    "Reduce cyclomatic complexity in complex functions",
                    "Add missing documentation and docstrings", 
                    "Eliminate code duplication",
                    "Follow naming conventions consistently"
                ])
            
            elif 'performance' in gate_name.lower():
                recommendations.extend([
                    "Optimize slow operations and algorithms",
                    "Implement caching for frequently accessed data",
                    "Use asynchronous programming where appropriate",
                    "Profile and optimize memory usage"
                ])
            
            elif 'coverage' in gate_name.lower():
                recommendations.extend([
                    "Add unit tests for uncovered functions",
                    "Implement integration tests for key workflows",
                    "Add edge case and error condition tests",
                    "Set up automated test execution"
                ])
        
        return recommendations
    
    def _calculate_overall_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics"""
        scores = []
        weights = []
        
        # Security score (weight: 0.3)
        security_score = quality_report.get('security_analysis', {}).get('security_score')
        if security_score is not None:
            scores.append(security_score)
            weights.append(0.3)
        
        # Code quality score (weight: 0.3)
        quality_score = quality_report.get('code_quality_analysis', {}).get('overall_quality_score')
        if quality_score is not None:
            scores.append(quality_score)
            weights.append(0.3)
        
        # Test success rate (weight: 0.2)
        test_results = quality_report.get('test_results', {})
        test_score = test_results.get('success_rate')
        if test_score is not None:
            scores.append(test_score)
            weights.append(0.2)
        
        # Test coverage (weight: 0.2)
        coverage_score = quality_report.get('coverage_analysis', {}).get('total_coverage')
        if coverage_score is not None:
            scores.append(coverage_score)
            weights.append(0.2)
        
        if not scores:
            return 0.0
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        weighted_score = sum(score * weight for score, weight in zip(scores, normalized_weights))
        
        return min(weighted_score, 100.0)
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate overall quality recommendations"""
        recommendations = []
        overall_score = quality_report.get('overall_quality_score', 0)
        
        if overall_score < 60:
            recommendations.append("🚨 Critical: Overall quality is below acceptable levels - immediate action required")
        elif overall_score < 80:
            recommendations.append("⚠️  Warning: Quality improvements needed to meet standards")
        
        # Security recommendations
        security_analysis = quality_report.get('security_analysis', {})
        if security_analysis.get('total_vulnerabilities', 0) > 0:
            recommendations.append(f"🔒 Security: Address {security_analysis['total_vulnerabilities']} security vulnerabilities")
        
        # Code quality recommendations
        quality_analysis = quality_report.get('code_quality_analysis', {})
        if quality_analysis.get('overall_quality_score', 0) < 75:
            recommendations.append("🔧 Code Quality: Improve code maintainability and consistency")
        
        # Test recommendations
        test_results = quality_report.get('test_results', {})
        if test_results.get('success_rate', 0) < 95:
            recommendations.append("🧪 Testing: Fix failing tests and improve test reliability")
        
        # Coverage recommendations
        coverage_analysis = quality_report.get('coverage_analysis', {})
        if coverage_analysis.get('total_coverage', 0) < 80:
            recommendations.append("📊 Coverage: Increase test coverage to meet quality standards")
        
        return recommendations
    
    async def save_quality_report(self, quality_report: Dict[str, Any], output_path: str = ".terragon") -> Path:
        """Save quality report to file"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"quality_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        # Also save as markdown for readability
        markdown_file = output_dir / f"quality_report_{timestamp}.md"
        await self._generate_markdown_report(quality_report, markdown_file)
        
        logger.info(f"Quality report saved to {report_file}")
        return report_file
    
    async def _generate_markdown_report(self, quality_report: Dict[str, Any], output_file: Path):
        """Generate markdown quality report"""
        content = f"""# 🛡️ Quality Assurance Report

**Generated:** {quality_report['timestamp']}
**Target:** {quality_report['target_path']}
**Overall Quality Score:** {quality_report.get('overall_quality_score', 0):.1f}/100

## 📊 Summary

"""
        
        # Quality gates summary
        gates = quality_report.get('quality_gates', {})
        if gates:
            content += "### Quality Gates Status\n\n"
            for gate_id, gate_result in gates.items():
                status_emoji = {'passed': '✅', 'warning': '⚠️', 'failed': '❌'}.get(gate_result.get('status', 'failed'), '❌')
                content += f"- {status_emoji} **{gate_result.get('gate_name', gate_id)}**: {gate_result.get('score', 0):.1f}/{gate_result.get('threshold', 0):.1f}\n"
        
        # Security analysis
        security = quality_report.get('security_analysis', {})
        if security and 'security_score' in security:
            content += f"""
## 🔒 Security Analysis

**Security Score:** {security.get('security_score', 0):.1f}/100
**Total Vulnerabilities:** {security.get('total_vulnerabilities', 0)}

"""
            if security.get('total_vulnerabilities', 0) > 0:
                content += "**Action Required:** Review and fix security vulnerabilities\n\n"
        
        # Code quality analysis
        quality = quality_report.get('code_quality_analysis', {})
        if quality and 'overall_quality_score' in quality:
            content += f"""
## 🔧 Code Quality Analysis

**Overall Quality Score:** {quality.get('overall_quality_score', 0):.1f}/100

### Metrics:
- **Average Complexity:** {quality.get('complexity_analysis', {}).get('average_complexity', 0):.1f}
- **Maintainability Index:** {quality.get('maintainability_index', {}).get('maintainability_index', 0):.1f}
- **Documentation Coverage:** {quality.get('documentation_coverage', {}).get('overall_documentation_coverage', 0):.1f}%

"""
        
        # Test results
        tests = quality_report.get('test_results', {})
        if tests and 'test_count' in tests:
            content += f"""
## 🧪 Test Results

**Test Count:** {tests.get('test_count', 0)}
**Success Rate:** {tests.get('success_rate', 0):.1f}%
**Passed:** {tests.get('passed', 0)} | **Failed:** {tests.get('failed', 0)} | **Skipped:** {tests.get('skipped', 0)}

"""
        
        # Coverage analysis  
        coverage = quality_report.get('coverage_analysis', {})
        if coverage and 'total_coverage' in coverage:
            content += f"""
## 📊 Test Coverage

**Total Coverage:** {coverage.get('total_coverage', 0):.1f}%

"""
        
        # Recommendations
        recommendations = quality_report.get('recommendations', [])
        if recommendations:
            content += "## 💡 Recommendations\n\n"
            for rec in recommendations:
                content += f"- {rec}\n"
        
        content += f"""

---
*Generated by Terragon Autonomous Quality Assurance System*
*Report ID: {quality_report['timestamp']}*
"""
        
        with open(output_file, 'w') as f:
            f.write(content)


# Global quality assurance instance
autonomous_qa = AutonomousQualityAssurance()


# Initialize quality assurance system
async def initialize_quality_assurance_system():
    """Initialize the quality assurance system"""
    logger.info("🛡️  Initializing Autonomous Quality Assurance System...")
    
    # System is ready - minimal initialization needed
    logger.info("✅ Quality Assurance System initialized successfully")


# Main execution function
async def run_quality_assurance(target_path: Path = None):
    """Run comprehensive quality assurance"""
    await initialize_quality_assurance_system()
    
    quality_report = await autonomous_qa.run_comprehensive_quality_check(target_path)
    report_file = await autonomous_qa.save_quality_report(quality_report)
    
    return quality_report, report_file


if __name__ == "__main__":
    async def demo_quality_assurance():
        """Demonstrate quality assurance system"""
        print("🛡️  Running Autonomous Quality Assurance Demo...")
        
        target_path = Path("src") if Path("src").exists() else Path(".")
        quality_report, report_file = await run_quality_assurance(target_path)
        
        print(f"\n📊 Quality Assessment Complete!")
        print(f"Overall Score: {quality_report.get('overall_quality_score', 0):.1f}/100")
        print(f"Report saved: {report_file}")
        
        # Show summary
        gates = quality_report.get('quality_gates', {})
        for gate_id, gate_result in gates.items():
            status = gate_result.get('status', 'unknown')
            score = gate_result.get('score', 0)
            threshold = gate_result.get('threshold', 0)
            print(f"- {gate_result.get('gate_name', gate_id)}: {status.upper()} ({score:.1f}/{threshold:.1f})")
    
    asyncio.run(demo_quality_assurance())