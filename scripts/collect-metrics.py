#!/usr/bin/env python3
"""
Metrics Collection Script for Observer Coordinator Insights.

This script collects comprehensive metrics about the project including
code quality, security, performance, and business metrics.
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and reports project metrics."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.metrics = {}
        self.timestamp = datetime.now(timezone.utc).isoformat()
        
    def load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all configured metrics."""
        logger.info("Starting metrics collection...")
        
        # Collect different categories of metrics
        self.collect_code_quality_metrics()
        self.collect_security_metrics()
        self.collect_performance_metrics()
        self.collect_reliability_metrics()
        self.collect_development_metrics()
        self.collect_business_metrics()
        self.collect_compliance_metrics()
        
        return {
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'collection_duration_seconds': time.time() - self.start_time
        }
    
    def collect_code_quality_metrics(self) -> None:
        """Collect code quality metrics."""
        logger.info("Collecting code quality metrics...")
        
        try:
            # Test coverage
            coverage = self.get_test_coverage()
            self.metrics['code_quality'] = {
                'test_coverage': coverage,
                'cyclomatic_complexity': self.get_cyclomatic_complexity(),
                'maintainability_index': self.get_maintainability_index(),
                'technical_debt_ratio': self.get_technical_debt_ratio(),
                'code_duplication': self.get_code_duplication(),
                'lines_of_code': self.get_lines_of_code(),
                'last_measured': self.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to collect code quality metrics: {e}")
            self.metrics['code_quality'] = {'error': str(e)}
    
    def get_test_coverage(self) -> float:
        """Get current test coverage percentage."""
        try:
            # Run coverage analysis
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', '--cov=src', '--cov-report=json'],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                # Parse coverage.json if it exists
                coverage_file = Path('coverage.json')
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        coverage_data = json.load(f)
                    return round(coverage_data.get('totals', {}).get('percent_covered', 0), 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not determine test coverage: {e}")
            return 0.0
    
    def get_cyclomatic_complexity(self) -> float:
        """Get average cyclomatic complexity."""
        try:
            # Use radon to calculate complexity
            result = subprocess.run(
                ['radon', 'cc', 'src/', '-j'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                total_functions = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item['type'] in ['function', 'method']:
                            total_complexity += item['complexity']
                            total_functions += 1
                
                return round(total_complexity / total_functions if total_functions > 0 else 0, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate cyclomatic complexity: {e}")
            return 0.0
    
    def get_maintainability_index(self) -> float:
        """Get maintainability index."""
        try:
            # Use radon to calculate maintainability index
            result = subprocess.run(
                ['radon', 'mi', 'src/', '-j'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                mi_data = json.loads(result.stdout)
                scores = [item['mi'] for item in mi_data.values() if 'mi' in item]
                return round(sum(scores) / len(scores) if scores else 0, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate maintainability index: {e}")
            return 0.0
    
    def get_technical_debt_ratio(self) -> float:
        """Get technical debt ratio."""
        try:
            # Run pylint to get code issues
            result = subprocess.run(
                ['pylint', 'src/', '--output-format=json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.stdout:
                issues = json.loads(result.stdout)
                total_lines = self.get_lines_of_code()
                if total_lines > 0:
                    debt_ratio = (len(issues) / total_lines) * 100
                    return round(debt_ratio, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate technical debt ratio: {e}")
            return 0.0
    
    def get_code_duplication(self) -> float:
        """Get code duplication percentage."""
        try:
            # Use a simple approach to detect duplication
            # In production, you might use tools like jscpd or similar
            
            # For now, return a placeholder
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate code duplication: {e}")
            return 0.0
    
    def get_lines_of_code(self) -> int:
        """Get total lines of code."""
        try:
            total_lines = 0
            for py_file in Path('src').rglob('*.py'):
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_lines += len(f.readlines())
            return total_lines
        except Exception as e:
            logger.warning(f"Could not count lines of code: {e}")
            return 0
    
    def collect_security_metrics(self) -> None:
        """Collect security metrics."""
        logger.info("Collecting security metrics...")
        
        try:
            self.metrics['security'] = {
                'vulnerability_count': self.get_vulnerability_count(),
                'dependency_freshness': self.get_dependency_freshness(),
                'secrets_exposure': self.get_secrets_exposure(),
                'last_scan': self.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to collect security metrics: {e}")
            self.metrics['security'] = {'error': str(e)}
    
    def get_vulnerability_count(self) -> Dict[str, int]:
        """Get vulnerability counts by severity."""
        try:
            # Run safety check
            result = subprocess.run(
                [sys.executable, '-m', 'safety', 'check', '--json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            vulnerabilities = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
            
            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        severity = vuln.get('severity', 'low').lower()
                        if severity in vulnerabilities:
                            vulnerabilities[severity] += 1
                        else:
                            vulnerabilities['low'] += 1
                except json.JSONDecodeError:
                    pass
            
            return vulnerabilities
        except Exception as e:
            logger.warning(f"Could not get vulnerability count: {e}")
            return {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
    
    def get_dependency_freshness(self) -> Dict[str, int]:
        """Get dependency freshness metrics."""
        try:
            # Check for outdated packages
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                return {
                    'outdated_dependencies': len(outdated),
                    'security_updates_available': 0,  # Would need additional checking
                    'last_check': self.timestamp
                }
            
            return {'outdated_dependencies': 0, 'security_updates_available': 0}
        except Exception as e:
            logger.warning(f"Could not check dependency freshness: {e}")
            return {'outdated_dependencies': 0, 'security_updates_available': 0}
    
    def get_secrets_exposure(self) -> Dict[str, int]:
        """Check for potential secrets exposure."""
        try:
            # Run bandit security scan
            result = subprocess.run(
                ['bandit', '-r', 'src/', '-f', 'json'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            secrets_count = 0
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    for issue in bandit_data.get('results', []):
                        if 'password' in issue.get('issue_text', '').lower() or \
                           'secret' in issue.get('issue_text', '').lower():
                            secrets_count += 1
                except json.JSONDecodeError:
                    pass
            
            return {
                'potential_secrets': secrets_count,
                'false_positives': 0,  # Would be manually tracked
                'last_scan': self.timestamp
            }
        except Exception as e:
            logger.warning(f"Could not check secrets exposure: {e}")
            return {'potential_secrets': 0, 'false_positives': 0}
    
    def collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        logger.info("Collecting performance metrics...")
        
        try:
            self.metrics['performance'] = {
                'build_time': self.get_build_time(),
                'test_execution_time': self.get_test_execution_time(),
                'clustering_performance': self.get_clustering_performance(),
                'last_measured': self.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")
            self.metrics['performance'] = {'error': str(e)}
    
    def get_build_time(self) -> float:
        """Get average build time."""
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, '-m', 'build'],
                capture_output=True,
                text=True,
                timeout=600
            )
            build_time = time.time() - start_time
            
            if result.returncode == 0:
                return round(build_time, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not measure build time: {e}")
            return 0.0
    
    def get_test_execution_time(self) -> float:
        """Get test execution time."""
        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v'],
                capture_output=True,
                text=True,
                timeout=300
            )
            test_time = time.time() - start_time
            
            if result.returncode == 0:
                return round(test_time, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not measure test execution time: {e}")
            return 0.0
    
    def get_clustering_performance(self) -> Dict[str, float]:
        """Get clustering algorithm performance metrics."""
        try:
            # Run a performance benchmark
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/performance/', '--benchmark-only', '--benchmark-json=benchmark.json'],
                capture_output=True,
                text=True,
                timeout=180
            )
            
            if result.returncode == 0 and Path('benchmark.json').exists():
                with open('benchmark.json', 'r') as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get('benchmarks', [])
                if benchmarks:
                    # Get clustering-specific benchmarks
                    clustering_benchmarks = [b for b in benchmarks if 'clustering' in b.get('name', '').lower()]
                    if clustering_benchmarks:
                        benchmark = clustering_benchmarks[0]  # Take first clustering benchmark
                        stats = benchmark.get('stats', {})
                        
                        # Calculate samples per second (inverse of mean time)
                        mean_time = stats.get('mean', 1.0)
                        samples_per_second = 1.0 / mean_time if mean_time > 0 else 0
                        
                        return {
                            'samples_per_second': round(samples_per_second, 2),
                            'mean_execution_time_ms': round(mean_time * 1000, 2),
                            'memory_usage_mb': 0  # Would need memory profiling
                        }
            
            return {'samples_per_second': 0, 'mean_execution_time_ms': 0, 'memory_usage_mb': 0}
        except Exception as e:
            logger.warning(f"Could not measure clustering performance: {e}")
            return {'samples_per_second': 0, 'mean_execution_time_ms': 0, 'memory_usage_mb': 0}
    
    def collect_reliability_metrics(self) -> None:
        """Collect reliability metrics."""
        logger.info("Collecting reliability metrics...")
        
        # These would typically come from monitoring systems in production
        self.metrics['reliability'] = {
            'uptime': 100.0,  # Placeholder
            'error_rate': 0.0,  # Placeholder
            'mean_time_to_recovery': 0,  # Placeholder
            'last_measured': self.timestamp
        }
    
    def collect_development_metrics(self) -> None:
        """Collect development process metrics."""
        logger.info("Collecting development metrics...")
        
        try:
            self.metrics['development'] = {
                'commit_frequency': self.get_commit_frequency(),
                'pull_request_cycle_time': 0,  # Would need GitHub API
                'code_review_coverage': 100,  # Placeholder
                'documentation_coverage': self.get_documentation_coverage(),
                'last_measured': self.timestamp
            }
        except Exception as e:
            logger.error(f"Failed to collect development metrics: {e}")
            self.metrics['development'] = {'error': str(e)}
    
    def get_commit_frequency(self) -> float:
        """Get commit frequency (commits per week)."""
        try:
            # Get commits from last week
            result = subprocess.run(
                ['git', 'log', '--since="1 week ago"', '--oneline'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                return len([c for c in commits if c.strip()])
            
            return 0
        except Exception as e:
            logger.warning(f"Could not get commit frequency: {e}")
            return 0
    
    def get_documentation_coverage(self) -> float:
        """Get documentation coverage percentage."""
        try:
            # Count Python files with docstrings
            total_files = 0
            documented_files = 0
            
            for py_file in Path('src').rglob('*.py'):
                total_files += 1
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Simple check for docstrings
                    if '"""' in content or "'''" in content:
                        documented_files += 1
            
            if total_files > 0:
                return round((documented_files / total_files) * 100, 2)
            
            return 0.0
        except Exception as e:
            logger.warning(f"Could not calculate documentation coverage: {e}")
            return 0.0
    
    def collect_business_metrics(self) -> None:
        """Collect business metrics."""
        logger.info("Collecting business metrics...")
        
        # These would typically come from application logs or analytics
        self.metrics['business'] = {
            'feature_usage': {
                'clustering_operations': {
                    'daily_count': 0,
                    'success_rate': 0,
                    'average_dataset_size': 0
                },
                'team_simulations': {
                    'daily_count': 0,
                    'success_rate': 0,
                    'average_team_size': 0
                }
            },
            'user_satisfaction': {
                'api_response_time': 0,
                'success_rate': 0
            },
            'last_measured': self.timestamp
        }
    
    def collect_compliance_metrics(self) -> None:
        """Collect compliance metrics."""
        logger.info("Collecting compliance metrics...")
        
        # These would typically come from audit logs and compliance tools
        self.metrics['compliance'] = {
            'data_retention': {
                'policy_compliance': 100,
                'data_age_distribution': {
                    '0_30_days': 0,
                    '31_90_days': 0,
                    '91_180_days': 0,
                    'over_180_days': 0
                }
            },
            'gdpr_compliance': {
                'anonymization_rate': 100,
                'audit_log_completeness': 100
            },
            'last_measured': self.timestamp
        }
    
    def save_metrics(self, metrics: Dict[str, Any], output_file: str = "metrics-report.json") -> None:
        """Save metrics to file."""
        try:
            # Ensure reports directory exists
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            output_path = reports_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            
            logger.info(f"Metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
    
    def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable summary report."""
        report = []
        report.append("📊 PROJECT METRICS SUMMARY")
        report.append("=" * 50)
        report.append(f"Generated: {metrics['timestamp']}")
        report.append(f"Collection Duration: {metrics.get('collection_duration_seconds', 0):.2f}s")
        report.append("")
        
        # Code Quality
        if 'code_quality' in metrics['metrics']:
            cq = metrics['metrics']['code_quality']
            report.append("🔍 CODE QUALITY")
            report.append(f"  Test Coverage: {cq.get('test_coverage', 0):.1f}%")
            report.append(f"  Complexity: {cq.get('cyclomatic_complexity', 0):.1f}")
            report.append(f"  Maintainability: {cq.get('maintainability_index', 0):.1f}")
            report.append(f"  Tech Debt: {cq.get('technical_debt_ratio', 0):.1f}%")
            report.append(f"  Lines of Code: {cq.get('lines_of_code', 0):,}")
            report.append("")
        
        # Security
        if 'security' in metrics['metrics']:
            sec = metrics['metrics']['security']
            vuln = sec.get('vulnerability_count', {})
            report.append("🔒 SECURITY")
            report.append(f"  Critical Vulnerabilities: {vuln.get('critical', 0)}")
            report.append(f"  High Vulnerabilities: {vuln.get('high', 0)}")
            report.append(f"  Medium Vulnerabilities: {vuln.get('medium', 0)}")
            report.append(f"  Low Vulnerabilities: {vuln.get('low', 0)}")
            
            deps = sec.get('dependency_freshness', {})
            report.append(f"  Outdated Dependencies: {deps.get('outdated_dependencies', 0)}")
            report.append("")
        
        # Performance
        if 'performance' in metrics['metrics']:
            perf = metrics['metrics']['performance']
            report.append("⚡ PERFORMANCE")
            report.append(f"  Build Time: {perf.get('build_time', 0):.1f}s")
            report.append(f"  Test Time: {perf.get('test_execution_time', 0):.1f}s")
            
            clustering = perf.get('clustering_performance', {})
            report.append(f"  Clustering Speed: {clustering.get('samples_per_second', 0):.1f} samples/sec")
            report.append("")
        
        # Development
        if 'development' in metrics['metrics']:
            dev = metrics['metrics']['development']
            report.append("👩‍💻 DEVELOPMENT")
            report.append(f"  Weekly Commits: {dev.get('commit_frequency', 0)}")
            report.append(f"  Doc Coverage: {dev.get('documentation_coverage', 0):.1f}%")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to collect and report metrics."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--output", default="metrics-report.json",
                        help="Output file name")
    parser.add_argument("--config", default=".github/project-metrics.json",
                        help="Metrics configuration file")
    parser.add_argument("--summary", action="store_true",
                        help="Print summary to console")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Collect metrics
    collector = MetricsCollector(args.config)
    collector.start_time = time.time()
    
    try:
        metrics = collector.collect_all_metrics()
        
        # Save metrics
        collector.save_metrics(metrics, args.output)
        
        # Print summary if requested
        if args.summary:
            summary = collector.generate_summary_report(metrics)
            print("\n" + summary)
        
        print(f"\n✅ Metrics collection completed successfully!")
        print(f"📄 Report saved to: reports/{args.output}")
        
    except KeyboardInterrupt:
        print("\n❌ Metrics collection cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()