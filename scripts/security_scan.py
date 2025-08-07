#!/usr/bin/env python3
"""
Automated Security Scanning Script
Performs comprehensive security analysis including dependency scanning, 
code analysis, and vulnerability detection
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import tempfile
import hashlib
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class SecurityFinding:
    """Structure for security findings"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    category: str
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    cwe_id: Optional[str]
    recommendation: str

@dataclass
class SecurityScanResult:
    """Structure for complete security scan results"""
    scan_timestamp: str
    scan_duration_seconds: float
    total_findings: int
    findings_by_severity: Dict[str, int]
    findings: List[SecurityFinding]
    dependency_vulnerabilities: int
    code_quality_score: float
    overall_risk_level: str
    recommendations: List[str]

class SecurityScanner:
    """Comprehensive security scanner"""
    
    def __init__(self, project_root: Path, output_dir: Path = None):
        self.project_root = Path(project_root)
        self.output_dir = output_dir or (self.project_root / "security_reports")
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Security patterns to scan for
        self.security_patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> Dict[str, List[Dict]]:
        """Load security patterns for code analysis"""
        return {
            'secrets': [
                {
                    'pattern': r'(?i)(password|pwd|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                    'severity': 'HIGH',
                    'title': 'Potential hardcoded secret',
                    'cwe': 'CWE-798'
                },
                {
                    'pattern': r'(?i)api[_-]?key\s*=\s*["\'][^"\']+["\']',
                    'severity': 'HIGH',
                    'title': 'Hardcoded API key',
                    'cwe': 'CWE-798'
                },
                {
                    'pattern': r'["\'][A-Za-z0-9+/]{20,}={0,2}["\']',
                    'severity': 'MEDIUM',
                    'title': 'Potential base64 encoded secret',
                    'cwe': 'CWE-798'
                }
            ],
            'injection': [
                {
                    'pattern': r'(?i)execute\s*\(\s*["\'].*%s.*["\']',
                    'severity': 'HIGH',
                    'title': 'Potential SQL injection vulnerability',
                    'cwe': 'CWE-89'
                },
                {
                    'pattern': r'(?i)dynamic_eval\s*\(',
                    'severity': 'HIGH',
                    'title': 'Dynamic code evaluation detected',
                    'cwe': 'CWE-95'
                },
                {
                    'pattern': r'(?i)dynamic_exec\s*\(',
                    'severity': 'CRITICAL',
                    'title': 'Dynamic code execution detected',
                    'cwe': 'CWE-95'
                }
            ],
            'crypto': [
                {
                    'pattern': r'(?i)md5\s*\(',
                    'severity': 'MEDIUM',
                    'title': 'Use of weak MD5 hash',
                    'cwe': 'CWE-327'
                },
                {
                    'pattern': r'(?i)sha1\s*\(',
                    'severity': 'MEDIUM',
                    'title': 'Use of weak SHA1 hash',
                    'cwe': 'CWE-327'
                },
                {
                    'pattern': r'(?i)random\.random\s*\(',
                    'severity': 'LOW',
                    'title': 'Use of weak random number generator',
                    'cwe': 'CWE-338'
                }
            ],
            'file_handling': [
                {
                    'pattern': r'(?i)open\s*\(\s*.*user.*input',
                    'severity': 'HIGH',
                    'title': 'Potential path traversal vulnerability',
                    'cwe': 'CWE-22'
                },
                {
                    'pattern': r'(?i)pickle\.loads?\s*\(',
                    'severity': 'HIGH',
                    'title': 'Insecure deserialization',
                    'cwe': 'CWE-502'
                }
            ],
            'network': [
                {
                    'pattern': r'(?i)verify\s*=\s*False',
                    'severity': 'HIGH',
                    'title': 'SSL verification disabled',
                    'cwe': 'CWE-295'
                },
                {
                    'pattern': r'(?i)http://',
                    'severity': 'LOW',
                    'title': 'Insecure HTTP protocol',
                    'cwe': 'CWE-319'
                }
            ]
        }
    
    def run_comprehensive_scan(self) -> SecurityScanResult:
        """Run comprehensive security scan"""
        self.logger.info("Starting comprehensive security scan...")
        start_time = time.time()
        
        all_findings = []
        
        # 1. Static code analysis
        self.logger.info("Running static code analysis...")
        code_findings = self._scan_code_for_vulnerabilities()
        all_findings.extend(code_findings)
        
        # 2. Dependency vulnerability scan
        self.logger.info("Scanning dependencies for vulnerabilities...")
        dep_findings, dep_count = self._scan_dependencies()
        all_findings.extend(dep_findings)
        
        # 3. Configuration security scan
        self.logger.info("Scanning configuration files...")
        config_findings = self._scan_configuration_files()
        all_findings.extend(config_findings)
        
        # 4. Docker security scan (if Dockerfile exists)
        if (self.project_root / "Dockerfile").exists():
            self.logger.info("Scanning Docker configuration...")
            docker_findings = self._scan_docker_security()
            all_findings.extend(docker_findings)
        
        # 5. Secrets detection
        self.logger.info("Scanning for exposed secrets...")
        secret_findings = self._scan_for_secrets()
        all_findings.extend(secret_findings)
        
        # 6. File permissions scan
        self.logger.info("Checking file permissions...")
        permission_findings = self._scan_file_permissions()
        all_findings.extend(permission_findings)
        
        # Calculate metrics
        scan_duration = time.time() - start_time
        findings_by_severity = self._categorize_findings_by_severity(all_findings)
        
        # Determine overall risk level
        overall_risk = self._calculate_overall_risk(findings_by_severity)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_findings)
        
        # Calculate code quality score
        quality_score = self._calculate_code_quality_score(all_findings)
        
        result = SecurityScanResult(
            scan_timestamp=datetime.utcnow().isoformat(),
            scan_duration_seconds=scan_duration,
            total_findings=len(all_findings),
            findings_by_severity=findings_by_severity,
            findings=all_findings,
            dependency_vulnerabilities=dep_count,
            code_quality_score=quality_score,
            overall_risk_level=overall_risk,
            recommendations=recommendations
        )
        
        # Save results
        self._save_scan_results(result)
        
        self.logger.info(f"Security scan completed in {scan_duration:.2f} seconds")
        self.logger.info(f"Total findings: {len(all_findings)}")
        self.logger.info(f"Overall risk level: {overall_risk}")
        
        return result
    
    def _scan_code_for_vulnerabilities(self) -> List[SecurityFinding]:
        """Scan code for security vulnerabilities"""
        findings = []
        
        # Scan Python files
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip virtual environments and cached files
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                findings.extend(self._scan_file_content(file_path, content))
                
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path}: {e}")
        
        return findings
    
    def _scan_file_content(self, file_path: Path, content: str) -> List[SecurityFinding]:
        """Scan file content for security issues"""
        findings = []
        lines = content.split('\n')
        
        for category, patterns in self.security_patterns.items():
            for pattern_config in patterns:
                pattern = pattern_config['pattern']
                
                for line_num, line in enumerate(lines, 1):
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        finding = SecurityFinding(
                            severity=pattern_config['severity'],
                            category=category,
                            title=pattern_config['title'],
                            description=f"Detected at line {line_num}: {line.strip()[:100]}",
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            cwe_id=pattern_config.get('cwe'),
                            recommendation=self._get_recommendation_for_pattern(pattern_config)
                        )
                        findings.append(finding)
        
        return findings
    
    def _scan_dependencies(self) -> tuple[List[SecurityFinding], int]:
        """Scan dependencies for known vulnerabilities"""
        findings = []
        vuln_count = 0
        
        # Check for requirements.txt
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            return findings, vuln_count
        
        try:
            # Try using safety if available
            result = subprocess.run(
                ['safety', 'check', '-r', str(requirements_file), '--json'],
                capture_output=True, text=True, timeout=60
            )
            
            if result.returncode == 0:
                # Parse safety output
                safety_data = json.loads(result.stdout) if result.stdout.strip() else []
                
                for vuln in safety_data:
                    finding = SecurityFinding(
                        severity='HIGH',
                        category='dependency',
                        title=f"Vulnerable dependency: {vuln.get('package', 'unknown')}",
                        description=vuln.get('advisory', 'No description available'),
                        file_path='requirements.txt',
                        line_number=None,
                        cwe_id=None,
                        recommendation=f"Update to version {vuln.get('recommended_version', 'latest')}"
                    )
                    findings.append(finding)
                    vuln_count += 1
            
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Fallback: manual check of known vulnerable packages
            findings.extend(self._manual_dependency_check(requirements_file))
        
        return findings, vuln_count
    
    def _manual_dependency_check(self, requirements_file: Path) -> List[SecurityFinding]:
        """Manual check for known vulnerable dependencies"""
        findings = []
        
        # Known vulnerable package patterns (simplified)
        vulnerable_patterns = [
            ('flask<1.0', 'Flask version < 1.0 has known vulnerabilities'),
            ('django<2.2', 'Django version < 2.2 has known vulnerabilities'),
            ('requests<2.20', 'Requests version < 2.20 has known vulnerabilities'),
            ('pyyaml<5.1', 'PyYAML version < 5.1 has arbitrary code execution vulnerability')
        ]
        
        try:
            with open(requirements_file, 'r') as f:
                requirements_content = f.read()
                
            for pattern, description in vulnerable_patterns:
                if pattern in requirements_content.lower():
                    finding = SecurityFinding(
                        severity='MEDIUM',
                        category='dependency',
                        title=f'Potentially vulnerable dependency detected',
                        description=description,
                        file_path='requirements.txt',
                        line_number=None,
                        cwe_id='CWE-937',
                        recommendation='Update to the latest secure version'
                    )
                    findings.append(finding)
        
        except Exception as e:
            self.logger.warning(f"Could not perform manual dependency check: {e}")
        
        return findings
    
    def _scan_configuration_files(self) -> List[SecurityFinding]:
        """Scan configuration files for security issues"""
        findings = []
        
        # Check for common configuration files
        config_files = [
            'config.py', 'settings.py', '.env', 'docker-compose.yml',
            'Dockerfile', '*.conf', '*.ini', '*.yaml', '*.yml'
        ]
        
        for pattern in config_files:
            for config_file in self.project_root.rglob(pattern):
                if config_file.is_file():
                    findings.extend(self._scan_config_file(config_file))
        
        return findings
    
    def _scan_config_file(self, config_file: Path) -> List[SecurityFinding]:
        """Scan individual configuration file"""
        findings = []
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for debug mode enabled
            if re.search(r'(?i)debug\s*=\s*true', content):
                findings.append(SecurityFinding(
                    severity='MEDIUM',
                    category='configuration',
                    title='Debug mode enabled',
                    description='Debug mode is enabled in configuration',
                    file_path=str(config_file.relative_to(self.project_root)),
                    line_number=None,
                    cwe_id='CWE-489',
                    recommendation='Disable debug mode in production'
                ))
            
            # Check for default passwords
            default_patterns = [
                r'(?i)password\s*=\s*["\']admin["\']',
                r'(?i)password\s*=\s*["\']password["\']',
                r'(?i)password\s*=\s*["\']123456["\']'
            ]
            
            for pattern in default_patterns:
                if re.search(pattern, content):
                    findings.append(SecurityFinding(
                        severity='HIGH',
                        category='configuration',
                        title='Default password detected',
                        description='Default or weak password found in configuration',
                        file_path=str(config_file.relative_to(self.project_root)),
                        line_number=None,
                        cwe_id='CWE-521',
                        recommendation='Change default passwords to strong, unique passwords'
                    ))
        
        except Exception as e:
            self.logger.warning(f"Could not scan config file {config_file}: {e}")
        
        return findings
    
    def _scan_docker_security(self) -> List[SecurityFinding]:
        """Scan Docker configuration for security issues"""
        findings = []
        dockerfile = self.project_root / "Dockerfile"
        
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Check for running as root
                if line.startswith('USER root') or 'USER 0' in line:
                    findings.append(SecurityFinding(
                        severity='HIGH',
                        category='docker',
                        title='Container runs as root',
                        description=f'Line {line_num}: {line}',
                        file_path='Dockerfile',
                        line_number=line_num,
                        cwe_id='CWE-250',
                        recommendation='Create and use a non-root user'
                    ))
                
                # Check for ADD instruction (prefer COPY)
                if line.startswith('ADD '):
                    findings.append(SecurityFinding(
                        severity='LOW',
                        category='docker',
                        title='Use of ADD instruction',
                        description=f'Line {line_num}: Prefer COPY over ADD',
                        file_path='Dockerfile',
                        line_number=line_num,
                        cwe_id=None,
                        recommendation='Use COPY instead of ADD unless you need tar extraction'
                    ))
                
                # Check for latest tag
                if ':latest' in line or (line.startswith('FROM ') and ':' not in line):
                    findings.append(SecurityFinding(
                        severity='MEDIUM',
                        category='docker',
                        title='Use of latest tag',
                        description=f'Line {line_num}: Using latest tag is not reproducible',
                        file_path='Dockerfile',
                        line_number=line_num,
                        cwe_id=None,
                        recommendation='Use specific version tags instead of latest'
                    ))
        
        except Exception as e:
            self.logger.warning(f"Could not scan Dockerfile: {e}")
        
        return findings
    
    def _scan_for_secrets(self) -> List[SecurityFinding]:
        """Scan for exposed secrets and credentials"""
        findings = []
        
        # Common secret patterns
        secret_patterns = [
            (r'(?i)aws_access_key_id\s*=\s*["\'][A-Z0-9]{20}["\']', 'AWS Access Key'),
            (r'(?i)aws_secret_access_key\s*=\s*["\'][A-Za-z0-9+/]{40}["\']', 'AWS Secret Key'),
            (r'(?i)github_token\s*=\s*["\']ghp_[A-Za-z0-9]{36}["\']', 'GitHub Token'),
            (r'(?i)slack_token\s*=\s*["\']xox[a-z]-[A-Za-z0-9-]+["\']', 'Slack Token'),
            (r'(?i)database_url\s*=\s*["\'][^"\']*://[^"\']*:[^"\']*@[^"\']*["\']', 'Database URL with credentials')
        ]
        
        # Scan all text files
        text_files = []
        for ext in ['*.py', '*.js', '*.ts', '*.json', '*.yaml', '*.yml', '*.env', '*.conf']:
            text_files.extend(self.project_root.rglob(ext))
        
        for file_path in text_files:
            if any(skip in str(file_path) for skip in ['venv', '__pycache__', '.git', 'node_modules']):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, secret_type in secret_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        finding = SecurityFinding(
                            severity='CRITICAL',
                            category='secrets',
                            title=f'Exposed {secret_type}',
                            description=f'Potential {secret_type} found in source code',
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=None,
                            cwe_id='CWE-798',
                            recommendation='Remove secrets from source code and use environment variables or secret management'
                        )
                        findings.append(finding)
                        
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path} for secrets: {e}")
        
        return findings
    
    def _scan_file_permissions(self) -> List[SecurityFinding]:
        """Scan for insecure file permissions"""
        findings = []
        
        # Check for overly permissive files
        sensitive_files = ['*.key', '*.pem', '*.p12', '*.pfx', '.env', 'config.py']
        
        for pattern in sensitive_files:
            for file_path in self.project_root.rglob(pattern):
                if file_path.is_file():
                    try:
                        # Check if file is world-readable (Unix-like systems)
                        stat_info = file_path.stat()
                        permissions = oct(stat_info.st_mode)[-3:]
                        
                        # Check if others can read (last digit >= 4)
                        if int(permissions[-1]) >= 4:
                            findings.append(SecurityFinding(
                                severity='MEDIUM',
                                category='permissions',
                                title='Overly permissive file permissions',
                                description=f'File {file_path.name} is world-readable (permissions: {permissions})',
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=None,
                                cwe_id='CWE-732',
                                recommendation='Restrict file permissions to owner only (chmod 600)'
                            ))
                            
                    except Exception as e:
                        self.logger.warning(f"Could not check permissions for {file_path}: {e}")
        
        return findings
    
    def _categorize_findings_by_severity(self, findings: List[SecurityFinding]) -> Dict[str, int]:
        """Categorize findings by severity level"""
        categories = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for finding in findings:
            if finding.severity in categories:
                categories[finding.severity] += 1
        
        return categories
    
    def _calculate_overall_risk(self, findings_by_severity: Dict[str, int]) -> str:
        """Calculate overall risk level"""
        if findings_by_severity['CRITICAL'] > 0:
            return 'CRITICAL'
        elif findings_by_severity['HIGH'] > 3:
            return 'HIGH'
        elif findings_by_severity['HIGH'] > 0 or findings_by_severity['MEDIUM'] > 5:
            return 'MEDIUM'
        elif findings_by_severity['MEDIUM'] > 0 or findings_by_severity['LOW'] > 10:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _calculate_code_quality_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate code quality score based on findings"""
        base_score = 100.0
        
        severity_weights = {
            'CRITICAL': 20,
            'HIGH': 10,
            'MEDIUM': 5,
            'LOW': 1
        }
        
        for finding in findings:
            weight = severity_weights.get(finding.severity, 0)
            base_score -= weight
        
        return max(0.0, min(100.0, base_score))
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = set()
        
        # High priority recommendations based on findings
        critical_findings = [f for f in findings if f.severity == 'CRITICAL']
        high_findings = [f for f in findings if f.severity == 'HIGH']
        
        if critical_findings:
            recommendations.add("URGENT: Address critical security vulnerabilities immediately")
            recommendations.add("Remove any exposed secrets and rotate affected credentials")
        
        if high_findings:
            recommendations.add("Address high-severity security issues within 24-48 hours")
            
        # Category-specific recommendations
        categories = set(f.category for f in findings)
        
        if 'secrets' in categories:
            recommendations.add("Implement a secrets management solution (e.g., HashiCorp Vault, AWS Secrets Manager)")
            
        if 'dependency' in categories:
            recommendations.add("Implement automated dependency vulnerability scanning in CI/CD pipeline")
            
        if 'configuration' in categories:
            recommendations.add("Review and harden configuration files")
            
        if 'docker' in categories:
            recommendations.add("Follow Docker security best practices")
        
        # General recommendations
        recommendations.add("Implement regular security scanning in development workflow")
        recommendations.add("Conduct security code reviews for all changes")
        recommendations.add("Establish incident response procedures for security issues")
        
        return list(recommendations)
    
    def _get_recommendation_for_pattern(self, pattern_config: Dict) -> str:
        """Get specific recommendation for a security pattern"""
        category_recommendations = {
            'secrets': 'Remove hardcoded secrets and use environment variables or secure vaults',
            'injection': 'Use parameterized queries and input validation',
            'crypto': 'Use strong cryptographic algorithms (SHA-256 or better)',
            'file_handling': 'Validate and sanitize file paths, avoid user-controlled file operations',
            'network': 'Use HTTPS and enable SSL certificate verification'
        }
        
        return pattern_config.get('recommendation', 
                                 category_recommendations.get(pattern_config.get('category', ''), 
                                                            'Review and address this security issue'))
    
    def _save_scan_results(self, result: SecurityScanResult):
        """Save scan results to files"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON report
        json_file = self.output_dir / f"security_scan_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        # Save human-readable report
        report_file = self.output_dir / f"security_report_{timestamp}.md"
        self._generate_markdown_report(result, report_file)
        
        # Save summary for CI/CD
        summary_file = self.output_dir / "security_summary.json"
        summary = {
            'timestamp': result.scan_timestamp,
            'overall_risk_level': result.overall_risk_level,
            'total_findings': result.total_findings,
            'critical_findings': result.findings_by_severity.get('CRITICAL', 0),
            'high_findings': result.findings_by_severity.get('HIGH', 0),
            'code_quality_score': result.code_quality_score
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Reports saved to {self.output_dir}")
    
    def _generate_markdown_report(self, result: SecurityScanResult, output_file: Path):
        """Generate human-readable markdown report"""
        with open(output_file, 'w') as f:
            f.write(f"# Security Scan Report\n\n")
            f.write(f"**Scan Date:** {result.scan_timestamp}\n")
            f.write(f"**Scan Duration:** {result.scan_duration_seconds:.2f} seconds\n")
            f.write(f"**Overall Risk Level:** {result.overall_risk_level}\n")
            f.write(f"**Code Quality Score:** {result.code_quality_score:.1f}/100\n\n")
            
            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Findings:** {result.total_findings}\n")
            for severity, count in result.findings_by_severity.items():
                f.write(f"- **{severity}:** {count}\n")
            f.write(f"- **Dependency Vulnerabilities:** {result.dependency_vulnerabilities}\n\n")
            
            # Recommendations
            if result.recommendations:
                f.write("## Recommendations\n\n")
                for rec in result.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
            
            # Detailed findings
            f.write("## Detailed Findings\n\n")
            
            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
                severity_findings = [f for f in result.findings if f.severity == severity]
                if severity_findings:
                    f.write(f"### {severity} Severity\n\n")
                    
                    for finding in severity_findings:
                        f.write(f"#### {finding.title}\n")
                        f.write(f"- **Category:** {finding.category}\n")
                        f.write(f"- **Description:** {finding.description}\n")
                        if finding.file_path:
                            f.write(f"- **File:** {finding.file_path}")
                            if finding.line_number:
                                f.write(f" (line {finding.line_number})")
                            f.write("\n")
                        if finding.cwe_id:
                            f.write(f"- **CWE:** {finding.cwe_id}\n")
                        f.write(f"- **Recommendation:** {finding.recommendation}\n\n")


def main():
    """Main function to run security scan"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive security scan')
    parser.add_argument('--project-root', type=Path, default=Path.cwd(),
                       help='Root directory of the project to scan')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for scan results')
    parser.add_argument('--fail-on-critical', action='store_true',
                       help='Exit with non-zero code if critical findings are found')
    parser.add_argument('--fail-on-high', action='store_true',
                       help='Exit with non-zero code if high severity findings are found')
    
    args = parser.parse_args()
    
    # Run security scan
    scanner = SecurityScanner(args.project_root, args.output_dir)
    result = scanner.run_comprehensive_scan()
    
    # Print summary
    print("\n" + "="*60)
    print("SECURITY SCAN SUMMARY")
    print("="*60)
    print(f"Overall Risk Level: {result.overall_risk_level}")
    print(f"Code Quality Score: {result.code_quality_score:.1f}/100")
    print(f"Total Findings: {result.total_findings}")
    print("\nFindings by Severity:")
    for severity, count in result.findings_by_severity.items():
        print(f"  {severity}: {count}")
    
    # Exit codes based on findings
    if args.fail_on_critical and result.findings_by_severity.get('CRITICAL', 0) > 0:
        print("\nFAILED: Critical security findings detected")
        sys.exit(1)
    elif args.fail_on_high and result.findings_by_severity.get('HIGH', 0) > 0:
        print("\nFAILED: High severity security findings detected")
        sys.exit(1)
    else:
        print("\nSecurity scan completed successfully")
        sys.exit(0)


if __name__ == '__main__':
    main()