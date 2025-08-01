#!/usr/bin/env python3
"""
Workflow Validation Script for Observer Coordinator Insights.

This script validates GitHub Actions workflows for syntax, security,
best practices, and completeness.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml


class WorkflowValidator:
    """Validates GitHub Actions workflows for correctness and best practices."""
    
    def __init__(self, workflows_dir: str = "docs/github-workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.issues: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        
        # Security patterns to check for
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*[:=]\s*["\'][^"\']+["\']',
                r'token\s*[:=]\s*["\'][^"\']+["\']',
                r'key\s*[:=]\s*["\'][^"\']+["\']',
                r'secret\s*[:=]\s*["\'][^"\']+["\']',
            ],
            'command_injection': [
                r'\$\{\{.*github\.event\..*\}\}',
                r'\$\{\{.*github\.head_ref.*\}\}',
                r'\$\{\{.*github\.base_ref.*\}\}',
            ],
            'unsafe_checkout': [
                r'actions/checkout@v[12](?:\s|$)',
                r'actions/checkout(?!@)',
            ]
        }
        
        # Required workflow files
        self.required_workflows = {
            'ci.yml': 'Main CI pipeline',
            'release.yml': 'Release automation',
            'security-enhanced.yml': 'Security scanning',
        }
        
        # Best practice checks
        self.best_practices = {
            'pin_actions': True,
            'use_timeout': True,
            'limit_permissions': True,
            'validate_inputs': True,
            'use_official_actions': True,
        }
    
    def validate_all(self) -> Tuple[bool, Dict[str, Any]]:
        """Validate all workflows in the directory."""
        print(f"🔍 Validating workflows in {self.workflows_dir}")
        
        if not self.workflows_dir.exists():
            self.add_issue("critical", "Workflows directory not found", 
                          f"Directory {self.workflows_dir} does not exist")
            return False, self.get_results()
        
        # Find all workflow files
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            self.add_issue("high", "No workflow files found", 
                          f"No .yml or .yaml files found in {self.workflows_dir}")
            return False, self.get_results()
        
        print(f"📄 Found {len(workflow_files)} workflow files")
        
        # Validate each workflow
        all_valid = True
        for workflow_file in workflow_files:
            print(f"   Validating {workflow_file.name}...")
            is_valid = self.validate_workflow(workflow_file)
            all_valid = all_valid and is_valid
        
        # Check for required workflows
        self.check_required_workflows(workflow_files)
        
        # Generate summary
        results = self.get_results()
        return all_valid and len(self.issues) == 0, results
    
    def validate_workflow(self, workflow_path: Path) -> bool:
        """Validate a single workflow file."""
        try:
            # Load and parse YAML
            with open(workflow_path, 'r') as f:
                content = f.read()
                workflow = yaml.safe_load(content)
            
            if not workflow:
                self.add_issue("high", f"Empty workflow: {workflow_path.name}",
                              "Workflow file is empty or invalid")
                return False
            
            # Validate workflow structure
            self.validate_structure(workflow, workflow_path.name)
            
            # Security checks
            self.check_security(content, workflow, workflow_path.name)
            
            # Best practices
            self.check_best_practices(workflow, workflow_path.name)
            
            # Specific workflow validations
            self.validate_specific_workflow(workflow, workflow_path.name)
            
            return True
            
        except yaml.YAMLError as e:
            self.add_issue("high", f"YAML syntax error in {workflow_path.name}",
                          f"Invalid YAML syntax: {str(e)}")
            return False
        except Exception as e:
            self.add_issue("medium", f"Validation error for {workflow_path.name}",
                          f"Unexpected error: {str(e)}")
            return False
    
    def validate_structure(self, workflow: Dict[str, Any], filename: str) -> None:
        """Validate basic workflow structure."""
        required_fields = ['name', 'on', 'jobs']
        
        for field in required_fields:
            if field not in workflow:
                self.add_issue("high", f"Missing required field in {filename}",
                              f"Required field '{field}' is missing")
        
        # Validate jobs structure
        if 'jobs' in workflow:
            jobs = workflow['jobs']
            if not isinstance(jobs, dict) or not jobs:
                self.add_issue("high", f"Invalid jobs structure in {filename}",
                              "Jobs must be a non-empty dictionary")
            else:
                for job_name, job_config in jobs.items():
                    self.validate_job(job_config, job_name, filename)
    
    def validate_job(self, job: Dict[str, Any], job_name: str, filename: str) -> None:
        """Validate individual job configuration."""
        if not isinstance(job, dict):
            self.add_issue("high", f"Invalid job config in {filename}",
                          f"Job '{job_name}' must be a dictionary")
            return
        
        # Check for runs-on
        if 'runs-on' not in job:
            self.add_issue("high", f"Missing runs-on in {filename}",
                          f"Job '{job_name}' missing required 'runs-on' field")
        
        # Check steps
        if 'steps' in job:
            if not isinstance(job['steps'], list):
                self.add_issue("high", f"Invalid steps in {filename}",
                              f"Job '{job_name}' steps must be a list")
            else:
                self.validate_steps(job['steps'], job_name, filename)
    
    def validate_steps(self, steps: List[Dict[str, Any]], job_name: str, filename: str) -> None:
        """Validate job steps."""
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                self.add_issue("medium", f"Invalid step in {filename}",
                              f"Job '{job_name}' step {i} must be a dictionary")
                continue
            
            # Each step should have either 'uses' or 'run'
            if 'uses' not in step and 'run' not in step:
                self.add_issue("medium", f"Invalid step in {filename}",
                              f"Job '{job_name}' step {i} must have either 'uses' or 'run'")
            
            # Validate action versions
            if 'uses' in step:
                self.validate_action_version(step['uses'], job_name, filename)
    
    def validate_action_version(self, action: str, job_name: str, filename: str) -> None:
        """Validate that actions are pinned to specific versions."""
        # Check if action is pinned to a version
        if '@' not in action:
            self.add_warning("medium", f"Unpinned action in {filename}",
                           f"Job '{job_name}' uses unpinned action '{action}'")
        elif action.endswith('@main') or action.endswith('@master'):
            self.add_warning("medium", f"Action pinned to branch in {filename}",
                           f"Job '{job_name}' action '{action}' pinned to branch, not version")
        
        # Check for deprecated actions
        deprecated_actions = [
            'actions/checkout@v1',
            'actions/checkout@v2', 
            'actions/setup-python@v1',
            'actions/setup-python@v2',
        ]
        
        for deprecated in deprecated_actions:
            if action.startswith(deprecated):
                self.add_warning("low", f"Deprecated action in {filename}",
                               f"Job '{job_name}' uses deprecated action '{action}'")
    
    def check_security(self, content: str, workflow: Dict[str, Any], filename: str) -> None:
        """Perform security checks on workflow."""
        # Check for hardcoded secrets
        for pattern_name, patterns in self.security_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    self.add_issue("high", f"Security issue in {filename}",
                                  f"Potential {pattern_name}: {matches}")
        
        # Check permissions
        self.check_permissions(workflow, filename)
        
        # Check for pull_request_target usage
        if 'on' in workflow:
            triggers = workflow['on']
            if isinstance(triggers, dict) and 'pull_request_target' in triggers:
                self.add_warning("high", f"Potential security risk in {filename}",
                               "pull_request_target can be dangerous with untrusted code")
    
    def check_permissions(self, workflow: Dict[str, Any], filename: str) -> None:
        """Check workflow and job permissions."""
        # Global permissions
        if 'permissions' in workflow:
            self.validate_permissions(workflow['permissions'], filename, "workflow")
        
        # Job-level permissions
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if 'permissions' in job_config:
                    self.validate_permissions(job_config['permissions'], filename, f"job '{job_name}'")
    
    def validate_permissions(self, permissions: Any, filename: str, context: str) -> None:
        """Validate permission configuration."""
        if permissions == 'read-all' or permissions == 'write-all':
            self.add_warning("medium", f"Broad permissions in {filename}",
                           f"{context} uses broad permissions: {permissions}")
        
        if isinstance(permissions, dict):
            high_risk_permissions = ['contents: write', 'repository-projects: write', 'security-events: write']
            for perm, level in permissions.items():
                if level == 'write' and perm in ['contents', 'repository-projects', 'security-events']:
                    self.add_warning("low", f"Write permission in {filename}",
                                   f"{context} has write permission for {perm}")
    
    def check_best_practices(self, workflow: Dict[str, Any], filename: str) -> None:
        """Check workflow against best practices."""
        # Check for timeouts
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if 'timeout-minutes' not in job_config:
                    self.add_warning("low", f"Missing timeout in {filename}",
                                   f"Job '{job_name}' should have timeout-minutes set")
        
        # Check for concurrency control
        if 'concurrency' not in workflow and filename in ['ci.yml', 'release.yml']:
            self.add_warning("low", f"Missing concurrency control in {filename}",
                           "Workflow should have concurrency control to prevent conflicts")
    
    def validate_specific_workflow(self, workflow: Dict[str, Any], filename: str) -> None:
        """Perform workflow-specific validations."""
        if filename == 'ci.yml':
            self.validate_ci_workflow(workflow, filename)
        elif filename == 'release.yml':
            self.validate_release_workflow(workflow, filename)
        elif filename.startswith('security'):
            self.validate_security_workflow(workflow, filename)
    
    def validate_ci_workflow(self, workflow: Dict[str, Any], filename: str) -> None:
        """Validate CI-specific requirements."""
        required_checks = ['test', 'lint', 'typecheck']
        
        if 'jobs' in workflow:
            job_names = set(workflow['jobs'].keys())
            for check in required_checks:
                if not any(check in job_name.lower() for job_name in job_names):
                    self.add_warning("medium", f"Missing CI check in {filename}",
                                   f"CI workflow should include {check} job")
        
        # Check for matrix builds
        has_matrix = False
        if 'jobs' in workflow:
            for job_config in workflow['jobs'].values():
                if 'strategy' in job_config and 'matrix' in job_config['strategy']:
                    has_matrix = True
                    break
        
        if not has_matrix:
            self.add_warning("low", f"No matrix builds in {filename}",
                           "CI workflow should test multiple Python versions")
    
    def validate_release_workflow(self, workflow: Dict[str, Any], filename: str) -> None:
        """Validate release workflow requirements."""
        # Should only trigger on main branch
        if 'on' in workflow and 'push' in workflow['on']:
            push_config = workflow['on']['push']
            if isinstance(push_config, dict) and 'branches' in push_config:
                branches = push_config['branches']
                if 'main' not in branches and 'master' not in branches:
                    self.add_warning("medium", f"Release workflow trigger in {filename}",
                                   "Release workflow should trigger on main/master branch")
    
    def validate_security_workflow(self, workflow: Dict[str, Any], filename: str) -> None:
        """Validate security workflow requirements."""
        # Should have CodeQL or similar security scanning
        has_security_scan = False
        if 'jobs' in workflow:
            for job_config in workflow['jobs'].values():
                if 'steps' in job_config:
                    for step in job_config['steps']:
                        if 'uses' in step:
                            action = step['uses'].lower()
                            if 'codeql' in action or 'security' in action or 'scan' in action:
                                has_security_scan = True
                                break
        
        if not has_security_scan:
            self.add_warning("medium", f"No security scanning in {filename}",
                           "Security workflow should include vulnerability scanning")
    
    def check_required_workflows(self, workflow_files: List[Path]) -> None:
        """Check that all required workflows are present."""
        present_workflows = {f.name for f in workflow_files}
        
        for required_file, description in self.required_workflows.items():
            if required_file not in present_workflows:
                self.add_warning("medium", f"Missing required workflow",
                               f"Required workflow '{required_file}' ({description}) not found")
    
    def add_issue(self, severity: str, title: str, description: str) -> None:
        """Add a validation issue."""
        self.issues.append({
            'severity': severity,
            'title': title,
            'description': description,
            'type': 'issue'
        })
    
    def add_warning(self, severity: str, title: str, description: str) -> None:
        """Add a validation warning."""
        self.warnings.append({
            'severity': severity,
            'title': title,
            'description': description,
            'type': 'warning'
        })
    
    def get_results(self) -> Dict[str, Any]:
        """Get validation results summary."""
        all_items = self.issues + self.warnings
        
        severity_counts = {
            'critical': len([i for i in all_items if i['severity'] == 'critical']),
            'high': len([i for i in all_items if i['severity'] == 'high']),
            'medium': len([i for i in all_items if i['severity'] == 'medium']),
            'low': len([i for i in all_items if i['severity'] == 'low']),
        }
        
        return {
            'summary': {
                'total_issues': len(self.issues),
                'total_warnings': len(self.warnings),
                'severity_counts': severity_counts,
                'passed': len(self.issues) == 0
            },
            'issues': self.issues,
            'warnings': self.warnings
        }
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print validation results in a readable format."""
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🔍 WORKFLOW VALIDATION RESULTS")
        print("="*60)
        
        # Summary
        if summary['passed']:
            print("✅ All validations passed!")
        else:
            print("❌ Validation failed!")
        
        print(f"\n📊 Summary:")
        print(f"   Issues: {summary['total_issues']}")
        print(f"   Warnings: {summary['total_warnings']}")
        print(f"   Critical: {summary['severity_counts']['critical']}")
        print(f"   High: {summary['severity_counts']['high']}")
        print(f"   Medium: {summary['severity_counts']['medium']}")
        print(f"   Low: {summary['severity_counts']['low']}")
        
        # Issues
        if results['issues']:
            print(f"\n❌ Issues ({len(results['issues'])}):")
            for issue in results['issues']:
                severity_icon = {
                    'critical': '🔥',
                    'high': '🚨',
                    'medium': '⚠️',
                    'low': 'ℹ️'
                }.get(issue['severity'], '❓')
                
                print(f"   {severity_icon} [{issue['severity'].upper()}] {issue['title']}")
                print(f"      {issue['description']}")
        
        # Warnings
        if results['warnings']:
            print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
            for warning in results['warnings']:
                severity_icon = {
                    'critical': '🔥',
                    'high': '🚨', 
                    'medium': '⚠️',
                    'low': 'ℹ️'
                }.get(warning['severity'], '❓')
                
                print(f"   {severity_icon} [{warning['severity'].upper()}] {warning['title']}")
                print(f"      {warning['description']}")
        
        print("\n" + "="*60)


def main():
    """Main function to run workflow validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GitHub Actions workflows")
    parser.add_argument("--workflows-dir", default="docs/github-workflows",
                        help="Directory containing workflow files")
    parser.add_argument("--json", action="store_true",
                        help="Output results in JSON format")
    parser.add_argument("--fail-on-warning", action="store_true",
                        help="Exit with error code if warnings are found")
    
    args = parser.parse_args()
    
    # Run validation
    validator = WorkflowValidator(args.workflows_dir)
    is_valid, results = validator.validate_all()
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        validator.print_results(results)
    
    # Exit with appropriate code
    has_warnings = results['summary']['total_warnings'] > 0
    if not is_valid or (args.fail_on_warning and has_warnings):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()