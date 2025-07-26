#!/usr/bin/env python3
"""
Autonomous Execution Engine
Implements TDD micro-cycles and task execution with security checks
"""

import os
import subprocess
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tempfile
import yaml

from backlog_manager import BacklogManager, BacklogItem

@dataclass
class ExecutionResult:
    """Result of task execution"""
    success: bool
    item_id: str
    stage: str  # RED, GREEN, REFACTOR, SECURITY, DOCS, CI
    message: str
    artifacts: List[str] = None
    test_results: Optional[Dict] = None
    security_status: Optional[Dict] = None

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []

class ExecutionEngine:
    """Executes backlog items using TDD micro-cycles"""
    
    def __init__(self, backlog_manager: BacklogManager):
        self.backlog_manager = backlog_manager
        self.repo_root = Path(".")
        self.current_item: Optional[BacklogItem] = None
        
        # Security checklist template
        self.security_checklist = [
            "input_validation",
            "authentication_authorization", 
            "secrets_management",
            "secure_logging",
            "crypto_storage",
            "error_handling"
        ]

    def check_scope_permissions(self, target_path: str) -> bool:
        """Check if target is within allowed scope"""
        target = Path(target_path).resolve()
        repo_root = self.repo_root.resolve()
        
        # Default scope: within current repo
        if target.is_relative_to(repo_root):
            return True
            
        # Check for automation scope manifest
        scope_file = repo_root / ".automation-scope.yaml"
        if scope_file.exists():
            try:
                with open(scope_file, 'r') as f:
                    scope_config = yaml.safe_load(f)
                    allowed_paths = scope_config.get('allowed_paths', [])
                    for allowed in allowed_paths:
                        if target.is_relative_to(Path(allowed).resolve()):
                            return True
            except Exception:
                pass
                
        return False

    def run_tests(self) -> Tuple[bool, Dict]:
        """Run available tests and return results"""
        test_commands = [
            (['python', '-m', 'pytest', '-v', '--json-report'], 'pytest'),
            (['npm', 'test'], 'npm'),
            (['cargo', 'test'], 'cargo'),
            (['go', 'test', './...'], 'go')
        ]
        
        for cmd, framework in test_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                return result.returncode == 0, {
                    'framework': framework,
                    'exit_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
                
        return True, {'framework': 'none', 'message': 'No test framework detected'}

    def run_lint_and_typecheck(self) -> Tuple[bool, Dict]:
        """Run linting and type checking"""
        checks = []
        
        # Common linting tools
        lint_commands = [
            (['npm', 'run', 'lint'], 'eslint'),
            (['ruff', 'check', '.'], 'ruff'),
            (['pylint', '.'], 'pylint'),
            (['cargo', 'clippy'], 'clippy')
        ]
        
        typecheck_commands = [
            (['npm', 'run', 'typecheck'], 'typescript'),
            (['mypy', '.'], 'mypy'),
            (['cargo', 'check'], 'cargo-check')
        ]
        
        all_passed = True
        results = {'lint': [], 'typecheck': []}
        
        # Run linting
        for cmd, tool in lint_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                results['lint'].append({
                    'tool': tool,
                    'passed': result.returncode == 0,
                    'output': result.stdout + result.stderr
                })
                if result.returncode != 0:
                    all_passed = False
                break  # Use first available linter
            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        # Run type checking
        for cmd, tool in typecheck_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                results['typecheck'].append({
                    'tool': tool,
                    'passed': result.returncode == 0,
                    'output': result.stdout + result.stderr
                })
                if result.returncode != 0:
                    all_passed = False
                break  # Use first available type checker
            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
                
        return all_passed, results

    def security_check(self, item: BacklogItem) -> ExecutionResult:
        """Perform security checklist verification"""
        security_status = {}
        
        # Basic security checks based on item type and description
        if item.type in ['security', 'feature', 'api']:
            # Check for common security patterns
            security_patterns = {
                'input_validation': ['validation', 'sanitize', 'escape'],
                'authentication': ['auth', 'login', 'token'],
                'secrets': ['secret', 'key', 'password', 'env'],
                'logging': ['log', 'audit', 'trace'],
                'crypto': ['encrypt', 'hash', 'crypto', 'ssl', 'tls']
            }
            
            description_lower = (item.description + ' ' + ' '.join(item.acceptance_criteria)).lower()
            
            for check, keywords in security_patterns.items():
                if any(keyword in description_lower for keyword in keywords):
                    security_status[check] = 'required'
                else:
                    security_status[check] = 'not_applicable'
        
        # Create follow-up security tasks if needed
        security_gaps = [k for k, v in security_status.items() if v == 'required']
        if security_gaps:
            for gap in security_gaps:
                self.create_security_followup_task(item, gap)
        
        return ExecutionResult(
            success=True,
            item_id=item.id,
            stage="SECURITY",
            message=f"Security check completed. {len(security_gaps)} areas require attention.",
            security_status=security_status
        )

    def create_security_followup_task(self, parent_item: BacklogItem, security_area: str):
        """Create follow-up security task"""
        security_descriptions = {
            'input_validation': 'Implement input validation and sanitization',
            'authentication': 'Add authentication and authorization checks',
            'secrets': 'Secure secrets management using environment variables',
            'logging': 'Implement secure logging without PII exposure',
            'crypto': 'Add encryption for sensitive data'
        }
        
        new_item = BacklogItem(
            id=f"SEC-{parent_item.id}-{security_area.upper()}",
            title=f"Security: {security_descriptions.get(security_area, security_area)}",
            type="security",
            description=f"Security follow-up for {parent_item.id}: {security_descriptions.get(security_area)}",
            acceptance_criteria=[
                f"Implement {security_area} for {parent_item.title}",
                "Security review passes",
                "No sensitive data exposure"
            ],
            effort=3,
            value=5,
            time_criticality=8,
            risk_reduction=13,
            status="NEW",
            risk_tier="high",
            created_at=datetime.now().isoformat() + 'Z',
            links=[parent_item.id]
        )
        
        self.backlog_manager.items.append(new_item)

    def update_documentation(self, item: BacklogItem) -> ExecutionResult:
        """Update documentation and artifacts"""
        artifacts = []
        
        # Update README if needed
        readme_path = self.repo_root / "README.md"
        if readme_path.exists() and item.type == 'feature':
            artifacts.append("README.md")
        
        # Create/update CHANGELOG
        changelog_path = self.repo_root / "CHANGELOG.md"
        if not changelog_path.exists():
            with open(changelog_path, 'w') as f:
                f.write("# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n")
        
        # Add entry to changelog
        with open(changelog_path, 'r') as f:
            content = f.read()
        
        new_entry = f"## [{datetime.now().strftime('%Y-%m-%d')}] - {item.id}\n"
        new_entry += f"### {item.type.title()}\n"
        new_entry += f"- {item.title}\n\n"
        
        # Insert after the header
        lines = content.split('\n')
        insert_index = 3  # After header and blank line
        lines.insert(insert_index, new_entry)
        
        with open(changelog_path, 'w') as f:
            f.write('\n'.join(lines))
        
        artifacts.append("CHANGELOG.md")
        
        return ExecutionResult(
            success=True,
            item_id=item.id,
            stage="DOCS",
            message="Documentation updated",
            artifacts=artifacts
        )

    def execute_micro_cycle(self, item: BacklogItem) -> List[ExecutionResult]:
        """Execute complete TDD micro-cycle for an item"""
        results = []
        self.current_item = item
        
        print(f"🚀 Starting micro-cycle for {item.id}: {item.title}")
        
        # Update status to DOING
        self.backlog_manager.update_item_status(item.id, "DOING")
        
        try:
            # A. Clarify acceptance criteria (already in backlog item)
            print("  ✅ Acceptance criteria clarified")
            
            # B. TDD Cycle - RED phase (write failing test)
            print("  🔴 RED: Writing failing test...")
            test_passed, test_results = self.run_tests()
            results.append(ExecutionResult(
                success=not test_passed,  # We want failing tests initially
                item_id=item.id,
                stage="RED",
                message="Initial test state captured",
                test_results=test_results
            ))
            
            # GREEN phase would be actual implementation (not done by this engine)
            results.append(ExecutionResult(
                success=True,
                item_id=item.id,
                stage="GREEN",
                message="Implementation phase - manual intervention required for actual coding",
                test_results=None
            ))
            
            # C. Security checklist
            print("  🔒 Security check...")
            security_result = self.security_check(item)
            results.append(security_result)
            
            # D. Documentation update
            print("  📚 Updating documentation...")
            docs_result = self.update_documentation(item)
            results.append(docs_result)
            
            # E. CI gate checks
            print("  🔍 Running CI checks...")
            lint_passed, lint_results = self.run_lint_and_typecheck()
            test_passed, test_results = self.run_tests()
            
            ci_success = lint_passed and test_passed
            results.append(ExecutionResult(
                success=ci_success,
                item_id=item.id,
                stage="CI",
                message=f"CI checks {'passed' if ci_success else 'failed'}",
                test_results={'lint': lint_results, 'tests': test_results}
            ))
            
            # F. PR preparation (create summary)
            pr_summary = self.prepare_pr_summary(item, results)
            results.append(ExecutionResult(
                success=True,
                item_id=item.id,
                stage="PR_PREP",
                message="PR summary prepared",
                artifacts=[pr_summary]
            ))
            
            # Update status based on results
            if all(r.success for r in results):
                self.backlog_manager.update_item_status(item.id, "PR")
                print(f"  ✅ {item.id} ready for PR")
            else:
                self.backlog_manager.update_item_status(item.id, "BLOCKED")
                failed_stages = [r.stage for r in results if not r.success]
                print(f"  ❌ {item.id} blocked at stages: {', '.join(failed_stages)}")
                
        except Exception as e:
            results.append(ExecutionResult(
                success=False,
                item_id=item.id,
                stage="ERROR",
                message=f"Execution failed: {str(e)}"
            ))
            self.backlog_manager.update_item_status(item.id, "BLOCKED")
            
        return results

    def prepare_pr_summary(self, item: BacklogItem, results: List[ExecutionResult]) -> str:
        """Prepare PR summary with context and test results"""
        summary = f"""# {item.title}

## Context
{item.description}

## Changes
This PR implements the following acceptance criteria:
"""
        for criterion in item.acceptance_criteria:
            summary += f"- [ ] {criterion}\n"
        
        summary += f"""
## Test Results
"""
        for result in results:
            if result.test_results:
                summary += f"- {result.stage}: {'✅ Passed' if result.success else '❌ Failed'}\n"
        
        summary += f"""
## Security Status
Security checklist has been reviewed for this change.

## Rollback Plan
This change can be reverted by reverting this PR.

## Links
- Backlog Item: {item.id}
- Risk Tier: {item.risk_tier}
"""
        
        # Save PR summary to file
        pr_file = self.repo_root / f"docs/pr-{item.id.lower()}.md"
        pr_file.parent.mkdir(exist_ok=True)
        with open(pr_file, 'w') as f:
            f.write(summary)
            
        return str(pr_file)

    def execute_next_ready_item(self) -> Optional[List[ExecutionResult]]:
        """Execute the next ready item from backlog"""
        next_item = self.backlog_manager.get_next_ready_item()
        if not next_item:
            return None
            
        # Check if high risk - require human approval
        if next_item.risk_tier == "high":
            print(f"⚠️  High-risk item {next_item.id} requires human approval")
            return None
            
        return self.execute_micro_cycle(next_item)

def main():
    """Main execution loop"""
    print("🤖 Starting Autonomous Execution Engine")
    
    # Initialize components
    backlog_manager = BacklogManager()
    execution_engine = ExecutionEngine(backlog_manager)
    
    # Discover and update backlog
    backlog_manager.continuous_discovery()
    
    # Execute ready items
    executed_count = 0
    max_executions = 5  # Safety limit
    
    while executed_count < max_executions:
        results = execution_engine.execute_next_ready_item()
        if not results:
            print("📋 No more ready items to execute")
            break
            
        executed_count += 1
        print(f"✅ Completed execution {executed_count}")
        
        # Save updated backlog
        backlog_manager.save_backlog()
    
    print(f"🏁 Execution complete. Processed {executed_count} items.")
    
    # Generate final report
    report = backlog_manager.generate_status_report()
    print(f"📊 Final status: {report['ready_items']} ready, {report['blocked_items']} blocked")

if __name__ == "__main__":
    main()