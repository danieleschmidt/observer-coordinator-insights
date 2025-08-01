#!/usr/bin/env python3
"""
Automated Dependency Update Script for Observer Coordinator Insights.

This script automates the process of updating project dependencies,
checking for security vulnerabilities, and creating pull requests
for updates.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import semver


logger = logging.getLogger(__name__)


class DependencyUpdater:
    """Manages automated dependency updates."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.updates_available = []
        self.security_updates = []
        self.breaking_changes = []
        self.update_summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'total_updates': 0,
            'security_updates': 0,
            'breaking_changes': 0,
            'successful_updates': 0,
            'failed_updates': 0
        }
    
    def check_for_updates(self) -> Dict[str, Any]:
        """Check for available dependency updates."""
        logger.info("🔍 Checking for dependency updates...")
        
        # Check Python dependencies
        python_updates = self.check_python_dependencies()
        
        # Check npm dependencies (if applicable)
        npm_updates = []
        if Path('package.json').exists():
            npm_updates = self.check_npm_dependencies()
        
        # Combine all updates
        all_updates = python_updates + npm_updates
        
        # Categorize updates
        self.categorize_updates(all_updates)
        
        logger.info(f"Found {len(all_updates)} dependency updates available")
        logger.info(f"Security updates: {len(self.security_updates)}")
        logger.info(f"Breaking changes: {len(self.breaking_changes)}")
        
        return {
            'total_updates': len(all_updates),
            'security_updates': len(self.security_updates),
            'breaking_changes': len(self.breaking_changes),
            'updates': all_updates
        }
    
    def check_python_dependencies(self) -> List[Dict[str, Any]]:
        """Check for Python dependency updates."""
        updates = []
        
        try:
            # Get outdated packages
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                
                for package in outdated_packages:
                    update_info = {
                        'ecosystem': 'python',
                        'name': package['name'],
                        'current_version': package['version'],
                        'latest_version': package['latest_version'],
                        'type': package.get('latest_filetype', 'wheel')
                    }
                    
                    # Check if it's a security update
                    security_info = self.check_security_advisory(package['name'], package['version'])
                    if security_info:
                        update_info['security'] = security_info
                    
                    # Analyze version change type
                    update_info['change_type'] = self.analyze_version_change(
                        package['version'], 
                        package['latest_version']
                    )
                    
                    updates.append(update_info)
            
        except Exception as e:
            logger.error(f"Failed to check Python dependencies: {e}")
        
        return updates
    
    def check_npm_dependencies(self) -> List[Dict[str, Any]]:
        """Check for npm dependency updates."""
        updates = []
        
        try:
            # Check for outdated npm packages
            result = subprocess.run(
                ['npm', 'outdated', '--json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # npm outdated returns non-zero when packages are outdated
            if result.stdout:
                try:
                    outdated_packages = json.loads(result.stdout)
                    
                    for package_name, package_info in outdated_packages.items():
                        update_info = {
                            'ecosystem': 'npm',
                            'name': package_name,
                            'current_version': package_info['current'],
                            'latest_version': package_info['latest'],
                            'wanted_version': package_info['wanted']
                        }
                        
                        # Analyze version change type
                        update_info['change_type'] = self.analyze_version_change(
                            package_info['current'],
                            package_info['latest']
                        )
                        
                        updates.append(update_info)
                        
                except json.JSONDecodeError:
                    logger.warning("Could not parse npm outdated output")
            
        except FileNotFoundError:
            logger.info("npm not found, skipping npm dependency check")
        except Exception as e:
            logger.error(f"Failed to check npm dependencies: {e}")
        
        return updates
    
    def check_security_advisory(self, package_name: str, current_version: str) -> Optional[Dict[str, Any]]:
        """Check if a package has security advisories."""
        try:
            # Use safety database to check for known vulnerabilities
            result = subprocess.run(
                [sys.executable, '-m', 'safety', 'check', '--json', '--ignore', ''],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0 and result.stdout:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    for vuln in vulnerabilities:
                        if vuln.get('package_name', '').lower() == package_name.lower():
                            return {
                                'vulnerability_id': vuln.get('vulnerability_id'),
                                'severity': vuln.get('severity', 'unknown'),
                                'description': vuln.get('advisory', ''),
                                'affected_versions': vuln.get('specs', [])
                            }
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not check security advisory for {package_name}: {e}")
            return None
    
    def analyze_version_change(self, current: str, latest: str) -> str:
        """Analyze the type of version change."""
        try:
            # Clean version strings (remove prefixes like 'v', '=', etc.)
            current_clean = current.lstrip('v=~^')
            latest_clean = latest.lstrip('v=~^')
            
            # Parse versions
            current_version = semver.VersionInfo.parse(current_clean)
            latest_version = semver.VersionInfo.parse(latest_clean)
            
            # Determine change type
            if latest_version.major > current_version.major:
                return 'major'
            elif latest_version.minor > current_version.minor:
                return 'minor'
            elif latest_version.patch > current_version.patch:
                return 'patch'
            else:
                return 'prerelease'
                
        except Exception:
            # Fallback to simple string comparison
            if current != latest:
                return 'unknown'
            return 'none'
    
    def categorize_updates(self, updates: List[Dict[str, Any]]) -> None:
        """Categorize updates by type and severity."""
        self.updates_available = updates
        self.security_updates = [u for u in updates if u.get('security')]
        self.breaking_changes = [u for u in updates if u.get('change_type') == 'major']
    
    def create_update_plan(self) -> Dict[str, List[Dict[str, Any]]]:
        """Create an update plan with prioritized batches."""
        plan = {
            'immediate': [],  # Security updates
            'safe': [],       # Patch updates
            'minor': [],      # Minor updates
            'major': []       # Major updates (breaking changes)
        }
        
        for update in self.updates_available:
            if update.get('security'):
                plan['immediate'].append(update)
            elif update.get('change_type') == 'patch':
                plan['safe'].append(update)
            elif update.get('change_type') == 'minor':
                plan['minor'].append(update)
            elif update.get('change_type') == 'major':
                plan['major'].append(update)
            else:
                plan['safe'].append(update)  # Default to safe
        
        return plan
    
    def apply_updates(self, update_plan: Dict[str, List[Dict[str, Any]]], 
                     batch: str = 'safe') -> bool:
        """Apply updates from a specific batch."""
        if batch not in update_plan or not update_plan[batch]:
            logger.info(f"No updates in batch '{batch}'")
            return True
        
        logger.info(f"Applying {len(update_plan[batch])} updates from batch '{batch}'")
        
        success = True
        for update in update_plan[batch]:
            if self.dry_run:
                logger.info(f"[DRY RUN] Would update {update['name']} from {update['current_version']} to {update['latest_version']}")
                continue
            
            try:
                if update['ecosystem'] == 'python':
                    self.update_python_package(update)
                elif update['ecosystem'] == 'npm':
                    self.update_npm_package(update)
                
                self.update_summary['successful_updates'] += 1
                logger.info(f"✅ Updated {update['name']} to {update['latest_version']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to update {update['name']}: {e}")
                self.update_summary['failed_updates'] += 1
                success = False
        
        return success
    
    def update_python_package(self, update: Dict[str, Any]) -> None:
        """Update a Python package."""
        package_name = update['name']
        target_version = update['latest_version']
        
        # Update the package
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', f"{package_name}=={target_version}"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise Exception(f"pip install failed: {result.stderr}")
        
        # Update requirements files
        self.update_requirements_file(package_name, target_version)
    
    def update_npm_package(self, update: Dict[str, Any]) -> None:
        """Update an npm package."""
        package_name = update['name']
        target_version = update['latest_version']
        
        # Update the package
        result = subprocess.run(
            ['npm', 'install', f"{package_name}@{target_version}"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            raise Exception(f"npm install failed: {result.stderr}")
    
    def update_requirements_file(self, package_name: str, version: str) -> None:
        """Update requirements.txt with new package version."""
        requirements_files = ['requirements.txt', 'requirements-dev.txt']
        
        for req_file in requirements_files:
            if not Path(req_file).exists():
                continue
            
            try:
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                
                updated = False
                for i, line in enumerate(lines):
                    line_clean = line.strip().lower()
                    if line_clean.startswith(package_name.lower()):
                        # Update the version
                        lines[i] = f"{package_name}=={version}\n"
                        updated = True
                        break
                
                if updated:
                    with open(req_file, 'w') as f:
                        f.writelines(lines)
                    logger.debug(f"Updated {package_name} in {req_file}")
                        
            except Exception as e:
                logger.warning(f"Could not update {req_file}: {e}")
    
    def run_tests_after_update(self) -> bool:
        """Run tests to ensure updates don't break functionality."""
        if self.dry_run:
            logger.info("[DRY RUN] Would run tests after updates")
            return True
        
        logger.info("🧪 Running tests after updates...")
        
        try:
            # Run test suite
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode == 0:
                logger.info("✅ All tests passed after updates")
                return True
            else:
                logger.error("❌ Tests failed after updates")
                logger.error(result.stdout)
                logger.error(result.stderr)
                return False
                
        except Exception as e:
            logger.error(f"Failed to run tests: {e}")
            return False
    
    def create_commit(self, batch: str, updates: List[Dict[str, Any]]) -> bool:
        """Create a git commit for the updates."""
        if self.dry_run:
            logger.info(f"[DRY RUN] Would create commit for {batch} updates")
            return True
        
        try:
            # Stage changes
            subprocess.run(['git', 'add', '.'], check=True, timeout=30)
            
            # Create commit message
            commit_msg = self.generate_commit_message(batch, updates)
            
            # Commit changes
            subprocess.run(
                ['git', 'commit', '-m', commit_msg],
                check=True,
                timeout=30
            )
            
            logger.info(f"✅ Created commit for {batch} updates")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create commit: {e}")
            return False
    
    def generate_commit_message(self, batch: str, updates: List[Dict[str, Any]]) -> str:
        """Generate a commit message for the updates."""
        if batch == 'immediate':
            title = "security: update dependencies with security vulnerabilities"
        elif batch == 'safe':
            title = "deps: update patch-level dependencies"
        elif batch == 'minor':
            title = "deps: update minor version dependencies"
        elif batch == 'major':
            title = "deps: update major version dependencies (breaking changes)"
        else:
            title = f"deps: update {batch} dependencies"
        
        # Add details about specific updates
        details = []
        for update in updates[:5]:  # Limit to first 5 for readability
            details.append(f"- {update['name']}: {update['current_version']} → {update['latest_version']}")
        
        if len(updates) > 5:
            details.append(f"- ... and {len(updates) - 5} more")
        
        commit_msg = title
        if details:
            commit_msg += "\n\n" + "\n".join(details)
        
        # Add automation signature
        commit_msg += "\n\n🤖 Generated with [Claude Code](https://claude.ai/code)\n\nCo-Authored-By: Claude <noreply@anthropic.com>"
        
        return commit_msg
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of the update process."""
        report = []
        report.append("📦 DEPENDENCY UPDATE SUMMARY")
        report.append("=" * 50)
        report.append(f"Timestamp: {self.update_summary['timestamp']}")
        report.append(f"Total Updates Available: {len(self.updates_available)}")
        report.append(f"Security Updates: {len(self.security_updates)}")
        report.append(f"Breaking Changes: {len(self.breaking_changes)}")
        report.append(f"Successful Updates: {self.update_summary['successful_updates']}")
        report.append(f"Failed Updates: {self.update_summary['failed_updates']}")
        report.append("")
        
        # Security updates section
        if self.security_updates:
            report.append("🔒 SECURITY UPDATES")
            for update in self.security_updates:
                security = update.get('security', {})
                report.append(f"  • {update['name']}: {update['current_version']} → {update['latest_version']}")
                report.append(f"    Severity: {security.get('severity', 'unknown')}")
                report.append(f"    Advisory: {security.get('description', 'N/A')[:100]}...")
                report.append("")
        
        # Breaking changes section
        if self.breaking_changes:
            report.append("⚠️  BREAKING CHANGES")
            for update in self.breaking_changes:
                report.append(f"  • {update['name']}: {update['current_version']} → {update['latest_version']}")
            report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run dependency updates."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated dependency updates")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without making changes")
    parser.add_argument("--batch", choices=['immediate', 'safe', 'minor', 'major', 'all'],
                        default='safe',
                        help="Which batch of updates to apply")
    parser.add_argument("--skip-tests", action="store_true",
                        help="Skip running tests after updates")
    parser.add_argument("--auto-commit", action="store_true",
                        help="Automatically commit successful updates")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create updater
    updater = DependencyUpdater(dry_run=args.dry_run)
    
    try:
        # Check for updates
        update_info = updater.check_for_updates()
        
        if update_info['total_updates'] == 0:
            print("✅ All dependencies are up to date!")
            return 0
        
        # Create update plan
        update_plan = updater.create_update_plan()
        
        # Print summary
        summary = updater.generate_summary_report()
        print(summary)
        
        # Apply updates
        if args.batch == 'all':
            batches = ['immediate', 'safe', 'minor', 'major']
        else:
            batches = [args.batch]
        
        overall_success = True
        for batch in batches:
            if update_plan.get(batch):
                success = updater.apply_updates(update_plan, batch)
                overall_success = overall_success and success
                
                if success and not args.skip_tests:
                    test_success = updater.run_tests_after_update()
                    if not test_success:
                        logger.error(f"Tests failed after {batch} updates")
                        overall_success = False
                        break
                
                if success and args.auto_commit and not args.dry_run:
                    updater.create_commit(batch, update_plan[batch])
        
        if overall_success:
            print("\n✅ Dependency updates completed successfully!")
        else:
            print("\n❌ Some dependency updates failed!")
            return 1
        
    except KeyboardInterrupt:
        print("\n❌ Dependency update cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Dependency update failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())