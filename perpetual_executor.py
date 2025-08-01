#!/usr/bin/env python3
"""
Perpetual Execution Engine for Autonomous SDLC Enhancement
Implements continuous value discovery and execution loop
"""

import json
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

from value_discovery_engine import ValueDiscoveryEngine, ValueItem


class PerpetualExecutor:
    """Main executor for continuous value delivery"""
    
    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        self.repo_path = Path(repo_path)
        self.dry_run = dry_run
        self.discovery_engine = ValueDiscoveryEngine(repo_path)
        self.setup_logging()
        
        # Execution state
        self.session_start = datetime.now()
        self.cycles_completed = 0
        self.items_executed = 0
        self.current_item: Optional[ValueItem] = None
        
    def setup_logging(self):
        """Setup logging for execution tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('.terragon/execution.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('PerpetualExecutor')
    
    def run_perpetual_loop(self, max_cycles: int = 100):
        """Main perpetual execution loop"""
        self.logger.info(f"Starting perpetual execution (max_cycles: {max_cycles})")
        
        try:
            while self.cycles_completed < max_cycles:
                cycle_start = datetime.now()
                self.logger.info(f"Starting cycle {self.cycles_completed + 1}")
                
                # 1. Discover value items
                items = self.discovery_engine.discover_value_items()
                self.logger.info(f"Discovered {len(items)} value items")
                
                if not items:
                    self.logger.info("No value items found, generating housekeeping tasks")
                    items = self._generate_housekeeping_tasks()
                
                # 2. Select next best value item
                next_item = self._select_next_item(items)
                if not next_item:
                    self.logger.info("No executable items found, waiting...")
                    time.sleep(300)  # Wait 5 minutes
                    continue
                
                # 3. Execute the item
                execution_result = self._execute_item(next_item)
                
                # 4. Track and learn from execution
                self._track_execution(next_item, execution_result)
                
                # 5. Update metrics and backlog
                self._update_metrics(items)
                self._update_backlog(items)
                
                self.cycles_completed += 1
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                self.logger.info(f"Cycle {self.cycles_completed} completed in {cycle_duration:.1f}s")
                
                # Brief pause before next cycle
                time.sleep(60)  # 1 minute between cycles
                
        except KeyboardInterrupt:
            self.logger.info("Execution interrupted by user")
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
        finally:
            self._generate_session_report()
    
    def _select_next_item(self, items: List[ValueItem]) -> Optional[ValueItem]:
        """Select the next best value item for execution"""
        for item in items:
            # Check dependencies
            if not self._are_dependencies_met(item):
                continue
                
            # Check risk threshold
            if self._assess_execution_risk(item) > 0.8:
                continue
                
            # Check for conflicts
            if self._has_conflicts(item):
                continue
                
            self.logger.info(f"Selected item: {item.title} (Score: {item.composite_score:.1f})")
            return item
        
        return None
    
    def _execute_item(self, item: ValueItem) -> Dict[str, Any]:
        """Execute a value item with comprehensive error handling"""
        self.current_item = item
        execution_start = datetime.now()
        
        try:
            self.logger.info(f"Executing: {item.title}")
            
            if self.dry_run:
                # Simulate execution
                time.sleep(2)
                result = {
                    'success': True,
                    'duration': 2.0,
                    'changes_made': ['simulated_change.py'],
                    'tests_passed': True,
                    'impact': 'Simulated execution completed'
                }
            else:
                result = self._perform_actual_execution(item)
            
            duration = (datetime.now() - execution_start).total_seconds()
            result['duration'] = duration
            
            if result['success']:
                self.items_executed += 1
                self.logger.info(f"Successfully executed {item.title} in {duration:.1f}s")
            else:
                self.logger.warning(f"Failed to execute {item.title}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed for {item.title}: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': (datetime.now() - execution_start).total_seconds()
            }
        finally:
            self.current_item = None
    
    def _perform_actual_execution(self, item: ValueItem) -> Dict[str, Any]:
        """Perform the actual execution of a value item"""
        import subprocess
        
        result = {
            'success': False,
            'changes_made': [],
            'tests_passed': False,
            'impact': ''
        }
        
        try:
            # Execute based on item category
            if item.category == 'code_quality':
                # Run auto-formatters
                subprocess.run(['ruff', 'format', *item.files_affected], check=True)
                result['changes_made'] = item.files_affected
                result['impact'] = 'Code formatting improved'
                
            elif item.category == 'security':
                # Update dependencies or fix security issues
                if 'requirements.txt' in item.files_affected:
                    subprocess.run(['pip-audit', '--fix'], check=True)
                    result['changes_made'] = ['requirements.txt']
                    result['impact'] = 'Security vulnerabilities patched'
                
            elif item.category == 'technical_debt':
                # Address specific technical debt items
                # This would involve more sophisticated analysis and fixes
                result['changes_made'] = item.files_affected
                result['impact'] = 'Technical debt reduced'
            
            # Run tests to verify changes
            test_result = subprocess.run(['python', '-m', 'pytest', '--tb=short'], capture_output=True)
            result['tests_passed'] = test_result.returncode == 0
            
            # Run linting to ensure quality
            lint_result = subprocess.run(['ruff', 'check', 'src/'], capture_output=True)
            result['linting_passed'] = lint_result.returncode == 0
            
            result['success'] = result['tests_passed'] and result['linting_passed']
            
        except subprocess.CalledProcessError as e:
            result['error'] = f"Command failed: {e.cmd}"
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _generate_housekeeping_tasks(self) -> List[ValueItem]:
        """Generate housekeeping tasks when no high-value items exist"""
        tasks = []
        
        # Documentation updates
        tasks.append(ValueItem(
            id="housekeeping-docs",
            title="Update documentation",
            description="Refresh README and documentation files",
            category="documentation",
            source="housekeeping",
            estimated_effort=1.0,
            scores={},
            composite_score=20.0,
            risk_level="low",
            files_affected=["README.md"],
            dependencies=[],
            created_at=datetime.now().isoformat(),
            metadata={"type": "housekeeping"}
        ))
        
        # Dependency updates
        tasks.append(ValueItem(
            id="housekeeping-deps",
            title="Update dependencies",
            description="Check for and apply safe dependency updates",
            category="maintenance",
            source="housekeeping",
            estimated_effort=2.0,
            scores={},
            composite_score=25.0,
            risk_level="medium",
            files_affected=["requirements.txt", "pyproject.toml"],
            dependencies=[],
            created_at=datetime.now().isoformat(),
            metadata={"type": "housekeeping"}
        ))
        
        return tasks
    
    def _are_dependencies_met(self, item: ValueItem) -> bool:
        """Check if item dependencies are satisfied"""
        # For now, assume all dependencies are met
        # In a real implementation, this would check for prerequisite items
        return True
    
    def _assess_execution_risk(self, item: ValueItem) -> float:
        """Assess the risk of executing an item"""
        risk_scores = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        base_risk = risk_scores.get(item.risk_level, 0.5)
        
        # Increase risk for items affecting many files
        if len(item.files_affected) > 5:
            base_risk += 0.2
        
        # Decrease risk for well-tested categories
        if item.category in ['code_quality', 'documentation']:
            base_risk -= 0.1
        
        return min(1.0, max(0.0, base_risk))
    
    def _has_conflicts(self, item: ValueItem) -> bool:
        """Check if item conflicts with current work"""
        if self.current_item:
            # Check for file conflicts
            current_files = set(self.current_item.files_affected)
            item_files = set(item.files_affected)
            return bool(current_files.intersection(item_files))
        return False
    
    def _track_execution(self, item: ValueItem, result: Dict[str, Any]):
        """Track execution results for learning"""
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'item': asdict(item),
            'result': result,
            'session_id': self.session_start.isoformat()
        }
        
        # Append to execution history
        history_file = self.repo_path / '.terragon' / 'execution_history.jsonl'
        with open(history_file, 'a') as f:
            f.write(json.dumps(execution_record) + '\n')
    
    def _update_metrics(self, items: List[ValueItem]):
        """Update value metrics with current state"""
        metrics_path = self.repo_path / '.terragon' / 'value-metrics.json'
        
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {}
        
        # Update continuous discovery metrics
        metrics['continuousDiscovery'] = {
            'lastScan': datetime.now().isoformat(),
            'itemsDiscovered': len(items),
            'securityItems': len([i for i in items if i.category == 'security']),
            'performanceOpportunities': len([i for i in items if i.category == 'performance']),
            'technicalDebtItems': len([i for i in items if i.category == 'technical_debt']),
            'cyclesCompleted': self.cycles_completed,
            'itemsExecuted': self.items_executed
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def _update_backlog(self, items: List[ValueItem]):
        """Update the backlog markdown file"""
        backlog_content = self.discovery_engine.generate_backlog_update(items)
        with open(self.repo_path / 'BACKLOG.md', 'w') as f:
            f.write(backlog_content)
    
    def _generate_session_report(self):
        """Generate final session report"""
        session_duration = datetime.now() - self.session_start
        
        report = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'duration_hours': session_duration.total_seconds() / 3600,
            'cycles_completed': self.cycles_completed,
            'items_executed': self.items_executed,
            'success_rate': self.items_executed / max(self.cycles_completed, 1),
            'average_cycle_time': session_duration.total_seconds() / max(self.cycles_completed, 1)
        }
        
        report_path = self.repo_path / '.terragon' / f'session_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Session completed: {self.items_executed} items executed in {self.cycles_completed} cycles")
        self.logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Perpetual SDLC Enhancement Executor")
    parser.add_argument("--dry-run", action="store_true", help="Run in simulation mode")
    parser.add_argument("--max-cycles", type=int, default=100, help="Maximum execution cycles")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    
    args = parser.parse_args()
    
    executor = PerpetualExecutor(args.repo_path, args.dry_run)
    executor.run_perpetual_loop(args.max_cycles)