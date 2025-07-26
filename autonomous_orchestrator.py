#!/usr/bin/env python3
"""
Autonomous Senior Coding Assistant Orchestrator
Main execution loop implementing the "DO UNTIL DONE" philosophy
"""

import os
import sys
import time
import signal
import yaml
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess

from backlog_manager import BacklogManager, BacklogItem
from execution_engine import ExecutionEngine, ExecutionResult
from metrics_reporter import MetricsReporter

class AutonomousOrchestrator:
    """Main orchestrator implementing the autonomous execution loop"""
    
    def __init__(self, config_file: str = ".automation-scope.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
        self.backlog_manager = BacklogManager()
        self.execution_engine = ExecutionEngine(self.backlog_manager)
        self.metrics_reporter = MetricsReporter(self.backlog_manager)
        
        # Execution state
        self.session_start = datetime.now()
        self.operations_count = 0
        self.max_operations = self.config.get('limits', {}).get('max_operations_per_session', 100)
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def load_config(self) -> Dict[str, Any]:
        """Load automation scope configuration"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f)
        return {
            'allowed_paths': [],
            'require_approval': ['cross_repo_changes', 'ci_config_changes'],
            'limits': {'max_operations_per_session': 100}
        }

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False

    def sync_repo_and_ci(self) -> bool:
        """Sync repository state and check CI status"""
        print("🔄 Syncing repository state...")
        
        try:
            # Fetch latest changes
            result = subprocess.run(['git', 'fetch'], capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️  Git fetch failed: {result.stderr}")
                return False
            
            # Check if we're behind
            result = subprocess.run(['git', 'status', '-uno'], capture_output=True, text=True)
            if 'behind' in result.stdout:
                print("📥 Repository is behind remote. Consider pulling changes.")
                return False
            
            # Check for uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected. Repository not clean.")
                return False
                
            print("✅ Repository state synchronized")
            return True
            
        except subprocess.SubprocessError as e:
            print(f"❌ Repository sync failed: {e}")
            return False

    def discover_new_tasks(self):
        """Run continuous discovery to find new backlog items"""
        print("🔍 Discovering new tasks...")
        initial_count = len(self.backlog_manager.items)
        
        # Run discovery
        self.backlog_manager.continuous_discovery()
        
        new_count = len(self.backlog_manager.items) - initial_count
        if new_count > 0:
            print(f"  📝 Discovered {new_count} new items")
        else:
            print("  ✅ No new items discovered")

    def score_and_sort_backlog(self):
        """Update WSJF scores and sort backlog"""
        print("📊 Updating WSJF scores...")
        
        # Update aging for all items
        now = datetime.now()
        for item in self.backlog_manager.items:
            created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
            days_old = (now.astimezone() - created).days
            item.apply_aging(days_old)
        
        # Sort by WSJF score
        self.backlog_manager.items.sort(key=lambda x: x.wsjf_score, reverse=True)
        print(f"  ✅ Sorted {len(self.backlog_manager.items)} items by WSJF score")

    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get next ready item within scope"""
        ready_items = [item for item in self.backlog_manager.get_prioritized_backlog() 
                      if item.status == "READY"]
        
        if not ready_items:
            return None
            
        # Check scope for the highest priority item
        next_item = ready_items[0]
        
        # High-risk items require approval
        if next_item.risk_tier == "high":
            return self.escalate_for_human_approval(next_item)
        
        # Check if item requires approval based on type
        if next_item.type in self.config.get('require_approval', []):
            return self.escalate_for_human_approval(next_item)
            
        return next_item

    def escalate_for_human_approval(self, item: BacklogItem) -> Optional[BacklogItem]:
        """Escalate high-risk item for human approval"""
        print(f"⚠️  Item {item.id} requires human approval:")
        print(f"   Title: {item.title}")
        print(f"   Risk: {item.risk_tier}")
        print(f"   Type: {item.type}")
        print(f"   Description: {item.description[:100]}...")
        
        # In a real implementation, this would trigger a webhook or notification
        webhook_url = self.config.get('notifications', {}).get('approval_required_webhook')
        if webhook_url:
            self.send_approval_notification(webhook_url, item)
        
        # Mark as blocked pending approval
        self.backlog_manager.update_item_status(item.id, "BLOCKED")
        return None

    def send_approval_notification(self, webhook_url: str, item: BacklogItem):
        """Send notification for approval request"""
        # This would send actual webhook in production
        print(f"📞 Approval notification sent for {item.id}")

    def is_item_high_risk_or_ambiguous(self, item: BacklogItem) -> bool:
        """Check if item is high risk or has ambiguous requirements"""
        # High risk tiers
        if item.risk_tier == "high":
            return True
            
        # Check for ambiguous acceptance criteria
        if not item.acceptance_criteria or len(item.acceptance_criteria) < 2:
            return True
            
        # Check for large effort estimates (may need breaking down)
        if item.effort >= 13:
            return True
            
        # Check for security/auth/crypto related items
        sensitive_keywords = ['auth', 'security', 'crypto', 'password', 'token', 'migration']
        description_lower = item.description.lower()
        if any(keyword in description_lower for keyword in sensitive_keywords):
            return True
            
        return False

    def execute_micro_cycle(self, item: BacklogItem) -> List[ExecutionResult]:
        """Execute micro-cycle for an item"""
        if self.operations_count >= self.max_operations:
            print(f"⚠️  Operation limit ({self.max_operations}) reached")
            return []
            
        print(f"⚡ Executing micro-cycle for {item.id}")
        self.operations_count += 1
        
        # Execute the item
        results = self.execution_engine.execute_micro_cycle(item)
        
        return results

    def merge_and_log(self, item: BacklogItem, results: List[ExecutionResult]):
        """Merge changes and log completion"""
        if all(r.success for r in results):
            print(f"✅ {item.id} completed successfully")
            self.backlog_manager.update_item_status(item.id, "DONE")
            
            # Log completion
            completion_log = {
                'timestamp': datetime.now().isoformat() + 'Z',
                'item_id': item.id,
                'title': item.title,
                'cycle_time_hours': (datetime.now() - 
                                   datetime.fromisoformat(item.created_at.replace('Z', '+00:00')).astimezone()).total_seconds() / 3600,
                'results': [{'stage': r.stage, 'success': r.success, 'message': r.message} for r in results]
            }
            
            # Save completion log
            logs_dir = Path("docs/logs")
            logs_dir.mkdir(exist_ok=True)
            log_file = logs_dir / f"completion_{item.id.lower()}.json"
            
            import json
            with open(log_file, 'w') as f:
                json.dump(completion_log, f, indent=2)
                
        else:
            failed_stages = [r.stage for r in results if not r.success]
            print(f"❌ {item.id} failed at stages: {', '.join(failed_stages)}")

    def update_metrics(self):
        """Update and save metrics"""
        report = self.metrics_reporter.generate_comprehensive_report()
        self.metrics_reporter.save_json_report(report)
        
        # Print summary
        ready_count = report['backlog_size_by_status'].get('READY', 0)
        blocked_count = report['backlog_size_by_status'].get('BLOCKED', 0)
        print(f"📊 Metrics updated: {ready_count} ready, {blocked_count} blocked")

    def has_actionable_items(self) -> bool:
        """Check if there are actionable items in backlog"""
        ready_items = [item for item in self.backlog_manager.items if item.status == "READY"]
        return len(ready_items) > 0

    def run_macro_execution_loop(self):
        """Main execution loop - "DO UNTIL DONE" """
        print("🤖 Starting Autonomous Senior Coding Assistant")
        print(f"📋 Session limit: {self.max_operations} operations")
        print("=" * 60)
        
        iteration = 0
        
        while self.running and self.has_actionable_items() and self.operations_count < self.max_operations:
            iteration += 1
            print(f"\n🔄 Iteration {iteration} (Operations: {self.operations_count}/{self.max_operations})")
            
            try:
                # 1. Sync repository and CI
                if not self.sync_repo_and_ci():
                    print("⚠️  Repository sync issues - pausing execution")
                    break
                
                # 2. Discover new tasks
                self.discover_new_tasks()
                
                # 3. Score and sort backlog  
                self.score_and_sort_backlog()
                
                # 4. Get next ready item
                task = self.get_next_ready_item()
                if not task:
                    print("📋 No feasible tasks available")
                    break
                
                # 5. Check if high risk or ambiguous
                if self.is_item_high_risk_or_ambiguous(task):
                    print(f"🚨 Escalating {task.id} for human review")
                    self.escalate_for_human_approval(task)
                    continue
                
                # 6. Execute micro-cycle
                results = self.execute_micro_cycle(task)
                if not results:
                    break
                
                # 7. Merge and log
                self.merge_and_log(task, results)
                
                # 8. Update metrics
                self.update_metrics()
                
                # Save backlog state
                self.backlog_manager.save_backlog()
                
                # Brief pause between iterations
                time.sleep(1)
                
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error in execution loop: {e}")
                # Continue with next iteration unless it's a fatal error
                continue
        
        # Final reporting
        self.generate_final_report()

    def generate_final_report(self):
        """Generate final execution report"""
        session_duration = datetime.now() - self.session_start
        
        print("\n" + "=" * 60)
        print("🏁 AUTONOMOUS EXECUTION COMPLETE")
        print("=" * 60)
        
        # Generate comprehensive final report
        final_report = self.metrics_reporter.generate_comprehensive_report()
        
        # Add session metrics
        final_report['session_metrics'] = {
            'start_time': self.session_start.isoformat() + 'Z',
            'end_time': datetime.now().isoformat() + 'Z',
            'duration_minutes': round(session_duration.total_seconds() / 60, 2),
            'operations_executed': self.operations_count,
            'iterations_completed': self.operations_count
        }
        
        # Save final reports
        json_path = self.metrics_reporter.save_json_report(final_report)
        md_path = self.metrics_reporter.save_markdown_report(final_report)
        
        print(f"📊 Session Summary:")
        print(f"  Duration: {session_duration}")
        print(f"  Operations: {self.operations_count}")
        print(f"  Items completed: {len(final_report['completed_ids'])}")
        print(f"  Items remaining: {final_report['backlog_health']['total_items'] - len(final_report['completed_ids'])}")
        
        if final_report['recommendations']:
            print(f"\n💡 Key Recommendations:")
            for rec in final_report['recommendations'][:3]:
                print(f"  • {rec}")
        
        print(f"\n📄 Final reports saved:")
        print(f"  • {json_path}")
        print(f"  • {md_path}")
        
        # Exit criteria met?
        ready_items = final_report['backlog_size_by_status'].get('READY', 0)
        if ready_items == 0:
            print("\n🎉 SUCCESS: All actionable backlog items completed!")
        elif self.operations_count >= self.max_operations:
            print(f"\n⏱️  PAUSED: Operation limit reached ({self.max_operations})")
        else:
            print("\n⏸️  PAUSED: Awaiting human intervention or new tasks")

def main():
    """Main entry point"""
    orchestrator = AutonomousOrchestrator()
    orchestrator.run_macro_execution_loop()

if __name__ == "__main__":
    main()