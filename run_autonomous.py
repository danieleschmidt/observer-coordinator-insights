#!/usr/bin/env python3
"""
Entry point for Autonomous Backlog Management System
Provides multiple execution modes and utilities
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from autonomous_orchestrator import AutonomousOrchestrator
from backlog_manager import BacklogManager
from metrics_reporter import MetricsReporter
from execution_engine import ExecutionEngine

def run_discovery_only():
    """Run backlog discovery without execution"""
    print("🔍 Running discovery mode...")
    manager = BacklogManager()
    manager.continuous_discovery()
    manager.save_backlog()
    
    report = manager.generate_status_report()
    print(f"✅ Discovery complete:")
    print(f"  Total items: {report['total_items']}")
    print(f"  Ready items: {report['ready_items']}")
    print(f"  High risk items: {report['high_risk_items']}")

def run_metrics_only():
    """Generate metrics report without execution"""
    print("📊 Generating metrics report...")
    manager = BacklogManager()
    reporter = MetricsReporter(manager)
    
    report = reporter.generate_comprehensive_report()
    json_path = reporter.save_json_report(report)
    md_path = reporter.save_markdown_report(report)
    
    print(f"✅ Reports generated:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")

def run_single_item(item_id: str):
    """Execute a single backlog item"""
    print(f"⚡ Executing single item: {item_id}")
    
    manager = BacklogManager()
    engine = ExecutionEngine(manager)
    
    # Find the item
    item = None
    for candidate in manager.items:
        if candidate.id == item_id:
            item = candidate
            break
    
    if not item:
        print(f"❌ Item {item_id} not found")
        return
    
    if item.status != "READY":
        print(f"⚠️  Item {item_id} is not in READY status (current: {item.status})")
        return
    
    # Execute the item
    results = engine.execute_micro_cycle(item)
    
    # Show results
    success_count = sum(1 for r in results if r.success)
    print(f"✅ Execution complete: {success_count}/{len(results)} stages successful")
    
    for result in results:
        status = "✅" if result.success else "❌"
        print(f"  {status} {result.stage}: {result.message}")

def show_backlog_status():
    """Show current backlog status"""
    print("📋 Current backlog status:")
    
    manager = BacklogManager()
    report = manager.generate_status_report()
    
    print(f"\n📊 Summary:")
    print(f"  Total items: {report['total_items']}")
    print(f"  Ready: {report['ready_items']}")
    print(f"  High risk: {report['high_risk_items']}")
    print(f"  Blocked: {report['blocked_items']}")
    print(f"  Average WSJF: {report['avg_wsjf_score']}")
    
    print(f"\n🏆 Top priority items:")
    for item in report['top_priority_items']:
        print(f"  {item['id']}: {item['title']} (WSJF: {item['wsjf_score']})")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Autonomous Backlog Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_autonomous.py                    # Run full autonomous mode
  python run_autonomous.py --discovery       # Discovery only
  python run_autonomous.py --metrics         # Generate metrics only
  python run_autonomous.py --status          # Show backlog status
  python run_autonomous.py --item BL-001     # Execute single item
        """
    )
    
    parser.add_argument('--discovery', action='store_true',
                       help='Run discovery mode only')
    parser.add_argument('--metrics', action='store_true',
                       help='Generate metrics report only')
    parser.add_argument('--status', action='store_true',
                       help='Show current backlog status')
    parser.add_argument('--item', type=str,
                       help='Execute a single backlog item by ID')
    parser.add_argument('--config', type=str, default='.automation-scope.yaml',
                       help='Path to automation scope configuration')
    
    args = parser.parse_args()
    
    try:
        if args.discovery:
            run_discovery_only()
        elif args.metrics:
            run_metrics_only()
        elif args.status:
            show_backlog_status()
        elif args.item:
            run_single_item(args.item)
        else:
            # Run full autonomous mode
            print("🤖 Starting full autonomous mode...")
            orchestrator = AutonomousOrchestrator(args.config)
            orchestrator.run_macro_execution_loop()
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()