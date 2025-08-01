#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Enhancement Runner
Entry point for the autonomous system with multiple execution modes
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


def ensure_setup():
    """Ensure .terragon directory and basic config exists"""
    terragon_dir = Path(".terragon")
    terragon_dir.mkdir(exist_ok=True)
    
    # Create basic config if it doesn't exist
    config_file = terragon_dir / "config.yaml"
    if not config_file.exists():
        with open(config_file, "w") as f:
            f.write("""# Terragon Autonomous SDLC Configuration
enabled: true
maturity_level: advanced

scoring:
  weights:
    wsjf: 0.5
    ice: 0.1
    technical_debt: 0.3
    security: 0.1
  thresholds:
    min_score: 10
    max_risk: 0.8

execution:
  max_concurrent_tasks: 1
  test_requirements:
    min_coverage: 80
    performance_regression: 5
  rollback_triggers:
    - test_failure
    - build_failure
    - security_violation
""")
    
    print(f"✅ Terragon configuration ready at {config_file}")


def run_discovery():
    """Run value discovery"""
    print("🔍 Running Autonomous Value Discovery...")
    os.system("python3 simple_value_discovery.py")


def run_perpetual(dry_run=False, max_cycles=10):
    """Run perpetual execution"""
    print(f"🤖 Running Perpetual Execution (dry_run={dry_run}, max_cycles={max_cycles})")
    
    if Path("perpetual_executor.py").exists():
        cmd = f"python3 perpetual_executor.py --max-cycles {max_cycles}"
        if dry_run:
            cmd += " --dry-run"
        os.system(cmd)
    else:
        print("⚠️  Full perpetual executor not available (missing dependencies)")
        print("   Running discovery mode instead...")
        run_discovery()


def show_status():
    """Show current autonomous system status"""
    print("📊 Terragon Autonomous SDLC Status")
    print("=" * 40)
    
    # Check configuration
    config_file = Path(".terragon/config.yaml")
    if config_file.exists():
        print("✅ Configuration: Ready")
    else:
        print("❌ Configuration: Missing")
    
    # Check metrics
    metrics_file = Path(".terragon/value-metrics.json")
    if metrics_file.exists():
        print("✅ Value Metrics: Available")
        import json
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
            maturity = metrics.get('repositoryMetrics', {}).get('maturityPercentage', 'Unknown')
            print(f"   Repository Maturity: {maturity}%")
        except:
            pass
    else:
        print("❌ Value Metrics: Not initialized")
    
    # Check discovery
    discovery_file = Path(".terragon/discovered_items.json")
    if discovery_file.exists():
        print("✅ Value Discovery: Active")
        import json
        try:
            with open(discovery_file) as f:
                items = json.load(f)
            print(f"   Items Discovered: {len(items)}")
            if items:
                top_item = max(items, key=lambda x: x['score'])
                print(f"   Top Item: {top_item['title'][:50]}... (Score: {top_item['score']})")
        except:
            pass
    else:
        print("❌ Value Discovery: Not run")
    
    # Check backlog
    backlog_file = Path("AUTONOMOUS_BACKLOG.md")
    if backlog_file.exists():
        print("✅ Autonomous Backlog: Generated")
        print(f"   Last Updated: {datetime.fromtimestamp(backlog_file.stat().st_mtime)}")
    else:
        print("❌ Autonomous Backlog: Not generated")
    
    print("\n🔗 Quick Actions:")
    print("   python3 run_terragon_autonomous.py --discover    # Run value discovery")
    print("   python3 run_terragon_autonomous.py --perpetual   # Run autonomous execution")
    print("   python3 run_terragon_autonomous.py --dry-run     # Simulate execution")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Terragon Autonomous SDLC Enhancement System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_terragon_autonomous.py --setup          # Initialize system
  python3 run_terragon_autonomous.py --discover       # Run value discovery
  python3 run_terragon_autonomous.py --perpetual      # Run autonomous execution
  python3 run_terragon_autonomous.py --dry-run        # Simulate execution
  python3 run_terragon_autonomous.py --status         # Show system status
        """
    )
    
    parser.add_argument("--setup", action="store_true", 
                       help="Initialize Terragon configuration")
    parser.add_argument("--discover", action="store_true",
                       help="Run value discovery")
    parser.add_argument("--perpetual", action="store_true",
                       help="Run perpetual execution")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run in simulation mode")
    parser.add_argument("--status", action="store_true",
                       help="Show system status")
    parser.add_argument("--max-cycles", type=int, default=10,
                       help="Maximum execution cycles for perpetual mode")
    
    args = parser.parse_args()
    
    # Default action if no arguments
    if not any([args.setup, args.discover, args.perpetual, args.dry_run, args.status]):
        print("🤖 Terragon Autonomous SDLC Enhancement System")
        print("   Use --help for available options")
        show_status()
        return
    
    # Execute requested actions
    if args.setup:
        ensure_setup()
    
    if args.status:
        show_status()
    
    if args.discover:
        ensure_setup()
        run_discovery()
    
    if args.perpetual:
        ensure_setup()
        run_perpetual(dry_run=args.dry_run, max_cycles=args.max_cycles)
    
    if args.dry_run and not args.perpetual:
        ensure_setup()
        run_perpetual(dry_run=True, max_cycles=5)


if __name__ == "__main__":
    main()