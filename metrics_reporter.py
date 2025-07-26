#!/usr/bin/env python3
"""
Metrics and Reporting Infrastructure
Generates comprehensive status reports and tracks backlog health
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import subprocess

from backlog_manager import BacklogManager, BacklogItem

class MetricsReporter:
    """Generates and tracks backlog and execution metrics"""
    
    def __init__(self, backlog_manager: BacklogManager):
        self.backlog_manager = backlog_manager
        self.status_dir = Path("docs/status")
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git-based metrics"""
        metrics = {}
        
        try:
            # Get recent commits
            result = subprocess.run(['git', 'log', '--oneline', '-10'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                metrics['recent_commits'] = len(commits)
                metrics['latest_commit'] = commits[0] if commits else None
            
            # Get branch info
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                metrics['current_branch'] = result.stdout.strip()
            
            # Get uncommitted changes
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
                metrics['uncommitted_files'] = len(changes)
                
        except subprocess.SubprocessError:
            pass
            
        return metrics

    def collect_test_coverage(self) -> Dict[str, Any]:
        """Collect test coverage metrics"""
        coverage = {'available': False}
        
        # Try to get pytest coverage
        try:
            result = subprocess.run(['python', '-m', 'pytest', '--cov=.', '--cov-report=json'], 
                                  capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                # Look for coverage.json
                coverage_file = Path('coverage.json')
                if coverage_file.exists():
                    with open(coverage_file, 'r') as f:
                        cov_data = json.load(f)
                        coverage = {
                            'available': True,
                            'total_coverage': cov_data.get('totals', {}).get('percent_covered', 0),
                            'lines_covered': cov_data.get('totals', {}).get('covered_lines', 0),
                            'lines_total': cov_data.get('totals', {}).get('num_statements', 0)
                        }
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass
            
        return coverage

    def analyze_flaky_tests(self) -> List[Dict[str, Any]]:
        """Identify potentially flaky tests"""
        # This would analyze test history - simplified for now
        flaky_tests = []
        
        # Look for test files with "flaky" or "skip" annotations
        try:
            result = subprocess.run(['grep', '-r', '-n', '-i', r'flaky\|skip\|xfail', 'tests/', '.'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[:5]  # Limit to 5
                for line in lines:
                    if ':' in line:
                        file_path, content = line.split(':', 1)
                        flaky_tests.append({
                            'file': file_path,
                            'indicator': content.strip()[:100]
                        })
        except subprocess.SubprocessError:
            pass
            
        return flaky_tests

    def calculate_cycle_time_metrics(self) -> Dict[str, float]:
        """Calculate average cycle time for completed items"""
        completed_items = [item for item in self.backlog_manager.items 
                          if item.status == "DONE" and item.updated_at]
        
        if not completed_items:
            return {'avg_cycle_time_days': 0.0, 'completed_items': 0}
        
        total_days = 0
        for item in completed_items:
            created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
            completed = datetime.fromisoformat(item.updated_at.replace('Z', '+00:00'))
            cycle_time = (completed - created).days
            total_days += cycle_time
        
        return {
            'avg_cycle_time_days': round(total_days / len(completed_items), 2),
            'completed_items': len(completed_items),
            'fastest_cycle_days': min((datetime.fromisoformat(item.updated_at.replace('Z', '+00:00')) - 
                                     datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))).days 
                                    for item in completed_items),
            'slowest_cycle_days': max((datetime.fromisoformat(item.updated_at.replace('Z', '+00:00')) - 
                                     datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))).days 
                                    for item in completed_items)
        }

    def analyze_backlog_health(self) -> Dict[str, Any]:
        """Analyze overall backlog health metrics"""
        items = self.backlog_manager.items
        
        # Age analysis
        now = datetime.now()
        ages = []
        for item in items:
            created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
            age_days = (now.astimezone() - created).days
            ages.append(age_days)
        
        # WSJF distribution
        wsjf_scores = [item.wsjf_score for item in items]
        
        # Risk analysis
        risk_distribution = {}
        for item in items:
            risk_distribution[item.risk_tier] = risk_distribution.get(item.risk_tier, 0) + 1
        
        # Status flow analysis
        status_counts = {}
        for item in items:
            status_counts[item.status] = status_counts.get(item.status, 0) + 1
        
        # Identify bottlenecks
        bottlenecks = []
        if status_counts.get('DOING', 0) > 3:
            bottlenecks.append("Too many items in DOING state")
        if status_counts.get('BLOCKED', 0) > 2:
            bottlenecks.append("High number of blocked items")
        if status_counts.get('NEW', 0) > status_counts.get('REFINED', 0) * 2:
            bottlenecks.append("Large backlog of unrefined items")
        
        return {
            'total_items': len(items),
            'avg_age_days': round(sum(ages) / len(ages), 2) if ages else 0,
            'oldest_item_days': max(ages) if ages else 0,
            'avg_wsjf_score': round(sum(wsjf_scores) / len(wsjf_scores), 2) if wsjf_scores else 0,
            'highest_wsjf_score': max(wsjf_scores) if wsjf_scores else 0,
            'risk_distribution': risk_distribution,
            'status_distribution': status_counts,
            'bottlenecks': bottlenecks,
            'items_needing_aging': len([item for item in items 
                                      if (now.astimezone() - datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))).days > 30])
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        timestamp = datetime.now().isoformat() + 'Z'
        
        # Collect all metrics
        git_metrics = self.collect_git_metrics()
        coverage_metrics = self.collect_test_coverage()
        flaky_tests = self.analyze_flaky_tests()
        cycle_metrics = self.calculate_cycle_time_metrics()
        backlog_health = self.analyze_backlog_health()
        
        # Get prioritized backlog snapshot
        prioritized = self.backlog_manager.get_prioritized_backlog()
        wsjf_snapshot = [
            {
                'id': item.id,
                'title': item.title,
                'status': item.status,
                'wsjf_score': round(item.wsjf_score, 2),
                'risk_tier': item.risk_tier,
                'effort': item.effort,
                'value': item.value
            }
            for item in prioritized[:10]  # Top 10 items
        ]
        
        # Identify risks and blocks
        risks_and_blocks = []
        blocked_items = [item for item in self.backlog_manager.items if item.status == "BLOCKED"]
        high_risk_items = [item for item in self.backlog_manager.items if item.risk_tier == "high"]
        
        for item in blocked_items:
            risks_and_blocks.append({
                'type': 'blocked',
                'id': item.id,
                'title': item.title,
                'description': f"Item blocked: {item.description[:100]}..."
            })
        
        for item in high_risk_items[:3]:  # Limit to top 3 high-risk
            if item.status not in ["DONE", "BLOCKED"]:
                risks_and_blocks.append({
                    'type': 'high_risk',
                    'id': item.id,
                    'title': item.title,
                    'description': f"High-risk item: {item.description[:100]}..."
                })
        
        # Count open PRs (simulated)
        open_prs = len([item for item in self.backlog_manager.items if item.status == "PR"])
        
        report = {
            'timestamp': timestamp,
            'completed_ids': [item.id for item in self.backlog_manager.items if item.status == "DONE"],
            'coverage_delta': coverage_metrics,
            'flaky_tests': flaky_tests,
            'ci_summary': {
                'git_metrics': git_metrics,
                'test_status': 'available' if coverage_metrics['available'] else 'not_configured'
            },
            'open_prs': open_prs,
            'risks_or_blocks': risks_and_blocks,
            'backlog_size_by_status': backlog_health['status_distribution'],
            'avg_cycle_time': cycle_metrics['avg_cycle_time_days'],
            'wsjf_snapshot': wsjf_snapshot,
            'backlog_health': backlog_health,
            'recommendations': self.generate_recommendations(backlog_health, cycle_metrics)
        }
        
        return report

    def generate_recommendations(self, backlog_health: Dict, cycle_metrics: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Backlog health recommendations
        if backlog_health['avg_age_days'] > 60:
            recommendations.append("Consider pruning old backlog items or updating priorities")
        
        if len(backlog_health['bottlenecks']) > 0:
            recommendations.append(f"Address bottlenecks: {', '.join(backlog_health['bottlenecks'])}")
        
        if backlog_health['status_distribution'].get('NEW', 0) > 10:
            recommendations.append("Large number of unrefined items - schedule refinement sessions")
        
        # Cycle time recommendations
        if cycle_metrics['avg_cycle_time_days'] > 14:
            recommendations.append("Consider breaking down large items to reduce cycle time")
        
        # Risk recommendations
        high_risk_count = backlog_health['risk_distribution'].get('high', 0)
        if high_risk_count > 3:
            recommendations.append("High number of risky items - prioritize risk mitigation")
        
        # WSJF recommendations
        if backlog_health['avg_wsjf_score'] < 2.0:
            recommendations.append("Low average WSJF scores - review value and effort estimates")
        
        return recommendations

    def save_json_report(self, report: Dict[str, Any]) -> str:
        """Save report as JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"status_report_{timestamp}.json"
        filepath = self.status_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Also save as latest
        latest_path = self.status_dir / "latest_report.json"
        with open(latest_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        return str(filepath)

    def save_markdown_report(self, report: Dict[str, Any]) -> str:
        """Save report as Markdown file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"status_report_{timestamp}.md"
        filepath = self.status_dir / filename
        
        md_content = self.format_markdown_report(report)
        
        with open(filepath, 'w') as f:
            f.write(md_content)
        
        # Also save as latest
        latest_path = self.status_dir / "latest_report.md"
        with open(latest_path, 'w') as f:
            f.write(md_content)
            
        return str(filepath)

    def format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format report as Markdown"""
        md = f"""# Autonomous Backlog Status Report

**Generated:** {report['timestamp']}

## 📊 Executive Summary

- **Total Items:** {report['backlog_health']['total_items']}
- **Average Cycle Time:** {report['avg_cycle_time']} days
- **Open PRs:** {report['open_prs']}
- **Completed Today:** {len(report['completed_ids'])}

## 🏆 Top Priority Items (WSJF)

| ID | Title | Status | WSJF | Risk |
|----|-------|--------|------|------|
"""
        
        for item in report['wsjf_snapshot'][:5]:
            md += f"| {item['id']} | {item['title'][:40]}... | {item['status']} | {item['wsjf_score']} | {item['risk_tier']} |\n"
        
        md += f"""
## 📈 Backlog Health

- **Average Age:** {report['backlog_health']['avg_age_days']} days
- **Oldest Item:** {report['backlog_health']['oldest_item_days']} days
- **Average WSJF:** {report['backlog_health']['avg_wsjf_score']}

### Status Distribution
"""
        
        for status, count in report['backlog_size_by_status'].items():
            md += f"- **{status}:** {count}\n"
        
        if report['backlog_health']['bottlenecks']:
            md += "\n### ⚠️ Bottlenecks\n"
            for bottleneck in report['backlog_health']['bottlenecks']:
                md += f"- {bottleneck}\n"
        
        if report['risks_or_blocks']:
            md += "\n## 🚨 Risks & Blocks\n"
            for risk in report['risks_or_blocks']:
                md += f"- **{risk['type'].upper()}:** {risk['id']} - {risk['title']}\n"
        
        md += f"""
## 🧪 Quality Metrics

### Test Coverage
- **Available:** {'Yes' if report['coverage_delta']['available'] else 'No'}
"""
        
        if report['coverage_delta']['available']:
            md += f"- **Coverage:** {report['coverage_delta']['total_coverage']:.1f}%\n"
            md += f"- **Lines Covered:** {report['coverage_delta']['lines_covered']}/{report['coverage_delta']['lines_total']}\n"
        
        if report['flaky_tests']:
            md += "\n### Flaky Tests\n"
            for test in report['flaky_tests'][:3]:
                md += f"- {test['file']}: {test['indicator']}\n"
        
        if report['recommendations']:
            md += "\n## 💡 Recommendations\n"
            for rec in report['recommendations']:
                md += f"- {rec}\n"
        
        md += f"""
## 📋 Completed Items

{len(report['completed_ids'])} items completed: {', '.join(report['completed_ids'])}

---
*Report generated by Autonomous Backlog Management System*
"""
        
        return md

def main():
    """Generate and save comprehensive report"""
    print("📊 Generating metrics report...")
    
    # Initialize components
    backlog_manager = BacklogManager()
    reporter = MetricsReporter(backlog_manager)
    
    # Generate comprehensive report
    report = reporter.generate_comprehensive_report()
    
    # Save reports
    json_path = reporter.save_json_report(report)
    md_path = reporter.save_markdown_report(report)
    
    print(f"✅ Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    
    # Display key metrics
    print(f"\n📈 Key Metrics:")
    print(f"  Total items: {report['backlog_health']['total_items']}")
    print(f"  Ready items: {report['backlog_size_by_status'].get('READY', 0)}")
    print(f"  Blocked items: {report['backlog_size_by_status'].get('BLOCKED', 0)}")
    print(f"  Average WSJF: {report['backlog_health']['avg_wsjf_score']}")
    
    if report['recommendations']:
        print(f"\n💡 Top Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"  • {rec}")

if __name__ == "__main__":
    main()