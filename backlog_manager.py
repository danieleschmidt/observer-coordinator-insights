#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF scoring, backlog discovery, and execution tracking
"""

import yaml
import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class BacklogItem:
    """Normalized backlog item with WSJF scoring"""
    id: str
    title: str
    type: str
    description: str
    acceptance_criteria: List[str]
    effort: int  # 1-2-3-5-8-13 (Fibonacci)
    value: int   # 1-2-3-5-8-13 (Fibonacci)
    time_criticality: int  # 1-2-3-5-8-13 (Fibonacci)
    risk_reduction: int    # 1-2-3-5-8-13 (Fibonacci)
    status: str  # NEW → REFINED → READY → DOING → PR → DONE/BLOCKED
    risk_tier: str  # low, medium, high
    created_at: str
    updated_at: Optional[str] = None
    links: List[str] = None
    wsjf_score: float = 0.0
    aging_multiplier: float = 1.0

    def __post_init__(self):
        if self.links is None:
            self.links = []
        self.calculate_wsjf_score()

    def calculate_wsjf_score(self):
        """Calculate WSJF score: (Value + Time Criticality + Risk Reduction) / Effort"""
        cost_of_delay = self.value + self.time_criticality + self.risk_reduction
        self.wsjf_score = (cost_of_delay / self.effort) * self.aging_multiplier

    def apply_aging(self, days_old: int, max_multiplier: float = 2.0):
        """Apply aging multiplier to boost stale but important items"""
        if days_old > 30:  # Start aging after 30 days
            age_factor = min((days_old - 30) / 60, 1.0)  # Max aging at 90 days
            self.aging_multiplier = 1.0 + (age_factor * (max_multiplier - 1.0))
            self.calculate_wsjf_score()

class BacklogManager:
    """Manages backlog items with WSJF prioritization and continuous discovery"""
    
    def __init__(self, backlog_file: str = "backlog.yml"):
        self.backlog_file = backlog_file
        self.items: List[BacklogItem] = []
        self.repo_root = Path(".")
        self.load_backlog()

    def load_backlog(self):
        """Load backlog from YAML file"""
        if os.path.exists(self.backlog_file):
            with open(self.backlog_file, 'r') as f:
                data = yaml.safe_load(f)
                for item_data in data.get('items', []):
                    item = BacklogItem(**item_data)
                    # Apply aging based on creation date
                    created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                    days_old = (datetime.now().astimezone() - created).days
                    item.apply_aging(days_old)
                    self.items.append(item)

    def save_backlog(self):
        """Save backlog to YAML file with metadata"""
        data = {
            'items': [asdict(item) for item in self.items],
            'metadata': {
                'last_updated': datetime.now().isoformat() + 'Z',
                'total_items': len(self.items),
                'status_counts': self._get_status_counts(),
                'avg_wsjf_score': self._get_avg_wsjf_score(),
                'next_review': (datetime.now() + timedelta(days=1)).isoformat() + 'Z'
            }
        }
        
        with open(self.backlog_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def discover_todos_and_fixmes(self) -> List[BacklogItem]:
        """Discover TODO/FIXME comments in codebase"""
        new_items = []
        todo_pattern = re.compile(r'(TODO|FIXME|BUG|HACK):\s*(.+)', re.IGNORECASE)
        
        try:
            # Use git to find tracked files to avoid .git directory
            result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
            else:
                # Fallback to find command
                result = subprocess.run(['find', '.', '-type', 'f', '-not', '-path', './.git/*'], 
                                      capture_output=True, text=True)
                files = result.stdout.strip().split('\n')
            
            for file_path in files:
                if not file_path or file_path.startswith('./.git/'):
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        for line_num, line in enumerate(f, 1):
                            match = todo_pattern.search(line)
                            if match:
                                todo_type = match.group(1).upper()
                                description = match.group(2).strip()
                                
                                # Generate unique ID
                                item_id = f"AUTO-{todo_type}-{len(new_items)+1:03d}"
                                
                                # Determine effort and priority based on type
                                effort_map = {'TODO': 3, 'FIXME': 2, 'BUG': 5, 'HACK': 8}
                                value_map = {'TODO': 3, 'FIXME': 5, 'BUG': 8, 'HACK': 2}
                                
                                item = BacklogItem(
                                    id=item_id,
                                    title=f"{todo_type}: {description[:50]}...",
                                    type="technical_debt",
                                    description=f"Found in {file_path}:{line_num} - {description}",
                                    acceptance_criteria=[f"Resolve {todo_type} in {file_path}:{line_num}"],
                                    effort=effort_map.get(todo_type, 3),
                                    value=value_map.get(todo_type, 3),
                                    time_criticality=5 if todo_type == 'BUG' else 2,
                                    risk_reduction=8 if todo_type == 'BUG' else 3,
                                    status="NEW",
                                    risk_tier="medium" if todo_type == 'BUG' else "low",
                                    created_at=datetime.now().isoformat() + 'Z',
                                    links=[f"{file_path}:{line_num}"]
                                )
                                new_items.append(item)
                                
                except (UnicodeDecodeError, PermissionError):
                    continue
                    
        except subprocess.SubprocessError:
            pass
            
        return new_items

    def discover_failing_tests(self) -> List[BacklogItem]:
        """Discover failing tests and create backlog items"""
        new_items = []
        
        # Common test commands to try
        test_commands = [
            ['npm', 'test'],
            ['pytest'],
            ['python', '-m', 'pytest'],
            ['cargo', 'test'],
            ['go', 'test', './...']
        ]
        
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0 and result.stderr:
                    # Parse test failures
                    item = BacklogItem(
                        id=f"TEST-FAIL-{len(new_items)+1:03d}",
                        title=f"Fix failing tests ({cmd[0]})",
                        type="bug",
                        description=f"Test command '{' '.join(cmd)}' is failing",
                        acceptance_criteria=["All tests pass", "No test failures in CI"],
                        effort=5,
                        value=8,
                        time_criticality=8,
                        risk_reduction=13,
                        status="NEW",
                        risk_tier="high",
                        created_at=datetime.now().isoformat() + 'Z'
                    )
                    new_items.append(item)
                    break  # Only add one failing test item per discovery
            except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
                
        return new_items

    def add_discovered_items(self, new_items: List[BacklogItem]):
        """Add new items to backlog, avoiding duplicates"""
        existing_titles = {item.title for item in self.items}
        
        for item in new_items:
            if item.title not in existing_titles:
                self.items.append(item)

    def continuous_discovery(self):
        """Run continuous discovery process"""
        print("🔍 Running backlog discovery...")
        
        # Discover TODOs/FIXMEs
        todo_items = self.discover_todos_and_fixmes()
        if todo_items:
            print(f"  Found {len(todo_items)} TODO/FIXME items")
            self.add_discovered_items(todo_items)
        
        # Discover failing tests
        test_items = self.discover_failing_tests()
        if test_items:
            print(f"  Found {len(test_items)} failing test items")
            self.add_discovered_items(test_items)
        
        print(f"✅ Discovery complete. Total backlog items: {len(self.items)}")

    def get_prioritized_backlog(self) -> List[BacklogItem]:
        """Get backlog sorted by WSJF score (highest first)"""
        return sorted(self.items, key=lambda x: x.wsjf_score, reverse=True)

    def get_next_ready_item(self) -> Optional[BacklogItem]:
        """Get the highest priority READY item"""
        ready_items = [item for item in self.get_prioritized_backlog() 
                      if item.status == "READY"]
        return ready_items[0] if ready_items else None

    def update_item_status(self, item_id: str, new_status: str):
        """Update item status"""
        for item in self.items:
            if item.id == item_id:
                item.status = new_status
                item.updated_at = datetime.now().isoformat() + 'Z'
                break

    def _get_status_counts(self) -> Dict[str, int]:
        """Get count of items by status"""
        counts = {}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def _get_avg_wsjf_score(self) -> float:
        """Get average WSJF score"""
        if not self.items:
            return 0.0
        return sum(item.wsjf_score for item in self.items) / len(self.items)

    def generate_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        return {
            'timestamp': datetime.now().isoformat() + 'Z',
            'total_items': len(self.items),
            'status_counts': self._get_status_counts(),
            'avg_wsjf_score': round(self._get_avg_wsjf_score(), 2),
            'top_priority_items': [
                {'id': item.id, 'title': item.title, 'wsjf_score': round(item.wsjf_score, 2)}
                for item in self.get_prioritized_backlog()[:5]
            ],
            'ready_items': len([item for item in self.items if item.status == "READY"]),
            'high_risk_items': len([item for item in self.items if item.risk_tier == "high"]),
            'blocked_items': len([item for item in self.items if item.status == "BLOCKED"])
        }

def main():
    """Main execution function"""
    manager = BacklogManager()
    
    # Run discovery
    manager.continuous_discovery()
    
    # Save updated backlog
    manager.save_backlog()
    
    # Generate and display status report
    report = manager.generate_status_report()
    print("\n📊 Backlog Status Report:")
    print(f"  Total items: {report['total_items']}")
    print(f"  Ready items: {report['ready_items']}")
    print(f"  Average WSJF: {report['avg_wsjf_score']}")
    print(f"  High risk items: {report['high_risk_items']}")
    
    print("\n🏆 Top Priority Items:")
    for item in report['top_priority_items']:
        print(f"  {item['id']}: {item['title']} (WSJF: {item['wsjf_score']})")
    
    # Show next ready item
    next_item = manager.get_next_ready_item()
    if next_item:
        print(f"\n▶️  Next item to execute: {next_item.id} - {next_item.title}")
    else:
        print("\n⏸️  No READY items available")

if __name__ == "__main__":
    main()