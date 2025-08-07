#!/usr/bin/env python3
"""
Advanced Value Discovery Engine for Perpetual SDLC Enhancement
Implements continuous signal harvesting and intelligent work prioritization
"""

import json
import re
import ast
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import yaml


@dataclass
class ValueItem:
    """Represents a discovered value item with comprehensive scoring"""
    id: str
    title: str
    description: str
    category: str
    source: str
    estimated_effort: float
    scores: Dict[str, float]
    composite_score: float
    risk_level: str
    files_affected: List[str]
    dependencies: List[str]
    created_at: str
    metadata: Dict[str, Any]


class ValueDiscoveryEngine:
    """Advanced engine for continuous value discovery and prioritization"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config = self._load_config()
        self.metrics = self._load_metrics()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load Terragon configuration"""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load existing value metrics"""
        metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return {}
    
    def discover_value_items(self) -> List[ValueItem]:
        """Main value discovery orchestration"""
        items = []
        
        # 1. Code Analysis Discovery
        items.extend(self._discover_from_code())
        
        # 2. Static Analysis Discovery
        items.extend(self._discover_from_static_analysis())
        
        # 3. Dependency Discovery
        items.extend(self._discover_from_dependencies())
        
        # 4. Performance Discovery
        items.extend(self._discover_from_performance())
        
        # 5. Security Discovery
        items.extend(self._discover_from_security())
        
        # Score and prioritize all items
        for item in items:
            item.composite_score = self._calculate_composite_score(item)
        
        return sorted(items, key=lambda x: x.composite_score, reverse=True)
    
    def _discover_from_code(self) -> List[ValueItem]:
        """Discover value items from code analysis"""
        items = []
        
        # Find TODO/FIXME comments
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find TODO/FIXME patterns
                todo_pattern = r'#\s*(TODO|FIXME|HACK|BUG|DEPRECATED):\s*(.+)'
                matches = re.finditer(todo_pattern, content, re.IGNORECASE)
                
                for match in matches:
                    tag, description = match.groups()
                    items.append(ValueItem(
                        id=f"code-{tag.lower()}-{hash(f'{py_file}:{match.start()}')}",
                        title=f"Address {tag} in {py_file.name}",
                        description=description.strip(),
                        category="technical_debt",
                        source="code_analysis",
                        estimated_effort=self._estimate_effort_from_tag(tag),
                        scores={},
                        composite_score=0.0,
                        risk_level=self._assess_risk_from_tag(tag),
                        files_affected=[str(py_file)],
                        dependencies=[],
                        created_at=datetime.now().isoformat(),
                        metadata={"line": content[:match.start()].count('\n') + 1}
                    ))
            except Exception:
                continue
                
        return items
    
    def _discover_from_static_analysis(self) -> List[ValueItem]:
        """Discover items from static analysis tools"""
        items = []
        
        try:
            # Run ruff to find code quality issues
            result = subprocess.run(
                ["ruff", "check", "src/", "--output-format=json"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues[:10]:  # Limit to top 10
                    items.append(ValueItem(
                        id=f"ruff-{issue.get('code', 'unknown')}-{hash(issue.get('filename', ''))}",
                        title=f"Fix {issue.get('code', 'code quality')} issue",
                        description=issue.get('message', 'Static analysis issue'),
                        category="code_quality",
                        source="ruff",
                        estimated_effort=self._estimate_effort_from_severity(issue.get('code', '')),
                        scores={},
                        composite_score=0.0,
                        risk_level="medium",
                        files_affected=[issue.get('filename', '')],
                        dependencies=[],
                        created_at=datetime.now().isoformat(),
                        metadata={"line": issue.get('location', {}).get('row', 0)}
                    ))
        except Exception:
            pass
            
        return items
    
    def _discover_from_dependencies(self) -> List[ValueItem]:
        """Discover dependency-related value items"""
        items = []
        
        try:
            # Check for outdated packages using pip-audit
            result = subprocess.run(
                ["pip-audit", "--format=json", "--require", "requirements.txt"],
                capture_output=True, text=True, cwd=self.repo_path
            )
            
            if result.stdout:
                audit_results = json.loads(result.stdout)
                for vuln in audit_results.get('vulnerabilities', []):
                    items.append(ValueItem(
                        id=f"security-{vuln.get('id', 'unknown')}",
                        title=f"Update {vuln.get('package', 'package')} (security)",
                        description=f"Security vulnerability: {vuln.get('description', 'Unknown')}",
                        category="security",
                        source="pip-audit",
                        estimated_effort=2.0,
                        scores={},
                        composite_score=0.0,
                        risk_level="high",
                        files_affected=["requirements.txt"],
                        dependencies=[],
                        created_at=datetime.now().isoformat(),
                        metadata={"severity": vuln.get('fix_versions', [])}
                    ))
        except Exception:
            pass
            
        return items
    
    def _discover_from_performance(self) -> List[ValueItem]:
        """Discover performance optimization opportunities"""
        items = []
        
        # Look for performance anti-patterns in code
        performance_patterns = [
            (r'\.iterrows\(\)', "Use vectorized operations instead of iterrows()"),
            (r'pd\.concat.*for.*in', "Use list comprehension with single concat"),
            (r'time\.sleep\([0-9\.]+\)', "Consider async alternatives to blocking sleep"),
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, suggestion in performance_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        items.append(ValueItem(
                            id=f"perf-{hash(f'{py_file}:{match.start()}')}",
                            title=f"Performance optimization in {py_file.name}",
                            description=suggestion,
                            category="performance",
                            source="pattern_analysis",
                            estimated_effort=3.0,
                            scores={},
                            composite_score=0.0,
                            risk_level="low",
                            files_affected=[str(py_file)],
                            dependencies=[],
                            created_at=datetime.now().isoformat(),
                            metadata={"pattern": pattern}
                        ))
            except Exception:
                continue
                
        return items
    
    def _discover_from_security(self) -> List[ValueItem]:
        """Discover security enhancement opportunities"""
        items = []
        
        # Security patterns to look for
        security_patterns = [
            (r'open\(["\'].*["\'],\s*["\']w["\']', "File operations without proper error handling"),
            (r'subprocess\..*shell=True', "Shell injection risk with shell=True"),
            (r'dynamic_eval\s*\(', "Dynamic code evaluation detected"),
            (r'dynamic_exec\s*\(', "Dynamic code execution detected"),
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in security_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        items.append(ValueItem(
                            id=f"sec-{hash(f'{py_file}:{match.start()}')}",
                            title=f"Security enhancement in {py_file.name}",
                            description=description,
                            category="security",
                            source="security_analysis",
                            estimated_effort=4.0,
                            scores={},
                            composite_score=0.0,
                            risk_level="high",
                            files_affected=[str(py_file)],
                            dependencies=[],
                            created_at=datetime.now().isoformat(),
                            metadata={"pattern": pattern}
                        ))
            except Exception:
                continue
                
        return items
    
    def _calculate_composite_score(self, item: ValueItem) -> float:
        """Calculate comprehensive composite score using WSJF + ICE + Technical Debt"""
        weights = self.config.get('scoring', {}).get('weights', {}).get('advanced', {})
        
        # WSJF Components
        user_value = self._score_user_value(item)
        time_criticality = self._score_time_criticality(item)
        risk_reduction = self._score_risk_reduction(item)
        opportunity = self._score_opportunity(item)
        
        cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
        wsjf = cost_of_delay / max(item.estimated_effort, 0.5)
        
        # ICE Components
        impact = self._score_impact(item)
        confidence = self._score_confidence(item)
        ease = 10 - min(item.estimated_effort, 10)
        ice = impact * confidence * ease
        
        # Technical Debt Score
        tech_debt = self._score_technical_debt(item)
        
        # Security/Compliance boosts
        security_boost = 2.0 if item.category == "security" else 1.0
        compliance_boost = 1.8 if item.category == "compliance" else 1.0
        
        # Composite calculation
        composite = (
            weights.get('wsjf', 0.5) * self._normalize_score(wsjf, 0, 50) +
            weights.get('ice', 0.1) * self._normalize_score(ice, 0, 1000) +
            weights.get('technicalDebt', 0.3) * self._normalize_score(tech_debt, 0, 100) +
            weights.get('security', 0.1) * 50
        ) * security_boost * compliance_boost
        
        # Store individual scores
        item.scores = {
            'wsjf': wsjf,
            'ice': ice,
            'technicalDebt': tech_debt,
            'impact': impact,
            'confidence': confidence,
            'ease': ease
        }
        
        return composite
    
    def _score_user_value(self, item: ValueItem) -> float:
        """Score user/business value impact"""
        category_values = {
            'security': 9,
            'compliance': 8,
            'performance': 7,
            'technical_debt': 5,
            'code_quality': 4,
            'documentation': 3
        }
        return category_values.get(item.category, 5)
    
    def _score_time_criticality(self, item: ValueItem) -> float:
        """Score time criticality"""
        if item.category == 'security':
            return 9
        elif item.risk_level == 'high':
            return 7
        elif item.risk_level == 'medium':
            return 5
        return 3
    
    def _score_risk_reduction(self, item: ValueItem) -> float:
        """Score risk reduction benefit"""
        risk_values = {'high': 8, 'medium': 5, 'low': 2}
        return risk_values.get(item.risk_level, 3)
    
    def _score_opportunity(self, item: ValueItem) -> float:
        """Score opportunity enablement"""
        if item.category in ['performance', 'architecture']:
            return 6
        return 3
    
    def _score_impact(self, item: ValueItem) -> float:
        """Score business impact (1-10)"""
        category_impacts = {
            'security': 9,
            'compliance': 8,
            'performance': 7,
            'technical_debt': 6,
            'code_quality': 5
        }
        return category_impacts.get(item.category, 5)
    
    def _score_confidence(self, item: ValueItem) -> float:
        """Score execution confidence (1-10)"""
        if item.estimated_effort <= 2:
            return 9
        elif item.estimated_effort <= 5:
            return 7
        return 5
    
    def _score_technical_debt(self, item: ValueItem) -> float:
        """Score technical debt reduction"""
        if item.category == 'technical_debt':
            return 80
        elif item.category == 'code_quality':
            return 60
        elif item.category == 'performance':
            return 40
        return 20
    
    def _normalize_score(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize score to 0-100 range"""
        return max(0, min(100, ((value - min_val) / (max_val - min_val)) * 100))
    
    def _estimate_effort_from_tag(self, tag: str) -> float:
        """Estimate effort based on code tag"""
        effort_map = {
            'TODO': 2.0,
            'FIXME': 3.0,
            'HACK': 4.0,
            'BUG': 5.0,
            'DEPRECATED': 6.0
        }
        return effort_map.get(tag.upper(), 3.0)
    
    def _assess_risk_from_tag(self, tag: str) -> str:
        """Assess risk level from code tag"""
        risk_map = {
            'TODO': 'low',
            'FIXME': 'medium',
            'HACK': 'high',
            'BUG': 'high',
            'DEPRECATED': 'medium'
        }
        return risk_map.get(tag.upper(), 'medium')
    
    def _estimate_effort_from_severity(self, code: str) -> float:
        """Estimate effort from ruff error code"""
        if code.startswith('E'):  # Style errors
            return 1.0
        elif code.startswith('W'):  # Warnings
            return 1.5
        elif code.startswith('F'):  # Logical errors
            return 3.0
        elif code.startswith('S'):  # Security
            return 4.0
        return 2.0
    
    def generate_backlog_update(self, items: List[ValueItem]) -> str:
        """Generate markdown backlog with discovered items"""
        now = datetime.now().isoformat()
        
        content = f"""# 📊 Autonomous Value Backlog

Last Updated: {now}
Next Execution: {(datetime.now() + timedelta(hours=1)).isoformat()}

## 🎯 Next Best Value Item
"""
        
        if items:
            top_item = items[0]
            content += f"""**[{top_item.id}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.scores.get('wsjf', 0):.1f} | **ICE**: {top_item.scores.get('ice', 0):.0f} | **Tech Debt**: {top_item.scores.get('technicalDebt', 0):.0f}
- **Estimated Effort**: {top_item.estimated_effort} hours
- **Expected Impact**: {top_item.description}

## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
            
            for i, item in enumerate(items[:10], 1):
                content += f"| {i} | {item.id} | {item.title[:50]}... | {item.composite_score:.1f} | {item.category} | {item.estimated_effort} |\n"
            
        content += f"""

## 📈 Value Metrics
- **Items Discovered**: {len(items)}
- **Security Items**: {len([i for i in items if i.category == 'security'])}
- **Performance Items**: {len([i for i in items if i.category == 'performance'])}
- **Technical Debt**: {len([i for i in items if i.category == 'technical_debt'])}

## 🔄 Continuous Discovery Stats
- **Discovery Sources**:
  - Code Analysis: {len([i for i in items if i.source == 'code_analysis'])}
  - Static Analysis: {len([i for i in items if i.source == 'ruff'])}
  - Security Analysis: {len([i for i in items if i.source == 'security_analysis'])}
  - Performance Analysis: {len([i for i in items if i.source == 'pattern_analysis'])}
"""
        
        return content


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    items = engine.discover_value_items()
    
    print(f"Discovered {len(items)} value items")
    if items:
        print(f"Top item: {items[0].title} (Score: {items[0].composite_score:.1f})")
        
        # Generate backlog
        backlog_content = engine.generate_backlog_update(items)
        with open("BACKLOG.md", "w") as f:
            f.write(backlog_content)
        print("Updated BACKLOG.md")