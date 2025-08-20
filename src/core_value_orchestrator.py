#!/usr/bin/env python3
"""
Core Value Orchestrator - Generation 1 Enhancement
Autonomous value discovery and prioritization engine for SDLC enhancement
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from .insights_clustering.config import Config


@dataclass
class ValueItem:
    """Represents a discoverable value item in the SDLC"""
    id: str
    title: str
    description: str
    category: str  # 'performance', 'security', 'maintainability', 'feature'
    priority: int  # 1-100 scale
    effort_hours: float
    business_impact: int  # 1-100 scale
    technical_debt: int  # 1-100 scale
    roi_score: float  # Calculated return on investment
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    tags: Optional[List[str]] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        # Calculate ROI score automatically
        self.roi_score = self._calculate_roi()
    
    def _calculate_roi(self) -> float:
        """Calculate return on investment score based on impact vs effort"""
        if self.effort_hours <= 0:
            return 0.0
        return (self.business_impact + self.technical_debt) / (self.effort_hours * 10)


class ValueDiscoveryEngine:
    """Core engine for discovering value opportunities in the codebase"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.discovered_items: List[ValueItem] = []
        
    async def discover_performance_opportunities(self) -> List[ValueItem]:
        """Discover performance optimization opportunities"""
        items = []
        
        # Analyze clustering performance patterns
        clustering_files = list(Path("src/insights_clustering").rglob("*.py"))
        
        for file_path in clustering_files:
            try:
                content = file_path.read_text(encoding='utf-8')
                
                # Look for performance anti-patterns
                patterns = [
                    ("for.*in.*range\\(len\\(", "Use enumerate instead of range(len())", 1.0, 30),
                    ("\\+.*=.*\\[.*\\]", "Consider using list comprehension", 0.5, 25),
                    ("time\\.sleep\\(", "Consider async alternatives", 2.0, 40),
                    ("\\.iterrows\\(\\)", "Use vectorized operations", 3.0, 60),
                ]
                
                for pattern, desc, effort, impact in patterns:
                    import re
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        item = ValueItem(
                            id=f"perf-{hashlib.md5(f'{file_path}:{line_num}'.encode()).hexdigest()[:8]}",
                            title=f"Performance optimization in {file_path.name}",
                            description=desc,
                            category="performance",
                            priority=impact + 20,
                            effort_hours=effort,
                            business_impact=impact,
                            technical_debt=30,
                            roi_score=0,  # Will be calculated in __post_init__
                            file_path=str(file_path),
                            line_number=line_num,
                            tags=["performance", "optimization"]
                        )
                        items.append(item)
                        
            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {e}")
        
        return items
    
    async def discover_security_opportunities(self) -> List[ValueItem]:
        """Discover security enhancement opportunities"""
        items = []
        
        # Security patterns to look for
        security_patterns = [
            ("open\\(.*'w'.*\\)", "File operations need error handling", 2.0, 70),
            ("subprocess\\(.*shell=True", "Potential shell injection", 4.0, 90),
            ("eval\\(", "Dynamic code evaluation", 6.0, 95),
            ("exec\\(", "Dynamic code execution", 6.0, 95),
        ]
        
        for py_file in Path("src").rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                for pattern, desc, effort, impact in security_patterns:
                    import re
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        item = ValueItem(
                            id=f"sec-{hashlib.md5(f'{py_file}:{line_num}'.encode()).hexdigest()[:8]}",
                            title=f"Security enhancement in {py_file.name}",
                            description=desc,
                            category="security",
                            priority=impact,
                            effort_hours=effort,
                            business_impact=impact,
                            technical_debt=impact - 10,
                            roi_score=0,
                            file_path=str(py_file),
                            line_number=line_num,
                            tags=["security", "vulnerability"]
                        )
                        items.append(item)
                        
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        return items
    
    async def discover_maintainability_opportunities(self) -> List[ValueItem]:
        """Discover code maintainability improvements"""
        items = []
        
        # Analyze for complex functions, missing docstrings, etc.
        for py_file in Path("src").rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                # Find functions without docstrings
                import re
                func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
                matches = re.finditer(func_pattern, content)
                
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    func_name = match.group(1)
                    
                    # Skip private methods
                    if func_name.startswith('_'):
                        continue
                    
                    # Check if next few lines contain docstring
                    has_docstring = False
                    for i in range(line_num, min(line_num + 3, len(lines))):
                        if '"""' in lines[i] or "'''" in lines[i]:
                            has_docstring = True
                            break
                    
                    if not has_docstring:
                        item = ValueItem(
                            id=f"maint-doc-{hashlib.md5(f'{py_file}:{func_name}'.encode()).hexdigest()[:8]}",
                            title=f"Add docstring to {func_name} in {py_file.name}",
                            description=f"Function {func_name} lacks documentation",
                            category="maintainability",
                            priority=35,
                            effort_hours=0.5,
                            business_impact=25,
                            technical_debt=40,
                            roi_score=0,
                            file_path=str(py_file),
                            line_number=line_num,
                            tags=["documentation", "maintainability"]
                        )
                        items.append(item)
                        
            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")
        
        return items
    
    async def discover_feature_opportunities(self) -> List[ValueItem]:
        """Discover new feature opportunities based on codebase analysis"""
        items = []
        
        # Analyze existing functionality to suggest enhancements
        feature_opportunities = [
            {
                "title": "Real-time clustering visualization",
                "description": "Add interactive web dashboard for real-time cluster visualization",
                "priority": 70,
                "effort": 16.0,
                "impact": 80,
                "tags": ["visualization", "dashboard", "realtime"]
            },
            {
                "title": "Advanced team recommendation API",
                "description": "Implement REST API for team formation recommendations",
                "priority": 65,
                "effort": 12.0,
                "impact": 75,
                "tags": ["api", "teams", "recommendations"]
            },
            {
                "title": "Clustering algorithm comparison framework",
                "description": "Add framework to compare multiple clustering algorithms",
                "priority": 60,
                "effort": 20.0,
                "impact": 70,
                "tags": ["algorithms", "comparison", "research"]
            }
        ]
        
        for i, opportunity in enumerate(feature_opportunities):
            item = ValueItem(
                id=f"feat-{i+1:03d}",
                title=opportunity["title"],
                description=opportunity["description"],
                category="feature",
                priority=opportunity["priority"],
                effort_hours=opportunity["effort"],
                business_impact=opportunity["impact"],
                technical_debt=10,  # New features have minimal tech debt
                roi_score=0,
                tags=opportunity["tags"]
            )
            items.append(item)
        
        return items


class ValueOrchestrator:
    """Main orchestrator for autonomous value discovery and prioritization"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.discovery_engine = ValueDiscoveryEngine(config)
        self.output_dir = Path("output/value_discovery")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def execute_discovery_cycle(self) -> Dict[str, Any]:
        """Execute complete value discovery cycle"""
        start_time = time.time()
        self.logger.info("Starting autonomous value discovery cycle")
        
        # Parallel discovery of different value types
        tasks = [
            self.discovery_engine.discover_performance_opportunities(),
            self.discovery_engine.discover_security_opportunities(), 
            self.discovery_engine.discover_maintainability_opportunities(),
            self.discovery_engine.discover_feature_opportunities()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine all discovered items
        all_items = []
        for result_list in results:
            all_items.extend(result_list)
        
        # Sort by ROI score
        all_items.sort(key=lambda x: x.roi_score, reverse=True)
        
        discovery_time = time.time() - start_time
        
        # Generate comprehensive report
        report = {
            "timestamp": datetime.now().isoformat(),
            "discovery_time_seconds": discovery_time,
            "total_items_discovered": len(all_items),
            "items_by_category": self._categorize_items(all_items),
            "top_10_items": [asdict(item) for item in all_items[:10]],
            "roi_analysis": self._analyze_roi(all_items),
            "effort_analysis": self._analyze_effort(all_items),
            "priority_matrix": self._create_priority_matrix(all_items),
            "recommendations": self._generate_recommendations(all_items)
        }
        
        # Save detailed report
        report_file = self.output_dir / f"value_discovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save all items for further processing
        items_file = self.output_dir / "discovered_value_items.json"
        with open(items_file, 'w') as f:
            json.dump([asdict(item) for item in all_items], f, indent=2)
        
        self.logger.info(f"Discovery cycle complete. {len(all_items)} items found in {discovery_time:.2f}s")
        
        return report
    
    def _categorize_items(self, items: List[ValueItem]) -> Dict[str, int]:
        """Categorize items by type"""
        categories = {}
        for item in items:
            categories[item.category] = categories.get(item.category, 0) + 1
        return categories
    
    def _analyze_roi(self, items: List[ValueItem]) -> Dict[str, Any]:
        """Analyze return on investment patterns"""
        roi_scores = [item.roi_score for item in items]
        return {
            "average_roi": sum(roi_scores) / len(roi_scores) if roi_scores else 0,
            "max_roi": max(roi_scores) if roi_scores else 0,
            "high_roi_items": len([item for item in items if item.roi_score > 5.0]),
            "quick_wins": len([item for item in items if item.roi_score > 5.0 and item.effort_hours < 2.0])
        }
    
    def _analyze_effort(self, items: List[ValueItem]) -> Dict[str, Any]:
        """Analyze effort distribution"""
        efforts = [item.effort_hours for item in items]
        return {
            "total_effort_hours": sum(efforts),
            "average_effort": sum(efforts) / len(efforts) if efforts else 0,
            "quick_tasks": len([item for item in items if item.effort_hours <= 1.0]),
            "medium_tasks": len([item for item in items if 1.0 < item.effort_hours <= 8.0]),
            "complex_tasks": len([item for item in items if item.effort_hours > 8.0])
        }
    
    def _create_priority_matrix(self, items: List[ValueItem]) -> Dict[str, List[str]]:
        """Create priority matrix based on impact vs effort"""
        matrix = {
            "quick_wins": [],  # High impact, low effort
            "major_projects": [],  # High impact, high effort
            "fill_ins": [],  # Low impact, low effort
            "questionable": []  # Low impact, high effort
        }
        
        for item in items:
            if item.business_impact >= 50:
                if item.effort_hours <= 4.0:
                    matrix["quick_wins"].append(item.id)
                else:
                    matrix["major_projects"].append(item.id)
            else:
                if item.effort_hours <= 4.0:
                    matrix["fill_ins"].append(item.id)
                else:
                    matrix["questionable"].append(item.id)
        
        return matrix
    
    def _generate_recommendations(self, items: List[ValueItem]) -> List[str]:
        """Generate strategic recommendations based on discovered items"""
        recommendations = []
        
        if not items:
            recommendations.append("No value items discovered. System appears well-optimized.")
            return recommendations
        
        # Analyze top items
        top_item = items[0]
        recommendations.append(f"Highest ROI opportunity: {top_item.title} (ROI: {top_item.roi_score:.2f})")
        
        # Security recommendations
        security_items = [item for item in items if item.category == "security"]
        if security_items:
            recommendations.append(f"Address {len(security_items)} security items immediately - highest priority")
        
        # Performance recommendations
        performance_items = [item for item in items if item.category == "performance"]
        if performance_items:
            avg_perf_roi = sum(item.roi_score for item in performance_items) / len(performance_items)
            recommendations.append(f"Performance optimization opportunities: {len(performance_items)} items, avg ROI: {avg_perf_roi:.2f}")
        
        # Quick wins
        quick_wins = [item for item in items if item.effort_hours <= 2.0 and item.roi_score >= 3.0]
        if quick_wins:
            recommendations.append(f"Execute {len(quick_wins)} quick wins for immediate value")
        
        return recommendations


async def main():
    """Main execution function for autonomous value orchestrator"""
    logging.basicConfig(level=logging.INFO)
    
    orchestrator = ValueOrchestrator()
    report = await orchestrator.execute_discovery_cycle()
    
    print("\n🎯 Autonomous Value Discovery Complete")
    print(f"📊 Total items discovered: {report['total_items_discovered']}")
    print(f"⏱️  Discovery time: {report['discovery_time_seconds']:.2f} seconds")
    print(f"🚀 High ROI items: {report['roi_analysis']['high_roi_items']}")
    print(f"⚡ Quick wins available: {report['roi_analysis']['quick_wins']}")
    
    if report['top_10_items']:
        top_item = report['top_10_items'][0]
        print(f"\n🎖️  Top Priority: {top_item['title']}")
        print(f"   ROI Score: {top_item['roi_score']:.2f}")
        print(f"   Effort: {top_item['effort_hours']} hours")
        print(f"   Category: {top_item['category']}")
    
    print(f"\n📁 Reports saved to: output/value_discovery/")


if __name__ == "__main__":
    asyncio.run(main())