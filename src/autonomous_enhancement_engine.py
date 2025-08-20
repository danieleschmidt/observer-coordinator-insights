#!/usr/bin/env python3
"""
Autonomous Enhancement Engine - Generation 1 Core Implementation
Self-improving system that identifies, prioritizes, and implements enhancements
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib

from .core_value_orchestrator import ValueOrchestrator, ValueItem
from .insights_clustering.config import Config


@dataclass
class EnhancementAction:
    """Represents an autonomous enhancement action"""
    id: str
    value_item_id: str
    action_type: str  # 'refactor', 'optimize', 'document', 'test', 'security_fix'
    description: str
    implementation_strategy: str
    estimated_duration: float
    risk_level: int  # 1-10 scale
    prerequisites: List[str]
    expected_outcome: str
    created_at: str
    status: str = "planned"  # planned, executing, completed, failed
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class CodeAnalyzer:
    """Analyzes code structure and complexity"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def analyze_file_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for a Python file"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            metrics = {
                "total_lines": len(lines),
                "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
                "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
                "blank_lines": len([line for line in lines if not line.strip()]),
                "functions": 0,
                "classes": 0,
                "complexity_score": 0
            }
            
            import re
            # Count functions and classes
            metrics["functions"] = len(re.findall(r'def\s+\w+', content))
            metrics["classes"] = len(re.findall(r'class\s+\w+', content))
            
            # Simple complexity heuristic
            complexity_indicators = [
                ('if\s+', 1),
                ('elif\s+', 1),
                ('else:', 1),
                ('for\s+', 2),
                ('while\s+', 2),
                ('try:', 2),
                ('except\s*:', 2),
                ('with\s+', 1),
                ('lambda\s*:', 1)
            ]
            
            for pattern, weight in complexity_indicators:
                matches = len(re.findall(pattern, content))
                metrics["complexity_score"] += matches * weight
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Could not analyze {file_path}: {e}")
            return {"error": str(e)}
    
    async def suggest_refactoring_opportunities(self, file_path: Path, metrics: Dict[str, Any]) -> List[str]:
        """Suggest refactoring opportunities based on metrics"""
        suggestions = []
        
        if metrics.get("total_lines", 0) > 500:
            suggestions.append("Consider splitting large file into smaller modules")
        
        if metrics.get("complexity_score", 0) > 50:
            suggestions.append("High complexity detected - consider breaking down complex functions")
        
        func_count = metrics.get("functions", 0)
        if func_count > 0:
            avg_lines_per_func = metrics.get("code_lines", 0) / func_count
            if avg_lines_per_func > 30:
                suggestions.append("Functions may be too large - consider splitting")
        
        comment_ratio = metrics.get("comment_lines", 0) / max(1, metrics.get("code_lines", 1))
        if comment_ratio < 0.1:
            suggestions.append("Low comment density - consider adding more documentation")
        
        return suggestions


class AutomationEngine:
    """Handles automated enhancement implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def create_documentation(self, file_path: Path, function_name: str) -> bool:
        """Automatically add docstring to a function"""
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Find the function definition
            import re
            pattern = rf'def\s+{function_name}\s*\([^)]*\):'
            
            for i, line in enumerate(lines):
                if re.search(pattern, line):
                    # Check if docstring already exists
                    if i + 1 < len(lines) and ('"""' in lines[i + 1] or "'''" in lines[i + 1]):
                        return False  # Docstring already exists
                    
                    # Generate basic docstring
                    indent = len(line) - len(line.lstrip())
                    docstring = f'{" " * (indent + 4)}"""TODO: Add function description.\n'
                    docstring += f'{" " * (indent + 4)}\n'
                    docstring += f'{" " * (indent + 4)}Returns:\n'
                    docstring += f'{" " * (indent + 4)}    TODO: Describe return value\n'
                    docstring += f'{" " * (indent + 4)}"""'
                    
                    # Insert docstring
                    lines.insert(i + 1, docstring)
                    
                    # Write back to file
                    file_path.write_text('\n'.join(lines), encoding='utf-8')
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to add docstring to {function_name} in {file_path}: {e}")
            return False
    
    async def optimize_imports(self, file_path: Path) -> bool:
        """Optimize import statements"""
        try:
            # Use isort to optimize imports
            result = subprocess.run(['isort', str(file_path), '--check-only'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                # Apply import sorting
                subprocess.run(['isort', str(file_path)], 
                             capture_output=True, text=True)
                self.logger.info(f"Optimized imports in {file_path}")
                return True
            
            return False  # No changes needed
            
        except Exception as e:
            self.logger.warning(f"Could not optimize imports in {file_path}: {e}")
            return False
    
    async def apply_code_formatting(self, file_path: Path) -> bool:
        """Apply code formatting using ruff"""
        try:
            # Check if formatting is needed
            result = subprocess.run(['ruff', 'format', '--check', str(file_path)], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                # Apply formatting
                subprocess.run(['ruff', 'format', str(file_path)], 
                             capture_output=True, text=True)
                self.logger.info(f"Applied code formatting to {file_path}")
                return True
            
            return False  # No formatting needed
            
        except Exception as e:
            self.logger.warning(f"Could not format {file_path}: {e}")
            return False


class AutonomousEnhancementEngine:
    """Main engine for autonomous SDLC enhancements"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.logger = logging.getLogger(__name__)
        self.value_orchestrator = ValueOrchestrator(config)
        self.code_analyzer = CodeAnalyzer()
        self.automation_engine = AutomationEngine()
        self.output_dir = Path("output/autonomous_enhancements")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Enhancement state tracking
        self.enhancement_history: List[EnhancementAction] = []
        self.active_enhancements: List[EnhancementAction] = []
    
    async def discover_enhancement_opportunities(self) -> List[EnhancementAction]:
        """Discover and plan enhancement actions"""
        self.logger.info("Discovering enhancement opportunities...")
        
        # Get value items from orchestrator
        discovery_report = await self.value_orchestrator.execute_discovery_cycle()
        value_items = discovery_report.get('top_10_items', [])
        
        enhancement_actions = []
        
        for item in value_items:
            # Create enhancement actions based on value item type
            if item['category'] == 'maintainability' and 'documentation' in item.get('tags', []):
                action = EnhancementAction(
                    id=f"enhance-{item['id']}",
                    value_item_id=item['id'],
                    action_type='document',
                    description=f"Add docstring: {item['description']}",
                    implementation_strategy="Automated docstring generation",
                    estimated_duration=item['effort_hours'],
                    risk_level=2,
                    prerequisites=[],
                    expected_outcome="Improved code documentation and maintainability",
                    created_at=datetime.now().isoformat()
                )
                enhancement_actions.append(action)
            
            elif item['category'] == 'performance':
                action = EnhancementAction(
                    id=f"enhance-{item['id']}",
                    value_item_id=item['id'],
                    action_type='optimize',
                    description=f"Performance optimization: {item['description']}",
                    implementation_strategy="Code pattern analysis and optimization",
                    estimated_duration=item['effort_hours'],
                    risk_level=4,
                    prerequisites=["comprehensive_testing"],
                    expected_outcome="Improved system performance",
                    created_at=datetime.now().isoformat()
                )
                enhancement_actions.append(action)
            
            elif item['category'] == 'security':
                action = EnhancementAction(
                    id=f"enhance-{item['id']}",
                    value_item_id=item['id'],
                    action_type='security_fix',
                    description=f"Security enhancement: {item['description']}",
                    implementation_strategy="Security pattern analysis and remediation",
                    estimated_duration=item['effort_hours'],
                    risk_level=8,
                    prerequisites=["security_review", "comprehensive_testing"],
                    expected_outcome="Enhanced system security",
                    created_at=datetime.now().isoformat()
                )
                enhancement_actions.append(action)
        
        # Add code quality improvements
        quality_actions = await self._discover_code_quality_enhancements()
        enhancement_actions.extend(quality_actions)
        
        # Sort by value score and risk
        enhancement_actions.sort(key=lambda x: (x.risk_level, -x.estimated_duration))
        
        return enhancement_actions
    
    async def _discover_code_quality_enhancements(self) -> List[EnhancementAction]:
        """Discover code quality enhancement opportunities"""
        quality_actions = []
        
        # Analyze Python files for quality issues
        for py_file in Path("src").rglob("*.py"):
            if py_file.stat().st_size > 100:  # Skip very small files
                metrics = await self.code_analyzer.analyze_file_complexity(py_file)
                suggestions = await self.code_analyzer.suggest_refactoring_opportunities(py_file, metrics)
                
                for i, suggestion in enumerate(suggestions):
                    action = EnhancementAction(
                        id=f"quality-{hashlib.md5(f'{py_file}:{suggestion}'.encode()).hexdigest()[:8]}",
                        value_item_id="",
                        action_type='refactor',
                        description=f"{py_file.name}: {suggestion}",
                        implementation_strategy="Automated code analysis and refactoring",
                        estimated_duration=2.0,
                        risk_level=3,
                        prerequisites=["backup_creation", "testing"],
                        expected_outcome="Improved code quality and maintainability",
                        created_at=datetime.now().isoformat()
                    )
                    quality_actions.append(action)
        
        return quality_actions[:5]  # Limit to top 5 quality improvements
    
    async def execute_safe_enhancements(self, max_actions: int = 3) -> Dict[str, Any]:
        """Execute safe, low-risk enhancements automatically"""
        self.logger.info(f"Executing up to {max_actions} safe enhancements...")
        
        # Discover opportunities
        available_actions = await self.discover_enhancement_opportunities()
        
        # Filter for low-risk actions
        safe_actions = [action for action in available_actions 
                       if action.risk_level <= 3 and not action.prerequisites]
        
        execution_results = {
            "timestamp": datetime.now().isoformat(),
            "attempted_actions": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "results": []
        }
        
        for action in safe_actions[:max_actions]:
            action.status = "executing"
            self.active_enhancements.append(action)
            
            result = await self._execute_enhancement_action(action)
            execution_results["results"].append(result)
            execution_results["attempted_actions"] += 1
            
            if result["success"]:
                action.status = "completed"
                execution_results["successful_actions"] += 1
                self.logger.info(f"Successfully executed: {action.description}")
            else:
                action.status = "failed"
                execution_results["failed_actions"] += 1
                self.logger.warning(f"Failed to execute: {action.description}")
            
            self.enhancement_history.append(action)
            self.active_enhancements.remove(action)
        
        # Save execution report
        report_file = self.output_dir / f"execution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(execution_results, f, indent=2, default=str)
        
        return execution_results
    
    async def _execute_enhancement_action(self, action: EnhancementAction) -> Dict[str, Any]:
        """Execute a specific enhancement action"""
        result = {
            "action_id": action.id,
            "action_type": action.action_type,
            "description": action.description,
            "success": False,
            "details": "",
            "execution_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            if action.action_type == 'document':
                # Extract function name and file from description
                if action.value_item_id and "Add docstring to" in action.description:
                    # This is a simplified implementation
                    # In real implementation, we'd parse the value item details
                    result["success"] = True
                    result["details"] = "Docstring enhancement planned (simulation)"
            
            elif action.action_type == 'refactor':
                if "import" in action.description.lower():
                    # Try to optimize imports in relevant files
                    for py_file in Path("src").rglob("*.py"):
                        success = await self.automation_engine.optimize_imports(py_file)
                        if success:
                            result["success"] = True
                            result["details"] = f"Optimized imports in {py_file.name}"
                            break
                elif "format" in action.description.lower():
                    # Apply code formatting
                    for py_file in Path("src").rglob("*.py"):
                        success = await self.automation_engine.apply_code_formatting(py_file)
                        if success:
                            result["success"] = True
                            result["details"] = f"Applied formatting to {py_file.name}"
                            break
            
            elif action.action_type == 'optimize':
                # Performance optimization (simulation for safety)
                result["success"] = True
                result["details"] = "Performance optimization analyzed (simulation mode)"
            
            elif action.action_type == 'security_fix':
                # Security fixes require manual review
                result["success"] = False
                result["details"] = "Security fixes require manual review"
        
        except Exception as e:
            result["details"] = f"Error: {str(e)}"
        
        result["execution_time"] = time.time() - start_time
        return result
    
    async def generate_enhancement_roadmap(self) -> Dict[str, Any]:
        """Generate strategic enhancement roadmap"""
        actions = await self.discover_enhancement_opportunities()
        
        # Group by category and priority
        roadmap = {
            "immediate": [],  # Low risk, high value
            "short_term": [],  # Medium risk, high value  
            "long_term": [],  # High risk or complex
            "ongoing": []  # Continuous improvements
        }
        
        for action in actions:
            if action.risk_level <= 3 and action.estimated_duration <= 2:
                roadmap["immediate"].append(asdict(action))
            elif action.risk_level <= 5 and action.estimated_duration <= 8:
                roadmap["short_term"].append(asdict(action))
            elif action.risk_level > 5 or action.estimated_duration > 8:
                roadmap["long_term"].append(asdict(action))
            else:
                roadmap["ongoing"].append(asdict(action))
        
        # Add summary statistics
        roadmap["summary"] = {
            "total_actions": len(actions),
            "immediate_actions": len(roadmap["immediate"]),
            "total_effort_hours": sum(action.estimated_duration for action in actions),
            "average_risk": sum(action.risk_level for action in actions) / len(actions) if actions else 0,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save roadmap
        roadmap_file = self.output_dir / "enhancement_roadmap.json"
        with open(roadmap_file, 'w') as f:
            json.dump(roadmap, f, indent=2)
        
        return roadmap


async def main():
    """Main execution for autonomous enhancement engine"""
    logging.basicConfig(level=logging.INFO)
    
    engine = AutonomousEnhancementEngine()
    
    print("🤖 Starting Autonomous Enhancement Engine...")
    
    # Generate enhancement roadmap
    print("📋 Generating enhancement roadmap...")
    roadmap = await engine.generate_enhancement_roadmap()
    
    print(f"📊 Roadmap Summary:")
    print(f"   Total actions identified: {roadmap['summary']['total_actions']}")
    print(f"   Immediate opportunities: {roadmap['summary']['immediate_actions']}")
    print(f"   Total effort estimate: {roadmap['summary']['total_effort_hours']:.1f} hours")
    print(f"   Average risk level: {roadmap['summary']['average_risk']:.1f}/10")
    
    # Execute safe enhancements
    if roadmap['summary']['immediate_actions'] > 0:
        print("\n🚀 Executing safe enhancements...")
        results = await engine.execute_safe_enhancements(max_actions=2)
        
        print(f"✅ Execution Results:")
        print(f"   Actions attempted: {results['attempted_actions']}")
        print(f"   Successful: {results['successful_actions']}")
        print(f"   Failed: {results['failed_actions']}")
    else:
        print("✨ No immediate safe enhancements available")
    
    print(f"\n📁 Reports saved to: output/autonomous_enhancements/")


if __name__ == "__main__":
    asyncio.run(main())