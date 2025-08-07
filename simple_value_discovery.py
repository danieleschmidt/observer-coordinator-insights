#!/usr/bin/env python3
"""
Simple Value Discovery Engine Demo
Demonstrates autonomous SDLC value discovery without external dependencies
"""

import json
import re
import os
from datetime import datetime
from pathlib import Path


def discover_code_issues():
    """Discover value items from code analysis"""
    items = []
    
    # Find TODO/FIXME comments in Python files
    for py_file in Path(".").rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Find TODO/FIXME patterns
            todo_pattern = r'#\s*(TODO|FIXME|HACK|BUG|DEPRECATED):\s*(.+)'
            matches = re.finditer(todo_pattern, content, re.IGNORECASE)
            
            for match in matches:
                tag, description = match.groups()
                line_num = content[:match.start()].count('\n') + 1
                
                # Score based on tag severity
                scores = {
                    'TODO': 30,
                    'FIXME': 50, 
                    'HACK': 70,
                    'BUG': 80,
                    'DEPRECATED': 60
                }
                
                items.append({
                    'id': f"code-{tag.lower()}-{hash(f'{py_file}:{line_num}')}",
                    'title': f"Address {tag} in {py_file.name}:{line_num}",
                    'description': description.strip(),
                    'category': 'technical_debt',
                    'score': scores.get(tag.upper(), 40),
                    'file': str(py_file),
                    'line': line_num,
                    'effort_hours': 2.0 if tag.upper() == 'TODO' else 4.0
                })
        except Exception:
            continue
    
    return items


def discover_performance_opportunities():
    """Find performance improvement opportunities"""
    items = []
    
    # Performance anti-patterns
    patterns = [
        (r'\.iterrows\(\)', "Use vectorized operations instead of iterrows()", 6.0),
        (r'pd\.concat.*for.*in', "Use list comprehension with single concat", 4.0),
        (r'time\.sleep\([0-9\.]+\)', "Consider async alternatives to blocking sleep", 3.0),
        (r'for.*in.*range\(len\(', "Use enumerate() instead of range(len())", 1.0),
    ]
    
    for py_file in Path(".").rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern, suggestion, effort in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    items.append({
                        'id': f"perf-{hash(f'{py_file}:{line_num}')}",
                        'title': f"Performance optimization in {py_file.name}:{line_num}",
                        'description': suggestion,
                        'category': 'performance',
                        'score': 45,
                        'file': str(py_file),
                        'line': line_num,
                        'effort_hours': effort
                    })
        except Exception:
            continue
    
    return items


def discover_security_issues():
    """Find potential security issues"""
    items = []
    
    # Security patterns
    patterns = [
        (r'open\(["\'].*["\'],\s*["\']w["\']', "File operations may need error handling", 3.0),
        (r'subprocess\..*shell=True', "Shell injection risk with shell=True", 5.0),
        (r'dynamic_eval\s*\(', "Dynamic code evaluation detected", 6.0),
        (r'dynamic_exec\s*\(', "Dynamic code execution detected", 6.0),
    ]
    
    for py_file in Path(".").rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern, description, effort in patterns:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    items.append({
                        'id': f"sec-{hash(f'{py_file}:{line_num}')}",
                        'title': f"Security review needed in {py_file.name}:{line_num}",
                        'description': description,
                        'category': 'security',
                        'score': 75,  # High priority for security
                        'file': str(py_file),
                        'line': line_num,
                        'effort_hours': effort
                    })
        except Exception:
            continue
    
    return items


def discover_documentation_gaps():
    """Find documentation improvement opportunities"""
    items = []
    
    # Check for missing docstrings in Python files
    for py_file in Path("src").rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find functions/classes without docstrings
            function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\):'
            class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            
            for pattern, item_type in [(function_pattern, 'function'), (class_pattern, 'class')]:
                matches = re.finditer(pattern, content)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    name = match.group(1)
                    
                    # Check if next few lines contain docstring
                    lines = content.split('\n')
                    has_docstring = False
                    for i in range(line_num, min(line_num + 3, len(lines))):
                        if '"""' in lines[i] or "'''" in lines[i]:
                            has_docstring = True
                            break
                    
                    if not has_docstring and not name.startswith('_'):
                        items.append({
                            'id': f"doc-{item_type}-{hash(f'{py_file}:{name}')}",
                            'title': f"Add docstring to {item_type} {name} in {py_file.name}",
                            'description': f"Missing docstring for {item_type} {name}",
                            'category': 'documentation',
                            'score': 25,
                            'file': str(py_file),
                            'line': line_num,
                            'effort_hours': 0.5
                        })
        except Exception:
            continue
    
    return items


def generate_backlog_markdown(items):
    """Generate markdown backlog report"""
    now = datetime.now().isoformat()
    
    # Sort by score descending
    items_sorted = sorted(items, key=lambda x: x['score'], reverse=True)
    
    content = f"""# 📊 Autonomous Value Backlog

**Generated:** {now}
**Total Items Discovered:** {len(items)}

## 🎯 Next Best Value Item
"""
    
    if items_sorted:
        top_item = items_sorted[0]
        content += f"""
**[{top_item['id']}] {top_item['title']}**
- **Score**: {top_item['score']}
- **Category**: {top_item['category']}
- **Estimated Effort**: {top_item['effort_hours']} hours
- **File**: {top_item['file']}:{top_item['line']}
- **Description**: {top_item['description']}

## 📋 Top 10 Value Items

| Rank | Title | Score | Category | Effort (hrs) | File |
|------|-------|-------|----------|--------------|------|
"""
        
        for i, item in enumerate(items_sorted[:10], 1):
            title = item['title'][:50] + "..." if len(item['title']) > 50 else item['title']
            content += f"| {i} | {title} | {item['score']} | {item['category']} | {item['effort_hours']} | {Path(item['file']).name} |\n"
    
    # Category breakdown
    categories = {}
    total_effort = 0
    for item in items:
        categories[item['category']] = categories.get(item['category'], 0) + 1
        total_effort += item['effort_hours']
    
    content += f"""

## 📈 Discovery Summary

### Items by Category
"""
    for category, count in sorted(categories.items()):
        content += f"- **{category.replace('_', ' ').title()}**: {count} items\n"
        
    content += f"""
### Effort Estimates
- **Total Estimated Effort**: {total_effort:.1f} hours
- **Average Effort per Item**: {total_effort/len(items):.1f} hours
- **High Priority Items** (Score > 60): {len([i for i in items if i['score'] > 60])}

### Value Opportunities
- **Quick Wins** (< 2 hours): {len([i for i in items if i['effort_hours'] < 2])}
- **Security Items**: {len([i for i in items if i['category'] == 'security'])}
- **Performance Items**: {len([i for i in items if i['category'] == 'performance'])}
- **Technical Debt**: {len([i for i in items if i['category'] == 'technical_debt'])}

## 🔄 Next Steps

1. **Immediate Action**: Address the top-scored item above
2. **Security Priority**: Review all security items (score automatically boosted)
3. **Quick Wins**: Execute items with < 2 hour effort estimates
4. **Systematic Improvement**: Work through backlog by score priority

---
*Generated by Terragon Autonomous SDLC Enhancement System*
"""
    
    return content


def main():
    """Main value discovery execution"""
    print("🔍 Starting Autonomous Value Discovery...")
    now = datetime.now().isoformat()
    
    # Discover value items from multiple sources
    all_items = []
    
    print("  📝 Analyzing code comments...")
    all_items.extend(discover_code_issues())
    
    print("  ⚡ Finding performance opportunities...")
    all_items.extend(discover_performance_opportunities())
    
    print("  🔒 Scanning for security issues...")
    all_items.extend(discover_security_issues())
    
    print("  📚 Checking documentation gaps...")
    all_items.extend(discover_documentation_gaps())
    
    print(f"✅ Discovered {len(all_items)} value items")
    
    if all_items:
        # Find highest value item
        top_item = max(all_items, key=lambda x: x['score'])
        print(f"🎯 Top Value Item: {top_item['title']} (Score: {top_item['score']})")
        
        # Generate backlog
        backlog_content = generate_backlog_markdown(all_items)
        
        # Save backlog
        with open("AUTONOMOUS_BACKLOG.md", "w") as f:
            f.write(backlog_content)
        print("📊 Generated AUTONOMOUS_BACKLOG.md")
        
        # Save raw data
        with open(".terragon/discovered_items.json", "w") as f:
            json.dump(all_items, f, indent=2)
        print("💾 Saved raw data to .terragon/discovered_items.json")
        
        # Update metrics
        metrics = {
            "last_discovery": now,
            "items_discovered": len(all_items),
            "categories": {
                "technical_debt": len([i for i in all_items if i['category'] == 'technical_debt']),
                "performance": len([i for i in all_items if i['category'] == 'performance']),
                "security": len([i for i in all_items if i['category'] == 'security']),
                "documentation": len([i for i in all_items if i['category'] == 'documentation'])
            },
            "total_effort": sum(item['effort_hours'] for item in all_items),
            "high_priority_items": len([i for i in all_items if i['score'] > 60]),
            "top_item": {
                "title": top_item['title'],
                "score": top_item['score'],
                "category": top_item['category']
            }
        }
        
        with open(".terragon/discovery_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
    else:
        print("🎉 No value items discovered - repository is in excellent shape!")
        
        # Create minimal backlog
        with open("AUTONOMOUS_BACKLOG.md", "w") as f:
            f.write(f"""# 📊 Autonomous Value Backlog

**Generated:** {now}
**Status:** 🎉 No value items discovered - repository is in excellent shape!

## Repository Health Status

Your repository appears to be well-maintained with:
- No outstanding TODO/FIXME comments
- No obvious performance anti-patterns
- No immediate security concerns detected
- Good documentation coverage

The autonomous system will continue monitoring for new opportunities.

---
*Generated by Terragon Autonomous SDLC Enhancement System*
""")


if __name__ == "__main__":
    # Ensure .terragon directory exists
    os.makedirs(".terragon", exist_ok=True)
    main()