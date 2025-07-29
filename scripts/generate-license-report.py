#!/usr/bin/env python3
"""
Generate comprehensive license report for all dependencies.
This script analyzes all project dependencies and generates reports
about their licenses for compliance purposes.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Any
import argparse


def run_command(cmd: List[str]) -> str:
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(cmd)}: {e}")
        return ""


def get_pip_licenses() -> List[Dict[str, Any]]:
    """Get licenses for pip packages."""
    try:
        output = run_command(["pip-licenses", "--format", "json", "--with-urls"])
        if output:
            return json.loads(output)
    except (json.JSONDecodeError, FileNotFoundError):
        print("Warning: pip-licenses not found or returned invalid JSON")
    return []


def analyze_licenses(licenses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze license compatibility and categorize them."""
    
    # License categories based on compatibility and restrictions
    permissive_licenses = {
        "MIT", "BSD", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Apache-2.0", 
        "Apache Software License", "Apache License", "Apache"
    }
    
    copyleft_weak = {
        "LGPL-2.1", "LGPL-3.0", "LGPL", "MPL-2.0", "Mozilla Public License 2.0"
    }
    
    copyleft_strong = {
        "GPL-2.0", "GPL-3.0", "GPL", "AGPL-3.0", "AGPL"
    }
    
    proprietary = {
        "Commercial", "Proprietary", "All Rights Reserved"
    }
    
    unknown = {"UNKNOWN", "Unknown", ""}
    
    categorized = {
        "permissive": [],
        "copyleft_weak": [],
        "copyleft_strong": [],
        "proprietary": [],
        "unknown": [],
        "other": []
    }
    
    license_counts = {}
    
    for pkg in licenses:
        license_name = pkg.get("License", "").strip()
        pkg_name = pkg.get("Name", "unknown")
        
        # Count licenses
        license_counts[license_name] = license_counts.get(license_name, 0) + 1
        
        # Categorize
        if any(perm in license_name for perm in permissive_licenses):
            categorized["permissive"].append(pkg)
        elif any(weak in license_name for weak in copyleft_weak):
            categorized["copyleft_weak"].append(pkg)
        elif any(strong in license_name for strong in copyleft_strong):
            categorized["copyleft_strong"].append(pkg)
        elif any(prop in license_name for prop in proprietary):
            categorized["proprietary"].append(pkg)
        elif license_name in unknown:
            categorized["unknown"].append(pkg)
        else:
            categorized["other"].append(pkg)
    
    return {
        "categorized": categorized,
        "license_counts": license_counts,
        "total_packages": len(licenses)
    }


def generate_html_report(analysis: Dict[str, Any], output_file: Path) -> None:
    """Generate an HTML license report."""
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>License Report - Observer Coordinator Insights</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .category {{ margin: 20px 0; }}
        .category h3 {{ color: #333; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
        .package {{ margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 3px; }}
        .package-name {{ font-weight: bold; }}
        .license {{ color: #666; font-style: italic; }}
        .summary {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; border-radius: 5px; }}
        .error {{ background: #f8d7da; border: 1px solid #f5c6cb; padding: 10px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>License Report</h1>
        <p><strong>Project:</strong> Observer Coordinator Insights</p>
        <p><strong>Generated:</strong> {analysis.get('timestamp', 'Unknown')}</p>
        <p><strong>Total Packages:</strong> {analysis['total_packages']}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <ul>
            <li><strong>Permissive licenses:</strong> {len(analysis['categorized']['permissive'])} packages</li>
            <li><strong>Weak copyleft:</strong> {len(analysis['categorized']['copyleft_weak'])} packages</li>
            <li><strong>Strong copyleft:</strong> {len(analysis['categorized']['copyleft_strong'])} packages</li>
            <li><strong>Proprietary:</strong> {len(analysis['categorized']['proprietary'])} packages</li>
            <li><strong>Unknown/Other:</strong> {len(analysis['categorized']['unknown']) + len(analysis['categorized']['other'])} packages</li>
        </ul>
    </div>
"""

    # Add warnings for problematic licenses
    if analysis['categorized']['copyleft_strong']:
        html_content += '''
    <div class="error">
        <h3>⚠️ Strong Copyleft Licenses Detected</h3>
        <p>This project contains dependencies with strong copyleft licenses (GPL). 
        This may require the entire project to be licensed under a compatible copyleft license.</p>
    </div>
'''

    if analysis['categorized']['unknown']:
        html_content += '''
    <div class="warning">
        <h3>⚠️ Unknown Licenses Detected</h3>
        <p>Some dependencies have unknown or unspecified licenses. 
        Please review these manually for compliance.</p>
    </div>
'''

    # License counts table
    html_content += '''
    <h2>License Distribution</h2>
    <table>
        <tr><th>License</th><th>Package Count</th></tr>
'''
    
    for license_name, count in sorted(analysis['license_counts'].items(), key=lambda x: x[1], reverse=True):
        html_content += f"<tr><td>{license_name or 'Unknown'}</td><td>{count}</td></tr>"
    
    html_content += "</table>"

    # Detailed breakdown by category
    for category_name, packages in analysis['categorized'].items():
        if packages:
            category_title = category_name.replace('_', ' ').title()
            html_content += f'''
    <div class="category">
        <h3>{category_title} ({len(packages)} packages)</h3>
'''
            for pkg in packages:
                html_content += f'''
        <div class="package">
            <span class="package-name">{pkg.get('Name', 'Unknown')}</span> 
            v{pkg.get('Version', 'Unknown')} - 
            <span class="license">{pkg.get('License', 'Unknown')}</span>
        </div>
'''
            html_content += "</div>"

    html_content += '''
    <div class="footer">
        <p><em>This report was generated automatically. Please review manually for accuracy.</em></p>
    </div>
</body>
</html>
'''

    output_file.write_text(html_content)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate license report for project dependencies")
    parser.add_argument("--output", "-o", type=Path, default="license-report.html",
                      help="Output file path (default: license-report.html)")
    parser.add_argument("--format", choices=["html", "json"], default="html",
                      help="Output format (default: html)")
    
    args = parser.parse_args()
    
    print("Collecting license information...")
    licenses = get_pip_licenses()
    
    if not licenses:
        print("No license information found. Make sure pip-licenses is installed:")
        print("pip install pip-licenses")
        sys.exit(1)
    
    print(f"Found {len(licenses)} packages")
    
    print("Analyzing licenses...")
    analysis = analyze_licenses(licenses)
    
    # Add timestamp
    from datetime import datetime
    analysis['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if args.format == "json":
        with open(args.output.with_suffix('.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"JSON report written to {args.output.with_suffix('.json')}")
    else:
        generate_html_report(analysis, args.output)
        print(f"HTML report written to {args.output}")
    
    # Print summary to console
    print("\nLicense Summary:")
    print(f"  Total packages: {analysis['total_packages']}")
    print(f"  Permissive: {len(analysis['categorized']['permissive'])}")
    print(f"  Weak copyleft: {len(analysis['categorized']['copyleft_weak'])}")
    print(f"  Strong copyleft: {len(analysis['categorized']['copyleft_strong'])}")
    print(f"  Proprietary: {len(analysis['categorized']['proprietary'])}")
    print(f"  Unknown/Other: {len(analysis['categorized']['unknown']) + len(analysis['categorized']['other'])}")
    
    # Exit with warning if problematic licenses found
    if analysis['categorized']['copyleft_strong'] or analysis['categorized']['proprietary']:
        print("\n⚠️  Warning: Potentially problematic licenses detected. Please review the report.")
        sys.exit(1)
    
    if analysis['categorized']['unknown']:
        print("\n⚠️  Warning: Unknown licenses detected. Please review the report.")
    
    print("\n✅ License analysis complete")


if __name__ == "__main__":
    main()