#!/usr/bin/env python3
"""
Generate Software Bill of Materials (SBOM) for Observer Coordinator Insights.

This script creates an SBOM in SPDX format listing all dependencies,
their versions, licenses, and security information for compliance purposes.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import toml


def get_package_info() -> Dict[str, Any]:
    """Get package information from pyproject.toml."""
    try:
        with open("pyproject.toml", "r") as f:
            data = toml.load(f)
        
        project_info = data.get("project", {})
        return {
            "name": project_info.get("name", "observer-coordinator-insights"),
            "version": project_info.get("version", "0.1.0"),
            "description": project_info.get("description", ""),
            "authors": project_info.get("authors", []),
            "license": project_info.get("license", {}),
            "homepage": project_info.get("urls", {}).get("Homepage", ""),
            "repository": project_info.get("urls", {}).get("Repository", ""),
        }
    except Exception as e:
        print(f"Warning: Could not read pyproject.toml: {e}")
        return {
            "name": "observer-coordinator-insights",
            "version": "0.1.0",
            "description": "Multi-agent orchestration for organizational analytics",
            "authors": [{"name": "Terragon Labs"}],
            "license": {"text": "Apache-2.0"},
            "homepage": "",
            "repository": "",
        }


def get_installed_packages() -> List[Dict[str, Any]]:
    """Get list of installed packages with versions and licenses."""
    packages = []
    
    try:
        # Get package list with pip list
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        pip_packages = json.loads(result.stdout)
        
        # Get license information
        try:
            license_result = subprocess.run(
                [sys.executable, "-m", "pip_licenses", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            license_data = json.loads(license_result.stdout)
            license_map = {pkg["Name"].lower(): pkg["License"] for pkg in license_data}
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: pip-licenses not available, license info will be limited")
            license_map = {}
        
        for pkg in pip_packages:
            package_info = {
                "name": pkg["name"],
                "version": pkg["version"],
                "license": license_map.get(pkg["name"].lower(), "Unknown"),
                "type": "python-package"
            }
            
            # Try to get additional metadata
            try:
                show_result = subprocess.run(
                    [sys.executable, "-m", "pip", "show", pkg["name"]],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                metadata = {}
                for line in show_result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip().lower()] = value.strip()
                
                package_info.update({
                    "summary": metadata.get("summary", ""),
                    "homepage": metadata.get("home-page", ""),
                    "author": metadata.get("author", ""),
                    "author_email": metadata.get("author-email", ""),
                })
                
            except subprocess.CalledProcessError:
                pass
            
            packages.append(package_info)
            
    except subprocess.CalledProcessError as e:
        print(f"Error getting package list: {e}")
        return []
    
    return sorted(packages, key=lambda x: x["name"].lower())


def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        import platform
        return {
            "os": platform.system(),
            "os_version": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        }
    except Exception:
        return {
            "os": "Unknown",
            "os_version": "Unknown", 
            "architecture": "Unknown",
            "python_version": "Unknown",
            "python_implementation": "Unknown",
        }


def generate_spdx_sbom(package_info: Dict[str, Any], packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate SBOM in SPDX format."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    # SPDX document
    sbom = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"{package_info['name']}-{package_info['version']}-sbom",
        "documentNamespace": f"https://github.com/terragon-labs/{package_info['name']}/sbom-{timestamp}",
        "creationInfo": {
            "created": timestamp,
            "creators": ["Tool: generate-sbom.py"],
            "licenseListVersion": "3.19"
        },
        "packages": [],
        "relationships": []
    }
    
    # Add main package
    main_package = {
        "SPDXID": "SPDXRef-Package-Root",
        "name": package_info["name"],
        "downloadLocation": package_info.get("repository", "NOASSERTION"),
        "filesAnalyzed": False,
        "homepage": package_info.get("homepage", "NOASSERTION"),
        "licenseConcluded": package_info.get("license", {}).get("text", "NOASSERTION"),
        "licenseDeclared": package_info.get("license", {}).get("text", "NOASSERTION"),
        "copyrightText": "NOASSERTION",
        "versionInfo": package_info["version"],
        "supplier": f"Organization: {package_info.get('authors', [{}])[0].get('name', 'Unknown')}",
        "description": package_info.get("description", "")
    }
    sbom["packages"].append(main_package)
    
    # Add dependencies
    for i, pkg in enumerate(packages):
        dep_id = f"SPDXRef-Package-{pkg['name'].replace('-', '').replace('.', '').replace('_', '')}-{i}"
        
        dependency = {
            "SPDXID": dep_id,
            "name": pkg["name"],
            "downloadLocation": "NOASSERTION",
            "filesAnalyzed": False,
            "homepage": pkg.get("homepage", "NOASSERTION"),
            "licenseConcluded": pkg.get("license", "NOASSERTION"),
            "licenseDeclared": pkg.get("license", "NOASSERTION"), 
            "copyrightText": "NOASSERTION",
            "versionInfo": pkg["version"],
            "supplier": f"Person: {pkg.get('author', 'Unknown')}",
            "description": pkg.get("summary", "")
        }
        sbom["packages"].append(dependency)
        
        # Add relationship
        sbom["relationships"].append({
            "spdxElementId": "SPDXRef-Package-Root",
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": dep_id
        })
    
    return sbom


def generate_cyclonedx_sbom(package_info: Dict[str, Any], packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate SBOM in CycloneDX format."""
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.4",
        "serialNumber": f"urn:uuid:{package_info['name']}-{timestamp}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "Terragon Labs",
                    "name": "generate-sbom.py",
                    "version": "1.0.0"
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": f"{package_info['name']}@{package_info['version']}",
                "name": package_info["name"],
                "version": package_info["version"],
                "description": package_info.get("description", ""),
                "licenses": [
                    {
                        "license": {
                            "id": package_info.get("license", {}).get("text", "Apache-2.0")
                        }
                    }
                ]
            }
        },
        "components": []
    }
    
    # Add dependencies
    for pkg in packages:
        component = {
            "type": "library",
            "bom-ref": f"{pkg['name']}@{pkg['version']}",
            "name": pkg["name"],
            "version": pkg["version"],
            "description": pkg.get("summary", ""),
            "licenses": [
                {
                    "license": {
                        "name": pkg.get("license", "Unknown")
                    }
                }
            ],
            "purl": f"pkg:pypi/{pkg['name']}@{pkg['version']}"
        }
        
        if pkg.get("homepage"):
            component["externalReferences"] = [
                {
                    "type": "website",
                    "url": pkg["homepage"]
                }
            ]
        
        sbom["components"].append(component)
    
    return sbom


def save_sbom(sbom: Dict[str, Any], filename: str) -> None:
    """Save SBOM to file."""
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(sbom, f, indent=2, sort_keys=True)
    
    print(f"SBOM saved to {filepath}")


def generate_vulnerability_report(packages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate vulnerability report using safety."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "safety", "check", "--json"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {"vulnerabilities": [], "status": "clean"}
        else:
            try:
                vulnerabilities = json.loads(result.stdout)
                return {
                    "vulnerabilities": vulnerabilities,
                    "status": "vulnerabilities_found",
                    "count": len(vulnerabilities)
                }
            except json.JSONDecodeError:
                return {
                    "vulnerabilities": [],
                    "status": "error",
                    "error": result.stderr
                }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "vulnerabilities": [],
            "status": "tool_not_available",
            "error": "Safety tool not installed"
        }


def main():
    """Main function to generate SBOM."""
    print("Generating Software Bill of Materials (SBOM)...")
    
    # Get package and system information
    package_info = get_package_info()
    packages = get_installed_packages()
    system_info = get_system_info()
    
    print(f"Found {len(packages)} packages")
    
    # Generate SPDX SBOM
    spdx_sbom = generate_spdx_sbom(package_info, packages)
    save_sbom(spdx_sbom, "sbom-spdx.json")
    
    # Generate CycloneDX SBOM 
    cyclonedx_sbom = generate_cyclonedx_sbom(package_info, packages)
    save_sbom(cyclonedx_sbom, "sbom-cyclonedx.json")
    
    # Generate vulnerability report
    print("Checking for vulnerabilities...")
    vuln_report = generate_vulnerability_report(packages)
    
    # Generate summary report
    summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "package_info": package_info,
        "system_info": system_info,
        "dependency_count": len(packages),
        "vulnerability_status": vuln_report["status"],
        "vulnerability_count": vuln_report.get("count", 0),
        "formats_generated": ["SPDX", "CycloneDX"],
        "packages": [
            {
                "name": pkg["name"],
                "version": pkg["version"],
                "license": pkg.get("license", "Unknown")
            }
            for pkg in packages
        ]
    }
    
    if vuln_report["status"] == "vulnerabilities_found":
        summary["vulnerabilities"] = vuln_report["vulnerabilities"]
    
    save_sbom(summary, "sbom-summary.json")
    
    print("\n📋 SBOM Generation Complete!")
    print(f"   • Total packages: {len(packages)}")
    print(f"   • Vulnerability status: {vuln_report['status']}")
    print(f"   • Files generated in reports/ directory")
    
    if vuln_report["status"] == "vulnerabilities_found":
        print(f"   ⚠️  {vuln_report['count']} vulnerabilities found!")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n❌ SBOM generation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating SBOM: {e}")
        sys.exit(1)