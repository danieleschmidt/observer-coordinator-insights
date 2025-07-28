# Workflow Requirements

## Overview

This document outlines the workflow requirements and setup process for automated CI/CD pipelines.

## Required GitHub Actions

### Core Workflows
* **CI Pipeline** (`ci.yml`) - Testing, linting, and security checks
* **Release Automation** (`release.yml`) - Automated package publishing
* **Security Scanning** (`sbom-diff.yml`) - SBOM generation and vulnerability tracking
* **PR Management** (`auto-rebase.yml`) - Automated PR maintenance

### Workflow Templates
Templates are available in `docs/github-workflows/` and must be manually copied to `.github/workflows/`.

## Prerequisites

### Repository Setup
* Branch protection rules configured
* Required secrets added to repository settings
* Security features enabled (Dependabot, code scanning)

### Manual Installation
```bash
# Copy workflow files (requires admin access)
cp docs/github-workflows/*.yml .github/workflows/
```

## Security Requirements

### Required Secrets
* `PYPI_TOKEN` - Package publishing authentication
* Additional secrets per individual workflow documentation

### Branch Protection
* Minimum 1 reviewer required
* Status checks must pass before merge
* Administrator bypass restrictions

## Resources

* [GitHub Actions Documentation](https://docs.github.com/en/actions)
* [Workflow Syntax Reference](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
* [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
* Setup guide: `../SETUP_REQUIRED.md`

## Manual Setup Required

Due to permission limitations, workflow files must be manually installed by repository administrators.