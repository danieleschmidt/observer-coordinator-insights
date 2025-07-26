# GitHub Actions Workflows

Due to GitHub App permission restrictions, the CI/CD workflows need to be manually created. This directory contains all the workflow templates that should be placed in `.github/workflows/`.

## Required Workflows

1. **ci.yml** - Main CI pipeline with testing, linting, and security scanning
2. **release.yml** - Automated releases with semantic versioning and signing
3. **sbom-diff.yml** - Weekly vulnerability scanning and SBOM monitoring
4. **auto-rebase.yml** - Automatic PR rebasing for conflict resolution

## Setup Instructions

1. Create the `.github/workflows/` directory in your repository
2. Copy the workflow files from this directory to `.github/workflows/`
3. Ensure the following secrets are configured in your repository settings:
   - `PYPI_TOKEN` (for PyPI publishing)
   - `GITHUB_TOKEN` (automatically provided by GitHub)

## Workflow Features

### CI Pipeline (ci.yml)
- Multi-Python version testing (3.9, 3.10, 3.11)
- Automated linting with ruff
- Type checking with mypy
- Security scanning with CodeQL, Safety, and Bandit
- OWASP dependency checking
- Coverage reporting
- SBOM generation

### Release Pipeline (release.yml)
- Semantic versioning
- Sigstore artifact signing
- PyPI publishing
- Container builds with multi-arch support
- GitHub releases with artifacts

### Security Monitoring (sbom-diff.yml)
- Weekly SBOM vulnerability scanning
- Automated issue creation for critical vulnerabilities
- Dependency drift monitoring

### Auto-rebase (auto-rebase.yml)
- Automatic PR rebasing onto target branch
- Smart conflict resolution using git rerere
- Automatic labeling and commenting

## Permissions Required

The GitHub App needs the following permissions to run these workflows:
- `actions: read`
- `contents: write`
- `security-events: write`
- `pull-requests: write`
- `issues: write`