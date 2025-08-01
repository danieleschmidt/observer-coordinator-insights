# CI/CD Setup Guide

This guide provides step-by-step instructions for setting up the complete CI/CD pipeline for the Observer Coordinator Insights project.

## Prerequisites

- Repository owner/admin access
- GitHub App permissions configured
- Required secrets configured
- Branch protection rules set up

## Quick Setup

### 1. Create Workflow Directory Structure

```bash
# In your repository root
mkdir -p .github/workflows
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/PULL_REQUEST_TEMPLATE
```

### 2. Copy Workflow Files

Copy all workflow files from `docs/github-workflows/` to `.github/workflows/`:

```bash
cp docs/github-workflows/*.yml .github/workflows/
```

### 3. Configure Repository Secrets

Go to **Settings → Secrets and Variables → Actions** and add:

#### Required Secrets
- `PYPI_TOKEN`: PyPI API token for package publishing
- `DOCKER_USERNAME`: Docker Hub username  
- `DOCKER_PASSWORD`: Docker Hub password/token
- `SLACK_WEBHOOK`: Slack webhook URL for notifications (optional)

#### Optional Secrets
- `SONAR_TOKEN`: SonarCloud token for code quality analysis
- `CODECOV_TOKEN`: Codecov token for coverage reporting
- `SNYK_TOKEN`: Snyk token for vulnerability scanning

### 4. Configure Repository Variables

Add repository variables for configuration:

- `PYTHON_VERSIONS`: JSON array of Python versions to test (default: `["3.9", "3.10", "3.11"]`)
- `REGISTRY_URL`: Container registry URL (default: `docker.io`)
- `ENABLE_SECURITY_SCANNING`: Enable security scans (default: `true`)

## Detailed Configuration

### Branch Protection Rules

Configure branch protection for `main` branch:

1. Go to **Settings → Branches**
2. Add rule for `main` branch
3. Configure the following:

```yaml
Required status checks:
  - test (3.9)
  - test (3.10)  
  - test (3.11)
  - lint
  - typecheck
  - security-scan
  - build

Additional settings:
  - Require branches to be up to date: ✓
  - Require review from code owners: ✓
  - Dismiss stale reviews: ✓
  - Require status checks to pass: ✓
  - Restrict pushes to matching branches: ✓
  - Allow force pushes: ✗
  - Allow deletions: ✗
```

### Repository Settings

#### General Settings
- **Allow merge commits**: ✓
- **Allow squash merging**: ✓
- **Allow rebase merging**: ✓
- **Automatically delete head branches**: ✓

#### Security Settings
- **Enable vulnerability alerts**: ✓
- **Enable dependency graph**: ✓
- **Enable Dependabot alerts**: ✓
- **Enable Dependabot security updates**: ✓
- **Enable secret scanning**: ✓
- **Enable push protection**: ✓

### Workflow Permissions

Each workflow requires specific permissions:

#### CI Workflow (ci.yml)
```yaml
permissions:
  contents: read
  security-events: write
  actions: read
  checks: write
  pull-requests: write
```

#### Release Workflow (release.yml)
```yaml  
permissions:
  contents: write
  packages: write
  pull-requests: read
  actions: read
```

#### Security Workflow (security-enhanced.yml)
```yaml
permissions:
  contents: read
  security-events: write
  actions: read
```

## Workflow Overview

### 1. Continuous Integration (ci.yml)

**Triggers**: Push to `main`, PRs, daily schedule
**Duration**: ~5-10 minutes

**Stages**:
1. **Setup** (1 min)
   - Checkout code
   - Setup Python matrix
   - Cache dependencies

2. **Quality** (2-3 min)
   - Lint with ruff
   - Type check with mypy
   - Format check

3. **Security** (1-2 min)
   - CodeQL analysis
   - Bandit security scan
   - Safety vulnerability check
   - OWASP dependency check

4. **Testing** (3-5 min)
   - Unit tests
   - Integration tests
   - Coverage reporting
   - Performance tests

5. **Build** (1-2 min)
   - Package build
   - Docker image build
   - SBOM generation

### 2. Release Pipeline (release.yml)

**Triggers**: Push to `main` (with semantic commits)
**Duration**: ~10-15 minutes

**Stages**:
1. **Preparation** (1 min)
   - Semantic release analysis
   - Version calculation
   - Changelog generation

2. **Build** (3-5 min)
   - Multi-arch Docker builds
   - Python package build
   - Documentation build

3. **Security** (2-3 min)
   - Image vulnerability scan
   - SBOM generation
   - Artifact signing with Sigstore

4. **Publish** (3-5 min)
   - PyPI publishing
   - Docker registry push
   - GitHub release creation

5. **Notify** (1 min)
   - Slack notifications
   - Status updates

### 3. Security Monitoring (security-enhanced.yml)

**Triggers**: Weekly schedule, security PRs
**Duration**: ~3-5 minutes

**Stages**:
1. **Dependency Scan**
   - OWASP dependency check
   - Snyk vulnerability scan
   - License compliance check

2. **Container Scan**
   - Image vulnerability assessment
   - Configuration security check
   - Runtime security analysis

3. **Code Scan**
   - CodeQL deep analysis
   - Semgrep security rules
   - Custom security policies

4. **Reporting**
   - Security dashboard update
   - Issue creation for findings
   - Compliance reporting

## Monitoring and Alerts

### Success Metrics

Monitor these key metrics for CI/CD health:

- **Build Success Rate**: > 95%
- **Average Build Time**: < 10 minutes
- **Test Coverage**: > 90%
- **Security Scan Pass Rate**: 100%
- **Deployment Success Rate**: > 98%

### Alert Configuration

Set up alerts for:

- **Failed builds on main branch**
- **Security vulnerabilities found**
- **Long-running builds (> 15 minutes)**
- **Test coverage drops below threshold**
- **Deployment failures**

### Dashboard Setup

Create monitoring dashboards tracking:

1. **Build Metrics**
   - Success/failure rates
   - Build duration trends
   - Queue times

2. **Quality Metrics**
   - Test coverage over time
   - Code quality scores
   - Technical debt trends

3. **Security Metrics**
   - Vulnerability counts
   - Security scan results
   - Compliance status

## Troubleshooting

### Common Issues

#### 1. Build Failures

**Symptom**: CI build fails consistently
**Investigation**:
```bash
# Check recent workflow runs
gh run list --limit 10

# View specific run logs
gh run view [RUN_ID]

# Check for common issues:
# - Dependency conflicts
# - Environment variable issues
# - Test failures
# - Resource limitations
```

**Solutions**:
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Check environment variables in workflow
- Review test failures and fix
- Increase runner resources if needed

#### 2. Security Scan Failures

**Symptom**: Security scans block PRs
**Investigation**:
```bash
# Check security scan results
gh api repos/{owner}/{repo}/code-scanning/alerts

# Review vulnerability details
# Check if false positive
# Assess actual risk level
```

**Solutions**:
- Update vulnerable dependencies
- Add exceptions for false positives
- Implement security fixes
- Document risk acceptance

#### 3. Slow Builds

**Symptom**: Builds take > 15 minutes
**Investigation**:
```bash
# Analyze build logs for bottlenecks
# Check test execution times
# Review dependency installation time
# Monitor resource usage
```

**Solutions**:
- Optimize test suite (parallel execution)
- Improve caching strategy
- Reduce dependency installation time
- Use matrix builds efficiently

#### 4. Deployment Failures

**Symptom**: Releases fail to deploy
**Investigation**:
```bash
# Check deployment logs
# Verify credentials and permissions
# Test deployment locally
# Check environment configuration
```

**Solutions**:
- Update deployment credentials
- Fix configuration issues
- Test deployment process locally
- Improve error handling

### Performance Optimization

#### Caching Strategy

```yaml
# Optimize dependency caching
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-
```

#### Parallel Execution

```yaml
# Use matrix builds for parallel testing
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11"]
    test-type: [unit, integration, performance]
  fail-fast: false
```

#### Resource Management

```yaml
# Optimize resource usage
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    strategy:
      max-parallel: 3
```

## Best Practices

### 1. Workflow Design

- **Keep workflows focused**: One workflow per concern
- **Use matrix builds**: Test across multiple environments
- **Implement proper caching**: Speed up builds
- **Set appropriate timeouts**: Prevent hanging jobs
- **Use secrets properly**: Never log sensitive data

### 2. Security

- **Principle of least privilege**: Minimal permissions
- **Validate inputs**: Prevent injection attacks
- **Use official actions**: Trusted marketplace actions
- **Pin action versions**: Use specific SHA or tags
- **Regular security reviews**: Audit workflows quarterly

### 3. Maintenance

- **Regular updates**: Keep actions and dependencies current
- **Monitor performance**: Track build times and success rates
- **Documentation**: Keep setup guides current
- **Testing**: Test workflow changes in feature branches
- **Backup**: Version control all configurations

## Migration Guide

### From Jenkins

1. **Analyze existing pipeline**
   - Document current stages
   - Identify dependencies
   - Note custom scripts

2. **Map to GitHub Actions**
   - Convert stages to jobs
   - Migrate environment variables
   - Replace plugins with actions

3. **Test migration**
   - Run parallel pipelines
   - Compare results
   - Validate artifacts

### From GitLab CI

1. **Convert .gitlab-ci.yml**
   - Map stages to jobs
   - Convert before_script/after_script
   - Migrate cache configuration

2. **Update deployment**
   - Configure new secrets
   - Update registry settings
   - Test deployment process

## Support and Resources

### Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/reference/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides)

### Community
- [GitHub Community Forum](https://github.community/)
- [Actions Marketplace](https://github.com/marketplace?type=actions)
- [Awesome Actions](https://github.com/sdras/awesome-actions)

### Support Channels
- **Internal**: #devops Slack channel
- **External**: GitHub Support
- **Community**: Stack Overflow (github-actions tag)