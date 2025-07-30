# Branch Protection Configuration Guide

## Overview

This document outlines the recommended branch protection rules for maintaining code quality and security in an advanced SDLC environment.

## Repository Context

**Current Maturity: ADVANCED (85%)**
- Comprehensive testing framework with 5 test layers
- Advanced security scanning and compliance
- Production-ready monitoring and observability
- Well-designed CI/CD workflows (awaiting deployment)

## Branch Protection Strategy

### Main Branch Protection

Configure the following protection rules for the `main` branch:

#### Required Status Checks
```yaml
required_status_checks:
  strict: true  # Require branches to be up to date before merging
  contexts:
    # Core CI Checks
    - "CI Tests (Python 3.9)"
    - "CI Tests (Python 3.10)" 
    - "CI Tests (Python 3.11)"
    - "CI Tests (Python 3.12)"
    
    # Code Quality Gates
    - "Code Quality / Ruff Linting"
    - "Code Quality / MyPy Type Checking"
    - "Code Quality / Test Coverage (>80%)"
    
    # Security Gates
    - "Security Scan / Bandit Static Analysis"
    - "Security Scan / Safety Dependency Check"
    - "Security Scan / Container Security"
    
    # Performance Gates
    - "Performance Tests / Benchmark Regression"
    - "Performance Tests / Load Testing"
```

#### Pull Request Requirements
```yaml
required_pull_request_reviews:
  required_approving_review_count: 1
  dismiss_stale_reviews: true
  require_code_owner_reviews: true  # If CODEOWNERS file exists
  require_review_from_code_owners: true
  dismiss_stale_reviews: true
  require_fresh_reviews: true
```

#### Additional Protection Rules
```yaml
additional_settings:
  enforce_admins: false  # Allow admins to bypass in emergencies
  allow_force_pushes: false
  allow_deletions: false
  required_linear_history: true  # Enforce clean git history
  required_conversation_resolution: true  # Require PR comment resolution
```

## Implementation Steps

### Step 1: Configure via GitHub Web Interface

1. Navigate to **Settings** → **Branches**
2. Click **Add rule** or edit existing rule for `main`
3. Configure settings as outlined above

### Step 2: Configure via GitHub CLI

```bash
# Install GitHub CLI if not available
# Configure branch protection
gh api repos/:owner/:repo/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["CI Tests","Security Scan","Code Quality"]}' \
  --field enforce_admins=false \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true}' \
  --field restrictions=null
```

### Step 3: Configure via Terraform (Optional)

```hcl
resource "github_branch_protection" "main" {
  repository_id = github_repository.main.id
  pattern       = "main"
  
  required_status_checks {
    strict = true
    contexts = [
      "CI Tests (Python 3.9)",
      "CI Tests (Python 3.10)",
      "CI Tests (Python 3.11)", 
      "CI Tests (Python 3.12)",
      "Code Quality / Ruff Linting",
      "Security Scan / Bandit Static Analysis"
    ]
  }
  
  required_pull_request_reviews {
    required_approving_review_count = 1
    dismiss_stale_reviews          = true
    require_code_owner_reviews     = true
  }
  
  enforce_admins         = false
  allows_force_pushes    = false
  allows_deletions       = false
  required_linear_history = true
}
```

## Development Branch Strategy

### Feature Branches
- **Naming Convention**: `feature/description` or `feat/ticket-number`
- **Protection**: Minimal protection, allow force pushes for development
- **Merge Strategy**: Squash merge to main with descriptive commit messages

### Release Branches
- **Naming Convention**: `release/v1.2.3`
- **Protection**: Similar to main but allow specific release managers
- **Merge Strategy**: Merge commit to preserve release history

### Hotfix Branches
- **Naming Convention**: `hotfix/critical-issue-description`
- **Protection**: Expedited review process with single approver
- **Merge Strategy**: Fast-forward merge with immediate deployment

## Quality Gates Integration

### Test Coverage Requirements
```yaml
coverage_gates:
  minimum_coverage: 80%
  coverage_diff_threshold: -2%  # Don't allow coverage to drop by more than 2%
  coverage_formats: ["lcov", "cobertura"]
```

### Performance Regression Detection
```yaml
performance_gates:
  benchmark_regression_threshold: 10%  # Fail if performance degrades by >10%
  memory_usage_threshold: 5%
  response_time_threshold: 100ms
```

### Security Compliance Gates
```yaml
security_gates:
  vulnerability_threshold: "medium"  # Block high/critical vulnerabilities
  license_compliance: true
  sbom_generation: required
  container_security_scan: required
```

## Emergency Procedures

### Hotfix Process
1. **Create hotfix branch** from main
2. **Implement minimal fix** with focused testing
3. **Emergency review** by single senior developer
4. **Fast-track merge** with immediate deployment
5. **Post-deployment verification** and monitoring

### Bypass Procedures
```yaml
emergency_bypass:
  conditions:
    - "Production outage (P0/P1)"
    - "Security vulnerability remediation"
    - "Critical business requirement"
  
  requirements:
    - Admin approval required
    - Post-bypass PR required within 24 hours
    - Incident report documentation
    - Retrospective review scheduled
```

## Monitoring and Metrics

### Branch Protection Metrics
Track the following metrics to ensure effective protection:

```yaml
metrics:
  pull_request_metrics:
    - average_review_time: "< 4 hours"
    - approval_rate: "> 95%"
    - merge_success_rate: "> 98%"
  
  quality_metrics:
    - failed_status_checks: "< 5%"
    - security_gate_failures: "< 1%"
    - coverage_regression_blocks: "< 3%"
  
  process_metrics:
    - bypass_frequency: "< 1 per month"
    - hotfix_frequency: "< 2 per month"
    - rollback_frequency: "< 1 per quarter"
```

### Alerting Rules
```yaml
alerts:
  - name: "High bypass frequency"
    condition: "bypasses > 2 per week"
    severity: "warning"
  
  - name: "Security gate failure spike"
    condition: "security_failures > 5 per day"
    severity: "critical"
  
  - name: "Review bottleneck"
    condition: "avg_review_time > 8 hours"
    severity: "warning"
```

## Continuous Improvement

### Monthly Review Process
1. **Analyze protection effectiveness** using collected metrics
2. **Review bypass incidents** and adjust rules if needed
3. **Update status check requirements** based on new tooling
4. **Gather developer feedback** on process friction
5. **Adjust thresholds** based on team capacity and quality goals

### Integration with SDLC Maturity
As the repository evolves:
- **Add new status checks** for emerging tools and practices
- **Adjust quality thresholds** based on improved capabilities
- **Implement additional automation** to reduce manual review burden
- **Enhance security requirements** based on threat landscape

---

**Note**: These branch protection rules are designed for an advanced SDLC environment. Adjust thresholds and requirements based on your team's specific needs and capabilities.