# Manual Setup Requirements

## GitHub Repository Settings

### Branch Protection Rules
* Enable protection for `main` branch
* Require PR reviews (minimum 1)
* Require status checks to pass
* Restrict push access to administrators

### Repository Configuration
* Set repository topics: `insights-discovery`, `clustering`, `team-composition`
* Configure homepage URL from package.json
* Enable security features:
  - Dependency graph
  - Dependabot alerts
  - Code scanning alerts

## GitHub Actions Setup

### Workflow Files
Copy workflows from `docs/github-workflows/` to `.github/workflows/`:
* `ci.yml` - Main CI pipeline
* `release.yml` - Release automation  
* `sbom-diff.yml` - Security scanning
* `auto-rebase.yml` - PR management

### Required Secrets
Configure in repository settings:
* `PYPI_TOKEN` - For package publishing
* Additional secrets per workflow documentation

## External Integrations

### Monitoring (Optional)
* Code coverage reporting
* Security scanning tools
* Performance monitoring

## Resources

* [GitHub Repository Settings](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features)
* [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)