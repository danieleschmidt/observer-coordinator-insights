# Security Policy

## Reporting Security Vulnerabilities

**Do not report security vulnerabilities through public GitHub issues.**

### Private Reporting

Please report security vulnerabilities via:
* GitHub Security Advisories (preferred)
* Email to project maintainers

### Response Timeline

* Initial response: within 48 hours
* Status updates: every 72 hours
* Resolution target: 90 days
## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. Please follow responsible disclosure:

### How to Report

1. **Email**: Send details to security@terragon-labs.com
2. **GitHub**: Use private vulnerability reporting (preferred)
3. **Encrypted**: Use our PGP key if needed

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if any)

### Response Timeline

- **Acknowledgment**: Within 24 hours
- **Initial Assessment**: Within 72 hours  
- **Status Updates**: Weekly until resolved
- **Resolution**: Target 30 days for high/critical issues

### Security Measures

- All dependencies scanned with Dependabot
- Code scanned with CodeQL and Bandit
- Container images scanned with Trivy
- Pre-commit hooks prevent common vulnerabilities
- No secrets or PII in logs or repositories

### Scope

This policy covers:
* Code vulnerabilities
* Dependency issues
* Configuration problems

### Resources

* [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories)
* [Coordinated Vulnerability Disclosure](https://about.gitlab.com/security/disclosure/)
- The observer-coordinator-insights application
- Associated Docker images
- Documentation and configuration files

Out of scope:
- Third-party dependencies (report to upstream)
- Infrastructure not directly controlled by us