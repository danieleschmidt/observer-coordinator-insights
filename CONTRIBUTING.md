# Contributing to Observer Coordinator Insights

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites
- Python 3.9+
- Git
- Docker (optional, for containerized development)

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/observer-coordinator-insights.git
   cd observer-coordinator-insights
   ```

2. **Set up Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   pip install -e .
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Run Tests**
   ```bash
   pytest
   ```

## Development Workflow

### Branch Naming
- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `docs/description` - Documentation updates

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
type(scope): description

feat(clustering): add hierarchical clustering algorithm
fix(parser): handle missing energy values correctly
docs(readme): update installation instructions
test(unit): add tests for data validation
```

### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following our style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Locally**
   ```bash
   # Run all tests
   pytest

   # Run linting
   ruff check src/ tests/
   ruff format src/ tests/

   # Type checking
   mypy src/
   ```

4. **Submit Pull Request**
   - Fill out the PR template completely
   - Link related issues
   - Request review from maintainers

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Fast, isolated tests
├── integration/    # Component interaction tests
├── e2e/           # End-to-end workflow tests
├── performance/   # Performance and load tests
├── security/      # Security-focused tests
└── fixtures/      # Test data and utilities
```

### Writing Tests
```python
import pytest
from src.insights_clustering.parser import DataParser

def test_parser_validates_required_columns():
    """Test that parser correctly validates required columns."""
    parser = DataParser()
    
    # Test with missing columns
    invalid_data = pd.DataFrame({'employee_id': ['EMP001']})
    
    with pytest.raises(ValidationError, match="Missing required columns"):
        parser.validate(invalid_data)
```

### Test Coverage
- Maintain >80% code coverage
- Focus on critical paths and edge cases
- Include both positive and negative test cases

## Code Style Guidelines

### Python Code Style
- Follow PEP 8
- Use Ruff for linting and formatting
- Maximum line length: 88 characters
- Use type hints for all functions

```python
from typing import List, Dict, Optional
import pandas as pd

def process_insights_data(
    data: pd.DataFrame, 
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Process Insights Discovery data for clustering analysis.
    
    Args:
        data: Employee insights data
        config: Processing configuration
        
    Returns:
        List of processed employee profiles
        
    Raises:
        ValidationError: If data format is invalid
    """
    # Implementation here
    pass
```

### Documentation
- Use Google-style docstrings
- Include type hints
- Provide examples for complex functions
- Update README.md for user-facing changes

## Security Guidelines

### Data Handling
- Never commit real employee data
- Use anonymized test data only
- Implement proper data validation
- Follow privacy by design principles

### Code Security
- No hardcoded secrets or credentials
- Validate all user inputs
- Use parameterized queries
- Follow OWASP guidelines

### Dependencies
- Keep dependencies updated
- Review new dependencies for security
- Use `safety` to check for vulnerabilities

## Performance Guidelines

### Optimization
- Profile before optimizing
- Use appropriate data structures
- Consider memory usage for large datasets
- Cache expensive computations when appropriate

### Benchmarking
```python
import pytest

@pytest.mark.benchmark
def test_clustering_performance(benchmark):
    """Benchmark clustering algorithm performance."""
    large_dataset = generate_test_data(n_employees=1000)
    
    result = benchmark(cluster_employees, large_dataset)
    
    assert len(result.clusters) > 0
```

## Documentation

### Types of Documentation
- **API Documentation**: Generated from docstrings
- **User Guides**: Step-by-step tutorials
- **Architecture Docs**: System design and decisions
- **Runbooks**: Operational procedures

### Documentation Updates
- Update docs with code changes
- Include screenshots for UI changes
- Add examples for new features
- Review for clarity and accuracy

## Issue Reporting

### Bug Reports
Include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages and stack traces
- Sample data (anonymized)

### Feature Requests
Include:
- Use case description
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

## Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Email**: security@terragon-labs.com for security issues

### Getting Help
- Check existing issues and documentation
- Provide minimal reproducible examples
- Be specific about your environment and use case

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Invited to join the contributors team (for significant contributions)

## License

By contributing, you agree that your contributions will be licensed under the same Apache 2.0 License that covers the project.

---

Thank you for contributing to Observer Coordinator Insights! 🎉