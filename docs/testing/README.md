# Testing Guide

This document provides comprehensive guidance on testing practices for the Observer Coordinator Insights project.

## Testing Philosophy

Our testing strategy follows the testing pyramid approach:

- **Unit Tests** (70%): Fast, isolated tests for individual functions and classes
- **Integration Tests** (20%): Tests for component interactions and data flow
- **End-to-End Tests** (10%): Full system tests simulating real user workflows

## Test Structure

```
tests/
├── conftest.py              # Global pytest configuration and fixtures
├── fixtures/                # Static test data and configuration files
├── unit/                    # Unit tests (fast, isolated)
├── integration/             # Integration tests (medium speed)
├── e2e/                     # End-to-end tests (slower)
├── performance/             # Performance and load tests
└── security/                # Security and vulnerability tests
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
make test
npm run test

# Run tests with coverage
make test-coverage
npm run coverage

# Run specific test categories
pytest tests/unit/ -v                    # Unit tests only
pytest tests/integration/ -v             # Integration tests only
pytest tests/e2e/ -v                    # E2E tests only
pytest tests/performance/ -v             # Performance tests only
pytest tests/security/ -v               # Security tests only
```

### Using Markers

Tests are automatically marked based on their location and can be run selectively:

```bash
# Run only unit tests
pytest -m unit

# Run only fast tests (exclude performance tests)
pytest -m "not slow"

# Run integration and e2e tests
pytest -m "integration or e2e"

# Run security tests
pytest -m security
```

### Development Workflow

```bash
# Run tests on file save (requires pytest-watch)
ptw tests/

# Run tests with verbose output and stop on first failure
pytest -v -x

# Run tests with pdb debugging on failure
pytest --pdb

# Run tests with coverage and open HTML report
pytest --cov=src --cov-report=html && open htmlcov/index.html
```

## Writing Tests

### Unit Tests

Unit tests should focus on testing individual functions or classes in isolation:

```python
import pytest
from src.insights_clustering.parser import DataParser

class TestDataParser:
    def test_parse_valid_csv(self, sample_csv_file):
        parser = DataParser()
        result = parser.parse(sample_csv_file)
        
        assert result is not None
        assert len(result) > 0
        assert all(col in result.columns for col in ['red_energy', 'blue_energy'])
    
    def test_parse_invalid_file_raises_error(self):
        parser = DataParser()
        
        with pytest.raises(FileNotFoundError):
            parser.parse("nonexistent_file.csv")
    
    @pytest.mark.parametrize("invalid_data", [
        None,
        "",
        [],
        {}
    ])
    def test_parse_invalid_data_types(self, invalid_data):
        parser = DataParser()
        
        with pytest.raises(ValueError):
            parser.validate_data(invalid_data)
```

### Integration Tests

Integration tests verify that components work together correctly:

```python
import pytest
from src.insights_clustering.clustering import ClusteringEngine
from src.insights_clustering.parser import DataParser

class TestClusteringPipeline:
    def test_full_clustering_pipeline(self, sample_csv_file, sample_config):
        # Test complete data processing pipeline
        parser = DataParser()
        engine = ClusteringEngine(sample_config['clustering'])
        
        # Parse data
        data = parser.parse(sample_csv_file)
        
        # Perform clustering
        clusters = engine.fit_predict(data)
        
        # Verify results
        assert len(clusters) == len(data)
        assert len(set(clusters)) == sample_config['clustering']['n_clusters']
```

### End-to-End Tests

E2E tests simulate complete user workflows:

```python
def test_complete_analysis_workflow(sample_csv_file, sample_config_file):
    """Test the complete analysis workflow from CSV input to report output."""
    from src.main import main
    
    # Run complete analysis
    result = main([
        "--input", str(sample_csv_file),
        "--config", str(sample_config_file),
        "--output", "test_results.json"
    ])
    
    assert result == 0  # Success exit code
    
    # Verify output file exists and contains expected data
    import json
    with open("test_results.json") as f:
        results = json.load(f)
    
    assert "clusters" in results
    assert "team_recommendations" in results
    assert len(results["clusters"]) > 0
```

### Performance Tests

Performance tests ensure the system meets performance requirements:

```python
import pytest

@pytest.mark.performance
def test_clustering_performance(large_dataset, benchmark):
    """Test clustering performance with large dataset."""
    from src.insights_clustering.clustering import ClusteringEngine
    
    engine = ClusteringEngine({"n_clusters": 5, "random_state": 42})
    
    # Benchmark the clustering operation
    result = benchmark(engine.fit_predict, large_dataset)
    
    assert len(result) == len(large_dataset)
    # Clustering should complete within reasonable time (handled by benchmark)

@pytest.mark.performance  
def test_memory_usage(large_dataset):
    """Test memory usage with large datasets."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    # Perform memory-intensive operation
    from src.insights_clustering.clustering import ClusteringEngine
    engine = ClusteringEngine({"n_clusters": 5})
    engine.fit_predict(large_dataset)
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Memory increase should be reasonable (less than 500MB for test)
    assert memory_increase < 500 * 1024 * 1024
```

### Security Tests

Security tests verify data protection and vulnerability resistance:

```python
@pytest.mark.security
def test_data_anonymization(sample_insights_data):
    """Test that PII is properly anonymized."""
    from src.insights_clustering.parser import DataParser
    
    parser = DataParser(anonymize=True)
    anonymized_data = parser.anonymize_data(sample_insights_data)
    
    # Verify no PII remains in anonymized data
    assert "name" not in anonymized_data.columns
    assert "email" not in anonymized_data.columns
    assert all(isinstance(id_val, str) and len(id_val) == 36 
              for id_val in anonymized_data["employee_id"])

@pytest.mark.security
def test_sql_injection_resistance(security_test_data):
    """Test resistance to SQL injection attempts."""
    from src.insights_clustering.validator import DataValidator
    
    validator = DataValidator()
    
    for injection_attempt in security_test_data["injection_attempts"]:
        with pytest.raises(ValueError, match="Invalid data format"):
            validator.validate_input(injection_attempt)
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
minversion = 6.0
addopts = -ra -q --strict-markers --strict-config
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    performance: Performance tests
    security: Security tests
    slow: Slow running tests
filterwarnings =
    error
    ignore::UserWarning
    ignore::DeprecationWarning
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/venv/*",
    "*/__pycache__/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

## Continuous Integration

Tests are automatically run in CI/CD pipeline:

1. **Unit Tests**: Run on every commit
2. **Integration Tests**: Run on pull requests
3. **Performance Tests**: Run nightly
4. **Security Tests**: Run on security-related changes

## Best Practices

### Writing Good Tests

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use Fixtures**: Leverage pytest fixtures for reusable test data and setup
5. **Mock External Dependencies**: Use mocks to isolate units under test

### Test Data Management

1. **Use Realistic Data**: Test data should resemble real-world data
2. **Anonymize Sensitive Data**: Never use real PII in tests
3. **Version Test Data**: Track changes to test data files
4. **Clean Up**: Remove temporary files and data after tests

### Performance Considerations

1. **Fast Unit Tests**: Unit tests should complete in milliseconds
2. **Parallel Execution**: Use pytest-xdist for parallel test execution
3. **Selective Testing**: Use markers to run only relevant test subsets
4. **Profile Slow Tests**: Identify and optimize slow-running tests

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PYTHONPATH includes src directory
2. **Fixture Not Found**: Check fixture scope and location
3. **Test Data Issues**: Verify test data files exist and are valid
4. **Flaky Tests**: Use proper test isolation and avoid timing dependencies

### Debugging Tests

```bash
# Run with verbose output
pytest -v -s

# Drop into debugger on failure
pytest --pdb

# Run specific test with debugging
pytest tests/unit/test_parser.py::test_specific_function --pdb

# Show local variables on failure
pytest --tb=long

# Capture and display stdout/stderr
pytest -s --capture=no
```

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Best Practices](https://docs.pytest.org/en/stable/goodpractices.html)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Mock Documentation](https://docs.python.org/3/library/unittest.mock.html)