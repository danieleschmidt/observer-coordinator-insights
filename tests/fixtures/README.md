# Test Fixtures

This directory contains static test data files and fixtures used across the test suite.

## Structure

```
fixtures/
├── data/                   # Sample data files
│   ├── valid_insights.csv     # Valid Insights Discovery data
│   ├── invalid_insights.csv   # Invalid data for error testing
│   └── large_dataset.csv      # Large dataset for performance testing
├── configs/               # Configuration files for testing
│   ├── test_config.yml       # Standard test configuration
│   ├── minimal_config.yml    # Minimal configuration
│   └── invalid_config.yml    # Invalid configuration for error testing
└── schemas/               # JSON schemas for validation testing
    ├── insights_schema.json  # Insights Discovery data schema
    └── config_schema.json    # Configuration schema
```

## Usage

Test fixtures are automatically loaded by pytest through the `conftest.py` configuration. Individual test files can access fixtures using the `test_data_dir` fixture:

```python
def test_example(test_data_dir):
    data_file = test_data_dir / "data" / "valid_insights.csv"
    assert data_file.exists()
```

## Data Format

### Insights Discovery CSV Format

The test CSV files follow the standard Insights Discovery format:

- `employee_id`: Unique identifier (anonymized in tests)
- `red_energy`: Red energy score (0-100)
- `blue_energy`: Blue energy score (0-100) 
- `green_energy`: Green energy score (0-100)
- `yellow_energy`: Yellow energy score (0-100)
- `thinking_preference`: Analytical, Practical, Conceptual, Creative
- `communication_style`: Direct, Inspiring, Harmonious, Reflective

### Configuration Format

Test configuration files use YAML format with the following structure:

```yaml
clustering:
  algorithm: kmeans
  n_clusters: 4
  random_state: 42

data:
  anonymize: true
  validation_level: strict
  retention_days: 180

output:
  format: json
  include_visualizations: true
  export_clusters: true

security:
  encrypt_data: true
  log_access: true
```

## Adding New Fixtures

1. Create the appropriate subdirectory if it doesn't exist
2. Add your fixture file with a descriptive name
3. Update this README with documentation
4. Add any necessary fixtures to `conftest.py` if dynamic loading is needed

## Security Considerations

- All test data is anonymized and contains no real PII
- Sensitive configuration values use test-only values
- Test data is excluded from version control if containing any real data patterns