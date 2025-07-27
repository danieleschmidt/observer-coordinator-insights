"""Pytest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_insights_data() -> pd.DataFrame:
    """Generate sample Insights Discovery data for testing."""
    data = {
        "employee_id": [f"EMP{i:03d}" for i in range(1, 101)],
        "cool_blue": [20 + i % 40 for i in range(100)],
        "earth_green": [15 + i % 35 for i in range(100)],
        "sunshine_yellow": [25 + i % 30 for i in range(100)],
        "fiery_red": [10 + i % 45 for i in range(100)],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def large_insights_data() -> pd.DataFrame:
    """Generate large dataset for performance testing."""
    import numpy as np
    
    np.random.seed(42)
    size = 5000
    
    data = {
        "employee_id": [f"EMP{i:05d}" for i in range(1, size + 1)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
    }
    return pd.DataFrame(data)


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


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.parent / "src"))
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def mock_clustering_result():
    """Mock clustering result for testing."""
    import numpy as np
    
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
        from sklearn.preprocessing import MinMaxScaler
        
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


# Performance testing markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


# Test data cleanup
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