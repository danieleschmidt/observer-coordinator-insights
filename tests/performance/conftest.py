"""Configuration for performance tests."""

import pytest
import numpy as np
import pandas as pd
from typing import Generator


@pytest.fixture
def small_dataset() -> pd.DataFrame:
    """Generate a small dataset for quick performance tests."""
    np.random.seed(42)
    data = {
        'employee_id': range(100),
        'red': np.random.randint(0, 101, 100),
        'blue': np.random.randint(0, 101, 100),
        'yellow': np.random.randint(0, 101, 100),
        'green': np.random.randint(0, 101, 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def medium_dataset() -> pd.DataFrame:
    """Generate a medium dataset for standard performance tests."""
    np.random.seed(42)
    data = {
        'employee_id': range(1000),
        'red': np.random.randint(0, 101, 1000),
        'blue': np.random.randint(0, 101, 1000),
        'yellow': np.random.randint(0, 101, 1000),
        'green': np.random.randint(0, 101, 1000),
    }
    return pd.DataFrame(data)


@pytest.fixture
def large_dataset() -> pd.DataFrame:
    """Generate a large dataset for stress performance tests."""
    np.random.seed(42)
    data = {
        'employee_id': range(10000),
        'red': np.random.randint(0, 101, 10000),
        'blue': np.random.randint(0, 101, 10000),
        'yellow': np.random.randint(0, 101, 10000),
        'green': np.random.randint(0, 101, 10000),
    }
    return pd.DataFrame(data)


@pytest.fixture
def benchmark_config() -> dict:
    """Configuration for benchmark tests."""
    return {
        'min_rounds': 5,
        'max_time': 30.0,
        'warmup': True,
        'disable_gc': True,
    }