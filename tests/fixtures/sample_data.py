"""
Test fixtures for sample Insights Discovery data.
This module provides mock data for testing without exposing real employee information.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import uuid


def generate_mock_insights_data(n_employees: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generate mock Insights Discovery data for testing.
    
    Args:
        n_employees: Number of mock employees to generate
        seed: Random seed for reproducible results
        
    Returns:
        DataFrame with mock Insights Discovery data
    """
    np.random.seed(seed)
    
    # Color energy distribution (should sum to 100 for each person)
    def generate_color_energies():
        # Generate 4 random values and normalize to sum to 100
        values = np.random.dirichlet([1, 1, 1, 1]) * 100
        return {
            'cool_blue': round(values[0], 1),
            'earth_green': round(values[1], 1), 
            'sunshine_yellow': round(values[2], 1),
            'fiery_red': round(values[3], 1)
        }
    
    data = []
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
    roles = ['Manager', 'Senior', 'Junior', 'Lead', 'Director', 'Specialist']
    
    for i in range(n_employees):
        energies = generate_color_energies()
        
        employee_data = {
            'employee_id': f"EMP{i+1:04d}",
            'anonymous_id': str(uuid.uuid4()),
            'department': np.random.choice(departments),
            'role_level': np.random.choice(roles),
            'tenure_years': np.random.randint(0, 15),
            'team_size': np.random.randint(1, 12),
            **energies,
            'created_at': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 365)),
            'last_assessment': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 90))
        }
        data.append(employee_data)
    
    return pd.DataFrame(data)


def get_test_clustering_config() -> Dict[str, Any]:
    """
    Get test configuration for clustering algorithms.
    
    Returns:
        Dictionary with test clustering configuration
    """
    return {
        'clustering': {
            'algorithm': 'kmeans',
            'n_clusters': 4,
            'random_state': 42,
            'max_iter': 300,
            'n_init': 10,
            'tolerance': 1e-4
        },
        'validation': {
            'min_cluster_size': 5,
            'max_cluster_size': 20,
            'silhouette_threshold': 0.3,
            'calinski_harabasz_threshold': 10.0
        },
        'output': {
            'save_plots': False,
            'plot_format': 'png',
            'include_individual_profiles': False,
            'anonymize_output': True
        }
    }


def get_malformed_data_samples() -> List[pd.DataFrame]:
    """
    Get samples of malformed data for negative testing.
    
    Returns:
        List of DataFrames with various data quality issues
    """
    samples = []
    
    # Missing columns
    samples.append(pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002'],
        'cool_blue': [25.0, 30.0],
        'earth_green': [25.0, 25.0]
        # Missing sunshine_yellow and fiery_red
    }))
    
    # Invalid energy values (negative)
    samples.append(pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002'],
        'cool_blue': [-10.0, 30.0],
        'earth_green': [25.0, 25.0],
        'sunshine_yellow': [30.0, 25.0],
        'fiery_red': [55.0, 20.0]
    }))
    
    # Energy values don't sum to 100
    samples.append(pd.DataFrame({
        'employee_id': ['EMP001', 'EMP002'],
        'cool_blue': [25.0, 30.0],
        'earth_green': [25.0, 25.0],
        'sunshine_yellow': [25.0, 25.0],
        'fiery_red': [30.0, 15.0]  # Sums to 105 and 95
    }))
    
    # Duplicate employee IDs
    samples.append(pd.DataFrame({
        'employee_id': ['EMP001', 'EMP001'],
        'cool_blue': [25.0, 30.0],
        'earth_green': [25.0, 25.0],
        'sunshine_yellow': [25.0, 25.0],
        'fiery_red': [25.0, 20.0]
    }))
    
    return samples


def get_edge_case_data() -> List[pd.DataFrame]:
    """
    Get edge case datasets for boundary testing.
    
    Returns:
        List of DataFrames with edge cases
    """
    edge_cases = []
    
    # Single employee (minimum viable dataset)
    edge_cases.append(pd.DataFrame({
        'employee_id': ['EMP001'],
        'cool_blue': [25.0],
        'earth_green': [25.0],
        'sunshine_yellow': [25.0],
        'fiery_red': [25.0]
    }))
    
    # All employees have identical profiles
    identical_data = {
        'employee_id': [f'EMP{i:03d}' for i in range(1, 11)],
        'cool_blue': [25.0] * 10,
        'earth_green': [25.0] * 10,
        'sunshine_yellow': [25.0] * 10,
        'fiery_red': [25.0] * 10
    }
    edge_cases.append(pd.DataFrame(identical_data))
    
    # Extreme energy distributions
    extreme_data = {
        'employee_id': ['EMP001', 'EMP002', 'EMP003', 'EMP004'],
        'cool_blue': [100.0, 0.0, 0.0, 0.0],
        'earth_green': [0.0, 100.0, 0.0, 0.0],
        'sunshine_yellow': [0.0, 0.0, 100.0, 0.0],
        'fiery_red': [0.0, 0.0, 0.0, 100.0]
    }
    edge_cases.append(pd.DataFrame(extreme_data))
    
    return edge_cases


def get_performance_test_data(size: str = 'medium') -> pd.DataFrame:
    """
    Generate datasets of various sizes for performance testing.
    
    Args:
        size: Size category ('small', 'medium', 'large', 'xlarge')
        
    Returns:
        DataFrame with the requested size
    """
    size_mapping = {
        'small': 100,
        'medium': 1000,
        'large': 10000,
        'xlarge': 50000
    }
    
    n_employees = size_mapping.get(size, 1000)
    return generate_mock_insights_data(n_employees=n_employees)


# Test data constants
SAMPLE_EMPLOYEE_PROFILES = {
    'blue_dominant': {'cool_blue': 60, 'earth_green': 15, 'sunshine_yellow': 15, 'fiery_red': 10},
    'green_dominant': {'cool_blue': 15, 'earth_green': 60, 'sunshine_yellow': 15, 'fiery_red': 10},
    'yellow_dominant': {'cool_blue': 15, 'earth_green': 15, 'sunshine_yellow': 60, 'fiery_red': 10},
    'red_dominant': {'cool_blue': 10, 'earth_green': 15, 'sunshine_yellow': 15, 'fiery_red': 60},
    'balanced': {'cool_blue': 25, 'earth_green': 25, 'sunshine_yellow': 25, 'fiery_red': 25}
}

TEAM_COMPOSITION_SCENARIOS = {
    'all_same_type': ['blue_dominant'] * 5,
    'mixed_balanced': ['blue_dominant', 'green_dominant', 'yellow_dominant', 'red_dominant', 'balanced'],
    'analyst_heavy': ['blue_dominant'] * 3 + ['green_dominant', 'balanced'],
    'creative_heavy': ['yellow_dominant'] * 3 + ['red_dominant', 'balanced'],
    'leadership_heavy': ['red_dominant'] * 3 + ['blue_dominant', 'yellow_dominant']
}