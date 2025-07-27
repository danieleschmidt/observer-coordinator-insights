"""End-to-end tests for the complete analysis pipeline."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.e2e
def test_complete_clustering_pipeline(temp_csv_file: str, temp_dir: str):
    """Test the complete clustering pipeline from CSV input to JSON output."""
    from src.insights_clustering.parser import InsightsParser
    from src.insights_clustering.validator import DataValidator
    from src.insights_clustering.clustering import ClusteringEngine
    
    # Parse input data
    parser = InsightsParser()
    data = parser.parse_csv(temp_csv_file)
    
    # Validate data
    validator = DataValidator()
    assert validator.validate(data), "Data validation failed"
    
    # Perform clustering
    engine = ClusteringEngine()
    result = engine.cluster(data)
    
    # Verify results
    assert "labels" in result
    assert "centers" in result
    assert "n_clusters" in result
    assert len(result["labels"]) == len(data)


@pytest.mark.e2e
def test_team_simulation_pipeline(sample_insights_data: pd.DataFrame, temp_dir: str):
    """Test the complete team simulation pipeline."""
    from src.insights_clustering.clustering import ClusteringEngine
    from src.team_simulator.simulator import TeamSimulator
    
    # Perform clustering first
    engine = ClusteringEngine()
    clustering_result = engine.cluster(sample_insights_data)
    
    # Run team simulation
    simulator = TeamSimulator()
    teams = simulator.generate_teams(
        data=sample_insights_data,
        cluster_labels=clustering_result["labels"],
        team_size=5,
        num_teams=3
    )
    
    # Verify team composition
    assert len(teams) == 3
    for team in teams:
        assert len(team["members"]) == 5
        assert "diversity_score" in team
        assert "predicted_performance" in team


@pytest.mark.e2e
def test_autonomous_orchestrator_execution(temp_csv_file: str, temp_dir: str):
    """Test the autonomous orchestrator with a complete workflow."""
    import sys
    import subprocess
    
    # Create a test configuration
    test_config = {
        "input_file": temp_csv_file,
        "output_dir": temp_dir,
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 4
        },
        "simulation": {
            "team_size": 5,
            "num_teams": 3
        }
    }
    
    config_file = os.path.join(temp_dir, "test_config.yml")
    import yaml
    with open(config_file, "w") as f:
        yaml.dump(test_config, f)
    
    # Run the orchestrator
    cmd = [
        sys.executable,
        "autonomous_orchestrator.py",
        "--config", config_file,
        "--output-dir", temp_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    
    # Verify execution
    assert result.returncode == 0, f"Orchestrator failed: {result.stderr}"
    
    # Check output files
    expected_files = [
        "clustering_results.json",
        "team_recommendations.json",
        "cluster_visualization.png"
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(temp_dir, file_name)
        assert os.path.exists(file_path), f"Expected output file {file_name} not found"


@pytest.mark.e2e
@pytest.mark.slow
def test_large_dataset_processing(large_insights_data: pd.DataFrame, temp_dir: str):
    """Test processing of large datasets (performance validation)."""
    import time
    
    from src.insights_clustering.clustering import ClusteringEngine
    
    start_time = time.time()
    
    # Process large dataset
    engine = ClusteringEngine()
    result = engine.cluster(large_insights_data)
    
    processing_time = time.time() - start_time
    
    # Performance assertions
    assert processing_time < 30, f"Processing took too long: {processing_time:.2f}s"
    assert len(result["labels"]) == len(large_insights_data)
    assert result["n_clusters"] > 0


@pytest.mark.e2e
def test_error_handling_invalid_data(temp_dir: str):
    """Test error handling with invalid input data."""
    from src.insights_clustering.parser import InsightsParser
    from src.insights_clustering.validator import DataValidator
    
    # Create invalid CSV file
    invalid_data = pd.DataFrame({
        "employee_id": ["EMP001", "EMP002"],
        "invalid_column": [1, 2],
        "another_invalid": [3, 4]
    })
    
    invalid_file = os.path.join(temp_dir, "invalid.csv")
    invalid_data.to_csv(invalid_file, index=False)
    
    # Parse and validate
    parser = InsightsParser()
    data = parser.parse_csv(invalid_file)
    
    validator = DataValidator()
    is_valid = validator.validate(data)
    
    # Should fail validation
    assert not is_valid, "Invalid data should not pass validation"


@pytest.mark.e2e
def test_privacy_compliance(sample_insights_data: pd.DataFrame, temp_dir: str):
    """Test privacy compliance features."""
    from src.insights_clustering.validator import DataValidator
    
    # Add some PII-like data
    data_with_pii = sample_insights_data.copy()
    data_with_pii["email"] = [f"emp{i}@company.com" for i in range(len(data_with_pii))]
    data_with_pii["phone"] = [f"555-{i:04d}" for i in range(len(data_with_pii))]
    
    validator = DataValidator()
    
    # Check PII detection
    pii_detected = validator.detect_pii(data_with_pii)
    assert pii_detected, "PII should be detected in the dataset"
    
    # Test anonymization
    anonymized_data = validator.anonymize_data(data_with_pii)
    
    # Verify PII is removed/anonymized
    assert "email" not in anonymized_data.columns or all(
        anonymized_data["email"].str.startswith("ANON_")
    )
    
    # Verify core data is preserved
    assert "cool_blue" in anonymized_data.columns
    assert len(anonymized_data) == len(data_with_pii)


@pytest.mark.e2e
def test_visualization_generation(sample_insights_data: pd.DataFrame, temp_dir: str):
    """Test visualization generation in the pipeline."""
    from src.insights_clustering.clustering import ClusteringEngine
    
    engine = ClusteringEngine()
    result = engine.cluster(sample_insights_data)
    
    # Generate visualization
    viz_path = os.path.join(temp_dir, "cluster_wheel.png")
    engine.create_visualization(
        data=sample_insights_data,
        labels=result["labels"],
        centers=result["centers"],
        output_path=viz_path
    )
    
    # Verify visualization file is created
    assert os.path.exists(viz_path), "Visualization file should be created"
    assert os.path.getsize(viz_path) > 0, "Visualization file should not be empty"


@pytest.mark.e2e
def test_configuration_management(temp_dir: str):
    """Test configuration management and validation."""
    import yaml
    
    # Create test configuration
    config = {
        "clustering": {
            "algorithm": "kmeans",
            "n_clusters": 5,
            "random_state": 42
        },
        "simulation": {
            "iterations": 100,
            "team_size_range": [3, 8]
        },
        "privacy": {
            "anonymize_data": True,
            "retention_days": 180
        }
    }
    
    config_file = os.path.join(temp_dir, "test_config.yml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    
    # Test configuration loading
    from src.main import load_configuration
    loaded_config = load_configuration(config_file)
    
    assert loaded_config["clustering"]["algorithm"] == "kmeans"
    assert loaded_config["privacy"]["retention_days"] == 180