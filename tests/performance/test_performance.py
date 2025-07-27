"""Performance tests for the clustering system."""

import time
from typing import Any, Dict

import pandas as pd
import pytest


@pytest.mark.performance
@pytest.mark.slow
def test_clustering_performance_small_dataset(sample_insights_data: pd.DataFrame):
    """Test clustering performance with small dataset (100 employees)."""
    from src.insights_clustering.clustering import ClusteringEngine
    
    engine = ClusteringEngine()
    
    start_time = time.time()
    result = engine.cluster(sample_insights_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 2.0, f"Small dataset processing too slow: {processing_time:.2f}s"
    assert len(result["labels"]) == len(sample_insights_data)


@pytest.mark.performance
@pytest.mark.slow
def test_clustering_performance_medium_dataset():
    """Test clustering performance with medium dataset (1000 employees)."""
    import numpy as np
    from src.insights_clustering.clustering import ClusteringEngine
    
    # Generate medium dataset
    np.random.seed(42)
    size = 1000
    
    data = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(size)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
    })
    
    engine = ClusteringEngine()
    
    start_time = time.time()
    result = engine.cluster(data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 5.0, f"Medium dataset processing too slow: {processing_time:.2f}s"
    assert len(result["labels"]) == len(data)


@pytest.mark.performance
@pytest.mark.slow
def test_clustering_performance_large_dataset(large_insights_data: pd.DataFrame):
    """Test clustering performance with large dataset (5000 employees)."""
    from src.insights_clustering.clustering import ClusteringEngine
    
    engine = ClusteringEngine()
    
    start_time = time.time()
    result = engine.cluster(large_insights_data)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 15.0, f"Large dataset processing too slow: {processing_time:.2f}s"
    assert len(result["labels"]) == len(large_insights_data)


@pytest.mark.performance
def test_memory_usage_clustering():
    """Test memory usage during clustering operations."""
    import tracemalloc
    import numpy as np
    from src.insights_clustering.clustering import ClusteringEngine
    
    # Generate test dataset
    np.random.seed(42)
    size = 2000
    
    data = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(size)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
    })
    
    # Start memory tracking
    tracemalloc.start()
    
    engine = ClusteringEngine()
    result = engine.cluster(data)
    
    # Get memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Convert to MB
    peak_mb = peak / 1024 / 1024
    
    # Memory assertions (should not exceed 100MB for 2000 employees)
    assert peak_mb < 100, f"Memory usage too high: {peak_mb:.2f}MB"
    assert len(result["labels"]) == len(data)


@pytest.mark.performance
def test_team_simulation_performance(sample_insights_data: pd.DataFrame):
    """Test team simulation performance."""
    from src.insights_clustering.clustering import ClusteringEngine
    from src.team_simulator.simulator import TeamSimulator
    
    # Perform clustering first
    engine = ClusteringEngine()
    clustering_result = engine.cluster(sample_insights_data)
    
    simulator = TeamSimulator()
    
    start_time = time.time()
    teams = simulator.generate_teams(
        data=sample_insights_data,
        cluster_labels=clustering_result["labels"],
        team_size=5,
        num_teams=10  # Generate more teams for performance test
    )
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    # Performance assertions
    assert processing_time < 3.0, f"Team simulation too slow: {processing_time:.2f}s"
    assert len(teams) == 10


@pytest.mark.performance
def test_concurrent_clustering():
    """Test concurrent clustering operations."""
    import concurrent.futures
    import numpy as np
    from src.insights_clustering.clustering import ClusteringEngine
    
    def cluster_dataset(seed: int) -> Dict[str, Any]:
        """Cluster a dataset with given random seed."""
        np.random.seed(seed)
        size = 500
        
        data = pd.DataFrame({
            "employee_id": [f"EMP{seed}_{i:03d}" for i in range(size)],
            "cool_blue": np.random.randint(10, 60, size),
            "earth_green": np.random.randint(10, 60, size),
            "sunshine_yellow": np.random.randint(10, 60, size),
            "fiery_red": np.random.randint(10, 60, size),
        })
        
        engine = ClusteringEngine()
        return engine.cluster(data)
    
    start_time = time.time()
    
    # Run 4 concurrent clustering operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(cluster_dataset, seed) for seed in range(4)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Should complete all 4 operations in reasonable time
    assert processing_time < 20.0, f"Concurrent clustering too slow: {processing_time:.2f}s"
    assert len(results) == 4
    
    # Verify all results are valid
    for result in results:
        assert "labels" in result
        assert "centers" in result
        assert len(result["labels"]) == 500


@pytest.mark.performance
def test_data_loading_performance(temp_dir: str):
    """Test data loading performance with large CSV files."""
    import numpy as np
    from src.insights_clustering.parser import InsightsParser
    
    # Create large test CSV
    np.random.seed(42)
    size = 10000
    
    large_data = pd.DataFrame({
        "employee_id": [f"EMP{i:05d}" for i in range(size)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR"], size),
        "location": np.random.choice(["NYC", "SF", "LA", "Chicago"], size),
    })
    
    csv_file = f"{temp_dir}/large_test_data.csv"
    large_data.to_csv(csv_file, index=False)
    
    parser = InsightsParser()
    
    start_time = time.time()
    loaded_data = parser.parse_csv(csv_file)
    end_time = time.time()
    
    loading_time = end_time - start_time
    
    # Loading should be fast
    assert loading_time < 5.0, f"Data loading too slow: {loading_time:.2f}s"
    assert len(loaded_data) == size


@pytest.mark.performance
def test_visualization_performance(sample_insights_data: pd.DataFrame, temp_dir: str):
    """Test visualization generation performance."""
    from src.insights_clustering.clustering import ClusteringEngine
    
    engine = ClusteringEngine()
    result = engine.cluster(sample_insights_data)
    
    start_time = time.time()
    
    # Generate multiple visualizations
    for i in range(5):
        viz_path = f"{temp_dir}/test_viz_{i}.png"
        engine.create_visualization(
            data=sample_insights_data,
            labels=result["labels"],
            centers=result["centers"],
            output_path=viz_path
        )
    
    end_time = time.time()
    generation_time = end_time - start_time
    
    # Visualization generation should be reasonably fast
    assert generation_time < 10.0, f"Visualization generation too slow: {generation_time:.2f}s"


@pytest.mark.performance
def test_algorithm_comparison_performance():
    """Compare performance of different clustering algorithms."""
    import numpy as np
    from src.insights_clustering.clustering import ClusteringEngine
    
    # Generate test dataset
    np.random.seed(42)
    size = 1000
    
    data = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(size)],
        "cool_blue": np.random.randint(10, 60, size),
        "earth_green": np.random.randint(10, 60, size),
        "sunshine_yellow": np.random.randint(10, 60, size),
        "fiery_red": np.random.randint(10, 60, size),
    })
    
    algorithms = ["kmeans"]  # Add more algorithms as implemented
    performance_results = {}
    
    for algorithm in algorithms:
        engine = ClusteringEngine(algorithm=algorithm)
        
        start_time = time.time()
        result = engine.cluster(data)
        end_time = time.time()
        
        performance_results[algorithm] = {
            "time": end_time - start_time,
            "clusters": result["n_clusters"],
            "inertia": result.get("inertia", 0)
        }
    
    # K-means should be reasonably fast
    assert performance_results["kmeans"]["time"] < 5.0, "K-means algorithm too slow"
    
    # Log performance comparison for analysis
    print("\nAlgorithm Performance Comparison:")
    for algo, metrics in performance_results.items():
        print(f"{algo}: {metrics['time']:.2f}s, {metrics['clusters']} clusters")