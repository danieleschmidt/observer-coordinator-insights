# Performance Guide

This guide provides comprehensive information about performance characteristics, optimization strategies, and monitoring for the observer-coordinator-insights library.

## Performance Benchmarks

### Clustering Performance

Based on our performance test suite, here are the expected performance characteristics:

| Dataset Size | Records | Clusters | Average Time | Memory Usage |
|--------------|---------|----------|--------------|--------------|
| Small        | 100     | 4        | < 0.1s       | ~10MB        |
| Medium       | 1,000   | 5        | < 1s         | ~50MB        |
| Large        | 10,000  | 8        | < 10s        | ~200MB       |
| Very Large   | 50,000  | 10       | < 60s        | ~500MB       |

### Performance Test Commands

Run performance benchmarks locally:

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run specific performance tests
pytest tests/performance/test_clustering_performance.py::TestClusteringPerformance::test_large_dataset_performance -v

# Run with benchmark output
pytest tests/performance/ --benchmark-only --benchmark-sort=mean

# Generate benchmark comparison
pytest tests/performance/ --benchmark-autosave --benchmark-compare
```

## Algorithm Complexity

### K-means Clustering
- **Time Complexity**: O(n × k × i × d)
  - n = number of data points
  - k = number of clusters
  - i = number of iterations
  - d = number of dimensions (4 for Insights Discovery)
- **Space Complexity**: O(n × d + k × d)

### Team Simulation
- **Time Complexity**: O(n! / (t! × (n-t)!)) for exact optimization
- **Practical Complexity**: O(n × t × iterations) for heuristic approach
  - n = number of employees
  - t = team size
  - iterations = optimization iterations

## Performance Optimization Strategies

### 1. Data Preprocessing

```python
# Efficient data loading
parser = InsightsDataParser()
data = parser.parse_csv("large_dataset.csv", 
                       chunk_size=1000,  # Process in chunks
                       dtype_optimization=True)  # Use optimal dtypes
```

### 2. Clustering Optimization

```python
# Use optimal cluster parameters
clusterer = KMeansClusterer(
    n_clusters=4,
    random_state=42,
    n_init=5,  # Reduce from default 10 for speed
    max_iter=100,  # Reduce from default 300 for large datasets
    algorithm='lloyd'  # Most efficient for small-medium datasets
)

# For large datasets, use mini-batch k-means
from sklearn.cluster import MiniBatchKMeans
large_clusterer = MiniBatchKMeans(
    n_clusters=8,
    batch_size=1000,
    random_state=42
)
```

### 3. Memory Optimization

```python
import gc
import pandas as pd

# Process large datasets in chunks
def process_large_dataset(file_path, chunk_size=5000):
    results = []
    
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        parser = InsightsDataParser()
        parser.data = chunk
        features = parser.get_clustering_features()
        
        clusterer = KMeansClusterer(n_clusters=4)
        clusterer.fit(features)
        
        results.append(clusterer.get_cluster_assignments())
        
        # Force garbage collection
        del chunk, features, clusterer
        gc.collect()
    
    return pd.concat(results)
```

### 4. Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def parallel_clustering(datasets, n_processes=None):
    """Process multiple datasets in parallel"""
    if n_processes is None:
        n_processes = mp.cpu_count() - 1
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [
            executor.submit(process_single_dataset, dataset) 
            for dataset in datasets
        ]
        results = [future.result() for future in futures]
    
    return results

def process_single_dataset(dataset_path):
    parser = InsightsDataParser()
    data = parser.parse_csv(dataset_path)
    
    clusterer = KMeansClusterer(n_clusters=4)
    clusterer.fit(parser.get_clustering_features())
    
    return {
        'path': dataset_path,
        'clusters': clusterer.get_cluster_assignments(),
        'metrics': clusterer.get_cluster_quality_metrics()
    }
```

## Performance Monitoring

### 1. Runtime Monitoring

```python
import time
import psutil
import os
from contextlib import contextmanager

@contextmanager
def performance_monitor():
    """Context manager for monitoring performance"""
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    start_cpu = process.cpu_percent()
    
    try:
        yield
    finally:
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()
        
        print(f"Execution time: {end_time - start_time:.2f}s")
        print(f"Memory usage: {end_memory - start_memory:.2f}MB")
        print(f"CPU usage: {end_cpu:.1f}%")

# Usage
with performance_monitor():
    clusterer = KMeansClusterer(n_clusters=4)
    clusterer.fit(features)
```

### 2. Profiling Tools

```bash
# Profile with cProfile
python -m cProfile -o clustering_profile.prof -m src.main data.csv

# Analyze profile
python -c "
import pstats
p = pstats.Stats('clustering_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler clustering_script.py

# Line-by-line profiling
pip install line_profiler
kernprof -l -v clustering_script.py
```

### 3. Continuous Performance Monitoring

```python
# Add to main execution
import logging
from datetime import datetime

def log_performance_metrics(func):
    """Decorator to log performance metrics"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        logging.info(f"Performance metrics for {func.__name__}:")
        logging.info(f"  Execution time: {end_time - start_time:.2f}s")
        logging.info(f"  Memory delta: {(end_memory - start_memory) / 1024 / 1024:.2f}MB")
        logging.info(f"  Timestamp: {datetime.now().isoformat()}")
        
        return result
    return wrapper

# Apply to key functions
@log_performance_metrics
def perform_clustering(data):
    clusterer = KMeansClusterer(n_clusters=4)
    return clusterer.fit(data)
```

## Performance Troubleshooting

### Common Performance Issues

1. **Slow Clustering on Large Datasets**
   - **Symptom**: Clustering takes > 60s for 10k records
   - **Solution**: Use MiniBatchKMeans or reduce max_iter parameter
   - **Example**:
     ```python
     # Instead of regular KMeans
     clusterer = KMeansClusterer(n_clusters=4, max_iter=50)
     
     # Or use MiniBatch approach
     from sklearn.cluster import MiniBatchKMeans
     clusterer = MiniBatchKMeans(n_clusters=4, batch_size=1000)
     ```

2. **High Memory Usage**
   - **Symptom**: Memory usage > 1GB for < 50k records
   - **Solution**: Process in chunks, optimize data types
   - **Example**:
     ```python
     # Optimize data types
     data = data.astype({
         'fiery_red': 'float32',
         'sunshine_yellow': 'float32',
         'earth_green': 'float32',
         'cool_blue': 'float32'
     })
     ```

3. **Team Simulation Bottleneck**
   - **Symptom**: Team generation takes > 30s for < 1000 employees
   - **Solution**: Reduce iteration count, use heuristic optimization
   - **Example**:
     ```python
     # Reduce iterations for large datasets
     teams = simulator.recommend_optimal_teams(
         num_teams=3, 
         iterations=3 if len(employees) > 500 else 10
     )
     ```

### Performance Best Practices

1. **Data Validation**
   - Run validation separately from clustering for large datasets
   - Use sampling for initial quality checks

2. **Resource Management**
   - Monitor memory usage with large datasets
   - Use generators for processing streams of data
   - Clean up intermediate results

3. **Algorithmic Choices**
   - Use appropriate cluster counts (typically 2-12)
   - Consider data characteristics when choosing algorithms
   - Profile before optimizing

4. **Infrastructure Scaling**
   - Use multiple CPU cores for parallel processing
   - Consider distributed computing for very large datasets
   - Monitor system resources during execution

## Performance Testing in CI/CD

The repository includes automated performance testing:

```yaml
# .github/workflows/performance.yml
name: Performance Tests
on:
  pull_request:
    paths: ['src/**', 'tests/performance/**']

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-benchmark
    - name: Run performance tests
      run: |
        pytest tests/performance/ --benchmark-only --benchmark-json=benchmark.json
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        auto-push: true
```

## Hardware Recommendations

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4GB
- **Storage**: 1GB free space
- **Dataset**: < 1,000 employees

### Recommended Configuration
- **CPU**: 4+ cores, 3.0+ GHz
- **RAM**: 8GB+
- **Storage**: 10GB+ SSD
- **Dataset**: < 10,000 employees

### High-Performance Setup
- **CPU**: 8+ cores, 3.5+ GHz (or GPU acceleration)
- **RAM**: 32GB+
- **Storage**: NVMe SSD
- **Dataset**: 50,000+ employees
- **Network**: High-bandwidth for distributed processing

## Future Performance Improvements

Planned optimizations for future releases:

1. **GPU Acceleration**: CUDA support for large datasets
2. **Distributed Computing**: Spark/Dask integration
3. **Approximate Algorithms**: Faster clustering for very large datasets
4. **Caching**: Intelligent result caching
5. **Streaming**: Real-time processing capabilities