# Performance Benchmarking

This directory contains automated performance benchmarking configuration and results for the Observer Coordinator Insights system.

## Overview

The benchmarking system provides:
- **Automated Performance Testing**: Continuous monitoring of clustering and data processing performance
- **Regression Detection**: Alerts when performance degrades beyond acceptable thresholds
- **Load Testing**: Simulation of concurrent usage scenarios
- **Memory Profiling**: Tracking memory usage patterns and optimization opportunities
- **CI/CD Integration**: Automated benchmarks on pull requests and main branch updates

## Configuration

The benchmarking system is configured via `config.yml`:

### Benchmark Categories

1. **Clustering Benchmarks**: K-means performance across different dataset sizes
2. **Data Processing Benchmarks**: CSV parsing, validation, and normalization
3. **Team Simulation Benchmarks**: Team composition and recommendation algorithms
4. **Load Testing**: Concurrent user simulation and batch processing

### Performance Thresholds

- Small datasets (100 employees): < 100ms
- Medium datasets (1000 employees): < 500ms  
- Large datasets (10000 employees): < 2000ms
- Memory limits scale from 25MB to 500MB based on dataset size

## Running Benchmarks

### Local Development

```bash
# Run all benchmarks
python -m pytest tests/performance/ --benchmark-only

# Run specific benchmark category
python -m pytest tests/performance/test_clustering_benchmarks.py --benchmark-only

# Run with profiling enabled
python -m pytest tests/performance/ --benchmark-only --profile

# Generate HTML report
python -m pytest tests/performance/ --benchmark-only --benchmark-json=results.json
python -c "import json; from datetime import datetime; print('Benchmark completed:', datetime.now())"
```

### CI/CD Integration

Benchmarks automatically run:
- On all pull requests (performance impact assessment)
- On pushes to main branch (baseline establishment)
- Weekly scheduled runs (trend monitoring)
- Manual trigger via GitHub Actions

## Results Interpretation

### Performance Metrics

- **Duration**: Execution time in milliseconds
- **Memory**: Peak memory usage in MB
- **CPU**: CPU utilization percentage
- **Throughput**: Operations per second

### Regression Detection

The system flags performance regressions when:
- Execution time increases > 10% compared to baseline
- Memory usage increases > 15% compared to baseline
- Throughput decreases > 10% compared to baseline

### Load Testing Results

Load tests measure:
- **Concurrent Performance**: System behavior under simultaneous requests
- **Scalability**: Performance degradation patterns as load increases  
- **Resource Utilization**: CPU, memory, and I/O under load
- **Error Rates**: Failure rates at different load levels

## Optimization Guidelines

### Performance Targets

| Dataset Size | Target Duration | Memory Limit | Notes |
|-------------|----------------|--------------|-------|
| Small (100) | < 100ms | < 50MB | Real-time processing |
| Medium (1K) | < 500ms | < 100MB | Interactive response |
| Large (10K) | < 2000ms | < 500MB | Batch processing |

### Common Optimizations

1. **Algorithm Efficiency**: Use optimized clustering algorithms (scikit-learn)
2. **Data Structures**: Leverage NumPy arrays for numerical computations
3. **Memory Management**: Implement data streaming for large datasets
4. **Caching**: Cache intermediate results for repeated operations
5. **Parallel Processing**: Use multiprocessing for independent operations

## Troubleshooting

### Performance Issues

- **Memory Leaks**: Check for unreferenced large objects
- **Inefficient Algorithms**: Profile with cProfile for bottlenecks
- **Data Loading**: Optimize CSV parsing with pandas chunk reading
- **Clustering Convergence**: Tune K-means parameters for faster convergence

### Benchmark Failures

- **Timeout**: Increase time limits or optimize algorithms
- **Memory Exceeded**: Implement streaming or reduce batch sizes
- **Flaky Tests**: Add warmup runs and statistical significance testing
- **Environment Issues**: Ensure consistent test environment conditions

## Monitoring Integration

Results integrate with:
- **GitHub Actions**: Automated PR comments with performance impact
- **Prometheus**: Metrics export for long-term monitoring  
- **Grafana**: Performance dashboards and alerting
- **Slack/Email**: Regression alerts and weekly reports

## Contributing

When adding new benchmarks:

1. Add benchmark configuration to `config.yml`
2. Implement test in `tests/performance/`
3. Set appropriate performance thresholds
4. Document expected behavior and optimization notes
5. Verify CI/CD integration works correctly

For questions or issues, refer to the main project documentation or create an issue.