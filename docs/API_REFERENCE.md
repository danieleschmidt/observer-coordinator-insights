# API Reference

This document provides comprehensive API documentation for using `observer-coordinator-insights` as a Python library.

## Installation

```bash
pip install observer-coordinator-insights
```

## Quick Start

```python
from observer_coordinator_insights import InsightsDataParser, KMeansClusterer, TeamCompositionSimulator

# Parse Insights Discovery data
parser = InsightsDataParser()
data = parser.parse_csv("employee_data.csv")

# Perform clustering
clusterer = KMeansClusterer(n_clusters=4)
clusterer.fit(parser.get_clustering_features())

# Simulate team compositions
simulator = TeamCompositionSimulator()
teams = simulator.recommend_optimal_teams(3)
```

## Core Modules

### `insights_clustering`

#### `InsightsDataParser`

Handles parsing and preprocessing of Insights Discovery CSV data.

```python
from observer_coordinator_insights import InsightsDataParser

parser = InsightsDataParser()
```

**Methods:**

- **`parse_csv(file_path: Path) -> pd.DataFrame`**
  - Parse Insights Discovery CSV file
  - **Parameters:** `file_path` - Path to CSV file
  - **Returns:** DataFrame with parsed employee data
  - **Raises:** `ValueError` if file format invalid

- **`get_clustering_features() -> pd.DataFrame`**
  - Extract numerical features suitable for clustering
  - **Returns:** DataFrame with normalized personality dimensions

- **`get_employee_metadata() -> pd.DataFrame`**
  - Get non-clustering metadata (names, roles, etc.)
  - **Returns:** DataFrame with employee metadata

**Example:**
```python
parser = InsightsDataParser()
data = parser.parse_csv("employees.csv")
features = parser.get_clustering_features()
metadata = parser.get_employee_metadata()
```

#### `KMeansClusterer`

Performs K-means clustering on employee personality data.

```python
from observer_coordinator_insights import KMeansClusterer

clusterer = KMeansClusterer(n_clusters=4, random_state=42)
```

**Constructor Parameters:**
- `n_clusters` (int): Number of clusters to create (default: 4)
- `random_state` (int): Random seed for reproducibility (default: 42)

**Methods:**

- **`fit(features: pd.DataFrame) -> KMeansClusterer`**
  - Fit clustering model to feature data
  - **Parameters:** `features` - DataFrame with numerical features
  - **Returns:** Self for method chaining

- **`predict(features: pd.DataFrame) -> np.ndarray`**
  - Predict cluster assignments for new data
  - **Parameters:** `features` - DataFrame with same structure as training data
  - **Returns:** Array of cluster labels

- **`get_cluster_assignments() -> pd.Series`**
  - Get cluster assignments for fitted data
  - **Returns:** Series with cluster labels

- **`get_cluster_centroids() -> pd.DataFrame`**
  - Get cluster centroids in original feature space
  - **Returns:** DataFrame with centroid coordinates

- **`get_cluster_quality_metrics() -> Dict[str, float]`**
  - Calculate clustering quality metrics
  - **Returns:** Dictionary with silhouette_score, calinski_harabasz_score

- **`find_optimal_clusters(features: pd.DataFrame, max_clusters: int = 10) -> Dict[int, Dict]`**
  - Find optimal number of clusters using multiple metrics
  - **Parameters:** 
    - `features` - Feature data
    - `max_clusters` - Maximum clusters to test
  - **Returns:** Dictionary mapping cluster counts to quality metrics

**Example:**
```python
clusterer = KMeansClusterer(n_clusters=5)
clusterer.fit(features)

# Get results
assignments = clusterer.get_cluster_assignments()
centroids = clusterer.get_cluster_centroids()
metrics = clusterer.get_cluster_quality_metrics()

print(f"Silhouette score: {metrics['silhouette_score']:.3f}")
```

#### `DataValidator`

Validates data quality and format compliance.

```python
from observer_coordinator_insights import DataValidator

validator = DataValidator()
```

**Methods:**

- **`validate_data_quality(data: pd.DataFrame) -> Dict[str, Any]`**
  - Comprehensive data quality validation
  - **Parameters:** `data` - DataFrame to validate
  - **Returns:** Dictionary with validation results:
    - `is_valid` (bool): Overall validation status
    - `quality_score` (float): Quality score 0-100
    - `errors` (List[str]): Validation errors
    - `warnings` (List[str]): Validation warnings

**Example:**
```python
validator = DataValidator()
results = validator.validate_data_quality(data)

if not results['is_valid']:
    print("Validation errors:")
    for error in results['errors']:
        print(f"  - {error}")
```

### `team_simulator`

#### `TeamCompositionSimulator`

Simulates and optimizes team compositions based on personality clusters.

```python
from observer_coordinator_insights import TeamCompositionSimulator

simulator = TeamCompositionSimulator()
```

**Methods:**

- **`load_employee_data(metadata: pd.DataFrame, cluster_assignments: pd.Series) -> None`**
  - Load employee data and cluster assignments
  - **Parameters:**
    - `metadata` - Employee metadata DataFrame
    - `cluster_assignments` - Series with cluster labels

- **`recommend_optimal_teams(num_teams: int, iterations: int = 5) -> List[Dict]`**
  - Generate optimal team compositions
  - **Parameters:**
    - `num_teams` - Number of teams to create
    - `iterations` - Number of optimization iterations
  - **Returns:** List of team compositions with balance scores

- **`get_team_recommendations_summary(compositions: List[Dict]) -> Dict[str, Any]`**
  - Summarize team recommendation results
  - **Parameters:** `compositions` - List of team compositions
  - **Returns:** Summary with best practices and insights

**Example:**
```python
simulator = TeamCompositionSimulator()
simulator.load_employee_data(metadata, cluster_assignments)

teams = simulator.recommend_optimal_teams(num_teams=3, iterations=10)
summary = simulator.get_team_recommendations_summary(teams)

best_team = teams[0]
print(f"Best composition balance score: {best_team['average_balance_score']:.2f}")
```

## CLI Usage

The package also provides a command-line interface:

```bash
# Basic clustering
insights-clustering data.csv --clusters 4 --output results/

# Optimize cluster count
insights-clustering data.csv --optimize-clusters --output results/

# Generate teams
insights-clustering data.csv --clusters 4 --teams 3 --output results/

# Validate data only
insights-clustering data.csv --validate-only
```

## Data Format Requirements

### Input CSV Format

The input CSV should contain Insights Discovery assessment results with these columns:

- `employee_id`: Unique identifier
- `name`: Employee name (will be anonymized)
- `role`: Job role/title
- `department`: Department/team
- `fiery_red`: Score for Fiery Red dimension (0-100)
- `sunshine_yellow`: Score for Sunshine Yellow dimension (0-100)
- `earth_green`: Score for Earth Green dimension (0-100)
- `cool_blue`: Score for Cool Blue dimension (0-100)

**Example:**
```csv
employee_id,name,role,department,fiery_red,sunshine_yellow,earth_green,cool_blue
EMP001,John Doe,Developer,Engineering,65,45,30,85
EMP002,Jane Smith,Designer,Product,40,80,70,35
```

### Output Formats

The library generates several output files:

- **`validation_report.json`**: Data quality assessment
- **`clustering_results.json`**: Cluster assignments and centroids
- **`team_compositions.json`**: Recommended team structures
- **`cluster_optimization.json`**: Optimal cluster analysis (if requested)

## Error Handling

All methods include comprehensive error handling and validation:

```python
try:
    parser = InsightsDataParser()
    data = parser.parse_csv("data.csv")
except ValueError as e:
    print(f"Data parsing failed: {e}")
except FileNotFoundError:
    print("Input file not found")
```

## Privacy and Security

- Employee names are automatically anonymized in output
- Data is encrypted during processing
- No PII is logged or persisted
- Compliance with GDPR data retention policies

## Performance Considerations

- Clustering complexity: O(n*k*i) where n=employees, k=clusters, i=iterations
- Memory usage: ~50MB per 1000 employees
- Recommended batch size: <5000 employees per analysis
- Use `find_optimal_clusters()` sparingly on large datasets

## Integration Examples

### With Pandas

```python
import pandas as pd
from observer_coordinator_insights import InsightsDataParser, KMeansClusterer

# Load data from various sources
df = pd.read_csv("employees.csv")
# or from database, API, etc.

parser = InsightsDataParser()
parser.data = df  # Direct data assignment
features = parser.get_clustering_features()

clusterer = KMeansClusterer()
clusterer.fit(features)
```

### With Jupyter Notebooks

```python
# Notebook-friendly visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Cluster visualization
centroids = clusterer.get_cluster_centroids()
sns.heatmap(centroids, annot=True, cmap='viridis')
plt.title('Cluster Centroids Heatmap')
plt.show()
```

### Batch Processing

```python
from pathlib import Path
from observer_coordinator_insights import InsightsDataParser, KMeansClusterer

def process_organization_data(data_dir: Path):
    """Process multiple department CSV files"""
    parser = InsightsDataParser()
    clusterer = KMeansClusterer()
    
    results = {}
    for csv_file in data_dir.glob("*.csv"):
        data = parser.parse_csv(csv_file)
        features = parser.get_clustering_features()
        clusterer.fit(features)
        
        results[csv_file.stem] = {
            'clusters': clusterer.get_cluster_assignments(),
            'metrics': clusterer.get_cluster_quality_metrics()
        }
    
    return results
```