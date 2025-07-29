"""
Load testing scenarios for Observer Coordinator Insights.

This module contains load testing scenarios using Locust to simulate
realistic usage patterns and identify performance bottlenecks.
"""

import random
from typing import Any, Dict

from locust import HttpUser, TaskSet, between, task


class InsightsAnalysisTaskSet(TaskSet):
    """Task set for insights analysis operations."""
    
    def on_start(self) -> None:
        """Setup tasks to run when a simulated user starts."""
        self.employee_ids = [f"EMP{i:04d}" for i in range(1, 1001)]
        self.departments = ["Engineering", "HR", "Marketing", "Sales", "Finance"]
    
    @task(3)
    def upload_insights_data(self) -> None:
        """Simulate uploading insights data."""
        # Generate sample data payload
        sample_data = []
        for _ in range(random.randint(10, 100)):
            sample_data.append({
                "employee_id": random.choice(self.employee_ids),
                "red_energy": random.randint(0, 100),
                "blue_energy": random.randint(0, 100),
                "yellow_energy": random.randint(0, 100),
                "green_energy": random.randint(0, 100),
                "department": random.choice(self.departments)
            })
        
        # Simulate API call (when API is implemented)
        with self.client.post(
            "/api/insights/upload",
            json={"data": sample_data},
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")

    @task(5)
    def run_clustering_analysis(self) -> None:
        """Simulate running clustering analysis."""
        cluster_config = {
            "algorithm": random.choice(["kmeans", "hierarchical"]),
            "n_clusters": random.randint(3, 8),
            "include_visualization": random.choice([True, False])
        }
        
        with self.client.post(
            "/api/clustering/analyze",
            json=cluster_config,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Analysis failed: {response.status_code}")

    @task(2)
    def get_team_recommendations(self) -> None:
        """Simulate getting team composition recommendations."""
        params = {
            "team_size": random.randint(3, 12),
            "project_type": random.choice(["development", "research", "operations"]),
            "diversity_weight": random.uniform(0.1, 1.0)
        }
        
        with self.client.get(
            "/api/teams/recommend",
            params=params,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Recommendations failed: {response.status_code}")

    @task(1)
    def export_results(self) -> None:
        """Simulate exporting analysis results."""
        export_config = {
            "format": random.choice(["json", "csv", "pdf"]),
            "include_visualizations": random.choice([True, False]),
            "anonymize": True
        }
        
        with self.client.post(
            "/api/export",
            json=export_config,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Export failed: {response.status_code}")


class DataValidationTaskSet(TaskSet):
    """Task set for data validation operations."""
    
    @task(2)
    def validate_data_format(self) -> None:
        """Simulate data format validation."""
        # Simulate various data validation scenarios
        validation_data = {
            "format": random.choice(["csv", "xlsx", "json"]),
            "schema_version": "1.0",
            "validate_pii": True
        }
        
        with self.client.post(
            "/api/validation/format",
            json=validation_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Validation failed: {response.status_code}")

    @task(1)
    def check_data_quality(self) -> None:
        """Simulate data quality checks."""
        with self.client.get("/api/validation/quality", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Quality check failed: {response.status_code}")


class AnalystUser(HttpUser):
    """Simulates a data analyst user behavior."""
    
    tasks = [InsightsAnalysisTaskSet]
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    weight = 3  # More analyst users
    
    def on_start(self) -> None:
        """Setup when user session starts."""
        # Simulate user authentication (when implemented)
        pass


class AdminUser(HttpUser):
    """Simulates an admin user behavior."""
    
    tasks = [DataValidationTaskSet]
    wait_time = between(2, 8)  # Longer wait times for admin tasks
    weight = 1  # Fewer admin users
    
    def on_start(self) -> None:
        """Setup when admin session starts."""
        # Simulate admin authentication (when implemented)
        pass


class BurstLoadUser(HttpUser):
    """Simulates burst load scenarios."""
    
    wait_time = between(0.1, 0.5)  # Very short wait times for burst load
    weight = 1
    
    @task
    def rapid_requests(self) -> None:
        """Generate rapid successive requests."""
        endpoints = [
            "/api/health",
            "/api/status", 
            "/api/metrics",
            "/api/clustering/status"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Burst request failed: {response.status_code}")


# Load testing scenarios configuration
LOAD_SCENARIOS = {
    "baseline": {
        "users": 10,
        "spawn_rate": 1,
        "run_time": "10m",
        "description": "Baseline load with normal user behavior"
    },
    "peak_load": {
        "users": 100,
        "spawn_rate": 5,
        "run_time": "15m", 
        "description": "Peak load simulation"
    },
    "stress_test": {
        "users": 500,
        "spawn_rate": 20,
        "run_time": "20m",
        "description": "Stress test to find breaking point"
    },
    "spike_test": {
        "users": 1000,
        "spawn_rate": 100,
        "run_time": "5m",
        "description": "Sudden spike in traffic"
    },
    "endurance": {
        "users": 50,
        "spawn_rate": 2,
        "run_time": "2h",
        "description": "Long-running endurance test"
    }
}


def custom_load_shape():
    """
    Define custom load patterns for advanced testing.
    
    This function can be used with Locust's LoadTestShape to create
    custom load patterns that simulate real-world traffic patterns.
    """
    stages = [
        # Warm-up phase
        {"duration": 60, "users": 10, "spawn_rate": 1},
        # Ramp-up phase
        {"duration": 300, "users": 50, "spawn_rate": 2},
        # Peak phase
        {"duration": 600, "users": 100, "spawn_rate": 5},
        # Sustain phase
        {"duration": 1200, "users": 100, "spawn_rate": 0},
        # Ramp-down phase
        {"duration": 1500, "users": 20, "spawn_rate": -2},
        # Cool-down phase
        {"duration": 1800, "users": 5, "spawn_rate": -1}
    ]
    return stages


# Performance benchmarks and thresholds
PERFORMANCE_THRESHOLDS = {
    "response_time": {
        "p50": 200,  # ms
        "p95": 500,  # ms
        "p99": 1000  # ms
    },
    "throughput": {
        "min_rps": 10,  # requests per second
        "target_rps": 50
    },
    "error_rate": {
        "max_error_rate": 0.01  # 1%
    },
    "resource_usage": {
        "max_cpu": 80,  # %
        "max_memory": 512,  # MB
        "max_disk_io": 100  # MB/s
    }
}