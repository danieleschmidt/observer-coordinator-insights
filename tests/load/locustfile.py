"""
Load testing configuration using Locust for the Observer Coordinator Insights system.
Tests performance under various load conditions.
"""

from locust import HttpUser, task, between
import json
import csv
from io import StringIO
import random
from typing import Dict, Any


class InsightsAnalysisUser(HttpUser):
    """Simulates a user performing insights analysis tasks."""
    
    wait_time = between(2, 10)  # Wait 2-10 seconds between tasks
    
    def on_start(self):
        """Setup before test starts."""
        self.test_data = self._generate_test_csv()
        self.config = self._get_test_config()
    
    def _generate_test_csv(self) -> str:
        """Generate test CSV data for upload."""
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        # Write header
        writer.writerow(['employee_id', 'cool_blue', 'earth_green', 'sunshine_yellow', 'fiery_red'])
        
        # Write test data (50 employees)
        for i in range(50):
            # Generate random color energies that sum to 100
            energies = [random.randint(10, 40) for _ in range(4)]
            total = sum(energies)
            energies = [round(e * 100 / total, 1) for e in energies]
            
            writer.writerow([f'EMP{i+1:04d}'] + energies)
        
        return csv_buffer.getvalue()
    
    def _get_test_config(self) -> Dict[str, Any]:
        """Get test configuration."""
        return {
            "clustering": {
                "algorithm": "kmeans",
                "n_clusters": 4,
                "random_state": 42
            },
            "output": {
                "generate_plots": False,
                "save_results": False
            }
        }
    
    @task(3)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def metrics_endpoint(self):
        """Test metrics endpoint."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics endpoint failed: {response.status_code}")
    
    @task(1)
    def ready_check(self):
        """Test readiness endpoint."""
        with self.client.get("/ready", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Ready check failed: {response.status_code}")
    
    @task(5)
    def upload_and_analyze(self):
        """Test data upload and analysis workflow."""
        # Step 1: Upload CSV data
        files = {'file': ('test_data.csv', self.test_data, 'text/csv')}
        
        with self.client.post('/api/upload', files=files, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Upload failed: {response.status_code}")
                return
            
            upload_result = response.json()
            if 'job_id' not in upload_result:
                response.failure("No job_id in upload response")
                return
            
            job_id = upload_result['job_id']
            response.success()
        
        # Step 2: Start analysis
        analysis_data = {
            'job_id': job_id,
            'config': self.config
        }
        
        with self.client.post('/api/analyze', 
                            json=analysis_data, 
                            catch_response=True) as response:
            if response.status_code == 202:  # Accepted
                response.success()
            else:
                response.failure(f"Analysis start failed: {response.status_code}")
                return
        
        # Step 3: Check analysis status (polling)
        max_checks = 30  # Maximum status checks
        for _ in range(max_checks):
            with self.client.get(f'/api/status/{job_id}', catch_response=True) as response:
                if response.status_code != 200:
                    response.failure(f"Status check failed: {response.status_code}")
                    break
                
                status_data = response.json()
                if status_data.get('status') == 'completed':
                    response.success()
                    break
                elif status_data.get('status') == 'failed':
                    response.failure("Analysis job failed")
                    break
                
                self.wait()  # Wait before next status check
        else:
            # Timeout reached
            response.failure("Analysis timed out")
    
    @task(2)
    def get_results(self):
        """Test results retrieval."""
        # Assume we have a completed job ID (in real scenario, would track from previous tasks)
        job_id = "test_job_123"
        
        with self.client.get(f'/api/results/{job_id}', catch_response=True) as response:
            if response.status_code == 200:
                results = response.json()
                if 'clusters' in results:
                    response.success()
                else:
                    response.failure("Invalid results format")
            elif response.status_code == 404:
                # Expected for non-existent job
                response.success()
            else:
                response.failure(f"Results retrieval failed: {response.status_code}")
    
    @task(1)
    def download_report(self):
        """Test report download."""
        job_id = "test_job_123"
        
        with self.client.get(f'/api/download/{job_id}', catch_response=True) as response:
            if response.status_code == 200:
                if response.headers.get('content-type', '').startswith('application/'):
                    response.success()
                else:
                    response.failure("Invalid content type for download")
            elif response.status_code == 404:
                # Expected for non-existent job
                response.success()
            else:
                response.failure(f"Download failed: {response.status_code}")


class AdminUser(HttpUser):
    """Simulates administrative user actions."""
    
    wait_time = between(5, 15)
    weight = 1  # Less frequent than regular users
    
    @task(2)
    def system_stats(self):
        """Test system statistics endpoint."""
        with self.client.get("/admin/stats", catch_response=True) as response:
            if response.status_code in [200, 401, 403]:  # OK or auth required
                response.success()
            else:
                response.failure(f"System stats failed: {response.status_code}")
    
    @task(1)
    def user_management(self):
        """Test user management endpoints."""
        with self.client.get("/admin/users", catch_response=True) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"User management failed: {response.status_code}")
    
    @task(1)
    def cleanup_jobs(self):
        """Test job cleanup endpoint."""
        with self.client.delete("/admin/cleanup", catch_response=True) as response:
            if response.status_code in [200, 204, 401, 403]:
                response.success()
            else:
                response.failure(f"Cleanup failed: {response.status_code}")


class HighVolumeUser(HttpUser):
    """Simulates high-volume batch processing user."""
    
    wait_time = between(30, 60)  # Longer wait times for batch operations
    weight = 0.5  # Even less frequent
    
    def on_start(self):
        """Setup large dataset for testing."""
        self.large_dataset = self._generate_large_csv(1000)  # 1000 employees
    
    def _generate_large_csv(self, num_employees: int) -> str:
        """Generate large CSV for stress testing."""
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        writer.writerow(['employee_id', 'cool_blue', 'earth_green', 'sunshine_yellow', 'fiery_red'])
        
        for i in range(num_employees):
            energies = [random.randint(10, 40) for _ in range(4)]
            total = sum(energies)
            energies = [round(e * 100 / total, 1) for e in energies]
            writer.writerow([f'BATCH_EMP{i+1:06d}'] + energies)
        
        return csv_buffer.getvalue()
    
    @task
    def batch_analysis(self):
        """Test large batch analysis."""
        files = {'file': ('large_dataset.csv', self.large_dataset, 'text/csv')}
        
        with self.client.post('/api/upload', 
                            files=files, 
                            timeout=60,  # Longer timeout for large files
                            catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Large batch upload failed: {response.status_code}")


# Load test scenarios
class QuickTestUser(HttpUser):
    """Quick test scenario for smoke testing."""
    
    wait_time = between(1, 3)
    
    @task
    def quick_health_check(self):
        """Quick health and metrics check."""
        endpoints = ["/health", "/ready", "/metrics"]
        
        for endpoint in endpoints:
            with self.client.get(endpoint, catch_response=True) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"{endpoint} failed: {response.status_code}")