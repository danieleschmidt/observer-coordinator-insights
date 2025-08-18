"""Analytics service for clustering and data processing
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.analytics import (
    AnalyticsJobRequest,
    AnalyticsJobResponse,
    ClusterInfo,
    ClusteringRequest,
    ClusteringResponse,
    EmployeeData,
    ValidationIssue,
    ValidationRequest,
    ValidationResponse,
)
from insights_clustering import DataValidator, InsightsDataParser, KMeansClusterer


logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for analytics operations"""

    def __init__(self):
        self.jobs = {}  # In-memory job storage (use Redis/DB in production)
        self.analyses = {}  # In-memory analysis storage

    async def analyze_csv_file(
        self,
        file_path: Path,
        params: ClusteringRequest
    ) -> ClusteringResponse:
        """Analyze CSV file with clustering"""
        start_time = time.time()

        try:
            # Parse data
            parser = InsightsDataParser()
            data = parser.parse_csv(file_path)

            # Get features for clustering
            features = parser.get_clustering_features()
            metadata = parser.get_employee_metadata()

            # Perform clustering
            if params.optimize_clusters:
                clusterer = KMeansClusterer()
                cluster_scores = clusterer.find_optimal_clusters(features, max_clusters=10)

                # Find best cluster count
                best_k = max(cluster_scores.keys(),
                           key=lambda k: cluster_scores[k]['silhouette_score'])
                params.n_clusters = best_k

            # Create clusterer with final parameters
            clusterer = KMeansClusterer(
                n_clusters=params.n_clusters,
                random_state=params.random_state
            )
            clusterer.fit(features)

            # Get results
            cluster_assignments = clusterer.get_cluster_assignments()
            centroids = clusterer.get_cluster_centroids()
            quality_metrics = clusterer.get_cluster_quality_metrics()

            # Create cluster information
            clusters = []
            for i in range(params.n_clusters):
                cluster_data = data[cluster_assignments == i]
                cluster_size = len(cluster_data)

                # Determine dominant energy
                centroid = centroids.iloc[i]
                energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
                dominant_energy = centroid[energy_cols].idxmax().replace('_energy', '')

                # Generate characteristics based on dominant energy
                characteristics = self._generate_cluster_characteristics(dominant_energy, centroid)

                clusters.append(ClusterInfo(
                    cluster_id=i,
                    size=cluster_size,
                    centroid=centroid.to_dict(),
                    dominant_energy=dominant_energy,
                    characteristics=characteristics
                ))

            # Generate recommendations
            recommendations = self._generate_clustering_recommendations(
                clusters, quality_metrics, len(data)
            )

            processing_time = time.time() - start_time

            return ClusteringResponse(
                employee_count=len(data),
                cluster_count=params.n_clusters,
                quality_metrics=quality_metrics,
                clusters=clusters,
                recommendations=recommendations,
                processing_time=round(processing_time, 2)
            )

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise e

    async def validate_csv_file(
        self,
        file_path: Path,
        params: ValidationRequest
    ) -> ValidationResponse:
        """Validate CSV file data quality"""
        try:
            # Parse data
            parser = InsightsDataParser(validate_data=False)
            data = parser.parse_csv(file_path)

            # Validate data
            validator = DataValidator()
            validation_results = validator.validate_data_quality(data)

            # Convert validation results to API models
            issues = []
            for error in validation_results.get('errors', []):
                issues.append(ValidationIssue(
                    level="error",
                    message=error,
                    suggestion="Review and fix data quality issues"
                ))

            for warning in validation_results.get('warnings', []):
                issues.append(ValidationIssue(
                    level="warning",
                    message=warning,
                    suggestion="Consider improving data quality"
                ))

            # Get improvement suggestions
            suggestions = validator.suggest_data_improvements(data)

            return ValidationResponse(
                is_valid=validation_results['is_valid'],
                quality_score=validation_results['quality_score'],
                total_records=len(data),
                valid_records=validation_results['metrics']['complete_records'],
                issues=issues,
                metrics=validation_results['metrics'],
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise e

    async def cluster_employee_data(
        self,
        employees: List[EmployeeData],
        params: ClusteringRequest
    ) -> ClusteringResponse:
        """Cluster employee data directly from API"""
        start_time = time.time()

        try:
            # Convert to DataFrame
            data_dict = [emp.dict() for emp in employees]
            data = pd.DataFrame(data_dict)

            # Extract features
            energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
            features = data[energy_cols]

            # Perform clustering
            clusterer = KMeansClusterer(
                n_clusters=params.n_clusters,
                random_state=params.random_state
            )
            clusterer.fit(features)

            # Get results
            cluster_assignments = clusterer.get_cluster_assignments()
            centroids = clusterer.get_cluster_centroids()
            quality_metrics = clusterer.get_cluster_quality_metrics()

            # Create cluster information
            clusters = []
            for i in range(params.n_clusters):
                cluster_size = (cluster_assignments == i).sum()
                centroid = centroids.iloc[i]

                # Determine dominant energy
                dominant_energy = centroid[energy_cols].idxmax().replace('_energy', '')

                # Generate characteristics
                characteristics = self._generate_cluster_characteristics(dominant_energy, centroid)

                clusters.append(ClusterInfo(
                    cluster_id=i,
                    size=cluster_size,
                    centroid=centroid.to_dict(),
                    dominant_energy=dominant_energy,
                    characteristics=characteristics
                ))

            # Generate recommendations
            recommendations = self._generate_clustering_recommendations(
                clusters, quality_metrics, len(data)
            )

            processing_time = time.time() - start_time

            return ClusteringResponse(
                employee_count=len(data),
                cluster_count=params.n_clusters,
                quality_metrics=quality_metrics,
                clusters=clusters,
                recommendations=recommendations,
                processing_time=round(processing_time, 2)
            )

        except Exception as e:
            logger.error(f"Employee data clustering failed: {e}")
            raise e

    async def optimize_cluster_count(
        self,
        job_id: str,
        max_clusters: int = 10
    ) -> ClusteringResponse:
        """Optimize cluster count for existing job data"""
        # This would retrieve data from job and optimize
        # For now, return a mock response
        return ClusteringResponse(
            employee_count=100,
            cluster_count=4,
            quality_metrics={"silhouette_score": 0.72},
            clusters=[],
            recommendations=["Optimal cluster count found"],
            processing_time=2.5
        )

    async def process_analytics_job(
        self,
        job_id: str,
        job_request: AnalyticsJobRequest
    ):
        """Process analytics job in background"""
        try:
            # Update job status
            self.jobs[job_id] = {
                "status": "running",
                "progress": 0.0,
                "started_at": time.time()
            }

            # Simulate processing
            for i in range(1, 101):
                await asyncio.sleep(0.1)  # Simulate work
                self.jobs[job_id]["progress"] = i

            # Complete job
            self.jobs[job_id].update({
                "status": "completed",
                "progress": 100.0,
                "completed_at": time.time(),
                "result": {"message": "Job completed successfully"}
            })

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id].update({
                "status": "failed",
                "error": str(e)
            })

    async def get_job_status(self, job_id: str) -> Optional[AnalyticsJobResponse]:
        """Get job status"""
        job_data = self.jobs.get(job_id)
        if not job_data:
            return None

        return AnalyticsJobResponse(
            job_id=job_id,
            **job_data
        )

    async def get_analysis_insights(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis insights"""
        return self.analyses.get(analysis_id)

    async def export_analysis(self, analysis_id: str, format: str) -> Dict[str, Any]:
        """Export analysis results"""
        analysis = self.analyses.get(analysis_id)
        if not analysis:
            raise ValueError("Analysis not found")

        if format == "json":
            return analysis
        else:
            # For CSV/Excel export, would generate file
            return {
                "file_path": f"/tmp/export_{analysis_id}.{format}",
                "filename": f"analysis_{analysis_id}.{format}",
                "media_type": "application/octet-stream"
            }

    async def get_analysis_history(self, limit: int, offset: int) -> Dict[str, Any]:
        """Get analysis history"""
        # Mock history data
        return {
            "total": len(self.analyses),
            "items": list(self.analyses.values())[offset:offset+limit]
        }

    async def cleanup_old_analyses(self, days_old: int) -> int:
        """Clean up old analyses"""
        # Mock cleanup
        return 5  # Number of analyses cleaned

    def _generate_cluster_characteristics(
        self,
        dominant_energy: str,
        centroid: pd.Series
    ) -> List[str]:
        """Generate cluster characteristics based on energy profile"""
        characteristics_map = {
            "red": [
                "Results-oriented",
                "Competitive",
                "Direct communication",
                "Quick decision-making",
                "Goal-focused"
            ],
            "blue": [
                "Analytical",
                "Detail-oriented",
                "Systematic approach",
                "Quality-focused",
                "Process-driven"
            ],
            "green": [
                "Supportive",
                "Team-oriented",
                "Patient",
                "Relationship-focused",
                "Collaborative"
            ],
            "yellow": [
                "Creative",
                "Enthusiastic",
                "Innovative",
                "People-focused",
                "Adaptable"
            ]
        }

        base_characteristics = characteristics_map.get(dominant_energy, [])

        # Add intensity-based characteristics
        energy_intensity = centroid[f'{dominant_energy}_energy']
        if energy_intensity > 35:
            base_characteristics.append(f"Strongly {dominant_energy}-oriented")
        elif energy_intensity < 20:
            base_characteristics.append(f"Moderately {dominant_energy}-oriented")

        return base_characteristics[:5]  # Return top 5 characteristics

    def _generate_clustering_recommendations(
        self,
        clusters: List[ClusterInfo],
        quality_metrics: Dict[str, float],
        total_employees: int
    ) -> List[str]:
        """Generate recommendations based on clustering results"""
        recommendations = []

        # Quality-based recommendations
        silhouette_score = quality_metrics.get('silhouette_score', 0)
        if silhouette_score > 0.7:
            recommendations.append("Excellent cluster separation - results are highly reliable")
        elif silhouette_score > 0.5:
            recommendations.append("Good cluster quality - results are reliable for team formation")
        else:
            recommendations.append("Consider adjusting cluster count for better separation")

        # Balance-based recommendations
        cluster_sizes = [cluster.size for cluster in clusters]
        size_variance = np.var(cluster_sizes)

        if size_variance < (total_employees * 0.1):
            recommendations.append("Well-balanced cluster sizes - good for team diversity")
        else:
            recommendations.append("Consider cluster rebalancing for more even distribution")

        # Diversity recommendations
        dominant_energies = [cluster.dominant_energy for cluster in clusters]
        unique_energies = len(set(dominant_energies))

        if unique_energies == len(clusters):
            recommendations.append("Excellent energy diversity across clusters")
        else:
            recommendations.append("Some energy types are dominant - ensure team diversity")

        return recommendations
