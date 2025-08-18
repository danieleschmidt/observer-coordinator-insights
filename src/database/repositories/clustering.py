"""Clustering repository for database operations
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from ..models.clustering import ClusteringResult, ClusterMember
from .base import BaseRepository


class ClusteringRepository(BaseRepository):
    """Repository for clustering data operations"""

    def __init__(self, db: Session):
        super().__init__(db, ClusteringResult)

    def get_with_members(self, clustering_id: str) -> Optional[ClusteringResult]:
        """Get clustering result with all cluster members"""
        return self.db.query(ClusteringResult).options(
            joinedload(ClusteringResult.cluster_members).joinedload(ClusterMember.employee)
        ).filter(ClusteringResult.id == clustering_id).first()

    def get_recent_results(self, limit: int = 10) -> List[ClusteringResult]:
        """Get most recent clustering results"""
        return self.db.query(ClusteringResult).order_by(
            desc(ClusteringResult.created_at)
        ).limit(limit).all()

    def get_best_results(self, limit: int = 10) -> List[ClusteringResult]:
        """Get clustering results with highest silhouette scores"""
        return self.db.query(ClusteringResult).filter(
            ClusteringResult.silhouette_score.isnot(None)
        ).order_by(
            desc(ClusteringResult.silhouette_score)
        ).limit(limit).all()

    def get_by_algorithm(self, algorithm: str) -> List[ClusteringResult]:
        """Get all results for a specific algorithm"""
        return self.db.query(ClusteringResult).filter(
            ClusteringResult.algorithm == algorithm
        ).order_by(desc(ClusteringResult.created_at)).all()

    def search_results(
        self,
        algorithm: str = None,
        min_clusters: int = None,
        max_clusters: int = None,
        min_silhouette: float = None,
        status: str = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ClusteringResult]:
        """Search clustering results with filters"""
        query = self.db.query(ClusteringResult)

        if algorithm:
            query = query.filter(ClusteringResult.algorithm == algorithm)

        if min_clusters:
            query = query.filter(ClusteringResult.n_clusters >= min_clusters)

        if max_clusters:
            query = query.filter(ClusteringResult.n_clusters <= max_clusters)

        if min_silhouette:
            query = query.filter(ClusteringResult.silhouette_score >= min_silhouette)

        if status:
            query = query.filter(ClusteringResult.status == status)

        return query.order_by(
            desc(ClusteringResult.created_at)
        ).offset(skip).limit(limit).all()

    def create_with_members(
        self,
        clustering_data: Dict[str, Any],
        member_data: List[Dict[str, Any]]
    ) -> ClusteringResult:
        """Create clustering result with cluster members"""
        # Create clustering result
        clustering_result = ClusteringResult(**clustering_data)
        self.db.add(clustering_result)
        self.db.flush()  # Get the ID without committing

        # Create cluster members
        for member_info in member_data:
            member_info['clustering_result_id'] = clustering_result.id
            cluster_member = ClusterMember(**member_info)
            self.db.add(cluster_member)

        self.db.commit()
        self.db.refresh(clustering_result)
        return clustering_result

    def get_cluster_members(self, clustering_id: str, cluster_id: int = None) -> List[ClusterMember]:
        """Get cluster members for a specific clustering result"""
        query = self.db.query(ClusterMember).filter(
            ClusterMember.clustering_result_id == clustering_id
        )

        if cluster_id is not None:
            query = query.filter(ClusterMember.cluster_id == cluster_id)

        return query.all()

    def get_employee_cluster_history(self, employee_id: int) -> List[ClusterMember]:
        """Get clustering history for a specific employee"""
        return self.db.query(ClusterMember).options(
            joinedload(ClusterMember.clustering_result)
        ).filter(
            ClusterMember.employee_id == employee_id
        ).order_by(desc(ClusterMember.created_at)).all()

    def get_cluster_statistics(self, clustering_id: str) -> Dict[str, Any]:
        """Get detailed statistics for a clustering result"""
        clustering_result = self.get(clustering_id)
        if not clustering_result:
            return None

        # Get cluster size distribution
        cluster_sizes = self.db.query(
            ClusterMember.cluster_id,
            func.count(ClusterMember.id).label('size')
        ).filter(
            ClusterMember.clustering_result_id == clustering_id
        ).group_by(ClusterMember.cluster_id).all()

        # Get cluster quality metrics
        cluster_quality = self.db.query(
            ClusterMember.cluster_id,
            func.avg(ClusterMember.distance_to_centroid).label('avg_distance'),
            func.avg(ClusterMember.silhouette_score).label('avg_silhouette'),
            func.count(ClusterMember.id).filter(ClusterMember.is_outlier == 1).label('outlier_count')
        ).filter(
            ClusterMember.clustering_result_id == clustering_id
        ).group_by(ClusterMember.cluster_id).all()

        return {
            'clustering_id': clustering_id,
            'total_members': clustering_result.total_employees,
            'n_clusters': clustering_result.n_clusters,
            'overall_quality': {
                'silhouette_score': clustering_result.silhouette_score,
                'calinski_harabasz_score': clustering_result.calinski_harabasz_score,
                'inertia': clustering_result.inertia
            },
            'cluster_sizes': {row.cluster_id: row.size for row in cluster_sizes},
            'cluster_quality': {
                row.cluster_id: {
                    'avg_distance_to_centroid': round(row.avg_distance, 3) if row.avg_distance else None,
                    'avg_silhouette_score': round(row.avg_silhouette, 3) if row.avg_silhouette else None,
                    'outlier_count': row.outlier_count
                }
                for row in cluster_quality
            }
        }

    def compare_clustering_results(self, clustering_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple clustering results"""
        results = self.db.query(ClusteringResult).filter(
            ClusteringResult.id.in_(clustering_ids)
        ).all()

        if not results:
            return {}

        comparison = {
            'results_count': len(results),
            'algorithms_used': list(set(r.algorithm for r in results)),
            'cluster_counts': [r.n_clusters for r in results],
            'quality_comparison': {
                'best_silhouette': {
                    'score': max(r.silhouette_score for r in results if r.silhouette_score),
                    'clustering_id': max(results, key=lambda x: x.silhouette_score or 0).id
                },
                'best_calinski_harabasz': {
                    'score': max(r.calinski_harabasz_score for r in results if r.calinski_harabasz_score),
                    'clustering_id': max(results, key=lambda x: x.calinski_harabasz_score or 0).id
                },
                'lowest_inertia': {
                    'score': min(r.inertia for r in results if r.inertia),
                    'clustering_id': min(results, key=lambda x: x.inertia or float('inf')).id
                }
            },
            'detailed_results': [
                {
                    'clustering_id': r.id,
                    'algorithm': r.algorithm,
                    'n_clusters': r.n_clusters,
                    'silhouette_score': r.silhouette_score,
                    'calinski_harabasz_score': r.calinski_harabasz_score,
                    'inertia': r.inertia,
                    'total_employees': r.total_employees,
                    'processing_time': r.processing_time,
                    'created_at': r.created_at
                }
                for r in results
            ]
        }

        return comparison

    def get_optimal_cluster_analysis(self, min_clusters: int = 2, max_clusters: int = 10) -> Dict[str, Any]:
        """Analyze clustering results to find optimal cluster count"""
        results = self.db.query(ClusteringResult).filter(
            ClusteringResult.n_clusters >= min_clusters,
            ClusteringResult.n_clusters <= max_clusters,
            ClusteringResult.silhouette_score.isnot(None)
        ).order_by(ClusteringResult.n_clusters).all()

        if not results:
            return {}

        # Group by cluster count and get best result for each
        cluster_analysis = {}
        for result in results:
            n_clusters = result.n_clusters
            if (n_clusters not in cluster_analysis or
                result.silhouette_score > cluster_analysis[n_clusters]['silhouette_score']):
                cluster_analysis[n_clusters] = {
                    'clustering_id': result.id,
                    'silhouette_score': result.silhouette_score,
                    'calinski_harabasz_score': result.calinski_harabasz_score,
                    'inertia': result.inertia,
                    'processing_time': result.processing_time
                }

        # Find optimal based on silhouette score
        optimal_clusters = max(cluster_analysis.keys(),
                             key=lambda k: cluster_analysis[k]['silhouette_score'])

        return {
            'optimal_clusters': optimal_clusters,
            'optimal_result': cluster_analysis[optimal_clusters],
            'all_cluster_counts': cluster_analysis,
            'recommendations': [
                f"Based on silhouette score, {optimal_clusters} clusters is optimal",
                f"Silhouette score: {cluster_analysis[optimal_clusters]['silhouette_score']:.3f}"
            ]
        }
