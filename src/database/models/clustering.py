"""
Clustering results data models
"""

from sqlalchemy import Column, String, Float, Integer, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from .base import BaseModel, UUIDMixin


class ClusteringResult(BaseModel, UUIDMixin):
    """Clustering analysis results"""
    
    __tablename__ = 'clustering_results'
    __table_args__ = (
        Index('idx_algorithm', 'algorithm'),
        Index('idx_quality_score', 'silhouette_score'),
    )
    
    # Analysis metadata
    algorithm = Column(
        String(50),
        nullable=False,
        default='kmeans',
        comment="Clustering algorithm used"
    )
    n_clusters = Column(
        Integer,
        nullable=False,
        comment="Number of clusters"
    )
    random_state = Column(
        Integer,
        comment="Random seed used for reproducibility"
    )
    
    # Quality metrics
    silhouette_score = Column(
        Float,
        comment="Silhouette score (-1 to 1, higher is better)"
    )
    calinski_harabasz_score = Column(
        Float,
        comment="Calinski-Harabasz score (higher is better)"
    )
    inertia = Column(
        Float,
        comment="Within-cluster sum of squared distances"
    )
    
    # Analysis parameters
    parameters = Column(
        Text,
        comment="JSON object with algorithm parameters"
    )
    
    # Dataset information
    total_employees = Column(
        Integer,
        nullable=False,
        comment="Total number of employees in analysis"
    )
    data_quality_score = Column(
        Float,
        comment="Input data quality score"
    )
    
    # Processing metadata
    processing_time = Column(
        Float,
        comment="Processing time in seconds"
    )
    status = Column(
        String(20),
        default='completed',
        comment="Analysis status (pending, running, completed, failed)"
    )
    error_message = Column(
        Text,
        comment="Error message if analysis failed"
    )
    
    # Cluster centroids and characteristics
    centroids = Column(
        Text,
        comment="JSON object with cluster centroids"
    )
    cluster_characteristics = Column(
        Text,
        comment="JSON object with cluster characteristics and descriptions"
    )
    
    # Recommendations and insights
    recommendations = Column(
        Text,
        comment="JSON array of analysis recommendations"
    )
    insights = Column(
        Text,
        comment="JSON object with key insights from analysis"
    )
    
    # Relationships
    cluster_members = relationship(
        "ClusterMember",
        back_populates="clustering_result",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<ClusteringResult(id={self.id}, algorithm='{self.algorithm}', n_clusters={self.n_clusters})>"
    
    @property
    def cluster_distribution(self):
        """Get cluster size distribution"""
        cluster_sizes = {}
        for member in self.cluster_members:
            cluster_id = member.cluster_id
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        return cluster_sizes
    
    @property
    def average_cluster_size(self):
        """Get average cluster size"""
        if self.n_clusters > 0:
            return self.total_employees / self.n_clusters
        return 0


class ClusterMember(BaseModel):
    """Association between employees and clustering results"""
    
    __tablename__ = 'cluster_members'
    __table_args__ = (
        Index('idx_clustering_result_id', 'clustering_result_id'),
        Index('idx_employee_id', 'employee_id'),
        Index('idx_cluster_id', 'cluster_id'),
    )
    
    # Foreign keys
    clustering_result_id = Column(
        String(36),
        ForeignKey('clustering_results.id', ondelete='CASCADE'),
        nullable=False,
        comment="Reference to clustering result"
    )
    employee_id = Column(
        Integer,
        ForeignKey('employees.id', ondelete='CASCADE'),
        nullable=False,
        comment="Reference to employee"
    )
    
    # Cluster assignment
    cluster_id = Column(
        Integer,
        nullable=False,
        comment="Assigned cluster ID (0-based)"
    )
    
    # Distance metrics
    distance_to_centroid = Column(
        Float,
        comment="Distance from employee to cluster centroid"
    )
    silhouette_score = Column(
        Float,
        comment="Individual silhouette score for this assignment"
    )
    
    # Confidence and quality
    assignment_confidence = Column(
        Float,
        comment="Confidence score for cluster assignment (0-1)"
    )
    is_outlier = Column(
        Integer,  # Using Integer as Boolean for SQLite compatibility
        default=0,
        comment="Whether this assignment is considered an outlier (0/1)"
    )
    
    # Analysis metadata
    features_used = Column(
        Text,
        comment="JSON array of features used for clustering"
    )
    normalized_features = Column(
        Text,
        comment="JSON object with normalized feature values"
    )
    
    # Relationships
    clustering_result = relationship(
        "ClusteringResult",
        back_populates="cluster_members"
    )
    employee = relationship(
        "Employee",
        back_populates="cluster_memberships"
    )
    
    def __repr__(self):
        return f"<ClusterMember(employee_id={self.employee_id}, cluster_id={self.cluster_id})>"