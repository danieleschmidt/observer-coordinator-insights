"""Team composition data models
"""

from sqlalchemy import Column, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import BaseModel, UUIDMixin


class TeamComposition(BaseModel, UUIDMixin):
    """Team composition analysis results"""

    __tablename__ = 'team_compositions'
    __table_args__ = (
        Index('idx_simulation_id', 'simulation_id'),
        Index('idx_balance_score', 'balance_score'),
        Index('idx_status', 'status'),
    )

    # Simulation metadata
    simulation_id = Column(
        String(36),
        nullable=False,
        index=True,
        comment="Unique simulation identifier"
    )
    team_number = Column(
        Integer,
        nullable=False,
        comment="Team number within simulation"
    )

    # Team characteristics
    team_name = Column(
        String(100),
        comment="Optional team name"
    )
    team_size = Column(
        Integer,
        nullable=False,
        comment="Number of team members"
    )

    # Balance and effectiveness scores
    balance_score = Column(
        Float,
        nullable=False,
        comment="Team balance score (0-100)"
    )
    effectiveness_score = Column(
        Float,
        comment="Predicted team effectiveness (0-100)"
    )
    diversity_score = Column(
        Float,
        comment="Team diversity score (0-100)"
    )

    # Energy distribution
    avg_red_energy = Column(
        Float,
        comment="Average red energy in team"
    )
    avg_blue_energy = Column(
        Float,
        comment="Average blue energy in team"
    )
    avg_green_energy = Column(
        Float,
        comment="Average green energy in team"
    )
    avg_yellow_energy = Column(
        Float,
        comment="Average yellow energy in team"
    )
    dominant_energy = Column(
        String(20),
        comment="Team's dominant energy type"
    )

    # Cluster distribution
    cluster_distribution = Column(
        Text,
        comment="JSON object with cluster representation"
    )
    cluster_diversity = Column(
        Float,
        comment="Cluster diversity score (0-100)"
    )

    # Team analysis
    strengths = Column(
        Text,
        comment="JSON array of team strengths"
    )
    potential_challenges = Column(
        Text,
        comment="JSON array of potential challenges"
    )
    recommendations = Column(
        Text,
        comment="JSON array of recommendations"
    )

    # Simulation parameters
    optimization_criteria = Column(
        String(50),
        default='balance',
        comment="What the team was optimized for"
    )
    constraints = Column(
        Text,
        comment="JSON object with team formation constraints"
    )

    # Status and lifecycle
    status = Column(
        String(20),
        default='active',
        comment="Team status (draft, active, archived)"
    )
    performance_rating = Column(
        Float,
        comment="Actual performance rating if available"
    )

    # Relationships
    members = relationship(
        "TeamMemberModel",
        back_populates="team_composition",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<TeamComposition(id={self.id}, simulation_id='{self.simulation_id}', team_number={self.team_number})>"

    @property
    def energy_profile(self):
        """Get team energy profile as dictionary"""
        return {
            'red_energy': self.avg_red_energy,
            'blue_energy': self.avg_blue_energy,
            'green_energy': self.avg_green_energy,
            'yellow_energy': self.avg_yellow_energy
        }

    @property
    def member_count(self):
        """Get actual member count from relationships"""
        return len(self.members)


class TeamMemberModel(BaseModel):
    """Association between employees and team compositions"""

    __tablename__ = 'team_members'
    __table_args__ = (
        Index('idx_team_composition_id', 'team_composition_id'),
        Index('idx_employee_id', 'employee_id'),
        Index('idx_role', 'role'),
    )

    # Foreign keys
    team_composition_id = Column(
        String(36),
        ForeignKey('team_compositions.id', ondelete='CASCADE'),
        nullable=False,
        comment="Reference to team composition"
    )
    employee_id = Column(
        Integer,
        ForeignKey('employees.id', ondelete='CASCADE'),
        nullable=False,
        comment="Reference to employee"
    )

    # Role and contribution
    role = Column(
        String(100),
        comment="Suggested role in team"
    )
    contribution_score = Column(
        Float,
        comment="Expected contribution score to team (0-100)"
    )
    fit_score = Column(
        Float,
        comment="How well employee fits in this team (0-100)"
    )

    # Skills and capabilities
    primary_skills = Column(
        Text,
        comment="JSON array of primary skills brought to team"
    )
    skill_gaps = Column(
        Text,
        comment="JSON array of skill gaps this member has"
    )

    # Team dynamics
    communication_style = Column(
        String(50),
        comment="Communication style preference"
    )
    leadership_potential = Column(
        Float,
        comment="Leadership potential score (0-100)"
    )
    collaboration_score = Column(
        Float,
        comment="Collaboration effectiveness score (0-100)"
    )

    # Performance and development
    performance_history = Column(
        Text,
        comment="JSON object with performance history if available"
    )
    development_areas = Column(
        Text,
        comment="JSON array of development areas"
    )

    # Assignment metadata
    assignment_reason = Column(
        Text,
        comment="Reason for assignment to this team"
    )
    assignment_confidence = Column(
        Float,
        comment="Confidence in this assignment (0-1)"
    )

    # Relationships
    team_composition = relationship(
        "TeamComposition",
        back_populates="members"
    )
    employee = relationship(
        "Employee",
        back_populates="team_memberships"
    )

    def __repr__(self):
        return f"<TeamMemberModel(employee_id={self.employee_id}, team_id={self.team_composition_id}, role='{self.role}')>"
