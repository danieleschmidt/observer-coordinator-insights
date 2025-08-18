"""Employee data model
"""

from sqlalchemy import Boolean, Column, Float, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from .base import BaseModel


class Employee(BaseModel):
    """Employee Insights Discovery data model"""

    __tablename__ = 'employees'
    __table_args__ = (
        Index('idx_employee_id', 'employee_id'),
        Index('idx_department', 'department'),
        Index('idx_is_active', 'is_active'),
    )

    # Core identification
    employee_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
        comment="Unique employee identifier"
    )

    # Insights Discovery energy data
    red_energy = Column(
        Float,
        nullable=False,
        comment="Red energy percentage (0-100)"
    )
    blue_energy = Column(
        Float,
        nullable=False,
        comment="Blue energy percentage (0-100)"
    )
    green_energy = Column(
        Float,
        nullable=False,
        comment="Green energy percentage (0-100)"
    )
    yellow_energy = Column(
        Float,
        nullable=False,
        comment="Yellow energy percentage (0-100)"
    )

    # Optional metadata
    first_name = Column(
        String(100),
        comment="Employee first name (optional, anonymized)"
    )
    last_name = Column(
        String(100),
        comment="Employee last name (optional, anonymized)"
    )
    department = Column(
        String(100),
        comment="Employee department"
    )
    role = Column(
        String(100),
        comment="Employee role/title"
    )
    team = Column(
        String(100),
        comment="Current team assignment"
    )

    # Skills and attributes
    skills = Column(
        Text,
        comment="JSON array of employee skills"
    )
    certifications = Column(
        Text,
        comment="JSON array of certifications"
    )
    experience_years = Column(
        Integer,
        comment="Years of experience"
    )

    # Status and lifecycle
    is_active = Column(
        Boolean,
        default=True,
        comment="Whether employee is currently active"
    )
    hire_date = Column(
        String(10),  # YYYY-MM-DD format
        comment="Employee hire date (anonymized to year/month if needed)"
    )

    # Data processing metadata
    data_source = Column(
        String(100),
        default='upload',
        comment="Source of employee data (upload, integration, etc.)"
    )
    anonymized = Column(
        Boolean,
        default=False,
        comment="Whether PII has been anonymized"
    )
    validation_score = Column(
        Float,
        comment="Data validation quality score (0-100)"
    )

    # Relationships
    cluster_memberships = relationship(
        "ClusterMember",
        back_populates="employee",
        cascade="all, delete-orphan"
    )
    team_memberships = relationship(
        "TeamMemberModel",
        back_populates="employee",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Employee(id={self.id}, employee_id='{self.employee_id}', department='{self.department}')>"

    @property
    def full_name(self):
        """Get full name if available"""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        elif self.first_name:
            return self.first_name
        else:
            return self.employee_id

    @property
    def energy_profile(self):
        """Get energy profile as dictionary"""
        return {
            'red_energy': self.red_energy,
            'blue_energy': self.blue_energy,
            'green_energy': self.green_energy,
            'yellow_energy': self.yellow_energy
        }

    @property
    def dominant_energy(self):
        """Get dominant energy type"""
        energies = self.energy_profile
        return max(energies, key=energies.get).replace('_energy', '')

    def validate_energy_sum(self):
        """Validate that energy values sum to approximately 100"""
        total = self.red_energy + self.blue_energy + self.green_energy + self.yellow_energy
        return 95 <= total <= 105  # Allow 5% tolerance

    def normalize_energies(self):
        """Normalize energy values to sum to 100"""
        total = self.red_energy + self.blue_energy + self.green_energy + self.yellow_energy
        if total > 0:
            self.red_energy = (self.red_energy / total) * 100
            self.blue_energy = (self.blue_energy / total) * 100
            self.green_energy = (self.green_energy / total) * 100
            self.yellow_energy = (self.yellow_energy / total) * 100

    def anonymize_pii(self):
        """Anonymize personally identifiable information"""
        if not self.anonymized:
            self.first_name = None
            self.last_name = None
            # Keep department and role as they're useful for analysis
            self.anonymized = True
