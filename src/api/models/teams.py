"""
Team composition and simulation API models
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime


class TeamMember(BaseModel):
    """Team member model"""
    employee_id: str = Field(..., description="Employee identifier")
    cluster: int = Field(..., description="Assigned cluster")
    energy_profile: Dict[str, float] = Field(..., description="Energy distribution")
    role: Optional[str] = Field(None, description="Suggested role in team")
    department: Optional[str] = Field(None, description="Employee department")
    skills: Optional[List[str]] = Field(None, description="Employee skills")
    
    class Config:
        schema_extra = {
            "example": {
                "employee_id": "EMP001",
                "cluster": 0,
                "energy_profile": {
                    "red_energy": 35.2,
                    "blue_energy": 25.1,
                    "green_energy": 20.3,
                    "yellow_energy": 19.4
                },
                "role": "Team Lead",
                "department": "Engineering",
                "skills": ["Python", "Leadership", "Analytics"]
            }
        }


class TeamComposition(BaseModel):
    """Team composition model"""
    team_id: int = Field(..., description="Team identifier")
    members: List[TeamMember] = Field(..., description="Team members")
    balance_score: float = Field(..., ge=0, le=100, description="Team balance score")
    size: int = Field(..., description="Team size")
    cluster_distribution: Dict[str, int] = Field(..., description="Cluster representation")
    energy_averages: Dict[str, float] = Field(..., description="Average energy distribution")
    dominant_energy: str = Field(..., description="Team's dominant energy type")
    effectiveness_score: float = Field(..., ge=0, le=100, description="Predicted team effectiveness")
    strengths: List[str] = Field(..., description="Team strengths")
    potential_challenges: List[str] = Field(..., description="Potential team challenges")
    
    class Config:
        schema_extra = {
            "example": {
                "team_id": 0,
                "members": [],
                "balance_score": 85.5,
                "size": 5,
                "cluster_distribution": {"cluster_0": 2, "cluster_1": 1, "cluster_2": 2},
                "energy_averages": {
                    "red_energy": 28.5,
                    "blue_energy": 26.2,
                    "green_energy": 23.1,
                    "yellow_energy": 22.2
                },
                "dominant_energy": "red",
                "effectiveness_score": 82.3,
                "strengths": ["Strong leadership", "Balanced perspectives", "Good decision-making"],
                "potential_challenges": ["May lack detailed analysis", "Could rush decisions"]
            }
        }


class TeamSimulationRequest(BaseModel):
    """Request model for team simulation"""
    target_teams: int = Field(default=3, ge=1, le=20, description="Number of teams to generate")
    min_team_size: int = Field(default=3, ge=2, le=15, description="Minimum team size")
    max_team_size: int = Field(default=8, ge=3, le=20, description="Maximum team size")
    iterations: int = Field(default=10, ge=1, le=100, description="Simulation iterations")
    optimize_for: str = Field(default="balance", description="Optimization target: balance, diversity, effectiveness")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Team formation constraints")
    
    @validator('max_team_size')
    def validate_team_sizes(cls, v, values):
        """Validate team size constraints"""
        if 'min_team_size' in values and v < values['min_team_size']:
            raise ValueError('max_team_size must be >= min_team_size')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "target_teams": 3,
                "min_team_size": 4,
                "max_team_size": 7,
                "iterations": 15,
                "optimize_for": "balance",
                "constraints": {
                    "department_diversity": True,
                    "skill_requirements": ["Python", "Leadership"]
                }
            }
        }


class TeamSimulationResponse(BaseModel):
    """Response model for team simulation"""
    success: bool = Field(default=True)
    simulation_id: str = Field(..., description="Unique simulation identifier")
    best_composition: Dict[str, Any] = Field(..., description="Best team composition found")
    all_compositions: List[Dict[str, Any]] = Field(..., description="All generated compositions")
    recommendations: Dict[str, Any] = Field(..., description="Team formation recommendations")
    statistics: Dict[str, Any] = Field(..., description="Simulation statistics")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "simulation_id": "sim_123456",
                "best_composition": {
                    "teams": [],
                    "average_balance_score": 85.2,
                    "total_employees_assigned": 15
                },
                "all_compositions": [],
                "recommendations": {
                    "key_insights": ["Teams show good balance", "High effectiveness predicted"],
                    "improvement_suggestions": ["Consider cross-training", "Regular team assessments"]
                },
                "statistics": {
                    "iterations_run": 15,
                    "best_score": 85.2,
                    "average_score": 78.6,
                    "score_variance": 12.3
                },
                "processing_time": 2.45
            }
        }


class TeamOptimizationRequest(BaseModel):
    """Request model for team optimization"""
    current_teams: List[TeamComposition] = Field(..., description="Current team compositions")
    optimization_goals: List[str] = Field(..., description="Optimization objectives")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Optimization constraints")
    max_changes: int = Field(default=3, ge=0, le=10, description="Maximum member swaps allowed")


class TeamOptimizationResponse(BaseModel):
    """Response model for team optimization"""
    success: bool = Field(default=True)
    optimized_teams: List[TeamComposition] = Field(..., description="Optimized team compositions")
    changes_made: List[Dict[str, Any]] = Field(..., description="Changes made during optimization")
    improvement_metrics: Dict[str, float] = Field(..., description="Improvement achieved")
    recommendations: List[str] = Field(..., description="Further optimization suggestions")


class TeamAssessmentRequest(BaseModel):
    """Request model for team assessment"""
    team_composition: TeamComposition = Field(..., description="Team to assess")
    assessment_criteria: List[str] = Field(..., description="Assessment criteria")
    benchmark_data: Optional[Dict[str, Any]] = Field(None, description="Benchmark comparison data")


class TeamAssessmentResponse(BaseModel):
    """Response model for team assessment"""
    success: bool = Field(default=True)
    overall_score: float = Field(..., ge=0, le=100, description="Overall team score")
    category_scores: Dict[str, float] = Field(..., description="Scores by category")
    strengths: List[str] = Field(..., description="Team strengths identified")
    areas_for_improvement: List[str] = Field(..., description="Areas needing improvement")
    action_items: List[str] = Field(..., description="Recommended action items")
    benchmark_comparison: Optional[Dict[str, Any]] = Field(None, description="Benchmark comparison results")