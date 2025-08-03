"""
Team composition and simulation API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.models.teams import (
    TeamSimulationRequest, TeamSimulationResponse, TeamOptimizationRequest,
    TeamOptimizationResponse, TeamAssessmentRequest, TeamAssessmentResponse,
    TeamComposition
)
from api.models.base import SuccessResponse
from api.services.teams import TeamsService

router = APIRouter()
teams_service = TeamsService()


@router.post("/simulate", response_model=TeamSimulationResponse)
async def simulate_teams(request: TeamSimulationRequest):
    """Generate optimal team compositions"""
    
    try:
        result = await teams_service.simulate_team_compositions(request)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team simulation failed: {str(e)}")


@router.post("/optimize", response_model=TeamOptimizationResponse)
async def optimize_teams(request: TeamOptimizationRequest):
    """Optimize existing team compositions"""
    
    try:
        result = await teams_service.optimize_team_compositions(request)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team optimization failed: {str(e)}")


@router.post("/assess", response_model=TeamAssessmentResponse)
async def assess_team(request: TeamAssessmentRequest):
    """Assess team composition effectiveness"""
    
    try:
        result = await teams_service.assess_team_composition(request)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Team assessment failed: {str(e)}")


@router.get("/recommendations/{simulation_id}")
async def get_team_recommendations(simulation_id: str):
    """Get detailed team formation recommendations"""
    
    try:
        recommendations = await teams_service.get_detailed_recommendations(simulation_id)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="Simulation not found")
        
        return SuccessResponse(
            message="Recommendations retrieved successfully",
            data=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@router.get("/templates")
async def get_team_templates():
    """Get predefined team composition templates"""
    
    try:
        templates = await teams_service.get_team_templates()
        
        return SuccessResponse(
            message="Team templates retrieved",
            data=templates
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")


@router.post("/compare")
async def compare_team_compositions(
    compositions: List[TeamComposition]
):
    """Compare multiple team compositions"""
    
    if len(compositions) < 2:
        raise HTTPException(status_code=400, detail="At least 2 compositions required for comparison")
    
    try:
        comparison = await teams_service.compare_compositions(compositions)
        
        return SuccessResponse(
            message="Team compositions compared successfully",
            data=comparison
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@router.get("/analytics/{team_id}")
async def get_team_analytics(team_id: int):
    """Get detailed analytics for a specific team"""
    
    try:
        analytics = await teams_service.get_team_analytics(team_id)
        
        if not analytics:
            raise HTTPException(status_code=404, detail="Team not found")
        
        return SuccessResponse(
            message="Team analytics retrieved",
            data=analytics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")