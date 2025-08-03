"""
Team repository for database operations
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func, desc, and_
from .base import BaseRepository
from ..models.team import TeamComposition, TeamMemberModel


class TeamRepository(BaseRepository):
    """Repository for team composition data operations"""
    
    def __init__(self, db: Session):
        super().__init__(db, TeamComposition)
    
    def get_with_members(self, team_id: str) -> Optional[TeamComposition]:
        """Get team composition with all members"""
        return self.db.query(TeamComposition).options(
            joinedload(TeamComposition.members).joinedload(TeamMemberModel.employee)
        ).filter(TeamComposition.id == team_id).first()
    
    def get_by_simulation(self, simulation_id: str) -> List[TeamComposition]:
        """Get all team compositions for a simulation"""
        return self.db.query(TeamComposition).filter(
            TeamComposition.simulation_id == simulation_id
        ).order_by(TeamComposition.team_number).all()
    
    def get_best_compositions(self, limit: int = 10) -> List[TeamComposition]:
        """Get team compositions with highest balance scores"""
        return self.db.query(TeamComposition).filter(
            TeamComposition.balance_score.isnot(None)
        ).order_by(
            desc(TeamComposition.balance_score)
        ).limit(limit).all()
    
    def search_compositions(
        self,
        simulation_id: str = None,
        min_balance_score: float = None,
        min_team_size: int = None,
        max_team_size: int = None,
        dominant_energy: str = None,
        status: str = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[TeamComposition]:
        """Search team compositions with filters"""
        query = self.db.query(TeamComposition)
        
        if simulation_id:
            query = query.filter(TeamComposition.simulation_id == simulation_id)
        
        if min_balance_score:
            query = query.filter(TeamComposition.balance_score >= min_balance_score)
        
        if min_team_size:
            query = query.filter(TeamComposition.team_size >= min_team_size)
        
        if max_team_size:
            query = query.filter(TeamComposition.team_size <= max_team_size)
        
        if dominant_energy:
            query = query.filter(TeamComposition.dominant_energy == dominant_energy)
        
        if status:
            query = query.filter(TeamComposition.status == status)
        
        return query.order_by(
            desc(TeamComposition.balance_score)
        ).offset(skip).limit(limit).all()
    
    def create_with_members(
        self, 
        team_data: Dict[str, Any], 
        member_data: List[Dict[str, Any]]
    ) -> TeamComposition:
        """Create team composition with members"""
        # Create team composition
        team = TeamComposition(**team_data)
        self.db.add(team)
        self.db.flush()  # Get the ID without committing
        
        # Create team members
        for member_info in member_data:
            member_info['team_composition_id'] = team.id
            team_member = TeamMemberModel(**member_info)
            self.db.add(team_member)
        
        self.db.commit()
        self.db.refresh(team)
        return team
    
    def get_team_members(self, team_id: str) -> List[TeamMemberModel]:
        """Get all members of a team"""
        return self.db.query(TeamMemberModel).options(
            joinedload(TeamMemberModel.employee)
        ).filter(
            TeamMemberModel.team_composition_id == team_id
        ).all()
    
    def get_employee_team_history(self, employee_id: int) -> List[TeamMemberModel]:
        """Get team history for a specific employee"""
        return self.db.query(TeamMemberModel).options(
            joinedload(TeamMemberModel.team_composition)
        ).filter(
            TeamMemberModel.employee_id == employee_id
        ).order_by(desc(TeamMemberModel.created_at)).all()
    
    def get_simulation_statistics(self, simulation_id: str) -> Dict[str, Any]:
        """Get statistics for a team simulation"""
        teams = self.get_by_simulation(simulation_id)
        
        if not teams:
            return {}
        
        # Calculate statistics
        balance_scores = [t.balance_score for t in teams if t.balance_score]
        effectiveness_scores = [t.effectiveness_score for t in teams if t.effectiveness_score]
        team_sizes = [t.team_size for t in teams]
        
        # Get energy distribution across teams
        energy_stats = self.db.query(
            func.avg(TeamComposition.avg_red_energy).label('avg_red'),
            func.avg(TeamComposition.avg_blue_energy).label('avg_blue'),
            func.avg(TeamComposition.avg_green_energy).label('avg_green'),
            func.avg(TeamComposition.avg_yellow_energy).label('avg_yellow')
        ).filter(
            TeamComposition.simulation_id == simulation_id
        ).first()
        
        return {
            'simulation_id': simulation_id,
            'total_teams': len(teams),
            'balance_scores': {
                'average': sum(balance_scores) / len(balance_scores) if balance_scores else 0,
                'min': min(balance_scores) if balance_scores else 0,
                'max': max(balance_scores) if balance_scores else 0,
                'distribution': balance_scores
            },
            'effectiveness_scores': {
                'average': sum(effectiveness_scores) / len(effectiveness_scores) if effectiveness_scores else 0,
                'min': min(effectiveness_scores) if effectiveness_scores else 0,
                'max': max(effectiveness_scores) if effectiveness_scores else 0,
                'distribution': effectiveness_scores
            },
            'team_sizes': {
                'average': sum(team_sizes) / len(team_sizes) if team_sizes else 0,
                'min': min(team_sizes) if team_sizes else 0,
                'max': max(team_sizes) if team_sizes else 0,
                'distribution': team_sizes
            },
            'energy_distribution': {
                'avg_red_energy': round(energy_stats.avg_red, 2) if energy_stats.avg_red else 0,
                'avg_blue_energy': round(energy_stats.avg_blue, 2) if energy_stats.avg_blue else 0,
                'avg_green_energy': round(energy_stats.avg_green, 2) if energy_stats.avg_green else 0,
                'avg_yellow_energy': round(energy_stats.avg_yellow, 2) if energy_stats.avg_yellow else 0
            },
            'best_team': max(teams, key=lambda t: t.balance_score or 0).id if teams else None
        }
    
    def compare_team_compositions(self, team_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple team compositions"""
        teams = self.db.query(TeamComposition).filter(
            TeamComposition.id.in_(team_ids)
        ).all()
        
        if not teams:
            return {}
        
        comparison = {
            'teams_count': len(teams),
            'best_balance': max(teams, key=lambda t: t.balance_score or 0),
            'best_effectiveness': max(teams, key=lambda t: t.effectiveness_score or 0),
            'most_diverse': max(teams, key=lambda t: t.diversity_score or 0),
            'average_scores': {
                'balance': sum(t.balance_score for t in teams if t.balance_score) / len([t for t in teams if t.balance_score]),
                'effectiveness': sum(t.effectiveness_score for t in teams if t.effectiveness_score) / len([t for t in teams if t.effectiveness_score]),
                'diversity': sum(t.diversity_score for t in teams if t.diversity_score) / len([t for t in teams if t.diversity_score])
            },
            'detailed_comparison': [
                {
                    'team_id': t.id,
                    'team_number': t.team_number,
                    'team_size': t.team_size,
                    'balance_score': t.balance_score,
                    'effectiveness_score': t.effectiveness_score,
                    'diversity_score': t.diversity_score,
                    'dominant_energy': t.dominant_energy,
                    'energy_profile': {
                        'red': t.avg_red_energy,
                        'blue': t.avg_blue_energy,
                        'green': t.avg_green_energy,
                        'yellow': t.avg_yellow_energy
                    }
                }
                for t in teams
            ]
        }
        
        return comparison
    
    def get_team_recommendations(self, team_id: str) -> Dict[str, Any]:
        """Get recommendations for improving a team composition"""
        team = self.get_with_members(team_id)
        if not team:
            return {}
        
        recommendations = {
            'team_id': team_id,
            'current_scores': {
                'balance': team.balance_score,
                'effectiveness': team.effectiveness_score,
                'diversity': team.diversity_score
            },
            'strengths': [],
            'areas_for_improvement': [],
            'specific_recommendations': []
        }
        
        # Analyze team composition
        if team.balance_score and team.balance_score > 80:
            recommendations['strengths'].append("Excellent energy balance")
        elif team.balance_score and team.balance_score < 60:
            recommendations['areas_for_improvement'].append("Energy balance could be improved")
            recommendations['specific_recommendations'].append("Consider adding members with complementary energy types")
        
        if team.team_size < 4:
            recommendations['areas_for_improvement'].append("Team size may be too small")
            recommendations['specific_recommendations'].append("Consider adding 1-2 more members for better collaboration")
        elif team.team_size > 8:
            recommendations['areas_for_improvement'].append("Team size may be too large")
            recommendations['specific_recommendations'].append("Consider splitting into smaller sub-teams")
        
        if team.diversity_score and team.diversity_score > 75:
            recommendations['strengths'].append("Good diversity in perspectives")
        elif team.diversity_score and team.diversity_score < 50:
            recommendations['areas_for_improvement'].append("Limited diversity")
            recommendations['specific_recommendations'].append("Include members from different clusters or departments")
        
        return recommendations
    
    def optimize_team_composition(self, team_id: str, optimization_goals: List[str]) -> Dict[str, Any]:
        """Suggest optimizations for a team composition"""
        team = self.get_with_members(team_id)
        if not team:
            return {}
        
        optimization_result = {
            'team_id': team_id,
            'optimization_goals': optimization_goals,
            'current_state': {
                'balance_score': team.balance_score,
                'effectiveness_score': team.effectiveness_score,
                'diversity_score': team.diversity_score,
                'team_size': team.team_size
            },
            'suggested_changes': [],
            'expected_improvements': {}
        }
        
        # Mock optimization logic - in production this would use sophisticated algorithms
        if 'balance' in optimization_goals and team.balance_score and team.balance_score < 75:
            optimization_result['suggested_changes'].append({
                'type': 'member_adjustment',
                'description': 'Adjust energy balance by swapping members',
                'expected_improvement': 10
            })
        
        if 'diversity' in optimization_goals and team.diversity_score and team.diversity_score < 70:
            optimization_result['suggested_changes'].append({
                'type': 'diversity_enhancement',
                'description': 'Add members from underrepresented clusters',
                'expected_improvement': 15
            })
        
        if 'effectiveness' in optimization_goals:
            optimization_result['suggested_changes'].append({
                'type': 'skill_optimization',
                'description': 'Ensure all required skills are covered',
                'expected_improvement': 8
            })
        
        return optimization_result