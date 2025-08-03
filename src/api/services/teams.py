"""
Teams service for composition simulation and optimization
"""

import uuid
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from team_simulator import TeamCompositionSimulator
from api.models.teams import (
    TeamSimulationRequest, TeamSimulationResponse, TeamOptimizationRequest,
    TeamOptimizationResponse, TeamAssessmentRequest, TeamAssessmentResponse,
    TeamComposition, TeamMember
)

class TeamsService:
    """Service for team composition operations"""
    
    def __init__(self):
        self.simulations = {}  # In-memory storage (use DB in production)
        
    async def simulate_team_compositions(
        self, 
        request: TeamSimulationRequest
    ) -> TeamSimulationResponse:
        """Generate team compositions based on clustering data"""
        
        start_time = time.time()
        simulation_id = str(uuid.uuid4())
        
        try:
            # Create simulator
            simulator = TeamCompositionSimulator(
                min_team_size=request.min_team_size,
                max_team_size=request.max_team_size
            )
            
            # Generate mock employee data for demonstration
            # In production, this would come from the clustering results
            mock_data = self._generate_mock_employee_data(100)
            cluster_assignments = np.random.randint(0, 4, 100)
            
            simulator.load_employee_data(mock_data, cluster_assignments)
            
            # Generate team compositions
            compositions = simulator.recommend_optimal_teams(
                request.target_teams, 
                iterations=request.iterations
            )
            
            if not compositions:
                raise ValueError("No valid team compositions generated")
            
            best_composition = compositions[0]
            
            # Get recommendations
            recommendations = simulator.get_team_recommendations_summary(compositions)
            
            # Calculate statistics
            statistics = self._calculate_simulation_statistics(compositions)
            
            processing_time = time.time() - start_time
            
            # Store simulation results
            self.simulations[simulation_id] = {
                "request": request,
                "compositions": compositions,
                "recommendations": recommendations,
                "created_at": time.time()
            }
            
            return TeamSimulationResponse(
                simulation_id=simulation_id,
                best_composition=best_composition,
                all_compositions=compositions,
                recommendations=recommendations,
                statistics=statistics,
                processing_time=round(processing_time, 2)
            )
            
        except Exception as e:
            raise e
    
    async def optimize_team_compositions(
        self,
        request: TeamOptimizationRequest
    ) -> TeamOptimizationResponse:
        """Optimize existing team compositions"""
        
        try:
            # Simulate optimization process
            optimized_teams = []
            changes_made = []
            
            for team in request.current_teams:
                # Mock optimization - in reality would run optimization algorithms
                optimized_team = team.copy()
                optimized_team.balance_score = min(100, team.balance_score + 5)
                optimized_teams.append(optimized_team)
                
                if team.balance_score < 80:
                    changes_made.append({
                        "team_id": team.team_id,
                        "change_type": "member_swap",
                        "description": f"Improved balance score from {team.balance_score} to {optimized_team.balance_score}"
                    })
            
            improvement_metrics = {
                "average_balance_improvement": 5.0,
                "total_changes": len(changes_made),
                "optimization_score": 85.5
            }
            
            recommendations = [
                "Regular team assessments recommended",
                "Consider cross-training for skill gaps",
                "Monitor team dynamics over time"
            ]
            
            return TeamOptimizationResponse(
                optimized_teams=optimized_teams,
                changes_made=changes_made,
                improvement_metrics=improvement_metrics,
                recommendations=recommendations
            )
            
        except Exception as e:
            raise e
    
    async def assess_team_composition(
        self,
        request: TeamAssessmentRequest
    ) -> TeamAssessmentResponse:
        """Assess team composition effectiveness"""
        
        try:
            team = request.team_composition
            
            # Calculate assessment scores
            category_scores = {
                "balance": team.balance_score,
                "diversity": self._calculate_diversity_score(team),
                "skills_coverage": self._calculate_skills_coverage(team),
                "communication_potential": self._calculate_communication_score(team),
                "leadership_potential": self._calculate_leadership_score(team)
            }
            
            overall_score = np.mean(list(category_scores.values()))
            
            # Generate insights
            strengths = self._identify_team_strengths(team, category_scores)
            areas_for_improvement = self._identify_improvement_areas(team, category_scores)
            action_items = self._generate_action_items(team, category_scores)
            
            # Benchmark comparison if provided
            benchmark_comparison = None
            if request.benchmark_data:
                benchmark_comparison = self._compare_to_benchmark(team, request.benchmark_data)
            
            return TeamAssessmentResponse(
                overall_score=round(overall_score, 1),
                category_scores=category_scores,
                strengths=strengths,
                areas_for_improvement=areas_for_improvement,
                action_items=action_items,
                benchmark_comparison=benchmark_comparison
            )
            
        except Exception as e:
            raise e
    
    async def get_detailed_recommendations(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed recommendations for a simulation"""
        
        simulation = self.simulations.get(simulation_id)
        if not simulation:
            return None
        
        return {
            "simulation_id": simulation_id,
            "detailed_analysis": simulation["recommendations"],
            "team_formation_guide": self._generate_formation_guide(simulation),
            "success_factors": self._identify_success_factors(simulation),
            "risk_mitigation": self._generate_risk_mitigation(simulation)
        }
    
    async def get_team_templates(self) -> Dict[str, Any]:
        """Get predefined team composition templates"""
        
        templates = {
            "innovation_team": {
                "name": "Innovation Team",
                "description": "High creativity and adaptability",
                "ideal_composition": {
                    "yellow_energy": 35,
                    "red_energy": 25,
                    "blue_energy": 20,
                    "green_energy": 20
                },
                "size_range": [4, 6],
                "characteristics": ["Creative", "Fast-paced", "Experimental"]
            },
            "analysis_team": {
                "name": "Analysis Team", 
                "description": "Detail-oriented and systematic",
                "ideal_composition": {
                    "blue_energy": 40,
                    "green_energy": 25,
                    "red_energy": 20,
                    "yellow_energy": 15
                },
                "size_range": [3, 5],
                "characteristics": ["Analytical", "Thorough", "Process-driven"]
            },
            "execution_team": {
                "name": "Execution Team",
                "description": "Results-focused and efficient",
                "ideal_composition": {
                    "red_energy": 35,
                    "blue_energy": 25,
                    "green_energy": 25,
                    "yellow_energy": 15
                },
                "size_range": [5, 8],
                "characteristics": ["Goal-oriented", "Decisive", "Action-focused"]
            },
            "support_team": {
                "name": "Support Team",
                "description": "Collaborative and relationship-focused",
                "ideal_composition": {
                    "green_energy": 35,
                    "yellow_energy": 25,
                    "blue_energy": 25,
                    "red_energy": 15
                },
                "size_range": [4, 7],
                "characteristics": ["Supportive", "Patient", "Team-oriented"]
            }
        }
        
        return templates
    
    async def compare_compositions(self, compositions: List[TeamComposition]) -> Dict[str, Any]:
        """Compare multiple team compositions"""
        
        comparison = {
            "summary": {
                "total_compositions": len(compositions),
                "average_balance_score": np.mean([c.balance_score for c in compositions]),
                "best_composition": max(compositions, key=lambda x: x.balance_score).team_id,
                "most_diverse": self._find_most_diverse_composition(compositions)
            },
            "detailed_comparison": [],
            "recommendations": []
        }
        
        for comp in compositions:
            comparison["detailed_comparison"].append({
                "team_id": comp.team_id,
                "balance_score": comp.balance_score,
                "size": comp.size,
                "dominant_energy": comp.dominant_energy,
                "effectiveness_score": comp.effectiveness_score,
                "relative_ranking": self._calculate_relative_ranking(comp, compositions)
            })
        
        # Generate comparison recommendations
        comparison["recommendations"] = self._generate_comparison_recommendations(compositions)
        
        return comparison
    
    async def get_team_analytics(self, team_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed analytics for a specific team"""
        
        # Mock analytics data - in production would query actual team data
        return {
            "team_id": team_id,
            "performance_metrics": {
                "productivity_score": 85.2,
                "collaboration_score": 78.5,
                "innovation_score": 82.1,
                "satisfaction_score": 89.3
            },
            "energy_distribution": {
                "red_energy": 28.5,
                "blue_energy": 26.2,
                "green_energy": 23.1,
                "yellow_energy": 22.2
            },
            "strengths": ["Strong leadership", "Good decision-making", "Balanced perspectives"],
            "challenges": ["May rush decisions", "Need more detailed analysis"],
            "recommendations": ["Regular check-ins", "Skill development sessions"]
        }
    
    def _generate_mock_employee_data(self, count: int) -> dict:
        """Generate mock employee data for demonstration"""
        import pandas as pd
        
        employees = []
        for i in range(count):
            # Generate random energy values that sum to 100
            energies = np.random.dirichlet([1, 1, 1, 1]) * 100
            employees.append({
                'employee_id': f'EMP{i:03d}',
                'red_energy': round(energies[0], 1),
                'blue_energy': round(energies[1], 1),
                'green_energy': round(energies[2], 1),
                'yellow_energy': round(energies[3], 1),
                'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'])
            })
        
        return pd.DataFrame(employees)
    
    def _calculate_simulation_statistics(self, compositions: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from simulation results"""
        
        scores = [comp['average_balance_score'] for comp in compositions]
        
        return {
            "iterations_run": len(compositions),
            "best_score": max(scores),
            "worst_score": min(scores),
            "average_score": round(np.mean(scores), 2),
            "score_variance": round(np.var(scores), 2),
            "score_std": round(np.std(scores), 2)
        }
    
    def _calculate_diversity_score(self, team: TeamComposition) -> float:
        """Calculate team diversity score"""
        # Mock calculation
        return min(100, len(team.cluster_distribution) * 25)
    
    def _calculate_skills_coverage(self, team: TeamComposition) -> float:
        """Calculate skills coverage score"""
        # Mock calculation
        return 75.0 + np.random.uniform(-10, 15)
    
    def _calculate_communication_score(self, team: TeamComposition) -> float:
        """Calculate communication potential score"""
        # Mock calculation based on team size and diversity
        size_factor = min(1.0, team.size / 6)  # Optimal around 6 people
        diversity_factor = len(team.cluster_distribution) / 4  # Max 4 clusters
        return (size_factor * 0.6 + diversity_factor * 0.4) * 100
    
    def _calculate_leadership_score(self, team: TeamComposition) -> float:
        """Calculate leadership potential score"""
        # Mock calculation
        return 70.0 + np.random.uniform(-5, 20)
    
    def _identify_team_strengths(self, team: TeamComposition, scores: Dict[str, float]) -> List[str]:
        """Identify team strengths based on scores"""
        strengths = []
        
        if scores['balance'] > 80:
            strengths.append("Excellent energy balance")
        if scores['diversity'] > 75:
            strengths.append("Good cluster diversity")
        if scores['communication_potential'] > 80:
            strengths.append("Strong communication potential")
        if team.size >= 5:
            strengths.append("Adequate team size for complex tasks")
        
        return strengths
    
    def _identify_improvement_areas(self, team: TeamComposition, scores: Dict[str, float]) -> List[str]:
        """Identify areas for improvement"""
        improvements = []
        
        if scores['balance'] < 70:
            improvements.append("Energy balance could be improved")
        if scores['diversity'] < 50:
            improvements.append("Consider adding more diverse perspectives")
        if scores['skills_coverage'] < 70:
            improvements.append("Skills gap analysis recommended")
        if team.size < 4:
            improvements.append("Team may be too small for complex projects")
        
        return improvements
    
    def _generate_action_items(self, team: TeamComposition, scores: Dict[str, float]) -> List[str]:
        """Generate specific action items"""
        actions = []
        
        if scores['balance'] < 80:
            actions.append("Review team composition for better energy balance")
        if scores['communication_potential'] < 75:
            actions.append("Implement team communication protocols")
        
        actions.append("Schedule regular team effectiveness reviews")
        actions.append("Consider team building activities")
        
        return actions
    
    def _compare_to_benchmark(self, team: TeamComposition, benchmark: Dict[str, Any]) -> Dict[str, Any]:
        """Compare team to benchmark data"""
        return {
            "vs_benchmark": {
                "balance_score": team.balance_score - benchmark.get('average_balance', 70),
                "effectiveness": team.effectiveness_score - benchmark.get('average_effectiveness', 75)
            },
            "percentile_ranking": min(100, max(0, team.balance_score + np.random.uniform(-10, 10)))
        }
    
    def _generate_formation_guide(self, simulation: Dict) -> Dict[str, Any]:
        """Generate team formation guide"""
        return {
            "step_by_step": [
                "Identify required skills and roles",
                "Select core team members from different clusters",
                "Balance energy types for team dynamics",
                "Consider department representation",
                "Validate team size and composition"
            ],
            "best_practices": [
                "Aim for cluster diversity",
                "Include complementary energy types",
                "Consider communication styles",
                "Plan for leadership roles"
            ]
        }
    
    def _identify_success_factors(self, simulation: Dict) -> List[str]:
        """Identify success factors from simulation"""
        return [
            "Balanced energy distribution",
            "Appropriate team size",
            "Clear role definitions",
            "Regular team assessments",
            "Ongoing communication"
        ]
    
    def _generate_risk_mitigation(self, simulation: Dict) -> Dict[str, List[str]]:
        """Generate risk mitigation strategies"""
        return {
            "communication_risks": [
                "Establish clear communication channels",
                "Regular team meetings",
                "Conflict resolution procedures"
            ],
            "performance_risks": [
                "Set clear goals and metrics",
                "Regular performance reviews",
                "Skills development programs"
            ],
            "team_dynamics_risks": [
                "Team building activities",
                "Role clarity workshops",
                "Leadership development"
            ]
        }
    
    def _find_most_diverse_composition(self, compositions: List[TeamComposition]) -> int:
        """Find the most diverse team composition"""
        max_diversity = 0
        most_diverse_id = compositions[0].team_id
        
        for comp in compositions:
            diversity_score = len(comp.cluster_distribution)
            if diversity_score > max_diversity:
                max_diversity = diversity_score
                most_diverse_id = comp.team_id
        
        return most_diverse_id
    
    def _calculate_relative_ranking(self, team: TeamComposition, all_teams: List[TeamComposition]) -> int:
        """Calculate relative ranking of team among all teams"""
        sorted_teams = sorted(all_teams, key=lambda x: x.balance_score, reverse=True)
        for i, t in enumerate(sorted_teams):
            if t.team_id == team.team_id:
                return i + 1
        return len(all_teams)
    
    def _generate_comparison_recommendations(self, compositions: List[TeamComposition]) -> List[str]:
        """Generate recommendations based on composition comparison"""
        recommendations = []
        
        avg_balance = np.mean([c.balance_score for c in compositions])
        if avg_balance > 80:
            recommendations.append("All compositions show good balance")
        else:
            recommendations.append("Consider rebalancing lower-scoring teams")
        
        size_variance = np.var([c.size for c in compositions])
        if size_variance > 4:
            recommendations.append("Consider standardizing team sizes")
        
        recommendations.append("Monitor team performance over time")
        
        return recommendations