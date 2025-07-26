"""
Team Composition Simulator
Generates and evaluates potential team compositions based on clustering results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


class TeamCompositionSimulator:
    """Simulates team compositions and evaluates effectiveness"""
    
    def __init__(self, min_team_size: int = 3, max_team_size: int = 8):
        self.min_team_size = min_team_size
        self.max_team_size = max_team_size
        self.employee_data = None
        self.cluster_assignments = None
        
    def load_employee_data(self, employee_data: pd.DataFrame, 
                          cluster_assignments: np.ndarray):
        """Load employee data and cluster assignments"""
        self.employee_data = employee_data.copy()
        self.cluster_assignments = cluster_assignments
        self.employee_data['cluster'] = cluster_assignments
        
    def generate_balanced_teams(self, num_teams: int) -> List[Dict]:
        """Generate balanced teams with diverse cluster representation"""
        if self.employee_data is None:
            raise ValueError("Employee data not loaded. Call load_employee_data first.")
        
        teams = []
        available_employees = self.employee_data.copy()
        
        for team_id in range(num_teams):
            team = self._create_balanced_team(available_employees)
            if team is not None:
                teams.append({
                    'team_id': team_id,
                    'members': team,
                    'composition': self._analyze_team_composition(team),
                    'balance_score': self._calculate_team_balance_score(team)
                })
                # Remove selected employees from available pool
                available_employees = available_employees[
                    ~available_employees['employee_id'].isin(team['employee_id'])
                ]
        
        return teams
    
    def _create_balanced_team(self, available_employees: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Create a single balanced team from available employees"""
        if len(available_employees) < self.min_team_size:
            return None
        
        # Get cluster distribution
        cluster_counts = available_employees['cluster'].value_counts()
        unique_clusters = cluster_counts.index.tolist()
        
        team_members = []
        target_size = min(self.max_team_size, len(available_employees))
        
        # Try to get at least one member from each cluster
        for cluster in unique_clusters:
            if len(team_members) >= target_size:
                break
            cluster_members = available_employees[available_employees['cluster'] == cluster]
            if len(cluster_members) > 0:
                # Select member with most balanced energy profile
                selected = self._select_balanced_member(cluster_members)
                team_members.append(selected)
        
        # Fill remaining spots with best-fit members
        while len(team_members) < target_size:
            remaining = available_employees[
                ~available_employees['employee_id'].isin([m['employee_id'] for m in team_members])
            ]
            if len(remaining) == 0:
                break
            
            best_fit = self._select_complementary_member(team_members, remaining)
            if best_fit is not None:
                team_members.append(best_fit)
            else:
                break
        
        if len(team_members) >= self.min_team_size:
            return pd.DataFrame(team_members)
        return None
    
    def _select_balanced_member(self, candidates: pd.DataFrame) -> Dict:
        """Select member with most balanced energy profile"""
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        # Calculate balance score for each candidate (lower variance = more balanced)
        balance_scores = []
        for _, candidate in candidates.iterrows():
            energy_values = [candidate[col] for col in energy_cols]
            balance_score = np.var(energy_values)  # Lower variance = more balanced
            balance_scores.append(balance_score)
        
        # Select candidate with most balanced profile
        best_idx = np.argmin(balance_scores)
        return candidates.iloc[best_idx].to_dict()
    
    def _select_complementary_member(self, current_team: List[Dict], 
                                   candidates: pd.DataFrame) -> Optional[Dict]:
        """Select member that best complements current team composition"""
        if len(candidates) == 0:
            return None
        
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        # Calculate current team average energy profile
        if current_team:
            team_energy_avg = {
                col: np.mean([member[col] for member in current_team])
                for col in energy_cols
            }
        else:
            team_energy_avg = {col: 25.0 for col in energy_cols}  # Target balanced
        
        # Find candidate that best balances the team
        best_score = float('inf')
        best_candidate = None
        
        for _, candidate in candidates.iterrows():
            # Calculate what team profile would be with this candidate
            new_team_size = len(current_team) + 1
            new_energy_avg = {}
            
            for col in energy_cols:
                current_total = team_energy_avg[col] * len(current_team)
                new_total = current_total + candidate[col]
                new_energy_avg[col] = new_total / new_team_size
            
            # Score how close this gets us to balanced (25% each energy)
            balance_score = sum(abs(new_energy_avg[col] - 25.0) for col in energy_cols)
            
            if balance_score < best_score:
                best_score = balance_score
                best_candidate = candidate.to_dict()
        
        return best_candidate
    
    def _analyze_team_composition(self, team: pd.DataFrame) -> Dict:
        """Analyze team composition metrics"""
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        composition = {
            'size': len(team),
            'cluster_distribution': team['cluster'].value_counts().to_dict(),
            'energy_averages': {col: team[col].mean() for col in energy_cols},
            'energy_diversity': {col: team[col].std() for col in energy_cols},
            'dominant_energy': None
        }
        
        # Determine dominant energy type
        avg_energies = composition['energy_averages']
        dominant_energy = max(avg_energies.keys(), key=lambda k: avg_energies[k])
        composition['dominant_energy'] = dominant_energy
        
        return composition
    
    def _calculate_team_balance_score(self, team: pd.DataFrame) -> float:
        """Calculate overall team balance score (higher is better)"""
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        
        # Energy balance score (closer to 25% each = better)
        energy_averages = [team[col].mean() for col in energy_cols]
        energy_balance = 100 - sum(abs(avg - 25.0) for avg in energy_averages)
        
        # Cluster diversity score
        unique_clusters = team['cluster'].nunique()
        max_possible_clusters = min(4, len(team))  # Assume max 4 clusters
        cluster_diversity = (unique_clusters / max_possible_clusters) * 100
        
        # Size appropriateness score
        if self.min_team_size <= len(team) <= self.max_team_size:
            size_score = 100
        else:
            size_score = 50  # Penalty for non-optimal size
        
        # Weighted overall score
        overall_score = (
            energy_balance * 0.5 +
            cluster_diversity * 0.3 +
            size_score * 0.2
        )
        
        return round(overall_score, 2)
    
    def recommend_optimal_teams(self, target_teams: int, 
                              iterations: int = 10) -> List[Dict]:
        """Generate multiple team compositions and recommend the best ones"""
        best_compositions = []
        
        for _ in range(iterations):
            teams = self.generate_balanced_teams(target_teams)
            total_score = sum(team['balance_score'] for team in teams)
            avg_score = total_score / len(teams) if teams else 0
            
            best_compositions.append({
                'teams': teams,
                'average_balance_score': avg_score,
                'total_employees_assigned': sum(team['composition']['size'] for team in teams)
            })
        
        # Sort by average balance score
        best_compositions.sort(key=lambda x: x['average_balance_score'], reverse=True)
        
        return best_compositions
    
    def get_team_recommendations_summary(self, compositions: List[Dict]) -> Dict:
        """Generate summary of team composition recommendations"""
        if not compositions:
            return {'error': 'No compositions provided'}
        
        best_composition = compositions[0]
        
        summary = {
            'recommended_composition': best_composition,
            'key_insights': [],
            'improvement_suggestions': []
        }
        
        # Generate insights
        for team in best_composition['teams']:
            comp = team['composition']
            if comp['size'] < self.min_team_size:
                summary['improvement_suggestions'].append(
                    f"Team {team['team_id']} is undersized ({comp['size']} members)"
                )
            
            if len(comp['cluster_distribution']) == 1:
                summary['improvement_suggestions'].append(
                    f"Team {team['team_id']} lacks cluster diversity"
                )
            
            if team['balance_score'] > 80:
                summary['key_insights'].append(
                    f"Team {team['team_id']} has excellent balance (score: {team['balance_score']})"
                )
        
        return summary