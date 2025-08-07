"""
End-to-End Tests with Realistic Organizational Scenarios
Tests complete workflows from data ingestion to business outcomes
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import tempfile
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.parser import InsightsDataParser
from insights_clustering.clustering import KMeansClusterer
from insights_clustering.neuromorphic_clustering import (
    NeuromorphicClusterer,
    NeuromorphicClusteringMethod
)
from insights_clustering.validator import DataValidator
from team_simulator.simulator import TeamCompositionSimulator
from insights_clustering.monitoring import ClusteringMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OrganizationalScenario:
    """Structure for organizational test scenarios"""
    name: str
    description: str
    employee_count: int
    department_distribution: Dict[str, float]
    personality_profiles: Dict[str, List[float]]
    business_objectives: List[str]
    success_criteria: Dict[str, Any]
    expected_challenges: List[str]


class OrganizationalDataGenerator:
    """Generate realistic organizational data for testing"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def create_startup_company(self, n_employees: int = 50) -> pd.DataFrame:
        """Create data for a startup company scenario"""
        data = []
        
        # Startup characteristics: High innovation, fast-paced, collaborative
        departments = {
            'Engineering': 0.45,  # Heavy on tech
            'Product': 0.15,
            'Sales': 0.20,
            'Marketing': 0.10,
            'Operations': 0.10
        }
        
        # Startup personality trends
        personality_trends = {
            'Engineering': ([25, 65, 20, 35], 8),    # Blue-heavy (analytical)
            'Product': ([40, 50, 30, 60], 10),       # Balanced but Yellow-heavy
            'Sales': ([70, 25, 30, 55], 12),         # Red-Yellow heavy
            'Marketing': ([35, 25, 25, 70], 10),     # Yellow-heavy (creative)
            'Operations': ([50, 45, 45, 25], 8)      # Red-Blue heavy
        }
        
        employee_id = 1
        
        for dept, proportion in departments.items():
            n_dept_employees = int(n_employees * proportion)
            base_profile, variation = personality_trends[dept]
            
            for i in range(n_dept_employees):
                # Generate personality with departmental bias
                noise = np.random.randn(4) * variation
                energies = np.array(base_profile) + noise
                energies = np.clip(energies, 1, 99)
                
                # Startup-specific adjustments
                # Higher Yellow (innovation) across all departments
                energies[3] *= 1.2
                
                # Normalize
                energies = (energies / np.sum(energies)) * 100
                
                # Generate other attributes
                experience = max(0, int(np.random.exponential(3)))  # Skewed toward junior
                
                # Performance influenced by personality fit
                personality_fit = self._calculate_startup_fit(energies)
                performance_base = 70 + personality_fit * 20
                performance = max(50, min(100, int(np.random.normal(performance_base, 10))))
                
                data.append({
                    'employee_id': f'STARTUP{employee_id:04d}',
                    'first_name': f'Employee{employee_id}',
                    'department': dept,
                    'position_level': self._assign_position_level(experience, performance),
                    'experience_years': experience,
                    'hire_date': self._generate_hire_date(12),  # Startup < 1 year old
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'performance_rating': performance,
                    'engagement_score': max(60, min(100, int(np.random.normal(80, 15)))),
                    'location': np.random.choice(['HQ', 'Remote'], p=[0.7, 0.3]),
                    'work_style': np.random.choice(['Individual', 'Collaborative', 'Mixed'], p=[0.2, 0.5, 0.3])
                })
                
                employee_id += 1
        
        return pd.DataFrame(data)
    
    def create_enterprise_company(self, n_employees: int = 500) -> pd.DataFrame:
        """Create data for a large enterprise scenario"""
        data = []
        
        # Enterprise characteristics: Structured, diverse, global
        departments = {
            'Engineering': 0.25,
            'Sales': 0.15,
            'Marketing': 0.10,
            'Finance': 0.12,
            'HR': 0.08,
            'Operations': 0.12,
            'Legal': 0.05,
            'IT': 0.08,
            'Strategy': 0.05
        }
        
        # More conservative personality distributions
        personality_trends = {
            'Engineering': ([30, 60, 25, 30], 10),
            'Sales': ([60, 25, 35, 45], 12),
            'Marketing': ([40, 30, 30, 55], 11),
            'Finance': ([35, 70, 25, 20], 8),
            'HR': ([25, 35, 65, 40], 9),
            'Operations': ([55, 50, 40, 25], 10),
            'Legal': ([40, 65, 30, 20], 7),
            'IT': ([30, 65, 30, 25], 9),
            'Strategy': ([45, 55, 30, 35], 8)
        }
        
        employee_id = 1
        
        for dept, proportion in departments.items():
            n_dept_employees = int(n_employees * proportion)
            base_profile, variation = personality_trends[dept]
            
            for i in range(n_dept_employees):
                # Generate experience (more senior in enterprise)
                if dept in ['Legal', 'Strategy', 'Finance']:
                    experience = max(2, int(np.random.normal(12, 8)))
                else:
                    experience = max(0, int(np.random.normal(8, 6)))
                
                # Personality with departmental bias
                noise = np.random.randn(4) * variation
                energies = np.array(base_profile) + noise
                
                # Enterprise-specific adjustments
                # More balanced personalities (less extreme)
                energies = energies * 0.9 + np.array([25, 25, 25, 25]) * 0.1
                energies = np.clip(energies, 5, 95)
                energies = (energies / np.sum(energies)) * 100
                
                # Performance distribution
                performance = max(60, min(100, int(np.random.normal(78, 12))))
                
                data.append({
                    'employee_id': f'ENT{employee_id:06d}',
                    'first_name': f'Employee{employee_id}',
                    'department': dept,
                    'position_level': self._assign_position_level(experience, performance),
                    'experience_years': experience,
                    'hire_date': self._generate_hire_date(60),  # Company 5 years old
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'performance_rating': performance,
                    'engagement_score': max(50, min(100, int(np.random.normal(72, 18)))),
                    'location': np.random.choice([
                        'New York', 'San Francisco', 'London', 'Singapore', 'Remote'
                    ], p=[0.3, 0.2, 0.15, 0.15, 0.2]),
                    'work_style': np.random.choice(['Individual', 'Collaborative', 'Mixed'], p=[0.4, 0.3, 0.3]),
                    'business_unit': np.random.choice([
                        'North America', 'Europe', 'Asia-Pacific', 'Corporate'
                    ], p=[0.4, 0.25, 0.2, 0.15])
                })
                
                employee_id += 1
        
        return pd.DataFrame(data)
    
    def create_consulting_firm(self, n_employees: int = 200) -> pd.DataFrame:
        """Create data for a consulting firm scenario"""
        data = []
        
        # Consulting characteristics: Client-focused, analytical, adaptable
        departments = {
            'Strategy Consulting': 0.35,
            'Technology Consulting': 0.25,
            'Operations Consulting': 0.20,
            'HR Consulting': 0.10,
            'Business Development': 0.10
        }
        
        # Consulting-specific personality profiles
        personality_trends = {
            'Strategy Consulting': ([55, 60, 30, 40], 10),
            'Technology Consulting': ([35, 70, 25, 35], 9),
            'Operations Consulting': ([50, 55, 35, 30], 8),
            'HR Consulting': ([30, 45, 60, 45], 10),
            'Business Development': ([65, 40, 45, 55], 12)
        }
        
        employee_id = 1
        
        for dept, proportion in departments.items():
            n_dept_employees = int(n_employees * proportion)
            base_profile, variation = personality_trends[dept]
            
            for i in range(n_dept_employees):
                # Consulting experience distribution (more senior-heavy)
                experience = max(1, int(np.random.gamma(2, 3)))
                
                # Personality generation
                noise = np.random.randn(4) * variation
                energies = np.array(base_profile) + noise
                
                # Consulting adjustments: Higher Red (client-facing) and Blue (analytical)
                energies[0] *= 1.1  # More assertive
                energies[1] *= 1.1  # More analytical
                
                energies = np.clip(energies, 10, 90)
                energies = (energies / np.sum(energies)) * 100
                
                # Performance linked to client interaction ability
                client_fit = (energies[0] + energies[3]) / 2  # Red + Yellow
                performance_base = 75 + (client_fit - 50) * 0.4
                performance = max(65, min(100, int(np.random.normal(performance_base, 8))))
                
                # Position levels in consulting
                if experience >= 10:
                    position = np.random.choice(['Partner', 'Principal'], p=[0.3, 0.7])
                elif experience >= 6:
                    position = np.random.choice(['Director', 'Senior Manager'], p=[0.4, 0.6])
                elif experience >= 3:
                    position = 'Manager'
                elif experience >= 1:
                    position = np.random.choice(['Senior Consultant', 'Consultant'], p=[0.4, 0.6])
                else:
                    position = 'Analyst'
                
                data.append({
                    'employee_id': f'CONS{employee_id:05d}',
                    'first_name': f'Employee{employee_id}',
                    'department': dept,
                    'position_level': position,
                    'experience_years': experience,
                    'hire_date': self._generate_hire_date(36),
                    'red_energy': round(energies[0], 2),
                    'blue_energy': round(energies[1], 2),
                    'green_energy': round(energies[2], 2),
                    'yellow_energy': round(energies[3], 2),
                    'performance_rating': performance,
                    'engagement_score': max(60, min(100, int(np.random.normal(75, 15)))),
                    'billable_hours_target': np.random.choice([1800, 2000, 2200], p=[0.3, 0.5, 0.2]),
                    'client_satisfaction': max(70, min(100, int(np.random.normal(85, 10)))),
                    'location': np.random.choice(['Office', 'Client Site', 'Remote'], p=[0.4, 0.4, 0.2])
                })
                
                employee_id += 1
        
        return pd.DataFrame(data)
    
    def _calculate_startup_fit(self, energies: np.ndarray) -> float:
        """Calculate how well personality fits startup culture"""
        # Startup values: Innovation (Yellow), Adaptability, Initiative (Red)
        yellow_fit = min(1.0, energies[3] / 50)  # Yellow energy
        red_fit = min(1.0, energies[0] / 50)     # Red energy
        adaptability = 1.0 - abs(50 - np.mean(energies)) / 50  # Balanced = adaptable
        
        return (yellow_fit + red_fit + adaptability) / 3
    
    def _assign_position_level(self, experience: int, performance: int) -> str:
        """Assign position level based on experience and performance"""
        score = experience * 3 + (performance - 70)  # Weighted score
        
        if score >= 25:
            return np.random.choice(['Senior', 'Lead', 'Manager'], p=[0.4, 0.3, 0.3])
        elif score >= 15:
            return np.random.choice(['Mid', 'Senior'], p=[0.6, 0.4])
        elif score >= 5:
            return np.random.choice(['Junior', 'Mid'], p=[0.7, 0.3])
        else:
            return 'Junior'
    
    def _generate_hire_date(self, company_age_months: int) -> str:
        """Generate realistic hire date"""
        days_ago = np.random.randint(30, company_age_months * 30)
        hire_date = datetime.now() - timedelta(days=days_ago)
        return hire_date.strftime('%Y-%m-%d')


class TestStartupScenario:
    """Test scenarios for startup companies"""
    
    def setup_method(self):
        """Setup startup scenario testing"""
        self.data_generator = OrganizationalDataGenerator(seed=42)
        self.monitor = ClusteringMonitor()
        
    def test_startup_team_formation(self):
        """Test team formation for a startup environment"""
        # Generate startup data
        startup_data = self.data_generator.create_startup_company(n_employees=50)
        
        with self.monitor.performance_monitor("startup_team_formation"):
            # Parse and validate data
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            startup_data.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                
                # Validate data quality
                validator = DataValidator()
                validation_results = validator.validate_data_quality(parsed_data)
                assert validation_results['is_valid']
                
                # Extract features for clustering
                features = parser.get_clustering_features()
                
                # Use neuromorphic clustering for better personality insights
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                    n_clusters=4,  # Small clusters for startup agility
                    random_state=42
                )
                
                clusterer.fit(features)
                labels = clusterer.get_cluster_assignments()
                
                # Get cluster interpretations
                interpretations = clusterer.get_cluster_interpretation()
                
                # Form cross-functional teams
                simulator = TeamCompositionSimulator(
                    min_team_size=6,   # Small agile teams
                    max_team_size=10
                )
                simulator.load_employee_data(parsed_data, labels)
                
                # Generate multiple team configurations
                team_configs = []
                for i in range(5):
                    teams = simulator.generate_balanced_teams(num_teams=4)
                    if teams:
                        avg_balance = np.mean([team['balance_score'] for team in teams])
                        team_configs.append({
                            'teams': teams,
                            'avg_balance': avg_balance,
                            'config_id': i
                        })
                
                # Select best configuration
                best_config = max(team_configs, key=lambda x: x['avg_balance'])
                
                # Startup-specific validations
                self._validate_startup_teams(
                    best_config['teams'], parsed_data, interpretations
                )
                
                # Performance assertions
                assert best_config['avg_balance'] > 45, "Teams not balanced enough for startup environment"
                assert len(best_config['teams']) == 4, "Expected 4 cross-functional teams"
                
                # Each team should have diverse departments
                for team in best_config['teams']:
                    team_df = pd.DataFrame(team['members'])
                    dept_diversity = len(team_df['department'].unique())
                    assert dept_diversity >= 3, f"Team lacks departmental diversity: {dept_diversity}"
                
                # Teams should have innovation potential (Yellow energy)
                for team in best_config['teams']:
                    team_df = pd.DataFrame(team['members'])
                    avg_yellow = team_df['yellow_energy'].mean()
                    assert avg_yellow > 35, f"Team lacks innovation potential: {avg_yellow}"
                
                logger.info(f"Startup team formation completed successfully")
                logger.info(f"Best configuration balance score: {best_config['avg_balance']:.2f}")
                
            finally:
                temp_path.unlink()
    
    def test_startup_scaling_simulation(self):
        """Test how team composition changes as startup scales"""
        # Test at different growth stages
        growth_stages = [
            ('seed', 25),
            ('series_a', 50), 
            ('series_b', 100)
        ]
        
        scaling_results = {}
        
        for stage_name, n_employees in growth_stages:
            startup_data = self.data_generator.create_startup_company(n_employees)
            
            # Process data
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            startup_data.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                features = parser.get_clustering_features()
                
                # Clustering
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    n_clusters=min(6, max(3, n_employees // 15)),  # Scale clusters with size
                    random_state=42
                )
                
                clusterer.fit(features)
                labels = clusterer.get_cluster_assignments()
                
                # Team formation
                simulator = TeamCompositionSimulator(min_team_size=5, max_team_size=12)
                simulator.load_employee_data(parsed_data, labels)
                
                n_teams = max(2, n_employees // 12)  # Scale teams with size
                teams = simulator.generate_balanced_teams(num_teams=n_teams)
                
                # Analyze scaling metrics
                avg_team_size = np.mean([len(team['members']) for team in teams])
                dept_diversity = np.mean([
                    len(pd.DataFrame(team['members'])['department'].unique()) 
                    for team in teams
                ])
                avg_balance = np.mean([team['balance_score'] for team in teams])
                
                scaling_results[stage_name] = {
                    'n_employees': n_employees,
                    'n_teams': len(teams),
                    'avg_team_size': avg_team_size,
                    'dept_diversity': dept_diversity,
                    'avg_balance': avg_balance
                }
                
            finally:
                temp_path.unlink()
        
        # Validate scaling patterns
        stages = list(scaling_results.keys())
        
        # Team size should remain reasonable as company grows
        for stage in stages:
            result = scaling_results[stage]
            assert 5 <= result['avg_team_size'] <= 12, f"Team size out of range at {stage}"
            assert result['avg_balance'] > 40, f"Balance too low at {stage}"
        
        # Department diversity should improve with scale
        seed_diversity = scaling_results['seed']['dept_diversity']
        series_b_diversity = scaling_results['series_b']['dept_diversity']
        assert series_b_diversity >= seed_diversity, "Diversity should improve with scale"
        
        logger.info("Startup scaling simulation completed")
        for stage, result in scaling_results.items():
            logger.info(f"{stage}: {result}")
    
    def _validate_startup_teams(self, teams: List[Dict], employee_data: pd.DataFrame, 
                               interpretations: Dict):
        """Validate teams meet startup-specific requirements"""
        for i, team in enumerate(teams):
            team_df = pd.DataFrame(team['members'])
            
            # Check for key startup capabilities in each team
            
            # 1. Technical capability (Engineering representation)
            eng_count = len(team_df[team_df['department'] == 'Engineering'])
            assert eng_count >= 1, f"Team {i} lacks technical capability"
            
            # 2. Customer-facing capability (Sales/Marketing)
            customer_facing = len(team_df[
                team_df['department'].isin(['Sales', 'Marketing', 'Product'])
            ])
            assert customer_facing >= 1, f"Team {i} lacks customer-facing capability"
            
            # 3. Innovation potential (high Yellow energy)
            avg_yellow = team_df['yellow_energy'].mean()
            assert avg_yellow > 30, f"Team {i} lacks innovation potential"
            
            # 4. Decision-making capability (some Red energy)
            max_red = team_df['red_energy'].max()
            assert max_red > 40, f"Team {i} may struggle with quick decisions"
            
            # 5. Team should not be too homogeneous
            personality_std = np.std([
                team_df['red_energy'].mean(),
                team_df['blue_energy'].mean(), 
                team_df['green_energy'].mean(),
                team_df['yellow_energy'].mean()
            ])
            assert personality_std > 5, f"Team {i} too personality-homogeneous"


class TestEnterpriseScenario:
    """Test scenarios for large enterprise companies"""
    
    def setup_method(self):
        """Setup enterprise scenario testing"""
        self.data_generator = OrganizationalDataGenerator(seed=42)
        self.monitor = ClusteringMonitor()
        
    def test_enterprise_cross_functional_teams(self):
        """Test cross-functional team formation in enterprise environment"""
        # Generate enterprise data
        enterprise_data = self.data_generator.create_enterprise_company(n_employees=300)
        
        with self.monitor.performance_monitor("enterprise_cross_functional_teams"):
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            enterprise_data.to_csv(temp_file.name, index=False)
            temp_path = Path(temp_file.name)
            
            try:
                # Parse and process data
                parser = InsightsDataParser()
                parsed_data = parser.parse_csv(temp_path)
                features = parser.get_clustering_features()
                
                # Use more clusters for enterprise complexity
                clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
                    n_clusters=8,  # More personality types in enterprise
                    random_state=42
                )
                
                clusterer.fit(features)
                labels = clusterer.get_cluster_assignments()
                
                # Form larger, more structured teams
                simulator = TeamCompositionSimulator(
                    min_team_size=8,
                    max_team_size=15  # Larger teams for enterprise
                )
                simulator.load_employee_data(parsed_data, labels)
                
                # Generate project teams for different business units
                project_teams = []
                business_units = parsed_data['business_unit'].unique()
                
                for bu in business_units:
                    bu_data = parsed_data[parsed_data['business_unit'] == bu]
                    if len(bu_data) >= 8:  # Enough people for a team
                        
                        # Create BU-specific simulator
                        bu_simulator = TeamCompositionSimulator(min_team_size=8, max_team_size=15)
                        bu_labels = labels[parsed_data['business_unit'] == bu]
                        bu_simulator.load_employee_data(bu_data, bu_labels)
                        
                        bu_teams = bu_simulator.generate_balanced_teams(
                            num_teams=max(1, len(bu_data) // 12)
                        )
                        
                        for team in bu_teams:
                            team['business_unit'] = bu
                            project_teams.append(team)
                
                # Validate enterprise team characteristics
                self._validate_enterprise_teams(project_teams, parsed_data)
                
                # Enterprise-specific assertions
                assert len(project_teams) >= 4, "Not enough cross-functional teams formed"
                
                # Each team should have senior leadership
                for team in project_teams:
                    team_df = pd.DataFrame(team['members'])
                    senior_count = len(team_df[
                        team_df['position_level'].isin(['Senior', 'Lead', 'Manager'])
                    ])
                    assert senior_count >= 2, "Team lacks senior leadership"
                
                # Teams should be geographically aware
                for team in project_teams:
                    team_df = pd.DataFrame(team['members'])
                    locations = team_df['location'].unique()
                    # Allow for some remote work but prefer co-location
                    if len(locations) > 3:
                        logger.warning(f"Team may have too much geographical dispersion: {locations}")
                
                logger.info(f"Enterprise cross-functional teams formed: {len(project_teams)}")
                
            finally:
                temp_path.unlink()
    
    def test_enterprise_leadership_development(self):
        """Test identification of leadership development candidates"""
        enterprise_data = self.data_generator.create_enterprise_company(n_employees=200)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        enterprise_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Clustering for leadership potential analysis
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=6,
                random_state=42
            )
            
            clusterer.fit(features)
            labels = clusterer.get_cluster_assignments()
            interpretations = clusterer.get_cluster_interpretation()
            
            # Identify leadership potential
            leadership_candidates = self._identify_leadership_potential(
                parsed_data, labels, interpretations
            )
            
            # Validate leadership identification
            assert len(leadership_candidates) > 0, "No leadership candidates identified"
            assert len(leadership_candidates) <= len(parsed_data) * 0.3, "Too many leadership candidates"
            
            # Leadership candidates should have appropriate traits
            for candidate in leadership_candidates:
                assert candidate['leadership_score'] > 60, "Leadership score too low"
                assert candidate['red_energy'] > 35, "Insufficient assertiveness for leadership"
                
                # Should have some experience or high performance
                assert (candidate['experience_years'] > 3 or 
                       candidate['performance_rating'] > 80), "Insufficient experience/performance"
            
            logger.info(f"Leadership development candidates identified: {len(leadership_candidates)}")
            
        finally:
            temp_path.unlink()
    
    def _validate_enterprise_teams(self, teams: List[Dict], employee_data: pd.DataFrame):
        """Validate teams meet enterprise requirements"""
        for i, team in enumerate(teams):
            team_df = pd.DataFrame(team['members'])
            
            # Enterprise team requirements
            
            # 1. Adequate size for enterprise projects
            assert 8 <= len(team_df) <= 15, f"Team {i} size inappropriate for enterprise"
            
            # 2. Balanced seniority levels
            seniority_levels = team_df['position_level'].value_counts()
            assert len(seniority_levels) >= 2, f"Team {i} lacks seniority diversity"
            
            # 3. Cross-departmental representation
            dept_count = len(team_df['department'].unique())
            assert dept_count >= 3, f"Team {i} lacks departmental diversity"
            
            # 4. Performance balance (mix of high and solid performers)
            avg_performance = team_df['performance_rating'].mean()
            assert avg_performance > 70, f"Team {i} average performance too low"
            
            # 5. Personality balance (not too extreme in any direction)
            personality_means = [
                team_df['red_energy'].mean(),
                team_df['blue_energy'].mean(),
                team_df['green_energy'].mean(),
                team_df['yellow_energy'].mean()
            ]
            
            # No personality dimension should dominate too much
            max_personality = max(personality_means)
            min_personality = min(personality_means)
            assert (max_personality - min_personality) < 30, f"Team {i} too personality-imbalanced"
    
    def _identify_leadership_potential(self, employee_data: pd.DataFrame, 
                                     labels: np.ndarray, 
                                     interpretations: Dict) -> List[Dict]:
        """Identify employees with leadership potential"""
        leadership_candidates = []
        
        for idx, (_, employee) in enumerate(employee_data.iterrows()):
            # Calculate leadership score based on multiple factors
            leadership_score = 0
            
            # Personality factors (40% weight)
            personality_score = (
                min(employee['red_energy'], 80) * 0.4 +      # Assertiveness (capped)
                min(employee['blue_energy'], 70) * 0.3 +     # Strategic thinking
                min(employee['green_energy'], 60) * 0.2 +    # Team building
                min(employee['yellow_energy'], 50) * 0.1     # Innovation
            ) * 0.4
            
            leadership_score += personality_score
            
            # Performance factor (30% weight)
            performance_score = (employee['performance_rating'] - 60) * 0.75  # Normalize to 0-30
            leadership_score += max(0, performance_score)
            
            # Experience factor (20% weight)
            experience_score = min(employee['experience_years'] * 2, 20)
            leadership_score += experience_score
            
            # Engagement factor (10% weight)  
            engagement_score = (employee.get('engagement_score', 70) - 60) * 0.25
            leadership_score += max(0, engagement_score)
            
            # Threshold for leadership potential
            if leadership_score > 60:
                candidate = {
                    'employee_id': employee['employee_id'],
                    'department': employee['department'],
                    'position_level': employee['position_level'],
                    'experience_years': employee['experience_years'],
                    'performance_rating': employee['performance_rating'],
                    'red_energy': employee['red_energy'],
                    'blue_energy': employee['blue_energy'],
                    'green_energy': employee['green_energy'],
                    'yellow_energy': employee['yellow_energy'],
                    'leadership_score': round(leadership_score, 2),
                    'cluster_id': labels[idx]
                }
                
                leadership_candidates.append(candidate)
        
        # Sort by leadership score
        leadership_candidates.sort(key=lambda x: x['leadership_score'], reverse=True)
        
        return leadership_candidates


class TestConsultingScenario:
    """Test scenarios for consulting firms"""
    
    def setup_method(self):
        """Setup consulting scenario testing"""
        self.data_generator = OrganizationalDataGenerator(seed=42)
        
    def test_consulting_project_team_formation(self):
        """Test project team formation for consulting engagements"""
        consulting_data = self.data_generator.create_consulting_firm(n_employees=150)
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        consulting_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            # Consulting-specific clustering
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                n_clusters=5,  # Consulting personality types
                random_state=42
            )
            
            clusterer.fit(features)
            labels = clusterer.get_cluster_assignments()
            
            # Simulate different types of consulting projects
            project_types = [
                {
                    'name': 'Strategy Engagement',
                    'required_skills': ['Strategy Consulting'],
                    'client_facing_ratio': 0.6,
                    'analytical_requirement': 'high',
                    'team_size': (4, 8)
                },
                {
                    'name': 'Technology Implementation',
                    'required_skills': ['Technology Consulting'],
                    'client_facing_ratio': 0.4,
                    'analytical_requirement': 'very_high', 
                    'team_size': (6, 12)
                },
                {
                    'name': 'Operations Transformation',
                    'required_skills': ['Operations Consulting', 'Strategy Consulting'],
                    'client_facing_ratio': 0.5,
                    'analytical_requirement': 'high',
                    'team_size': (5, 10)
                }
            ]
            
            project_teams = {}
            
            for project in project_types:
                # Filter consultants by required skills
                skill_match = parsed_data[
                    parsed_data['department'].isin(project['required_skills'])
                ]
                
                if len(skill_match) >= project['team_size'][0]:
                    # Create project-specific teams
                    simulator = TeamCompositionSimulator(
                        min_team_size=project['team_size'][0],
                        max_team_size=project['team_size'][1]
                    )
                    
                    # Filter labels for matching employees
                    project_labels = labels[parsed_data['department'].isin(project['required_skills'])]
                    simulator.load_employee_data(skill_match, project_labels)
                    
                    teams = simulator.generate_balanced_teams(num_teams=2)
                    
                    # Validate teams for consulting requirements
                    validated_teams = self._validate_consulting_teams(
                        teams, skill_match, project
                    )
                    
                    project_teams[project['name']] = validated_teams
            
            # Assertions for consulting team formation
            assert len(project_teams) > 0, "No project teams formed"
            
            for project_name, teams in project_teams.items():
                assert len(teams) > 0, f"No teams formed for {project_name}"
                
                for team in teams:
                    team_df = pd.DataFrame(team['members'])
                    
                    # Consulting teams should have hierarchy
                    position_levels = team_df['position_level'].unique()
                    assert len(position_levels) >= 2, f"Team lacks hierarchy for {project_name}"
                    
                    # Should have client-facing capability
                    avg_client_facing = (team_df['red_energy'] + team_df['yellow_energy']).mean() / 2
                    assert avg_client_facing > 40, f"Team lacks client-facing skills for {project_name}"
            
            logger.info(f"Consulting project teams formed for {len(project_teams)} project types")
            
        finally:
            temp_path.unlink()
    
    def test_consulting_client_satisfaction_prediction(self):
        """Test prediction of client satisfaction based on team composition"""
        consulting_data = self.data_generator.create_consulting_firm(n_employees=100)
        
        # Add client satisfaction data
        consulting_data['client_satisfaction'] = consulting_data.apply(
            lambda row: self._calculate_expected_client_satisfaction(row), axis=1
        )
        
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        consulting_data.to_csv(temp_file.name, index=False)
        temp_path = Path(temp_file.name)
        
        try:
            parser = InsightsDataParser()
            parsed_data = parser.parse_csv(temp_path)
            features = parser.get_clustering_features()
            
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
                n_clusters=4,
                random_state=42
            )
            
            clusterer.fit(features)
            labels = clusterer.get_cluster_assignments()
            
            # Analyze relationship between personality clusters and client satisfaction
            cluster_satisfaction = {}
            for cluster_id in np.unique(labels):
                cluster_mask = labels == cluster_id
                cluster_data = parsed_data[cluster_mask]
                avg_satisfaction = cluster_data['client_satisfaction'].mean()
                cluster_satisfaction[cluster_id] = avg_satisfaction
            
            # Form teams and predict satisfaction
            simulator = TeamCompositionSimulator(min_team_size=4, max_team_size=8)
            simulator.load_employee_data(parsed_data, labels)
            teams = simulator.generate_balanced_teams(num_teams=5)
            
            team_predictions = []
            for team in teams:
                team_df = pd.DataFrame(team['members'])
                
                # Predict team client satisfaction
                predicted_satisfaction = self._predict_team_client_satisfaction(team_df)
                team_predictions.append({
                    'team_id': team.get('team_id', 'unknown'),
                    'predicted_satisfaction': predicted_satisfaction,
                    'balance_score': team['balance_score']
                })
            
            # Validate predictions
            avg_predicted_satisfaction = np.mean([t['predicted_satisfaction'] for t in team_predictions])
            assert 70 <= avg_predicted_satisfaction <= 95, "Client satisfaction predictions out of range"
            
            # Better balanced teams should generally have higher client satisfaction
            sorted_teams = sorted(team_predictions, key=lambda x: x['balance_score'])
            high_balance_satisfaction = np.mean([
                t['predicted_satisfaction'] for t in sorted_teams[-2:]
            ])
            low_balance_satisfaction = np.mean([
                t['predicted_satisfaction'] for t in sorted_teams[:2]
            ])
            
            # This is a tendency, not a strict rule
            if high_balance_satisfaction > low_balance_satisfaction:
                logger.info("Higher team balance correlates with better client satisfaction")
            
            logger.info(f"Client satisfaction analysis completed")
            logger.info(f"Average predicted satisfaction: {avg_predicted_satisfaction:.1f}")
            
        finally:
            temp_path.unlink()
    
    def _validate_consulting_teams(self, teams: List[Dict], consultant_data: pd.DataFrame, 
                                  project_config: Dict) -> List[Dict]:
        """Validate teams meet consulting project requirements"""
        validated_teams = []
        
        for team in teams:
            team_df = pd.DataFrame(team['members'])
            
            # Check team size
            min_size, max_size = project_config['team_size']
            if not (min_size <= len(team_df) <= max_size):
                continue
            
            # Check client-facing requirement
            client_facing_score = (team_df['red_energy'] + team_df['yellow_energy']).mean() / 2
            required_threshold = {
                0.6: 45,  # High client-facing
                0.5: 40,  # Medium client-facing
                0.4: 35   # Lower client-facing
            }.get(project_config['client_facing_ratio'], 40)
            
            if client_facing_score < required_threshold:
                continue
            
            # Check analytical requirement
            analytical_score = team_df['blue_energy'].mean()
            analytical_requirements = {
                'very_high': 55,
                'high': 45,
                'medium': 35
            }
            
            required_analytical = analytical_requirements.get(
                project_config['analytical_requirement'], 35
            )
            
            if analytical_score < required_analytical:
                continue
            
            # Check for senior leadership
            senior_positions = ['Partner', 'Principal', 'Director']
            has_senior_lead = any(
                pos in senior_positions for pos in team_df['position_level']
            )
            
            if not has_senior_lead and len(team_df) > 5:  # Larger teams need senior lead
                continue
            
            validated_teams.append(team)
        
        return validated_teams
    
    def _calculate_expected_client_satisfaction(self, consultant: pd.Series) -> float:
        """Calculate expected client satisfaction for a consultant"""
        base_satisfaction = 75
        
        # Personality factors
        client_facing = (consultant['red_energy'] + consultant['yellow_energy']) / 2
        analytical = consultant['blue_energy']
        supportive = consultant['green_energy']
        
        # Position level factor
        position_multiplier = {
            'Partner': 1.2,
            'Principal': 1.15,
            'Director': 1.1,
            'Senior Manager': 1.05,
            'Manager': 1.0,
            'Senior Consultant': 0.95,
            'Consultant': 0.9,
            'Analyst': 0.85
        }.get(consultant['position_level'], 1.0)
        
        # Calculate satisfaction
        personality_score = (client_facing * 0.4 + analytical * 0.3 + supportive * 0.3) - 50
        satisfaction = base_satisfaction + personality_score * 0.3
        satisfaction *= position_multiplier
        
        # Add some noise
        satisfaction += np.random.normal(0, 3)
        
        return max(60, min(100, satisfaction))
    
    def _predict_team_client_satisfaction(self, team_df: pd.DataFrame) -> float:
        """Predict client satisfaction for a team"""
        # Team-level factors
        avg_client_facing = (team_df['red_energy'] + team_df['yellow_energy']).mean() / 2
        avg_analytical = team_df['blue_energy'].mean()
        avg_supportive = team_df['green_energy'].mean()
        
        # Team composition factors
        position_diversity = len(team_df['position_level'].unique())
        has_senior = any(pos in ['Partner', 'Principal', 'Director'] 
                        for pos in team_df['position_level'])
        
        # Base prediction
        base_score = 78
        
        # Personality contributions
        personality_contribution = (
            (avg_client_facing - 45) * 0.3 +
            (avg_analytical - 45) * 0.2 +
            (avg_supportive - 40) * 0.1
        )
        
        # Team structure contributions
        structure_contribution = position_diversity * 2
        if has_senior:
            structure_contribution += 5
        
        predicted_satisfaction = base_score + personality_contribution + structure_contribution
        
        return max(65, min(95, predicted_satisfaction))


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])