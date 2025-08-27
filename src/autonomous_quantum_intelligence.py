#!/usr/bin/env python3
"""Autonomous Quantum Intelligence System - Generation 3 Implementation
Self-learning quantum AI with adaptive optimization and autonomous decision making
"""

import asyncio
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import ParameterGrid

logger = logging.getLogger(__name__)


class QuantumIntelligenceCore:
    """Core quantum intelligence system with self-learning capabilities."""
    
    def __init__(self, intelligence_config: Optional[Dict] = None):
        self.config = intelligence_config or self._default_intelligence_config()
        
        # Learning components
        self.knowledge_base = {}
        self.experience_memory = []
        self.performance_history = {}
        self.adaptation_rules = []
        self.decision_patterns = {}
        
        # Intelligence metrics
        self.learning_rate = 0.1
        self.confidence_threshold = 0.8
        self.exploration_rate = 0.2  # Epsilon for exploration vs exploitation
        self.intelligence_score = 0.5  # Starts at 50%
        
        # Quantum learning parameters
        self.quantum_learning_depth = 3
        self.quantum_memory_capacity = 1000
        self.quantum_adaptation_rate = 0.05
        
        # Initialize intelligence systems
        self._initialize_intelligence_systems()
    
    def _default_intelligence_config(self) -> Dict[str, Any]:
        """Default configuration for quantum intelligence."""
        return {
            'learning_enabled': True,
            'adaptation_enabled': True,
            'memory_limit': 10000,
            'performance_window': 100,
            'confidence_decay': 0.01,
            'intelligence_growth_rate': 0.02,
            'quantum_coherence_threshold': 0.7,
            'autonomous_optimization': True
        }
    
    def _initialize_intelligence_systems(self):
        """Initialize all intelligence subsystems."""
        logger.info("🧠 Initializing Quantum Intelligence Systems")
        
        # Initialize knowledge base with fundamental patterns
        self.knowledge_base = {
            'clustering_patterns': {
                'optimal_cluster_sizes': {},
                'data_type_preferences': {},
                'parameter_correlations': {}
            },
            'performance_patterns': {
                'successful_configurations': [],
                'failure_modes': [],
                'optimization_paths': []
            },
            'adaptation_patterns': {
                'environment_responses': {},
                'scaling_behaviors': {},
                'resource_optimization': {}
            }
        }
        
        # Initialize basic adaptation rules
        self.adaptation_rules = [
            {
                'rule_id': 'performance_degradation',
                'condition': lambda metrics: metrics.get('silhouette_score', 0) < 0.5,
                'action': 'increase_quantum_depth',
                'confidence': 0.8
            },
            {
                'rule_id': 'high_error_rate',
                'condition': lambda metrics: metrics.get('error_rate', 0) > 0.1,
                'action': 'enable_error_correction',
                'confidence': 0.9
            },
            {
                'rule_id': 'low_efficiency',
                'condition': lambda metrics: metrics.get('parallel_efficiency', 0) < 0.6,
                'action': 'optimize_parallelization',
                'confidence': 0.7
            }
        ]
        
        logger.info("✅ Quantum Intelligence Systems Initialized")
    
    async def learn_from_experience(self, 
                                   operation_type: str,
                                   input_data: Dict[str, Any],
                                   results: Dict[str, Any],
                                   performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from operational experience to improve future performance."""
        try:
            learning_start = time.time()
            
            # Create experience record
            experience = {
                'timestamp': time.time(),
                'operation_type': operation_type,
                'input_hash': self._hash_input_data(input_data),
                'input_characteristics': self._analyze_input_characteristics(input_data),
                'configuration_used': input_data.get('configuration', {}),
                'results': results,
                'performance_metrics': performance_metrics,
                'success_score': self._calculate_success_score(results, performance_metrics)
            }
            
            # Add to experience memory
            self.experience_memory.append(experience)
            
            # Limit memory size
            if len(self.experience_memory) > self.config['memory_limit']:
                self.experience_memory = self.experience_memory[-self.config['memory_limit']:]
            
            # Extract learning insights
            learning_insights = await self._extract_learning_insights(experience)
            
            # Update knowledge base
            await self._update_knowledge_base(experience, learning_insights)
            
            # Adapt decision patterns
            await self._adapt_decision_patterns(experience, learning_insights)
            
            # Update intelligence score
            self._update_intelligence_score(experience, learning_insights)
            
            learning_time = time.time() - learning_start
            
            learning_summary = {
                'experience_recorded': True,
                'learning_insights': learning_insights,
                'intelligence_score': self.intelligence_score,
                'knowledge_base_size': len(self.experience_memory),
                'learning_time': learning_time,
                'adaptation_rules_updated': len(learning_insights.get('new_rules', [])),
                'patterns_discovered': len(learning_insights.get('patterns', []))
            }
            
            logger.info(f"🎓 Learning complete: intelligence={self.intelligence_score:.3f}, "
                       f"insights={len(learning_insights.get('patterns', []))}")
            
            return learning_summary
            
        except Exception as e:
            logger.error(f"❌ Learning from experience failed: {e}")
            return {'experience_recorded': False, 'error': str(e)}
    
    async def _extract_learning_insights(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """Extract actionable insights from experience."""
        insights = {
            'patterns': [],
            'correlations': [],
            'new_rules': [],
            'optimization_opportunities': []
        }
        
        # Pattern recognition
        similar_experiences = self._find_similar_experiences(experience)
        if len(similar_experiences) >= 3:
            pattern = self._identify_pattern(similar_experiences)
            if pattern:
                insights['patterns'].append(pattern)
        
        # Parameter correlation analysis
        if len(self.experience_memory) >= 10:
            correlations = self._analyze_parameter_correlations()
            insights['correlations'].extend(correlations)
        
        # Automatic rule generation
        if experience['success_score'] > 0.8:  # High success
            new_rule = self._generate_success_rule(experience)
            if new_rule:
                insights['new_rules'].append(new_rule)
        elif experience['success_score'] < 0.3:  # Failure
            failure_rule = self._generate_failure_avoidance_rule(experience)
            if failure_rule:
                insights['new_rules'].append(failure_rule)
        
        # Optimization opportunity detection
        optimizations = self._detect_optimization_opportunities(experience)
        insights['optimization_opportunities'].extend(optimizations)
        
        return insights
    
    def _find_similar_experiences(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find experiences similar to the current one."""
        similar = []
        
        target_characteristics = experience['input_characteristics']
        
        for past_experience in self.experience_memory[-100:]:  # Check last 100
            if past_experience['operation_type'] != experience['operation_type']:
                continue
            
            # Calculate similarity score
            similarity = self._calculate_similarity(
                target_characteristics,
                past_experience['input_characteristics']
            )
            
            if similarity > 0.7:  # High similarity threshold
                similar.append(past_experience)
        
        return similar
    
    def _calculate_similarity(self, chars1: Dict, chars2: Dict) -> float:
        """Calculate similarity between input characteristics."""
        common_keys = set(chars1.keys()) & set(chars2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1, val2 = chars1[key], chars2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-10)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0, sim))
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities)
    
    def _identify_pattern(self, experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Identify patterns in similar experiences."""
        if len(experiences) < 3:
            return None
        
        # Analyze success patterns
        success_scores = [exp['success_score'] for exp in experiences]
        avg_success = np.mean(success_scores)
        
        if avg_success > 0.7:  # Consistent success pattern
            # Find common configuration elements
            configs = [exp['configuration_used'] for exp in experiences]
            common_config = self._find_common_configuration_elements(configs)
            
            return {
                'pattern_type': 'success_configuration',
                'pattern_strength': avg_success,
                'common_elements': common_config,
                'sample_size': len(experiences),
                'confidence': min(1.0, len(experiences) / 10)
            }
        
        return None
    
    def _find_common_configuration_elements(self, configs: List[Dict]) -> Dict[str, Any]:
        """Find configuration elements common across successful cases."""
        if not configs:
            return {}
        
        # Find parameters that appear in all configs with similar values
        common_elements = {}
        
        all_keys = set()
        for config in configs:
            all_keys.update(config.keys())
        
        for key in all_keys:
            values = [config.get(key) for config in configs if key in config]
            if len(values) >= len(configs) * 0.8:  # Appears in 80% of configs
                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    # Numerical parameter
                    mean_val = np.mean([v for v in values if v is not None])
                    std_val = np.std([v for v in values if v is not None])
                    if std_val < mean_val * 0.2:  # Low variation
                        common_elements[key] = {
                            'type': 'numerical',
                            'value': mean_val,
                            'std': std_val,
                            'confidence': 1.0 - std_val / (mean_val + 1e-10)
                        }
                else:
                    # Categorical parameter
                    unique_values = list(set(values))
                    if len(unique_values) == 1:  # Same value in all cases
                        common_elements[key] = {
                            'type': 'categorical',
                            'value': unique_values[0],
                            'confidence': 1.0
                        }
        
        return common_elements
    
    def _analyze_parameter_correlations(self) -> List[Dict[str, Any]]:
        """Analyze correlations between parameters and performance."""
        if len(self.experience_memory) < 20:
            return []
        
        correlations = []
        recent_experiences = self.experience_memory[-100:]
        
        # Extract parameter-performance pairs
        param_performance_pairs = []
        for exp in recent_experiences:
            config = exp['configuration_used']
            success = exp['success_score']
            
            for param, value in config.items():
                if isinstance(value, (int, float)):
                    param_performance_pairs.append((param, value, success))
        
        # Group by parameter and analyze correlation
        param_groups = {}
        for param, value, success in param_performance_pairs:
            if param not in param_groups:
                param_groups[param] = {'values': [], 'successes': []}
            param_groups[param]['values'].append(value)
            param_groups[param]['successes'].append(success)
        
        for param, data in param_groups.items():
            if len(data['values']) >= 10:  # Minimum samples for correlation
                correlation = np.corrcoef(data['values'], data['successes'])[0, 1]
                if abs(correlation) > 0.5:  # Significant correlation
                    correlations.append({
                        'parameter': param,
                        'correlation': correlation,
                        'strength': 'strong' if abs(correlation) > 0.7 else 'moderate',
                        'direction': 'positive' if correlation > 0 else 'negative',
                        'sample_size': len(data['values'])
                    })
        
        return correlations
    
    def _generate_success_rule(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a rule based on successful experience."""
        config = experience['configuration_used']
        characteristics = experience['input_characteristics']
        
        if not config:
            return None
        
        # Create condition based on input characteristics
        conditions = []
        for key, value in characteristics.items():
            if isinstance(value, (int, float)):
                conditions.append(f"input_characteristics.get('{key}', 0) >= {value * 0.8}")
            elif isinstance(value, str):
                conditions.append(f"input_characteristics.get('{key}') == '{value}'")
        
        if not conditions:
            return None
        
        # Create action based on successful configuration
        actions = []
        for param, value in config.items():
            actions.append(f"set_{param}:{value}")
        
        return {
            'rule_id': f"success_rule_{int(time.time())}",
            'condition_text': " and ".join(conditions),
            'actions': actions,
            'confidence': experience['success_score'],
            'created_from_experience': experience['timestamp'],
            'rule_type': 'success_pattern'
        }
    
    def _generate_failure_avoidance_rule(self, experience: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a rule to avoid failure patterns."""
        config = experience['configuration_used']
        
        if not config:
            return None
        
        # Identify problematic parameters
        problematic_params = []
        for param, value in config.items():
            if isinstance(value, (int, float)):
                # Look for parameters that might be out of optimal range
                if param in ['quantum_depth'] and value > 5:
                    problematic_params.append((param, 'too_high', value))
                elif param in ['reservoir_size'] and value < 50:
                    problematic_params.append((param, 'too_low', value))
        
        if not problematic_params:
            return None
        
        # Create avoidance rule
        param, issue, value = problematic_params[0]  # Take first problematic parameter
        
        return {
            'rule_id': f"avoidance_rule_{int(time.time())}",
            'condition_text': f"proposed_config.get('{param}') == {value}",
            'action': 'flag_risky_configuration',
            'confidence': 1.0 - experience['success_score'],
            'created_from_failure': experience['timestamp'],
            'rule_type': 'failure_avoidance',
            'issue_type': issue
        }
    
    def _detect_optimization_opportunities(self, experience: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect opportunities for optimization."""
        opportunities = []
        
        metrics = experience['performance_metrics']
        
        # Check for efficiency opportunities
        if metrics.get('parallel_efficiency', 0) < 0.8:
            opportunities.append({
                'type': 'parallelization',
                'current_value': metrics.get('parallel_efficiency', 0),
                'target_value': 0.8,
                'potential_improvement': 0.8 - metrics.get('parallel_efficiency', 0),
                'recommendation': 'optimize_task_granularity'
            })
        
        # Check for memory optimization
        if metrics.get('cache_hit_rate', 0) < 0.6:
            opportunities.append({
                'type': 'caching',
                'current_value': metrics.get('cache_hit_rate', 0),
                'target_value': 0.8,
                'potential_improvement': 0.8 - metrics.get('cache_hit_rate', 0),
                'recommendation': 'improve_cache_strategy'
            })
        
        # Check for clustering quality
        if metrics.get('silhouette_score', 0) < 0.7:
            opportunities.append({
                'type': 'clustering_quality',
                'current_value': metrics.get('silhouette_score', 0),
                'target_value': 0.8,
                'potential_improvement': 0.8 - metrics.get('silhouette_score', 0),
                'recommendation': 'tune_clustering_parameters'
            })
        
        return opportunities
    
    async def _update_knowledge_base(self, experience: Dict[str, Any], insights: Dict[str, Any]):
        """Update the knowledge base with new insights."""
        # Update clustering patterns
        if experience['operation_type'] == 'clustering':
            cluster_count = experience['configuration_used'].get('n_clusters', 4)
            success_score = experience['success_score']
            
            # Update optimal cluster sizes knowledge
            data_size = experience['input_characteristics'].get('data_size', 0)
            size_category = self._categorize_data_size(data_size)
            
            if size_category not in self.knowledge_base['clustering_patterns']['optimal_cluster_sizes']:
                self.knowledge_base['clustering_patterns']['optimal_cluster_sizes'][size_category] = []
            
            self.knowledge_base['clustering_patterns']['optimal_cluster_sizes'][size_category].append({
                'cluster_count': cluster_count,
                'success_score': success_score,
                'timestamp': experience['timestamp']
            })
        
        # Update performance patterns
        if experience['success_score'] > 0.8:
            self.knowledge_base['performance_patterns']['successful_configurations'].append({
                'configuration': experience['configuration_used'],
                'success_score': experience['success_score'],
                'context': experience['input_characteristics']
            })
        elif experience['success_score'] < 0.3:
            self.knowledge_base['performance_patterns']['failure_modes'].append({
                'configuration': experience['configuration_used'],
                'failure_score': 1.0 - experience['success_score'],
                'context': experience['input_characteristics']
            })
        
        # Update with new patterns from insights
        for pattern in insights.get('patterns', []):
            if pattern['pattern_type'] == 'success_configuration':
                self.knowledge_base['performance_patterns']['optimization_paths'].append(pattern)
    
    async def _adapt_decision_patterns(self, experience: Dict[str, Any], insights: Dict[str, Any]):
        """Adapt decision-making patterns based on experience."""
        # Add new rules from insights
        for new_rule in insights.get('new_rules', []):
            if new_rule['confidence'] > self.confidence_threshold:
                self.adaptation_rules.append(new_rule)
                logger.info(f"Added new adaptation rule: {new_rule['rule_id']}")
        
        # Update exploration vs exploitation balance
        if experience['success_score'] > 0.8:
            # Successful exploration - slightly reduce exploration rate
            self.exploration_rate = max(0.05, self.exploration_rate * 0.99)
        elif experience['success_score'] < 0.3:
            # Failed exploration - increase exploration rate
            self.exploration_rate = min(0.4, self.exploration_rate * 1.01)
    
    def _update_intelligence_score(self, experience: Dict[str, Any], insights: Dict[str, Any]):
        """Update the overall intelligence score."""
        # Base update from experience success
        experience_impact = (experience['success_score'] - 0.5) * self.learning_rate
        
        # Bonus for generating insights
        insight_bonus = len(insights.get('patterns', [])) * 0.01
        insight_bonus += len(insights.get('new_rules', [])) * 0.02
        insight_bonus += len(insights.get('correlations', [])) * 0.015
        
        # Update intelligence score
        self.intelligence_score += experience_impact + insight_bonus
        self.intelligence_score = np.clip(self.intelligence_score, 0.0, 1.0)
        
        # Gradual growth toward optimal intelligence
        if self.intelligence_score < 0.8:
            self.intelligence_score += self.config['intelligence_growth_rate']
    
    async def autonomous_parameter_optimization(self, 
                                              operation_type: str,
                                              input_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomously optimize parameters based on learned patterns."""
        try:
            logger.info(f"🎯 Autonomous optimization for {operation_type}")
            
            # Start with base parameters
            base_params = self._get_base_parameters(operation_type)
            
            # Apply learned optimizations
            optimized_params = await self._apply_learned_optimizations(
                base_params, operation_type, input_characteristics
            )
            
            # Apply exploration vs exploitation
            if np.random.random() < self.exploration_rate:
                # Exploration: try new parameter combinations
                explored_params = self._explore_parameter_space(optimized_params)
                logger.info("🔍 Exploration mode: trying new parameter combinations")
            else:
                # Exploitation: use best known parameters
                explored_params = optimized_params
                logger.info("⚡ Exploitation mode: using best known parameters")
            
            # Final validation and adjustment
            final_params = self._validate_and_adjust_parameters(explored_params, input_characteristics)
            
            optimization_metadata = {
                'base_parameters': base_params,
                'learned_optimizations_applied': len(optimized_params) - len(base_params),
                'exploration_mode': np.random.random() < self.exploration_rate,
                'intelligence_score_used': self.intelligence_score,
                'confidence_level': self._calculate_parameter_confidence(final_params),
                'optimization_reasoning': self._generate_optimization_reasoning(
                    base_params, final_params, input_characteristics
                )
            }
            
            return {
                'optimized_parameters': final_params,
                'optimization_metadata': optimization_metadata,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Autonomous optimization failed: {e}")
            return {
                'optimized_parameters': self._get_base_parameters(operation_type),
                'success': False,
                'error': str(e)
            }
    
    def _get_base_parameters(self, operation_type: str) -> Dict[str, Any]:
        """Get base parameters for an operation type."""
        base_params = {
            'clustering': {
                'n_clusters': 4,
                'quantum_depth': 3,
                'neuromorphic_layers': 2,
                'reservoir_size': 100,
                'spectral_radius': 0.95,
                'leak_rate': 0.1,
                'quantum_noise_level': 0.01
            }
        }
        
        return base_params.get(operation_type, {})
    
    async def _apply_learned_optimizations(self, 
                                         base_params: Dict[str, Any],
                                         operation_type: str,
                                         input_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations learned from experience."""
        optimized_params = base_params.copy()
        
        # Apply successful configuration patterns
        successful_configs = self.knowledge_base['performance_patterns']['successful_configurations']
        
        if successful_configs:
            # Find most relevant successful configuration
            best_match = None
            best_similarity = 0
            
            for config_record in successful_configs[-20:]:  # Check recent successes
                similarity = self._calculate_similarity(
                    input_characteristics,
                    config_record['context']
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = config_record
            
            if best_match and best_similarity > 0.6:
                # Apply parameters from best matching successful case
                for param, value in best_match['configuration'].items():
                    if param in optimized_params:
                        # Weighted blend with base parameter
                        if isinstance(value, (int, float)) and isinstance(optimized_params[param], (int, float)):
                            weight = best_similarity * self.intelligence_score
                            optimized_params[param] = (
                                (1 - weight) * optimized_params[param] + 
                                weight * value
                            )
                        else:
                            optimized_params[param] = value
        
        # Apply correlation-based optimizations
        correlations = self._analyze_parameter_correlations()
        for correlation in correlations:
            if correlation['correlation'] > 0.7 and correlation['parameter'] in optimized_params:
                # Positive correlation - increase parameter value
                current_val = optimized_params[correlation['parameter']]
                if isinstance(current_val, (int, float)):
                    boost_factor = 1.0 + 0.1 * correlation['correlation']
                    optimized_params[correlation['parameter']] = current_val * boost_factor
        
        return optimized_params
    
    def _explore_parameter_space(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Explore parameter space for potential improvements."""
        explored_params = base_params.copy()
        
        # Define exploration ranges for each parameter type
        exploration_ranges = {
            'n_clusters': (2, 8),
            'quantum_depth': (1, 6),
            'neuromorphic_layers': (1, 4),
            'reservoir_size': (50, 300),
            'spectral_radius': (0.8, 0.99),
            'leak_rate': (0.01, 0.3),
            'quantum_noise_level': (0.001, 0.1)
        }
        
        # Randomly modify some parameters within reasonable ranges
        params_to_explore = np.random.choice(
            list(explored_params.keys()),
            size=min(3, len(explored_params)),
            replace=False
        )
        
        for param in params_to_explore:
            if param in exploration_ranges:
                min_val, max_val = exploration_ranges[param]
                explored_params[param] = np.random.uniform(min_val, max_val)
                
                # Round integers appropriately
                if param in ['n_clusters', 'quantum_depth', 'neuromorphic_layers', 'reservoir_size']:
                    explored_params[param] = int(explored_params[param])
        
        return explored_params
    
    def _validate_and_adjust_parameters(self, params: Dict[str, Any], 
                                       input_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters and make necessary adjustments."""
        adjusted_params = params.copy()
        
        # Apply business rules and constraints
        data_size = input_characteristics.get('data_size', 1000)
        
        # Adjust cluster count based on data size
        if data_size < 100:
            adjusted_params['n_clusters'] = min(adjusted_params.get('n_clusters', 4), 3)
        elif data_size > 10000:
            adjusted_params['n_clusters'] = min(adjusted_params.get('n_clusters', 4), 8)
        
        # Ensure quantum depth is reasonable
        adjusted_params['quantum_depth'] = max(1, min(6, adjusted_params.get('quantum_depth', 3)))
        
        # Ensure reservoir size is within computational limits
        adjusted_params['reservoir_size'] = max(10, min(500, adjusted_params.get('reservoir_size', 100)))
        
        # Ensure spectral radius is stable
        adjusted_params['spectral_radius'] = max(0.5, min(0.99, adjusted_params.get('spectral_radius', 0.95)))
        
        return adjusted_params
    
    def _calculate_parameter_confidence(self, params: Dict[str, Any]) -> float:
        """Calculate confidence in parameter selection."""
        # Base confidence from intelligence score
        confidence = self.intelligence_score
        
        # Boost confidence if parameters match known successful patterns
        successful_configs = self.knowledge_base['performance_patterns']['successful_configurations']
        
        if successful_configs:
            max_similarity = 0
            for config_record in successful_configs:
                similarity = self._calculate_similarity(params, config_record['configuration'])
                max_similarity = max(max_similarity, similarity)
            
            confidence = confidence * 0.7 + max_similarity * 0.3
        
        return min(1.0, confidence)
    
    def _generate_optimization_reasoning(self, 
                                       base_params: Dict[str, Any],
                                       final_params: Dict[str, Any],
                                       input_characteristics: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for optimization decisions."""
        reasoning = []
        
        for param, final_value in final_params.items():
            base_value = base_params.get(param, final_value)
            
            if base_value != final_value:
                if isinstance(base_value, (int, float)) and isinstance(final_value, (int, float)):
                    change_pct = ((final_value - base_value) / base_value) * 100 if base_value != 0 else 0
                    
                    if abs(change_pct) > 5:  # Significant change
                        direction = "increased" if change_pct > 0 else "decreased"
                        reasoning.append(
                            f"{param} {direction} by {abs(change_pct):.1f}% "
                            f"based on learned patterns"
                        )
        
        # Add context-based reasoning
        data_size = input_characteristics.get('data_size', 0)
        if data_size > 5000:
            reasoning.append("Parameters optimized for large dataset processing")
        elif data_size < 500:
            reasoning.append("Parameters optimized for small dataset efficiency")
        
        if not reasoning:
            reasoning.append("Parameters maintained at optimal baseline values")
        
        return reasoning
    
    def _calculate_success_score(self, results: Dict[str, Any], metrics: Dict[str, Any]) -> float:
        """Calculate overall success score for an operation."""
        scores = []
        
        # Clustering quality metrics
        if 'silhouette_score' in metrics:
            scores.append(metrics['silhouette_score'])
        
        if 'quantum_coherence' in metrics:
            scores.append(metrics['quantum_coherence'])
        
        if 'neuromorphic_stability' in metrics:
            scores.append(metrics['neuromorphic_stability'])
        
        # Performance metrics
        if 'parallel_efficiency' in metrics:
            scores.append(metrics['parallel_efficiency'])
        
        if 'cache_hit_rate' in metrics:
            scores.append(metrics['cache_hit_rate'])
        
        # Error rate (inverse score)
        if 'error_rate' in metrics:
            scores.append(1.0 - metrics['error_rate'])
        
        # Execution time score (faster is better, normalized)
        if 'execution_time' in metrics:
            # Assume 60 seconds is baseline, faster gets higher score
            time_score = max(0, min(1, 60 / metrics['execution_time']))
            scores.append(time_score)
        
        return np.mean(scores) if scores else 0.5
    
    def _hash_input_data(self, input_data: Dict[str, Any]) -> str:
        """Generate hash for input data for similarity comparison."""
        import hashlib
        
        # Create stable string representation
        stable_str = json.dumps(input_data, sort_keys=True, default=str)
        return hashlib.md5(stable_str.encode()).hexdigest()
    
    def _analyze_input_characteristics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze characteristics of input data."""
        characteristics = {}
        
        # Data size characteristics
        if 'data' in input_data:
            data = input_data['data']
            if hasattr(data, 'shape'):
                characteristics['data_size'] = data.shape[0]
                characteristics['feature_count'] = data.shape[1] if len(data.shape) > 1 else 1
                characteristics['data_type'] = 'numpy_array'
            elif isinstance(data, list):
                characteristics['data_size'] = len(data)
                characteristics['data_type'] = 'list'
        
        # Configuration characteristics
        if 'configuration' in input_data:
            config = input_data['configuration']
            characteristics.update({
                f"config_{k}": v for k, v in config.items()
                if isinstance(v, (int, float, str, bool))
            })
        
        # Operation characteristics
        characteristics['has_optimization'] = 'optimize' in input_data
        characteristics['has_validation'] = 'validate' in input_data
        
        return characteristics
    
    def _categorize_data_size(self, data_size: int) -> str:
        """Categorize data size for pattern matching."""
        if data_size < 100:
            return 'tiny'
        elif data_size < 1000:
            return 'small'
        elif data_size < 10000:
            return 'medium'
        elif data_size < 100000:
            return 'large'
        else:
            return 'huge'
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive intelligence system status."""
        recent_experiences = len([
            exp for exp in self.experience_memory
            if time.time() - exp['timestamp'] < 3600
        ])
        
        avg_recent_success = 0
        if recent_experiences > 0:
            recent_success_scores = [
                exp['success_score'] for exp in self.experience_memory
                if time.time() - exp['timestamp'] < 3600
            ]
            avg_recent_success = np.mean(recent_success_scores)
        
        return {
            'intelligence_score': self.intelligence_score,
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'confidence_threshold': self.confidence_threshold,
            'total_experiences': len(self.experience_memory),
            'recent_experiences_1h': recent_experiences,
            'average_recent_success': avg_recent_success,
            'adaptation_rules': len(self.adaptation_rules),
            'knowledge_base_patterns': len(self.knowledge_base['clustering_patterns']['optimal_cluster_sizes']),
            'successful_configurations': len(self.knowledge_base['performance_patterns']['successful_configurations']),
            'identified_failure_modes': len(self.knowledge_base['performance_patterns']['failure_modes']),
            'quantum_learning_depth': self.quantum_learning_depth,
            'quantum_memory_utilization': len(self.experience_memory) / self.quantum_memory_capacity,
            'intelligence_grade': self._grade_intelligence(self.intelligence_score),
            'system_maturity': self._assess_system_maturity()
        }
    
    def _grade_intelligence(self, score: float) -> str:
        """Grade intelligence level."""
        if score >= 0.9:
            return 'Genius'
        elif score >= 0.8:
            return 'Expert'
        elif score >= 0.7:
            return 'Advanced'
        elif score >= 0.6:
            return 'Intermediate'
        elif score >= 0.5:
            return 'Novice'
        else:
            return 'Learning'
    
    def _assess_system_maturity(self) -> str:
        """Assess overall system maturity."""
        factors = [
            len(self.experience_memory) >= 100,  # Sufficient experience
            self.intelligence_score >= 0.7,      # High intelligence
            len(self.adaptation_rules) >= 10,    # Rich rule set
            len(self.knowledge_base['performance_patterns']['successful_configurations']) >= 20  # Pattern library
        ]
        
        maturity_level = sum(factors) / len(factors)
        
        if maturity_level >= 0.75:
            return 'Mature'
        elif maturity_level >= 0.5:
            return 'Developing'
        else:
            return 'Emerging'


# Global quantum intelligence instance
quantum_intelligence = QuantumIntelligenceCore()