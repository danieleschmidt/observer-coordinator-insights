#!/usr/bin/env python3
"""Adaptive AI Engine for Autonomous Learning and Optimization
Generation 4 self-improving AI system with reinforcement learning
"""

import asyncio
import json
import logging
import time
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LearningStrategy(Enum):
    """Different learning strategies for adaptive optimization"""
    EXPLORATORY = "exploratory"
    EXPLOITATIVE = "exploitative"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"


@dataclass
class ExperienceMemory:
    """Experience replay memory for reinforcement learning"""
    state: np.ndarray
    action: Dict[str, Any]
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPerformance:
    """Performance tracking for different models"""
    model_id: str
    accuracy_scores: List[float] = field(default_factory=list)
    training_times: List[float] = field(default_factory=list)
    resource_usage: List[Dict[str, float]] = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)


class AdaptiveQLearning:
    """Q-Learning agent for hyperparameter optimization"""

    def __init__(self, action_space_size: int, state_space_size: int,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01

        # Q-table for state-action values
        self.q_table = np.random.uniform(
            low=-2, high=0, size=(state_space_size, action_space_size)
        )

        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.training_history = []

    def get_state_representation(self, performance_metrics: Dict[str, float]) -> int:
        """Convert performance metrics to discrete state"""
        # Discretize continuous metrics into state bins
        accuracy = performance_metrics.get('accuracy', 0.0)
        speed = performance_metrics.get('speed', 0.0)
        resource_usage = performance_metrics.get('resource_usage', 0.0)

        # Create state bins
        accuracy_bin = min(int(accuracy * 10), 9)
        speed_bin = min(int(speed * 10), 9)
        resource_bin = min(int(resource_usage * 10), 9)

        # Combine bins into single state
        state = accuracy_bin * 100 + speed_bin * 10 + resource_bin
        return min(state, self.state_space_size - 1)

    def choose_action(self, state: int, strategy: LearningStrategy = LearningStrategy.BALANCED) -> int:
        """Choose action using epsilon-greedy strategy with learning strategy"""
        if strategy == LearningStrategy.EXPLORATORY:
            epsilon = self.epsilon * 2  # More exploration
        elif strategy == LearningStrategy.EXPLOITATIVE:
            epsilon = self.epsilon * 0.5  # Less exploration
        elif strategy == LearningStrategy.AGGRESSIVE:
            epsilon = self.epsilon * 1.5
        elif strategy == LearningStrategy.CONSERVATIVE:
            epsilon = self.epsilon * 0.3
        else:  # BALANCED
            epsilon = self.epsilon

        if np.random.random() < epsilon:
            return np.random.randint(self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def store_experience(self, state: int, action: int, reward: float,
                        next_state: int, done: bool):
        """Store experience in replay buffer"""
        experience = ExperienceMemory(
            state=np.array([state]),
            action={'action_id': action},
            reward=reward,
            next_state=np.array([next_state]),
            done=done,
            timestamp=time.time()
        )
        self.memory.append(experience)

    def replay_training(self, batch_size: int = 32):
        """Train using experience replay"""
        if len(self.memory) < batch_size:
            return

        # Sample random batch
        batch = np.random.choice(self.memory, batch_size, replace=False)

        for experience in batch:
            state = int(experience.state[0])
            action = experience.action['action_id']
            reward = experience.reward
            next_state = int(experience.next_state[0])
            done = experience.done

            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.discount_factor * np.max(self.q_table[next_state])

            self.q_table[state, action] += self.learning_rate * (
                target - self.q_table[state, action]
            )

        # Decay epsilon
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.training_history.append({
            'timestamp': time.time(),
            'epsilon': self.epsilon,
            'q_table_mean': float(np.mean(self.q_table)),
            'memory_size': len(self.memory)
        })


class HyperparameterOptimizer:
    """Advanced hyperparameter optimization using multiple strategies"""

    def __init__(self):
        self.optimization_history = []
        self.best_configurations = {}
        self.performance_predictor = None

    def bayesian_optimization(self, parameter_space: Dict[str, Tuple[float, float]],
                            objective_function: Callable, n_iterations: int = 50) -> Dict[str, Any]:
        """Bayesian optimization for hyperparameter tuning"""
        logger.info("Starting Bayesian optimization...")

        best_params = None
        best_score = -np.inf
        observations = []

        for iteration in range(n_iterations):
            # Sample parameters
            params = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                if iteration == 0:  # Random initialization
                    params[param_name] = np.random.uniform(min_val, max_val)
                else:
                    # Use simple acquisition function
                    params[param_name] = self._acquisition_sample(
                        param_name, min_val, max_val, observations
                    )

            # Evaluate objective function
            score = objective_function(params)

            observations.append({
                'params': params.copy(),
                'score': score,
                'iteration': iteration
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()

            logger.debug(f"Iteration {iteration}: Score {score:.4f}, Best {best_score:.4f}")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': observations
        }

    def _acquisition_sample(self, param_name: str, min_val: float, max_val: float,
                          observations: List[Dict]) -> float:
        """Simple acquisition function for parameter sampling"""
        if len(observations) < 5:
            return np.random.uniform(min_val, max_val)

        # Extract parameter values and scores
        param_values = [obs['params'][param_name] for obs in observations]
        scores = [obs['score'] for obs in observations]

        # Find best performing region
        best_idx = np.argmax(scores)
        best_param = param_values[best_idx]

        # Sample around best parameter with exploration
        exploration_radius = (max_val - min_val) * 0.1
        candidate = np.random.normal(best_param, exploration_radius)

        return np.clip(candidate, min_val, max_val)

    def genetic_algorithm_optimization(self, parameter_space: Dict[str, Tuple[float, float]],
                                     objective_function: Callable,
                                     population_size: int = 20,
                                     generations: int = 30) -> Dict[str, Any]:
        """Genetic algorithm for hyperparameter optimization"""
        logger.info("Starting genetic algorithm optimization...")

        # Initialize population
        population = []
        for _ in range(population_size):
            individual = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                individual[param_name] = np.random.uniform(min_val, max_val)
            population.append(individual)

        best_individual = None
        best_fitness = -np.inf
        evolution_history = []

        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = objective_function(individual)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()

            evolution_history.append({
                'generation': generation,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'population_diversity': self._calculate_diversity(population)
            })

            # Selection and reproduction
            new_population = []

            # Elitism: Keep best individuals
            elite_count = max(1, population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())

            # Crossover and mutation
            while len(new_population) < population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)

                child = self._crossover(parent1, parent2)
                child = self._mutate(child, parameter_space)

                new_population.append(child)

            population = new_population

        return {
            'best_params': best_individual,
            'best_score': best_fitness,
            'evolution_history': evolution_history
        }

    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float],
                            tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm"""
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover operation for genetic algorithm"""
        child = {}
        for param_name in parent1:
            # Uniform crossover
            if np.random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    def _mutate(self, individual: Dict, parameter_space: Dict[str, Tuple[float, float]],
               mutation_rate: float = 0.1) -> Dict:
        """Mutation operation for genetic algorithm"""
        mutated = individual.copy()

        for param_name, (min_val, max_val) in parameter_space.items():
            if np.random.random() < mutation_rate:
                # Gaussian mutation
                current_val = mutated[param_name]
                mutation_strength = (max_val - min_val) * 0.1
                new_val = current_val + np.random.normal(0, mutation_strength)
                mutated[param_name] = np.clip(new_val, min_val, max_val)

        return mutated

    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0

        distances = []
        param_names = list(population[0].keys())

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = 0.0
                for param_name in param_names:
                    dist += (population[i][param_name] - population[j][param_name]) ** 2
                distances.append(np.sqrt(dist))

        return float(np.mean(distances))


class AdaptiveAIEngine:
    """Main adaptive AI engine coordinating all optimization strategies"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize components
        self.q_learning_agent = AdaptiveQLearning(
            action_space_size=self.config.get('action_space_size', 100),
            state_space_size=self.config.get('state_space_size', 1000)
        )

        self.hyperparameter_optimizer = HyperparameterOptimizer()

        # Performance tracking
        self.model_performances = {}
        self.learning_history = []
        self.optimization_strategies = {
            'q_learning': self.q_learning_agent,
            'bayesian': self.hyperparameter_optimizer.bayesian_optimization,
            'genetic': self.hyperparameter_optimizer.genetic_algorithm_optimization
        }

        # Adaptive parameters
        self.current_strategy = LearningStrategy.BALANCED
        self.strategy_performance = defaultdict(list)
        self.adaptation_threshold = 0.05

        logger.info("Adaptive AI Engine initialized")

    def register_model(self, model_id: str, initial_hyperparameters: Dict[str, Any]):
        """Register a new model for optimization"""
        self.model_performances[model_id] = ModelPerformance(
            model_id=model_id,
            hyperparameters=initial_hyperparameters.copy()
        )
        logger.info(f"Registered model: {model_id}")

    def update_model_performance(self, model_id: str, performance_metrics: Dict[str, float],
                               training_time: float, resource_usage: Dict[str, float]):
        """Update performance metrics for a model"""
        if model_id not in self.model_performances:
            self.register_model(model_id, {})

        model_perf = self.model_performances[model_id]

        # Calculate accuracy score from performance metrics
        accuracy_score = performance_metrics.get('silhouette_score', 0.0)
        model_perf.accuracy_scores.append(accuracy_score)
        model_perf.training_times.append(training_time)
        model_perf.resource_usage.append(resource_usage)
        model_perf.last_updated = time.time()

        # Update success rate
        recent_scores = model_perf.accuracy_scores[-10:]  # Last 10 scores
        model_perf.success_rate = np.mean([score > 0.5 for score in recent_scores])

        # Update Q-learning
        state = self.q_learning_agent.get_state_representation(performance_metrics)
        reward = self._calculate_reward(performance_metrics, training_time, resource_usage)

        # Store experience for Q-learning
        if len(model_perf.accuracy_scores) > 1:
            prev_metrics = {
                'accuracy': model_perf.accuracy_scores[-2],
                'speed': 1.0 / model_perf.training_times[-2] if model_perf.training_times[-2] > 0 else 0.0,
                'resource_usage': sum(model_perf.resource_usage[-2].values()) / len(model_perf.resource_usage[-2])
            }
            prev_state = self.q_learning_agent.get_state_representation(prev_metrics)
            action = 0  # Placeholder action

            self.q_learning_agent.store_experience(
                prev_state, action, reward, state, done=False
            )

        logger.debug(f"Updated performance for {model_id}: accuracy={accuracy_score:.3f}")

    def _calculate_reward(self, performance_metrics: Dict[str, float],
                         training_time: float, resource_usage: Dict[str, float]) -> float:
        """Calculate reward signal for reinforcement learning"""
        accuracy_reward = performance_metrics.get('silhouette_score', 0.0) * 10
        speed_reward = max(0, 2.0 - training_time)  # Prefer faster training
        resource_reward = max(0, 2.0 - sum(resource_usage.values()) / len(resource_usage))

        total_reward = accuracy_reward + speed_reward * 0.3 + resource_reward * 0.2
        return total_reward

    def adaptive_strategy_selection(self) -> LearningStrategy:
        """Adaptively select learning strategy based on recent performance"""
        if len(self.strategy_performance) < 2:
            return LearningStrategy.BALANCED

        # Calculate average performance for each strategy
        strategy_scores = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_scores[strategy] = np.mean(scores[-5:])  # Last 5 scores

        if not strategy_scores:
            return LearningStrategy.BALANCED

        # Select best performing strategy
        best_strategy = max(strategy_scores.keys(), key=lambda k: strategy_scores[k])

        # Add exploration chance
        if np.random.random() < 0.1:  # 10% exploration
            return np.random.choice(list(LearningStrategy))

        return LearningStrategy(best_strategy)

    def optimize_hyperparameters(self, model_id: str,
                                parameter_space: Dict[str, Tuple[float, float]],
                                objective_function: Callable,
                                optimization_method: str = 'auto') -> Dict[str, Any]:
        """Optimize hyperparameters using adaptive strategy selection"""
        if optimization_method == 'auto':
            # Adaptively choose optimization method
            if model_id in self.model_performances:
                perf = self.model_performances[model_id]
                if len(perf.accuracy_scores) < 5:
                    optimization_method = 'genetic'  # Exploration phase
                elif perf.success_rate > 0.7:
                    optimization_method = 'bayesian'  # Exploitation phase
                else:
                    optimization_method = 'q_learning'  # Learning phase
            else:
                optimization_method = 'genetic'

        logger.info(f"Optimizing {model_id} using {optimization_method} method")

        start_time = time.time()

        if optimization_method == 'bayesian':
            result = self.hyperparameter_optimizer.bayesian_optimization(
                parameter_space, objective_function
            )
        elif optimization_method == 'genetic':
            result = self.hyperparameter_optimizer.genetic_algorithm_optimization(
                parameter_space, objective_function
            )
        else:  # q_learning or fallback
            result = self._q_learning_optimization(
                model_id, parameter_space, objective_function
            )

        optimization_time = time.time() - start_time

        # Update learning history
        self.learning_history.append({
            'model_id': model_id,
            'optimization_method': optimization_method,
            'optimization_time': optimization_time,
            'best_score': result['best_score'],
            'timestamp': time.time()
        })

        # Update strategy performance
        self.strategy_performance[optimization_method].append(result['best_score'])

        logger.info(f"Optimization completed: {result['best_score']:.4f} in {optimization_time:.2f}s")

        return result

    def _q_learning_optimization(self, model_id: str,
                               parameter_space: Dict[str, Tuple[float, float]],
                               objective_function: Callable) -> Dict[str, Any]:
        """Q-learning based hyperparameter optimization"""
        best_params = None
        best_score = -np.inf
        optimization_history = []

        # Map parameters to action space
        param_names = list(parameter_space.keys())
        actions_per_param = max(10, 100 // len(param_names))

        for episode in range(50):
            # Select strategy adaptively
            strategy = self.adaptive_strategy_selection()

            # Generate parameters using Q-learning
            params = {}
            for i, (param_name, (min_val, max_val)) in enumerate(parameter_space.items()):
                # Use Q-learning to select parameter bin
                param_state = i * actions_per_param
                action = self.q_learning_agent.choose_action(param_state, strategy)

                # Convert action to parameter value
                param_val = min_val + (max_val - min_val) * (action / actions_per_param)
                params[param_name] = param_val

            # Evaluate parameters
            score = objective_function(params)

            optimization_history.append({
                'episode': episode,
                'params': params.copy(),
                'score': score,
                'strategy': strategy.value
            })

            if score > best_score:
                best_score = score
                best_params = params.copy()

            # Update Q-learning with reward
            reward = score
            for i, param_name in enumerate(param_names):
                param_state = i * actions_per_param
                action = int((params[param_name] - parameter_space[param_name][0]) /
                           (parameter_space[param_name][1] - parameter_space[param_name][0]) * actions_per_param)
                action = max(0, min(actions_per_param - 1, action))

                next_state = param_state  # Simplified
                self.q_learning_agent.store_experience(
                    param_state, action, reward, next_state, done=(episode == 49)
                )

            # Perform Q-learning update
            if episode % 10 == 0:
                self.q_learning_agent.replay_training()

        return {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_history': optimization_history
        }

    async def continuous_optimization(self, model_id: str,
                                    parameter_space: Dict[str, Tuple[float, float]],
                                    objective_function: Callable,
                                    optimization_interval: float = 3600):
        """Continuous background optimization"""
        logger.info(f"Starting continuous optimization for {model_id}")

        while True:
            try:
                # Perform optimization
                result = self.optimize_hyperparameters(
                    model_id, parameter_space, objective_function
                )

                # Update model with best parameters
                if model_id in self.model_performances:
                    self.model_performances[model_id].hyperparameters.update(
                        result['best_params']
                    )

                logger.info(f"Continuous optimization cycle completed for {model_id}")

            except Exception as e:
                logger.error(f"Error in continuous optimization: {e}")

            # Wait for next optimization cycle
            await asyncio.sleep(optimization_interval)

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        report = {
            'model_performances': {},
            'learning_history': self.learning_history,
            'strategy_performance': dict(self.strategy_performance),
            'q_learning_stats': {
                'epsilon': self.q_learning_agent.epsilon,
                'memory_size': len(self.q_learning_agent.memory),
                'training_episodes': len(self.q_learning_agent.training_history)
            },
            'current_strategy': self.current_strategy.value,
            'total_optimizations': len(self.learning_history)
        }

        # Add model performance summaries
        for model_id, perf in self.model_performances.items():
            report['model_performances'][model_id] = {
                'success_rate': perf.success_rate,
                'avg_accuracy': float(np.mean(perf.accuracy_scores)) if perf.accuracy_scores else 0.0,
                'avg_training_time': float(np.mean(perf.training_times)) if perf.training_times else 0.0,
                'best_hyperparameters': perf.hyperparameters,
                'total_evaluations': len(perf.accuracy_scores),
                'last_updated': perf.last_updated
            }

        return report

    def save_state(self, filepath: Path):
        """Save engine state to disk"""
        state = {
            'model_performances': {
                model_id: {
                    'model_id': perf.model_id,
                    'accuracy_scores': perf.accuracy_scores,
                    'training_times': perf.training_times,
                    'resource_usage': perf.resource_usage,
                    'hyperparameters': perf.hyperparameters,
                    'success_rate': perf.success_rate,
                    'last_updated': perf.last_updated
                } for model_id, perf in self.model_performances.items()
            },
            'learning_history': self.learning_history,
            'strategy_performance': dict(self.strategy_performance),
            'q_learning_state': {
                'q_table': self.q_learning_agent.q_table.tolist(),
                'epsilon': self.q_learning_agent.epsilon,
                'training_history': self.q_learning_agent.training_history
            },
            'current_strategy': self.current_strategy.value
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Saved adaptive AI engine state to {filepath}")

    def load_state(self, filepath: Path):
        """Load engine state from disk"""
        with open(filepath) as f:
            state = json.load(f)

        # Restore model performances
        self.model_performances = {}
        for model_id, perf_data in state['model_performances'].items():
            self.model_performances[model_id] = ModelPerformance(**perf_data)

        # Restore other state
        self.learning_history = state['learning_history']
        self.strategy_performance = defaultdict(list, state['strategy_performance'])

        # Restore Q-learning state
        q_state = state['q_learning_state']
        self.q_learning_agent.q_table = np.array(q_state['q_table'])
        self.q_learning_agent.epsilon = q_state['epsilon']
        self.q_learning_agent.training_history = q_state['training_history']

        self.current_strategy = LearningStrategy(state['current_strategy'])

        logger.info(f"Loaded adaptive AI engine state from {filepath}")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Initialize adaptive AI engine
    engine = AdaptiveAIEngine()

    # Example objective function
    def test_objective(params):
        # Simulate model performance based on hyperparameters
        learning_rate = params.get('learning_rate', 0.01)
        batch_size = params.get('batch_size', 32)

        # Simulate performance (normally this would train and evaluate a model)
        performance = 0.5 + 0.3 * np.sin(learning_rate * 100) + 0.2 * np.cos(batch_size / 10)
        performance += np.random.normal(0, 0.1)  # Add noise

        return np.clip(performance, 0, 1)

    # Test hyperparameter optimization
    parameter_space = {
        'learning_rate': (0.001, 0.1),
        'batch_size': (16, 128)
    }

    result = engine.optimize_hyperparameters(
        'test_model', parameter_space, test_objective, 'genetic'
    )

    print("Optimization Result:")
    print(f"Best Parameters: {result['best_params']}")
    print(f"Best Score: {result['best_score']:.4f}")

    # Generate optimization report
    report = engine.get_optimization_report()
    print("\nOptimization Report:")
    print(f"Total Optimizations: {report['total_optimizations']}")
    print(f"Current Strategy: {report['current_strategy']}")
