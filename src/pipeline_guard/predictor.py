"""Predictive Failure Analysis using Neuromorphic Computing
"""

import logging
import time


try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic functionality
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def zeros(size):
            return [0.0] * (size if isinstance(size, int) else size[0])
        @staticmethod
        def random():
            import random
            return random
        @staticmethod
        def dot(a, b):
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                return sum(x * y for x, y in zip(a, b))
            return 0.0
        @staticmethod
        def tanh(x):
            import math
            if isinstance(x, (list, tuple)):
                return [math.tanh(val) for val in x]
            return math.tanh(x)
        @staticmethod
        def exp(x):
            import math
            return math.exp(x)
        @staticmethod
        def sin(x):
            import math
            return math.sin(x)
        @staticmethod
        def cos(x):
            import math
            return math.cos(x)
        @staticmethod
        def pi():
            import math
            return math.pi

        class linalg:
            @staticmethod
            def eigvals(matrix):
                return [1.0]  # Mock eigenvalue
            @staticmethod
            def inv(matrix):
                return matrix  # Mock inverse

        class float64:
            pass

    np = MockNumpy()
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PredictionResult:
    """Prediction result for component failure"""
    component_name: str
    failure_probability: float
    predicted_failure_time: Optional[float]
    confidence: float
    contributing_factors: List[str]
    recommended_actions: List[str]
    timestamp: float


class NeuromorphicPredictor:
    """Neuromorphic computing-based failure predictor using Echo State Networks
    """

    def __init__(self, reservoir_size: int = 100, spectral_radius: float = 0.95):
        """Initialize neuromorphic predictor"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Echo State Network parameters
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius

        # Network components
        self.input_weights = None
        self.reservoir_weights = None
        self.output_weights = None
        self.reservoir_state = None

        # Training data
        self.training_data: List[Dict[str, Any]] = []
        self.is_trained = False

        # Feature normalization
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}

        self.logger.info("Neuromorphic predictor initialized")

    def initialize_network(self, input_size: int, output_size: int = 1) -> None:
        """Initialize the Echo State Network"""
        np.random.seed(42)  # For reproducibility

        # Input weights (random sparse matrix)
        self.input_weights = np.random.randn(self.reservoir_size, input_size) * 0.1

        # Reservoir weights (sparse random matrix with spectral radius control)
        self.reservoir_weights = np.random.randn(self.reservoir_size, self.reservoir_size)

        # Make sparse (connect only 10% of neurons)
        sparsity = 0.1
        mask = np.random.random((self.reservoir_size, self.reservoir_size)) < sparsity
        self.reservoir_weights *= mask

        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(self.reservoir_weights)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.reservoir_weights *= self.spectral_radius / max_eigenvalue

        # Initialize reservoir state
        self.reservoir_state = np.zeros(self.reservoir_size)

        # Output weights (will be trained)
        self.output_weights = np.random.randn(output_size, self.reservoir_size) * 0.01

        self.logger.info(f"ESN initialized: {input_size} inputs, {self.reservoir_size} reservoir, {output_size} outputs")

    def _extract_features(self, component_metrics: Dict[str, Any]) -> list:
        """Extract features from component metrics"""
        features = []

        # Basic features
        features.append(component_metrics.get('failure_count', 0))
        features.append(1.0 if component_metrics.get('critical', False) else 0.0)
        features.append(time.time() - component_metrics.get('last_failure', time.time()))

        # State encoding
        state_map = {
            'healthy': 0.0,
            'degraded': 0.3,
            'failing': 0.7,
            'critical': 1.0,
            'recovering': 0.5,
            'offline': 1.0
        }
        features.append(state_map.get(component_metrics.get('state', 'healthy'), 0.0))

        # Time-based features
        current_time = time.time()
        features.append(np.sin(2 * np.pi * (current_time % 86400) / 86400))  # Daily pattern
        features.append(np.cos(2 * np.pi * (current_time % 86400) / 86400))
        features.append(np.sin(2 * np.pi * (current_time % 604800) / 604800))  # Weekly pattern
        features.append(np.cos(2 * np.pi * (current_time % 604800) / 604800))

        return np.array(features)

    def _normalize_features(self, features: list, training: bool = False) -> list:
        """Normalize features for network input"""
        if training:
            # Calculate normalization parameters
            for i in range(len(features)):
                feature_name = f"feature_{i}"
                self.feature_means[feature_name] = np.mean(features[i]) if hasattr(features[i], '__iter__') else features[i]
                self.feature_stds[feature_name] = np.std(features[i]) if hasattr(features[i], '__iter__') else 1.0

                if self.feature_stds[feature_name] == 0:
                    self.feature_stds[feature_name] = 1.0

        # Apply normalization
        normalized = features.copy()
        for i in range(len(features)):
            feature_name = f"feature_{i}"
            if feature_name in self.feature_means:
                normalized[i] = (features[i] - self.feature_means[feature_name]) / self.feature_stds[feature_name]

        return normalized

    def _update_reservoir(self, input_vector: list) -> list:
        """Update reservoir state with new input"""
        # ESN update equation: x(t+1) = tanh(W_in * u(t) + W * x(t))
        input_activation = np.dot(self.input_weights, input_vector)
        reservoir_activation = np.dot(self.reservoir_weights, self.reservoir_state)

        self.reservoir_state = np.tanh(input_activation + reservoir_activation)

        return self.reservoir_state.copy()

    def train(self, training_data: List[Dict[str, Any]], failure_labels: List[bool]) -> None:
        """Train the neuromorphic predictor"""
        if len(training_data) != len(failure_labels):
            raise ValueError("Training data and labels must have same length")

        self.logger.info(f"Training neuromorphic predictor with {len(training_data)} samples")

        # Extract features
        feature_vectors = []
        for data in training_data:
            features = self._extract_features(data)
            feature_vectors.append(features)

        if not feature_vectors:
            self.logger.warning("No training data available")
            return

        # Initialize network if not done
        input_size = len(feature_vectors[0])
        if self.input_weights is None:
            self.initialize_network(input_size)

        # Normalize features
        all_features = np.array(feature_vectors)
        normalized_features = self._normalize_features(all_features, training=True)

        # Collect reservoir states
        reservoir_states = []
        self.reservoir_state = np.zeros(self.reservoir_size)  # Reset state

        for features in normalized_features:
            state = self._update_reservoir(features)
            reservoir_states.append(state)

        # Train output weights using ridge regression
        X = np.array(reservoir_states)
        y = np.array(failure_labels, dtype=float)

        # Add regularization
        reg_param = 1e-6
        XTX_reg = np.dot(X.T, X) + reg_param * np.eye(X.shape[1])

        try:
            self.output_weights = np.dot(np.dot(np.linalg.inv(XTX_reg), X.T), y).reshape(1, -1)
            self.is_trained = True
            self.logger.info("Neuromorphic predictor training completed")
        except np.linalg.LinAlgError:
            self.logger.error("Training failed - singular matrix")
            self.is_trained = False

    def predict(self, component_metrics: Dict[str, Any]) -> float:
        """Predict failure probability for component"""
        if not self.is_trained:
            # Return random prediction if not trained
            return np.random.random() * 0.5  # Low random value

        # Extract and normalize features
        features = self._extract_features(component_metrics)
        normalized_features = self._normalize_features(features)

        # Update reservoir and predict
        reservoir_state = self._update_reservoir(normalized_features)
        prediction = np.dot(self.output_weights, reservoir_state)[0]

        # Apply sigmoid to ensure [0,1] range
        probability = 1 / (1 + np.exp(-prediction))

        return float(probability)

    def save_model(self, filepath: Path) -> None:
        """Save the trained model"""
        model_data = {
            'input_weights': self.input_weights,
            'reservoir_weights': self.reservoir_weights,
            'output_weights': self.output_weights,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'reservoir_size': self.reservoir_size,
            'spectral_radius': self.spectral_radius,
            'is_trained': self.is_trained
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path) -> None:
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.input_weights = model_data['input_weights']
        self.reservoir_weights = model_data['reservoir_weights']
        self.output_weights = model_data['output_weights']
        self.feature_means = model_data['feature_means']
        self.feature_stds = model_data['feature_stds']
        self.reservoir_size = model_data['reservoir_size']
        self.spectral_radius = model_data['spectral_radius']
        self.is_trained = model_data['is_trained']

        # Reset reservoir state
        self.reservoir_state = np.zeros(self.reservoir_size)

        self.logger.info(f"Model loaded from {filepath}")


class FailurePredictor:
    """Main failure prediction system combining multiple prediction approaches
    """

    def __init__(self, model_save_path: Optional[Path] = None):
        """Initialize failure predictor"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Prediction models
        self.neuromorphic_predictor = NeuromorphicPredictor()
        self.statistical_predictor = StatisticalPredictor()

        # Historical data for training
        self.component_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failure_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Model persistence
        self.model_save_path = model_save_path or Path("models/failure_predictor.pkl")

        # Prediction cache
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # component -> (prediction, timestamp)
        self.cache_ttl = 300  # 5 minutes

        self.logger.info("Failure predictor initialized")

    def update_component_metrics(self, component_name: str, metrics: Dict[str, Any], failed: bool = False) -> None:
        """Update historical data for a component"""
        self.component_history[component_name].append(metrics)
        self.failure_history[component_name].append(failed)

        # Retrain models periodically
        if len(self.component_history[component_name]) % 50 == 0:
            self._retrain_models()

    def predict_component_failures(self, all_component_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Predict failure probabilities for all components"""
        predictions = {}

        for component_name, metrics in all_component_metrics.items():
            prediction = self.predict_single_component(component_name, metrics)
            predictions[component_name] = prediction

        return predictions

    def predict_single_component(self, component_name: str, metrics: Dict[str, Any]) -> float:
        """Predict failure probability for a single component"""
        # Check cache first
        if component_name in self.prediction_cache:
            prediction, timestamp = self.prediction_cache[component_name]
            if time.time() - timestamp < self.cache_ttl:
                return prediction

        # Get predictions from both models
        neuromorphic_pred = self.neuromorphic_predictor.predict(metrics)
        statistical_pred = self.statistical_predictor.predict(component_name, metrics)

        # Ensemble prediction (weighted average)
        if self.neuromorphic_predictor.is_trained:
            prediction = 0.7 * neuromorphic_pred + 0.3 * statistical_pred
        else:
            prediction = statistical_pred

        # Cache result
        self.prediction_cache[component_name] = (prediction, time.time())

        return prediction

    def get_detailed_prediction(self, component_name: str, metrics: Dict[str, Any]) -> PredictionResult:
        """Get detailed prediction with explanations"""
        failure_prob = self.predict_single_component(component_name, metrics)

        # Generate contributing factors
        contributing_factors = []
        if metrics.get('failure_count', 0) > 0:
            contributing_factors.append(f"Historical failures: {metrics.get('failure_count', 0)}")

        if metrics.get('state') in ['degraded', 'failing', 'critical']:
            contributing_factors.append(f"Current state: {metrics.get('state')}")

        if metrics.get('critical', False):
            contributing_factors.append("Critical component")

        # Generate recommendations
        recommendations = []
        if failure_prob > 0.8:
            recommendations.extend([
                "Immediate attention required",
                "Consider preventive maintenance",
                "Prepare backup systems"
            ])
        elif failure_prob > 0.6:
            recommendations.extend([
                "Monitor closely",
                "Schedule maintenance window",
                "Review component logs"
            ])
        elif failure_prob > 0.4:
            recommendations.append("Increase monitoring frequency")

        # Estimate failure time (simplified)
        predicted_failure_time = None
        if failure_prob > 0.5:
            # Very rough estimate based on probability
            hours_to_failure = (1 - failure_prob) * 48  # 0-48 hours
            predicted_failure_time = time.time() + (hours_to_failure * 3600)

        # Calculate confidence based on data availability
        confidence = min(len(self.component_history[component_name]) / 100, 1.0)

        return PredictionResult(
            component_name=component_name,
            failure_probability=failure_prob,
            predicted_failure_time=predicted_failure_time,
            confidence=confidence,
            contributing_factors=contributing_factors,
            recommended_actions=recommendations,
            timestamp=time.time()
        )

    def _retrain_models(self) -> None:
        """Retrain prediction models with accumulated data"""
        self.logger.info("Retraining prediction models")

        # Collect all training data
        training_data = []
        failure_labels = []

        for component_name in self.component_history:
            metrics_history = list(self.component_history[component_name])
            failure_history = list(self.failure_history[component_name])

            for metrics, failed in zip(metrics_history, failure_history):
                training_data.append(metrics)
                failure_labels.append(failed)

        if len(training_data) > 10:  # Minimum data requirement
            try:
                self.neuromorphic_predictor.train(training_data, failure_labels)

                # Save model if path provided
                if self.model_save_path:
                    self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    self.neuromorphic_predictor.save_model(self.model_save_path)

            except Exception as e:
                self.logger.error(f"Model retraining failed: {e}")

    def load_pretrained_model(self) -> None:
        """Load a pre-trained model if available"""
        if self.model_save_path.exists():
            try:
                self.neuromorphic_predictor.load_model(self.model_save_path)
                self.logger.info("Pre-trained model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load pre-trained model: {e}")

    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get prediction system statistics"""
        return {
            'neuromorphic_model_trained': self.neuromorphic_predictor.is_trained,
            'total_components_tracked': len(self.component_history),
            'total_data_points': sum(len(history) for history in self.component_history.values()),
            'cache_size': len(self.prediction_cache),
            'model_path': str(self.model_save_path) if self.model_save_path else None
        }


class StatisticalPredictor:
    """Simple statistical predictor as baseline/fallback
    """

    def __init__(self):
        """Initialize statistical predictor"""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Component statistics
        self.component_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'total_checks': 0,
            'failure_count': 0,
            'failure_rate': 0.0,
            'last_failure': None
        })

    def update_stats(self, component_name: str, failed: bool) -> None:
        """Update component statistics"""
        stats = self.component_stats[component_name]
        stats['total_checks'] += 1

        if failed:
            stats['failure_count'] += 1
            stats['last_failure'] = time.time()

        stats['failure_rate'] = stats['failure_count'] / stats['total_checks']

    def predict(self, component_name: str, metrics: Dict[str, Any]) -> float:
        """Simple statistical prediction"""
        stats = self.component_stats[component_name]

        # Base prediction on historical failure rate
        base_prediction = stats.get('failure_rate', 0.1)

        # Adjust based on current state
        state_multipliers = {
            'healthy': 0.5,
            'degraded': 1.5,
            'failing': 3.0,
            'critical': 5.0,
            'recovering': 1.2,
            'offline': 10.0
        }

        state = metrics.get('state', 'healthy')
        multiplier = state_multipliers.get(state, 1.0)

        # Adjust for recent failures
        failure_count = metrics.get('failure_count', 0)
        if failure_count > 0:
            recency_factor = 1 + (failure_count * 0.1)
            multiplier *= recency_factor

        prediction = min(base_prediction * multiplier, 1.0)

        return prediction
