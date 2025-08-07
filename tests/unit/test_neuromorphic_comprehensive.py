"""
Comprehensive Unit Tests for Neuromorphic Clustering Algorithms
Tests all neuromorphic components (ESN, SNN, LSM, Hybrid) with edge cases and error handling
Target: 95%+ code coverage with exhaustive validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
import time
import threading
import tempfile
import json
import warnings
from pathlib import Path
import sys
from dataclasses import asdict
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from insights_clustering.neuromorphic_clustering import (
    EchoStateNetwork,
    SpikingNeuralCluster, 
    LiquidStateMachine,
    NeuromorphicClusterer,
    NeuromorphicClusteringMethod,
    NeuromorphicException,
    NeuromorphicErrorType,
    CircuitBreakerState,
    CircuitBreaker,
    CircuitBreakerConfig,
    RetryManager,
    ClusteringMetrics,
    ResourceMetrics,
    get_correlation_id,
    set_correlation_id,
    resource_monitor,
    timeout_operation
)

# Global test configuration
TEST_CONFIG = {
    'seed': 42,
    'tolerance': 1e-6,
    'timeout_seconds': 30,
    'default_n_samples': 100,
    'min_quality_threshold': 0.7
}


class TestEchoStateNetwork:
    """Comprehensive tests for Echo State Network implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        np.random.seed(TEST_CONFIG['seed'])
        self.esn_params = {
            'reservoir_size': 50,
            'input_size': 4,
            'spectral_radius': 0.95,
            'sparsity': 0.1,
            'leaking_rate': 0.3,
            'random_state': TEST_CONFIG['seed']
        }
        self.esn = EchoStateNetwork(**self.esn_params)
        
    def test_esn_initialization(self):
        """Test ESN initialization with various parameters"""
        # Test default initialization
        esn = EchoStateNetwork()
        assert esn.reservoir_size == 100
        assert esn.input_size == 4
        assert esn.spectral_radius == 0.95
        
        # Test custom initialization
        custom_esn = EchoStateNetwork(reservoir_size=200, spectral_radius=0.8)
        assert custom_esn.reservoir_size == 200
        assert custom_esn.spectral_radius == 0.8
        
        # Verify weight matrix shapes
        assert self.esn.W_in.shape == (50, 4)
        assert self.esn.W_res.shape == (50, 50)
        
        # Verify spectral radius constraint
        eigenvals = np.linalg.eigvals(self.esn.W_res)
        max_eigenval = np.max(np.abs(eigenvals))
        assert abs(max_eigenval - self.esn_params['spectral_radius']) < TEST_CONFIG['tolerance']
        
    def test_esn_state_update(self):
        """Test ESN state update mechanism"""
        # Test with standard input
        input_vec = np.array([0.5, 0.3, 0.2, 0.4])
        initial_state = self.esn.state.copy()
        
        updated_state = self.esn.update_state(input_vec)
        
        # State should be updated
        assert not np.array_equal(initial_state, updated_state)
        assert updated_state.shape == (self.esn_params['reservoir_size'],)
        
        # State values should be in tanh range [-1, 1]
        assert np.all(updated_state >= -1.0)
        assert np.all(updated_state <= 1.0)
        
    def test_esn_sequence_processing(self):
        """Test ESN processing of temporal sequences"""
        # Create test sequence
        seq_length = 20
        sequence = np.random.rand(seq_length, 4)
        
        states = self.esn.process_sequence(sequence)
        
        # Verify output shape and properties
        assert states.shape == (seq_length, self.esn_params['reservoir_size'])
        assert len(self.esn.states_history) == seq_length
        
        # Test state evolution (should change over time)
        state_changes = np.diff(states, axis=0)
        assert np.any(np.abs(state_changes) > TEST_CONFIG['tolerance'])
        
    def test_esn_feature_extraction(self):
        """Test ESN feature extraction from states"""
        # Create test states
        seq_length = 15
        states = np.random.randn(seq_length, self.esn_params['reservoir_size'])
        
        features = self.esn.extract_features(states)
        
        # Verify all expected features are present
        expected_features = [
            'mean_activation', 'std_activation', 'max_activation',
            'final_state', 'activation_trend', 'stability_measure'
        ]
        assert all(feat in features for feat in expected_features)
        
        # Verify feature shapes
        assert features['mean_activation'].shape == (self.esn_params['reservoir_size'],)
        assert features['final_state'].shape == (self.esn_params['reservoir_size'],)
        
        # Test edge case: empty states
        empty_features = self.esn.extract_features(np.array([]))
        assert empty_features['final_state'].shape == (self.esn_params['reservoir_size'],)
        
    def test_esn_reproducibility(self):
        """Test ESN reproducibility with same random seed"""
        # Create two ESNs with same seed
        esn1 = EchoStateNetwork(random_state=42)
        esn2 = EchoStateNetwork(random_state=42)
        
        # Test same initialization
        assert np.array_equal(esn1.W_in, esn2.W_in)
        assert np.array_equal(esn1.W_res, esn2.W_res)
        
        # Test same sequence processing
        sequence = np.random.rand(10, 4)
        states1 = esn1.process_sequence(sequence)
        states2 = esn2.process_sequence(sequence)
        
        assert np.allclose(states1, states2, atol=TEST_CONFIG['tolerance'])
        
    def test_esn_edge_cases(self):
        """Test ESN with edge cases and boundary conditions"""
        # Test with zero input
        zero_input = np.zeros(4)
        state = self.esn.update_state(zero_input)
        assert state is not None
        assert not np.any(np.isnan(state))
        
        # Test with extreme values
        extreme_input = np.array([1000, -1000, 0, 0.5])
        state = self.esn.update_state(extreme_input)
        assert np.all(np.isfinite(state))
        
        # Test with single time step sequence
        single_seq = np.array([[0.5, 0.3, 0.2, 0.4]])
        states = self.esn.process_sequence(single_seq)
        assert states.shape == (1, self.esn_params['reservoir_size'])


class TestSpikingNeuralCluster:
    """Comprehensive tests for Spiking Neural Network implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        np.random.seed(TEST_CONFIG['seed'])
        self.snn_params = {
            'n_neurons': 30,
            'threshold': 1.0,
            'tau_membrane': 20.0,
            'tau_synapse': 5.0,
            'learning_rate': 0.01,
            'random_state': TEST_CONFIG['seed']
        }
        self.snn = SpikingNeuralCluster(**self.snn_params)
        
    def test_snn_initialization(self):
        """Test SNN initialization"""
        assert self.snn.n_neurons == 30
        assert self.snn.threshold == 1.0
        assert self.snn.membrane_potential.shape == (30,)
        assert self.snn.synaptic_weights.shape == (4, 30)
        assert len(self.snn.spike_times) == 30
        
    def test_snn_input_encoding(self):
        """Test conversion of energy values to spike trains"""
        # Test standard input
        energy_vec = np.array([0.8, 0.4, 0.2, 0.6])
        spike_trains = self.snn.encode_input(energy_vec, duration=100.0)
        
        assert len(spike_trains) == 4
        
        # Higher energy should result in more spikes
        n_spikes = [len(train) for train in spike_trains]
        expected_order = np.argsort(energy_vec)[::-1]  # Descending order
        actual_order = np.argsort(n_spikes)[::-1]
        
        # Correlation should be positive (not perfect due to randomness)
        correlation = np.corrcoef(energy_vec, n_spikes)[0, 1]
        assert correlation > 0.3
        
    def test_snn_dynamics_simulation(self):
        """Test SNN dynamics simulation"""
        # Create test spike trains
        input_spikes = [
            [10, 20, 30],  # High frequency
            [25, 50, 75],  # Medium frequency  
            [40],          # Low frequency
            [15, 45, 80]   # Medium frequency
        ]
        
        spike_response = self.snn.simulate_dynamics(input_spikes, duration=100.0)
        
        # Verify output shape
        assert spike_response.shape == (self.snn_params['n_neurons'], 100)
        
        # Check that some neurons spiked
        total_spikes = np.sum(spike_response)
        assert total_spikes > 0
        
        # Verify spike times were recorded
        assert any(len(times) > 0 for times in self.snn.spike_times)
        
    def test_snn_feature_extraction(self):
        """Test spike feature extraction"""
        # Create mock spike response
        n_neurons = self.snn_params['n_neurons']
        spike_response = np.random.binomial(1, 0.1, (n_neurons, 100))
        
        features = self.snn.extract_spike_features(spike_response)
        
        # Should have 2 features per neuron (firing rate + burst count)
        expected_length = n_neurons * 2
        assert features.shape == (expected_length,)
        
        # Features should be non-negative
        assert np.all(features >= 0)
        
    def test_snn_membrane_dynamics(self):
        """Test membrane potential dynamics"""
        # Test membrane potential evolution
        initial_potential = self.snn.membrane_potential.copy()
        
        # Simulate with no input (should decay)
        input_spikes = [[], [], [], []]
        self.snn.simulate_dynamics(input_spikes, duration=50.0)
        
        # With tau_membrane=20, potentials should mostly decay
        # (exact behavior depends on lateral connections)
        assert not np.array_equal(initial_potential, self.snn.membrane_potential)
        
    def test_snn_edge_cases(self):
        """Test SNN edge cases"""
        # Test with no spikes
        empty_spikes = [[], [], [], []]
        response = self.snn.simulate_dynamics(empty_spikes, duration=50.0)
        assert response.shape == (self.snn_params['n_neurons'], 50)
        
        # Test with very high spike rate
        high_freq_spikes = [list(range(0, 100, 2)) for _ in range(4)]
        response = self.snn.simulate_dynamics(high_freq_spikes, duration=100.0)
        assert np.sum(response) > 0  # Should produce some output spikes
        
        # Test feature extraction with no spikes
        no_spike_response = np.zeros((30, 100))
        features = self.snn.extract_spike_features(no_spike_response)
        assert np.all(features == 0)


class TestLiquidStateMachine:
    """Comprehensive tests for Liquid State Machine implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        np.random.seed(TEST_CONFIG['seed'])
        self.lsm_params = {
            'liquid_size': 40,
            'input_size': 4,
            'connection_prob': 0.3,
            'tau_membrane': 30.0,
            'random_state': TEST_CONFIG['seed']
        }
        self.lsm = LiquidStateMachine(**self.lsm_params)
        
    def test_lsm_initialization(self):
        """Test LSM initialization"""
        assert self.lsm.liquid_size == 40
        assert self.lsm.input_size == 4
        assert self.lsm.W_input.shape == (40, 4)
        assert self.lsm.W_liquid.shape == (40, 40)
        assert self.lsm.liquid_state.shape == (40,)
        
    def test_lsm_topology_creation(self):
        """Test liquid network topology"""
        # Verify connection matrix properties
        W = self.lsm.W_liquid
        
        # No self-connections
        assert np.all(np.diag(W) == 0)
        
        # Check sparsity (approximately correct)
        n_connections = np.count_nonzero(W)
        max_connections = 40 * 39  # No self-connections
        actual_density = n_connections / max_connections
        expected_density = self.lsm_params['connection_prob']
        
        # Allow some tolerance for randomness
        assert abs(actual_density - expected_density) < 0.1
        
    def test_lsm_state_update(self):
        """Test LSM state update"""
        input_vec = np.array([0.5, 0.3, 0.8, 0.1])
        initial_state = self.lsm.liquid_state.copy()
        
        new_state = self.lsm.update_liquid(input_vec)
        
        # State should change
        assert not np.array_equal(initial_state, new_state)
        assert new_state.shape == (40,)
        
        # Multiple updates should show evolution
        state_sequence = [new_state.copy()]
        for _ in range(5):
            new_state = self.lsm.update_liquid(input_vec)
            state_sequence.append(new_state.copy())
        
        # States should evolve (not all identical)
        states_array = np.array(state_sequence)
        state_variance = np.var(states_array, axis=0)
        assert np.sum(state_variance) > 0
        
    def test_lsm_temporal_processing(self):
        """Test LSM temporal sequence processing"""
        # Create test sequence
        sequence = np.random.rand(10, 4)
        
        states = self.lsm.process_temporal_sequence(sequence)
        
        assert states.shape == (10, 40)
        
        # Check state evolution across sequence
        state_changes = np.diff(states, axis=0)
        assert np.sum(np.abs(state_changes)) > 0
        
    def test_lsm_distance_dependent_connections(self):
        """Test distance-dependent connectivity in liquid"""
        W = self.lsm.W_liquid
        
        # Check that connections exist and have reasonable structure
        nonzero_weights = W[W != 0]
        assert len(nonzero_weights) > 0
        
        # Weights should be bounded (from exponential decay)
        assert np.all(np.abs(nonzero_weights) <= 1.0)
        
        # Should have both excitatory and inhibitory connections
        positive_weights = nonzero_weights[nonzero_weights > 0]
        negative_weights = nonzero_weights[nonzero_weights < 0]
        
        assert len(positive_weights) > 0
        # Note: negative weights depend on 20% probability, might not always occur
        
    def test_lsm_edge_cases(self):
        """Test LSM edge cases"""
        # Test with zero input
        zero_input = np.zeros(4)
        state = self.lsm.update_liquid(zero_input)
        assert not np.any(np.isnan(state))
        
        # Test with empty sequence
        empty_seq = np.array([]).reshape(0, 4)
        states = self.lsm.process_temporal_sequence(empty_seq)
        assert states.shape == (0, 40)


class TestNeuromorphicClusterer:
    """Comprehensive tests for main NeuromorphicClusterer class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        np.random.seed(TEST_CONFIG['seed'])
        set_correlation_id("test-correlation-id")
        
        # Create realistic test data
        self.test_data = self._create_test_data()
        
        # Default clusterer
        self.clusterer = NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.HYBRID_RESERVOIR,
            n_clusters=4,
            random_state=TEST_CONFIG['seed']
        )
        
    def _create_test_data(self, n_samples=100):
        """Create test data with known cluster structure"""
        data = []
        
        # Create 4 distinct personality archetypes
        archetypes = [
            [70, 20, 15, 20],  # Red-dominant
            [20, 70, 15, 20],  # Blue-dominant
            [15, 20, 70, 20],  # Green-dominant
            [20, 15, 20, 70]   # Yellow-dominant
        ]
        
        samples_per_type = n_samples // 4
        
        for i, archetype in enumerate(archetypes):
            for j in range(samples_per_type):
                # Add noise to archetype
                noise = np.random.randn(4) * 5
                energies = np.array(archetype) + noise
                energies = np.clip(energies, 0, 100)
                
                # Normalize to sum to 100
                energies = (energies / np.sum(energies)) * 100
                
                data.append({
                    'employee_id': f'EMP{i:02d}{j:03d}',
                    'red_energy': energies[0],
                    'blue_energy': energies[1],
                    'green_energy': energies[2],
                    'yellow_energy': energies[3],
                    'true_cluster': i
                })
        
        return pd.DataFrame(data)
        
    def test_clusterer_initialization(self):
        """Test NeuromorphicClusterer initialization"""
        # Test default initialization
        clusterer = NeuromorphicClusterer()
        assert clusterer.method == NeuromorphicClusteringMethod.HYBRID_RESERVOIR
        assert clusterer.n_clusters == 4
        assert clusterer.enable_fallback == True
        
        # Test custom parameters
        custom_clusterer = NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            n_clusters=6,
            enable_fallback=False
        )
        assert custom_clusterer.method == NeuromorphicClusteringMethod.ECHO_STATE_NETWORK
        assert custom_clusterer.n_clusters == 6
        assert custom_clusterer.enable_fallback == False
        
    def test_clusterer_fit_basic(self):
        """Test basic clustering fit functionality"""
        features = self.test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        self.clusterer.fit(features)
        
        # Verify training state
        assert self.clusterer.trained == True
        assert self.clusterer.cluster_labels is not None
        assert len(self.clusterer.cluster_labels) == len(features)
        
        # Verify cluster assignments
        unique_labels = np.unique(self.clusterer.cluster_labels)
        assert len(unique_labels) <= self.clusterer.n_clusters
        assert np.all(unique_labels >= 0)
        
    def test_all_neuromorphic_methods(self):
        """Test all neuromorphic clustering methods"""
        features = self.test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        methods_to_test = [
            NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
            NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
            NeuromorphicClusteringMethod.HYBRID_RESERVOIR
        ]
        
        for method in methods_to_test:
            clusterer = NeuromorphicClusterer(
                method=method,
                n_clusters=4,
                random_state=TEST_CONFIG['seed']
            )
            
            # Should fit without errors
            clusterer.fit(features)
            
            # Verify results
            assert clusterer.trained
            assert clusterer.cluster_labels is not None
            assert len(clusterer.cluster_labels) == len(features)
            
            # Test metrics calculation
            metrics = clusterer.get_clustering_metrics()
            assert isinstance(metrics, ClusteringMetrics)
            assert -1.0 <= metrics.silhouette_score <= 1.0
            
    def test_clustering_metrics_comprehensive(self):
        """Test comprehensive clustering metrics calculation"""
        features = self.test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        self.clusterer.fit(features)
        metrics = self.clusterer.get_clustering_metrics()
        
        # Verify all metric fields
        expected_fields = [
            'silhouette_score', 'calinski_harabasz_score', 'cluster_stability',
            'interpretability_score', 'temporal_coherence', 'computational_efficiency'
        ]
        
        for field in expected_fields:
            assert hasattr(metrics, field)
            value = getattr(metrics, field)
            assert value is not None
            assert not np.isnan(value)
            
        # Verify metric ranges
        assert -1.0 <= metrics.silhouette_score <= 1.0
        assert 0 <= metrics.cluster_stability <= 1.0
        assert 0 <= metrics.interpretability_score <= 1.0
        assert 0 <= metrics.temporal_coherence <= 1.0
        assert 0 <= metrics.computational_efficiency <= 1.0
        
    def test_cluster_interpretation(self):
        """Test cluster interpretation functionality"""
        features = self.test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        self.clusterer.fit(features)
        interpretations = self.clusterer.get_cluster_interpretation()
        
        # Should have interpretations for each cluster
        unique_clusters = np.unique(self.clusterer.cluster_labels)
        assert len(interpretations) == len(unique_clusters)
        
        # Verify interpretation structure
        for cluster_id, traits in interpretations.items():
            expected_traits = [
                'assertiveness', 'analytical', 'supportive', 'enthusiastic',
                'complexity', 'stability'
            ]
            assert all(trait in traits for trait in expected_traits)
            
            # All trait values should be in [0, 1]
            for trait, value in traits.items():
                assert 0 <= value <= 1
                
    def test_input_validation(self):
        """Test input validation and error handling"""
        # Test with None input
        with pytest.raises(Exception):
            self.clusterer.fit(None)
            
        # Test with empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(Exception):
            self.clusterer.fit(empty_df)
            
        # Test with missing columns
        incomplete_df = pd.DataFrame({
            'red_energy': [50, 60],
            'blue_energy': [30, 25]
            # Missing green_energy and yellow_energy
        })
        with pytest.raises(Exception):
            self.clusterer.fit(incomplete_df)
            
        # Test with insufficient samples
        tiny_df = pd.DataFrame({
            'red_energy': [50],
            'blue_energy': [30],
            'green_energy': [20], 
            'yellow_energy': [25]
        })
        # Should handle this case (might reduce clusters or use fallback)
        try:
            self.clusterer.fit(tiny_df)
        except Exception as e:
            assert "sample" in str(e).lower() or "cluster" in str(e).lower()
            
    def test_fallback_mechanism(self):
        """Test fallback to K-means when neuromorphic methods fail"""
        features = self.test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        # Create clusterer with fallback enabled
        clusterer_with_fallback = NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            n_clusters=4,
            enable_fallback=True,
            random_state=TEST_CONFIG['seed']
        )
        
        # Mock neuromorphic processing to fail
        with patch.object(clusterer_with_fallback, '_extract_neuromorphic_features_safe') as mock_extract:
            mock_extract.side_effect = Exception("Neuromorphic processing failed")
            
            # Should fallback to K-means
            clusterer_with_fallback.fit(features)
            
            # Verify fallback was used
            assert clusterer_with_fallback.trained
            assert clusterer_with_fallback.fallback_used
            assert clusterer_with_fallback.cluster_labels is not None
            
    def test_prediction_functionality(self):
        """Test prediction on new data"""
        # Split data for training and testing
        train_data = self.test_data.iloc[:80]
        test_data = self.test_data.iloc[80:]
        
        train_features = train_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        test_features = test_data[['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']]
        
        # Train clusterer
        self.clusterer.fit(train_features)
        
        # Test prediction (note: current implementation has limitations)
        # This tests the predict method even if it's not fully functional
        try:
            predictions = self.clusterer.predict(test_features)
            assert len(predictions) == len(test_features)
            assert all(0 <= pred <= self.clusterer.n_clusters for pred in predictions)
        except (ValueError, AttributeError) as e:
            # Current implementation might not fully support prediction
            assert "fitted" in str(e).lower() or "not implemented" in str(e).lower()
            

class TestCircuitBreakerAndResilience:
    """Test circuit breaker and resilience mechanisms"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=1,  # Short timeout for testing
            expected_exception=Exception
        )
        self.circuit_breaker = CircuitBreaker(self.circuit_config)
        
    def test_circuit_breaker_normal_operation(self):
        """Test circuit breaker in normal operation"""
        @self.circuit_breaker
        def successful_operation():
            return "success"
            
        # Should work normally
        result = successful_operation()
        assert result == "success"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        
    def test_circuit_breaker_failure_handling(self):
        """Test circuit breaker failure handling"""
        call_count = 0
        
        @self.circuit_breaker
        def failing_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("Operation failed")
            
        # Should fail and open circuit after threshold
        for i in range(self.circuit_config.failure_threshold):
            with pytest.raises(Exception):
                failing_operation()
                
        # Circuit should be open
        assert self.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Next call should fail immediately without calling function
        initial_count = call_count
        with pytest.raises(NeuromorphicException):
            failing_operation()
        assert call_count == initial_count  # Function not called
        
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism"""
        @self.circuit_breaker
        def intermittent_operation():
            if self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
                return "recovered"
            raise Exception("Still failing")
            
        # Force circuit to open
        self.circuit_breaker.state = CircuitBreakerState.OPEN
        self.circuit_breaker.last_failure_time = time.time() - 2  # Older than recovery timeout
        
        # Should attempt recovery
        result = intermittent_operation()
        assert result == "recovered"
        assert self.circuit_breaker.state == CircuitBreakerState.CLOSED
        
    def test_retry_manager(self):
        """Test retry manager functionality"""
        retry_manager = RetryManager(max_retries=3, base_delay=0.01)  # Fast delays for testing
        
        attempt_count = 0
        
        def failing_then_succeeding():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Not ready yet")
            return "success"
            
        result = retry_manager.retry_with_backoff(failing_then_succeeding)
        assert result == "success"
        assert attempt_count == 3
        
    def test_retry_manager_exhaustion(self):
        """Test retry manager when all retries are exhausted"""
        retry_manager = RetryManager(max_retries=2, base_delay=0.01)
        
        def always_failing():
            raise Exception("Always fails")
            
        with pytest.raises(NeuromorphicException):
            retry_manager.retry_with_backoff(always_failing)
            
    def test_timeout_operation(self):
        """Test timeout operation decorator"""
        @timeout_operation(timeout_seconds=0.1)
        def quick_operation():
            return "done"
            
        @timeout_operation(timeout_seconds=0.05)
        def slow_operation():
            time.sleep(0.1)
            return "should not reach here"
            
        # Quick operation should succeed
        result = quick_operation()
        assert result == "done"
        
        # Slow operation should timeout
        with pytest.raises(NeuromorphicException) as exc_info:
            slow_operation()
        assert exc_info.value.error_type == NeuromorphicErrorType.TIMEOUT_ERROR
        
    def test_resource_monitoring(self):
        """Test resource monitoring context manager"""
        with patch('psutil.Process') as mock_process:
            # Mock process metrics
            mock_instance = Mock()
            mock_instance.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
            mock_instance.cpu_percent.return_value = 25.0
            mock_process.return_value = mock_instance
            
            with resource_monitor():
                time.sleep(0.01)  # Small operation
                
            # Should complete without error and log metrics
            
    def test_correlation_id_management(self):
        """Test correlation ID management"""
        # Test initial correlation ID
        corr_id_1 = get_correlation_id()
        assert isinstance(corr_id_1, str)
        assert len(corr_id_1) > 0
        
        # Same thread should get same ID
        corr_id_2 = get_correlation_id()
        assert corr_id_1 == corr_id_2
        
        # Setting new ID should update
        new_id = "test-custom-id"
        set_correlation_id(new_id)
        assert get_correlation_id() == new_id


class TestNeuromorphicExceptions:
    """Test neuromorphic exception handling"""
    
    def test_neuromorphic_exception_creation(self):
        """Test NeuromorphicException creation and attributes"""
        context = {'input_size': 100, 'method': 'ESN'}
        
        exception = NeuromorphicException(
            "Test error message",
            error_type=NeuromorphicErrorType.DIMENSION_ERROR,
            correlation_id="test-correlation-123",
            context=context,
            recoverable=True
        )
        
        assert str(exception).startswith("dimension_error: Test error message")
        assert exception.error_type == NeuromorphicErrorType.DIMENSION_ERROR
        assert exception.correlation_id == "test-correlation-123"
        assert exception.context == context
        assert exception.recoverable == True
        assert exception.timestamp is not None
        
    def test_all_error_types(self):
        """Test all neuromorphic error types"""
        error_types = [
            NeuromorphicErrorType.MEMORY_ERROR,
            NeuromorphicErrorType.CONVERGENCE_ERROR,
            NeuromorphicErrorType.DIMENSION_ERROR,
            NeuromorphicErrorType.TIMEOUT_ERROR,
            NeuromorphicErrorType.STABILITY_ERROR,
            NeuromorphicErrorType.RESOURCE_ERROR
        ]
        
        for error_type in error_types:
            exception = NeuromorphicException(
                f"Test {error_type.value}",
                error_type=error_type,
                correlation_id="test",
                context={}
            )
            assert exception.error_type == error_type


class TestEdgeCasesAndErrorConditions:
    """Test edge cases and error conditions across all components"""
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints"""
        # Test with large reservoir size (might cause memory issues)
        try:
            large_esn = EchoStateNetwork(reservoir_size=10000)
            # If it succeeds, verify it's properly initialized
            assert large_esn.reservoir_size == 10000
            assert large_esn.W_res.shape == (10000, 10000)
        except MemoryError:
            # Expected for very large networks
            pass
            
    def test_invalid_parameters(self):
        """Test handling of invalid parameters"""
        # Test invalid spectral radius
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            esn = EchoStateNetwork(spectral_radius=-0.5)  # Negative spectral radius
            # Should still initialize but behavior might be unexpected
            assert esn.spectral_radius == -0.5
            
        # Test invalid sparsity
        esn = EchoStateNetwork(sparsity=1.5)  # > 1 sparsity
        # Should handle gracefully (clamp or interpret differently)
        assert esn.sparsity == 1.5
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme inputs"""
        esn = EchoStateNetwork(reservoir_size=50)
        
        # Test with very large inputs
        large_input = np.array([1e6, -1e6, 1e3, -1e3])
        state = esn.update_state(large_input)
        assert np.all(np.isfinite(state))
        assert not np.any(np.isnan(state))
        
        # Test with very small inputs
        small_input = np.array([1e-10, -1e-10, 1e-15, 1e-15])
        state = esn.update_state(small_input)
        assert np.all(np.isfinite(state))
        
    def test_concurrent_access(self):
        """Test thread safety (basic test)"""
        clusterer = NeuromorphicClusterer(
            method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            random_state=42
        )
        
        # Create test data
        test_data = pd.DataFrame({
            'red_energy': np.random.rand(50) * 100,
            'blue_energy': np.random.rand(50) * 100,
            'green_energy': np.random.rand(50) * 100,
            'yellow_energy': np.random.rand(50) * 100
        })
        
        results = []
        exceptions = []
        
        def clustering_task():
            try:
                local_clusterer = NeuromorphicClusterer(
                    method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                    random_state=42
                )
                local_clusterer.fit(test_data)
                results.append(local_clusterer.get_cluster_assignments())
            except Exception as e:
                exceptions.append(e)
                
        # Run multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=clustering_task)
            threads.append(thread)
            thread.start()
            
        for thread in threads:
            thread.join()
            
        # Should not have exceptions and should produce results
        assert len(exceptions) == 0
        assert len(results) > 0


class TestPerformanceAndBenchmarking:
    """Test performance characteristics and benchmarking"""
    
    def setup_method(self):
        """Setup performance test fixtures"""
        np.random.seed(TEST_CONFIG['seed'])
        
    def test_clustering_performance_scaling(self):
        """Test clustering performance with different data sizes"""
        sizes = [50, 100, 200]
        times = []
        
        for size in sizes:
            # Create test data
            test_data = pd.DataFrame({
                'red_energy': np.random.rand(size) * 100,
                'blue_energy': np.random.rand(size) * 100,
                'green_energy': np.random.rand(size) * 100,
                'yellow_energy': np.random.rand(size) * 100
            })
            
            clusterer = NeuromorphicClusterer(
                method=NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
                random_state=TEST_CONFIG['seed']
            )
            
            start_time = time.time()
            clusterer.fit(test_data)
            fit_time = time.time() - start_time
            times.append(fit_time)
            
            # Verify results
            assert clusterer.trained
            assert len(clusterer.get_cluster_assignments()) == size
            
        # Performance should scale reasonably (not exponentially)
        # This is a basic check - exact scaling depends on implementation
        assert all(t > 0 for t in times)
        
    def test_method_performance_comparison(self):
        """Compare performance of different neuromorphic methods"""
        test_data = pd.DataFrame({
            'red_energy': np.random.rand(100) * 100,
            'blue_energy': np.random.rand(100) * 100,
            'green_energy': np.random.rand(100) * 100,
            'yellow_energy': np.random.rand(100) * 100
        })
        
        methods = [
            NeuromorphicClusteringMethod.ECHO_STATE_NETWORK,
            NeuromorphicClusteringMethod.SPIKING_NEURAL_NETWORK,
            NeuromorphicClusteringMethod.LIQUID_STATE_MACHINE,
            NeuromorphicClusteringMethod.HYBRID_RESERVOIR
        ]
        
        performance_results = {}
        
        for method in methods:
            clusterer = NeuromorphicClusterer(
                method=method,
                random_state=TEST_CONFIG['seed']
            )
            
            start_time = time.time()
            clusterer.fit(test_data)
            fit_time = time.time() - start_time
            
            # Calculate metrics
            metrics = clusterer.get_clustering_metrics()
            
            performance_results[method.value] = {
                'fit_time': fit_time,
                'silhouette_score': metrics.silhouette_score,
                'cluster_stability': metrics.cluster_stability
            }
            
        # All methods should complete successfully
        assert len(performance_results) == len(methods)
        for method_name, results in performance_results.items():
            assert results['fit_time'] > 0
            assert -1.0 <= results['silhouette_score'] <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])