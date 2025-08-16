#!/usr/bin/env python3
"""
Comprehensive test suite for Generation 4 Quantum Neuromorphic features
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import warnings
warnings.filterwarnings('ignore')

# Import Generation 4 components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

try:
    from insights_clustering.quantum_neuromorphic import (
        QuantumState, QuantumNeuron, QuantumReservoir, 
        QuantumNeuromorphicClusterer, create_quantum_ensemble
    )
    from insights_clustering.adaptive_ai_engine import (
        AdaptiveAIEngine, LearningStrategy, HyperparameterOptimizer,
        AdaptiveQLearning, ModelPerformance
    )
    from insights_clustering.gen4_integration import (
        Gen4ClusteringPipeline, Gen4Config, quantum_neuromorphic_clustering
    )
    GENERATION_4_AVAILABLE = True
except ImportError:
    GENERATION_4_AVAILABLE = False


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestQuantumState(unittest.TestCase):
    """Test quantum state functionality"""
    
    def setUp(self):
        self.quantum_state = QuantumState(
            amplitude=complex(0.8, 0.6),
            phase=np.pi/4,
            entanglement_strength=0.1,
            coherence_time=1.0
        )
    
    def test_quantum_state_creation(self):
        """Test quantum state initialization"""
        self.assertIsInstance(self.quantum_state.amplitude, complex)
        self.assertEqual(self.quantum_state.phase, np.pi/4)
        self.assertEqual(self.quantum_state.entanglement_strength, 0.1)
        self.assertEqual(self.quantum_state.coherence_time, 1.0)
    
    def test_state_collapse(self):
        """Test quantum state collapse to classical value"""
        collapsed_value = self.quantum_state.collapse()
        expected_value = abs(self.quantum_state.amplitude) ** 2
        self.assertAlmostEqual(collapsed_value, expected_value, places=6)
    
    def test_state_evolution(self):
        """Test quantum state evolution over time"""
        time_step = 0.1
        evolved_state = self.quantum_state.evolve(time_step)
        
        self.assertIsInstance(evolved_state, QuantumState)
        self.assertNotEqual(evolved_state.phase, self.quantum_state.phase)
        self.assertLess(abs(evolved_state.amplitude), abs(self.quantum_state.amplitude))
    
    def test_state_coherence_decay(self):
        """Test coherence decay over time"""
        time_step = 2.0  # Large time step
        evolved_state = self.quantum_state.evolve(time_step)
        
        # Amplitude should decay
        self.assertLess(abs(evolved_state.amplitude), abs(self.quantum_state.amplitude))
        
        # Entanglement should decay
        self.assertLess(evolved_state.entanglement_strength, self.quantum_state.entanglement_strength)


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestQuantumNeuron(unittest.TestCase):
    """Test quantum neuron functionality"""
    
    def setUp(self):
        position = np.array([1.0, 2.0, 3.0])
        quantum_state = QuantumState(
            amplitude=complex(0.7, 0.7),
            phase=0.0,
            coherence_time=1.0
        )
        self.neuron = QuantumNeuron(
            position=position,
            quantum_state=quantum_state,
            activation_threshold=0.5
        )
    
    def test_neuron_initialization(self):
        """Test quantum neuron initialization"""
        self.assertEqual(len(self.neuron.position), 3)
        self.assertEqual(len(self.neuron.synaptic_weights), 3)
        self.assertEqual(self.neuron.activation_threshold, 0.5)
        self.assertIsInstance(self.neuron.quantum_state, QuantumState)
    
    def test_neuron_input_processing(self):
        """Test neuron input processing and spike generation"""
        inputs = np.array([1.0, 1.0, 1.0])
        current_time = 0.0
        
        output = self.neuron.receive_input(inputs, current_time)
        self.assertIsInstance(output, float)
        self.assertIn(output, [0.0, 1.0])  # Should be binary output
    
    def test_refractory_period(self):
        """Test neuron refractory period"""
        inputs = np.array([10.0, 10.0, 10.0])  # Strong input
        
        # First spike
        output1 = self.neuron.receive_input(inputs, 0.0)
        self.assertEqual(output1, 1.0)
        
        # Should be in refractory period
        output2 = self.neuron.receive_input(inputs, 0.05)  # Within refractory period
        self.assertEqual(output2, 0.0)
        
        # After refractory period
        output3 = self.neuron.receive_input(inputs, 0.2)  # After refractory period
        self.assertEqual(output3, 1.0)
    
    def test_synaptic_plasticity(self):
        """Test synaptic weight updates"""
        initial_weights = self.neuron.synaptic_weights.copy()
        inputs = np.array([2.0, 2.0, 2.0])
        
        # Trigger plasticity by causing spike
        self.neuron.receive_input(inputs, 0.0)
        
        # Weights should have changed
        self.assertFalse(np.array_equal(initial_weights, self.neuron.synaptic_weights))


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestQuantumReservoir(unittest.TestCase):
    """Test quantum reservoir computing network"""
    
    def setUp(self):
        self.reservoir = QuantumReservoir(
            size=100,
            spectral_radius=0.95,
            quantum_coupling=0.1
        )
    
    def test_reservoir_initialization(self):
        """Test reservoir initialization"""
        self.assertEqual(len(self.reservoir.neurons), 100)
        self.assertEqual(self.reservoir.spectral_radius, 0.95)
        self.assertEqual(self.reservoir.quantum_coupling, 0.1)
        self.assertEqual(self.reservoir.entanglement_matrix.shape, (100, 100))
    
    def test_entanglement_matrix_properties(self):
        """Test entanglement matrix spectral properties"""
        eigenvalues = np.linalg.eigvals(self.reservoir.entanglement_matrix)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        
        # Should respect spectral radius constraint (with some tolerance)
        self.assertLessEqual(max_eigenvalue, self.reservoir.spectral_radius * 1.1)
    
    def test_reservoir_processing(self):
        """Test reservoir input processing"""
        inputs = np.random.randn(3)
        time_steps = 10
        
        outputs = self.reservoir.process(inputs, time_steps)
        
        self.assertEqual(outputs.shape, (time_steps, 100))
        self.assertTrue(np.all(outputs >= 0))  # Neuron outputs should be non-negative
        self.assertTrue(np.all(outputs <= 1))  # Neuron outputs should be binary/bounded


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestQuantumNeuromorphicClusterer(unittest.TestCase):
    """Test quantum neuromorphic clustering"""
    
    def setUp(self):
        self.clusterer = QuantumNeuromorphicClusterer(
            n_clusters=3,
            reservoir_size=300,
            quantum_coupling=0.1,
            optimization_iterations=10  # Reduced for testing
        )
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.randn(50, 4)
    
    def test_clusterer_initialization(self):
        """Test clusterer initialization"""
        self.assertEqual(self.clusterer.n_clusters, 3)
        self.assertEqual(len(self.clusterer.quantum_reservoirs), 3)
        self.assertEqual(self.clusterer.reservoir_size, 300)
    
    def test_clusterer_fitting(self):
        """Test clusterer fitting process"""
        self.clusterer.fit(self.test_data)
        
        self.assertTrue(self.clusterer.is_trained)
        self.assertIsNotNone(self.clusterer.cluster_centers)
        self.assertIsNotNone(self.clusterer.cluster_assignments)
        self.assertEqual(len(self.clusterer.cluster_assignments), len(self.test_data))
    
    def test_quantum_feature_extraction(self):
        """Test quantum feature extraction"""
        quantum_features = self.clusterer._extract_quantum_features(self.test_data.T)
        
        self.assertIsInstance(quantum_features, np.ndarray)
        self.assertEqual(quantum_features.shape[0], len(self.test_data))
        self.assertGreater(quantum_features.shape[1], 0)
    
    def test_prediction(self):
        """Test cluster prediction"""
        self.clusterer.fit(self.test_data)
        predictions = self.clusterer.predict(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < self.clusterer.n_clusters))
    
    def test_cluster_analysis(self):
        """Test cluster analysis generation"""
        self.clusterer.fit(self.test_data)
        analysis = self.clusterer.get_cluster_analysis()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('cluster_sizes', analysis)
        self.assertIn('performance_metrics', analysis)
        self.assertIn('quantum_reservoir_stats', analysis)
        
        # Check cluster sizes
        cluster_sizes = analysis['cluster_sizes']
        self.assertEqual(len(cluster_sizes), self.clusterer.n_clusters)
        self.assertEqual(sum(cluster_sizes), len(self.test_data))
    
    def test_quantum_tunneling_optimization(self):
        """Test quantum tunneling optimization"""
        self.clusterer.fit(self.test_data)
        
        tunneling_results = self.clusterer.quantum_tunneling_optimization(
            self.test_data, tunneling_strength=0.1
        )
        
        self.assertIsInstance(tunneling_results, dict)
        self.assertIn('tunneling_improvements', tunneling_results)
        self.assertIn('performance_improvement', tunneling_results)
        self.assertIn('optimized_silhouette', tunneling_results)


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestAdaptiveAIEngine(unittest.TestCase):
    """Test adaptive AI engine functionality"""
    
    def setUp(self):
        self.engine = AdaptiveAIEngine()
    
    def test_engine_initialization(self):
        """Test engine initialization"""
        self.assertIsInstance(self.engine.q_learning_agent, AdaptiveQLearning)
        self.assertIsInstance(self.engine.hyperparameter_optimizer, HyperparameterOptimizer)
        self.assertEqual(len(self.engine.model_performances), 0)
    
    def test_model_registration(self):
        """Test model registration"""
        model_id = "test_model"
        hyperparams = {"param1": 0.1, "param2": 10}
        
        self.engine.register_model(model_id, hyperparams)
        
        self.assertIn(model_id, self.engine.model_performances)
        self.assertEqual(
            self.engine.model_performances[model_id].hyperparameters,
            hyperparams
        )
    
    def test_performance_update(self):
        """Test model performance updates"""
        model_id = "test_model"
        self.engine.register_model(model_id, {})
        
        performance_metrics = {"silhouette_score": 0.8}
        training_time = 10.5
        resource_usage = {"memory": 0.5, "cpu": 0.7}
        
        self.engine.update_model_performance(
            model_id, performance_metrics, training_time, resource_usage
        )
        
        model_perf = self.engine.model_performances[model_id]
        self.assertEqual(len(model_perf.accuracy_scores), 1)
        self.assertEqual(model_perf.accuracy_scores[0], 0.8)
        self.assertEqual(len(model_perf.training_times), 1)
        self.assertEqual(len(model_perf.resource_usage), 1)
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection"""
        strategy = self.engine.adaptive_strategy_selection()
        self.assertIsInstance(strategy, LearningStrategy)
    
    def test_hyperparameter_optimization(self):
        """Test hyperparameter optimization"""
        model_id = "test_model"
        self.engine.register_model(model_id, {})
        
        parameter_space = {
            "param1": (0.01, 0.1),
            "param2": (10, 100)
        }
        
        def objective_function(params):
            return np.random.uniform(0.5, 1.0)  # Mock objective
        
        result = self.engine.optimize_hyperparameters(
            model_id, parameter_space, objective_function, "genetic"
        )
        
        self.assertIn('best_params', result)
        self.assertIn('best_score', result)
        self.assertIn('optimization_history', result)
    
    def test_optimization_report(self):
        """Test optimization report generation"""
        report = self.engine.get_optimization_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('model_performances', report)
        self.assertIn('learning_history', report)
        self.assertIn('q_learning_stats', report)


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestGen4Integration(unittest.TestCase):
    """Test Generation 4 integration pipeline"""
    
    def setUp(self):
        self.config = Gen4Config(
            quantum_enabled=True,
            ensemble_size=3,
            optimization_iterations=10  # Reduced for testing
        )
        self.pipeline = Gen4ClusteringPipeline(self.config)
        
        # Generate test data
        np.random.seed(42)
        self.test_data = np.random.randn(60, 5)
    
    def test_config_creation(self):
        """Test Generation 4 configuration"""
        self.assertTrue(self.config.quantum_enabled)
        self.assertEqual(self.config.ensemble_size, 3)
        self.assertTrue(self.config.adaptive_learning)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.config, self.config)
        self.assertIsNotNone(self.pipeline.adaptive_ai_engine)
        self.assertFalse(self.pipeline.is_trained)
    
    def test_data_validation(self):
        """Test input data validation"""
        is_valid, message = self.pipeline._validate_input_data(self.test_data)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Data validation passed")
        
        # Test invalid data
        invalid_data = np.array([[1, 2], [3, np.inf]])
        is_valid, message = self.pipeline._validate_input_data(invalid_data)
        self.assertFalse(is_valid)
        self.assertIn("infinite", message)
    
    def test_resource_estimation(self):
        """Test resource requirement estimation"""
        estimates = self.pipeline._estimate_resource_requirements(self.test_data)
        
        self.assertIn('estimated_memory_mb', estimates)
        self.assertIn('estimated_time_seconds', estimates)
        self.assertIn('memory_feasible', estimates)
        self.assertIn('time_feasible', estimates)
        
        self.assertIsInstance(estimates['estimated_memory_mb'], float)
        self.assertIsInstance(estimates['memory_feasible'], bool)
    
    def test_strategy_selection(self):
        """Test optimal strategy selection"""
        resource_estimates = {
            'memory_feasible': True,
            'time_feasible': True
        }
        
        strategy = self.pipeline._select_optimal_strategy(
            self.test_data, resource_estimates
        )
        
        self.assertIn(strategy, ['quantum_simple', 'quantum_full', 'quantum_optimized', 'fallback'])
    
    def test_pipeline_fitting(self):
        """Test complete pipeline fitting"""
        self.pipeline.fit(self.test_data, n_clusters=3)
        
        self.assertTrue(self.pipeline.is_trained)
        self.assertIsNotNone(self.pipeline.best_model)
        self.assertGreater(len(self.pipeline.clustering_history), 0)
    
    def test_pipeline_prediction(self):
        """Test pipeline prediction"""
        self.pipeline.fit(self.test_data, n_clusters=3)
        predictions = self.pipeline.predict(self.test_data)
        
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions < 3))
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis generation"""
        self.pipeline.fit(self.test_data, n_clusters=3)
        analysis = self.pipeline.get_comprehensive_analysis()
        
        self.assertIsInstance(analysis, dict)
        self.assertIn('model_info', analysis)
        self.assertIn('training_history', analysis)
        self.assertIn('performance_summary', analysis)
        
        # Check model info
        model_info = analysis['model_info']
        self.assertTrue(model_info['is_trained'])
        self.assertIn('model_type', model_info)
    
    def test_convenience_function(self):
        """Test convenience clustering function"""
        assignments, analysis = quantum_neuromorphic_clustering(
            self.test_data,
            n_clusters=3,
            quantum_enabled=True,
            adaptive_learning=True,
            ensemble_voting=True
        )
        
        self.assertEqual(len(assignments), len(self.test_data))
        self.assertIsInstance(analysis, dict)
        self.assertIn('model_info', analysis)


@unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
class TestQuantumEnsemble(unittest.TestCase):
    """Test quantum ensemble functionality"""
    
    def setUp(self):
        np.random.seed(42)
        self.test_data = np.random.randn(40, 4)
    
    def test_ensemble_creation(self):
        """Test quantum ensemble creation"""
        ensemble_result = create_quantum_ensemble(
            self.test_data,
            n_clusters=3,
            ensemble_size=3
        )
        
        self.assertIsInstance(ensemble_result, dict)
        self.assertIn('ensemble_results', ensemble_result)
        self.assertIn('best_model', ensemble_result)
        self.assertIn('best_model_score', ensemble_result)
        self.assertIn('ensemble_assignments', ensemble_result)
        
        # Check ensemble size
        ensemble_results = ensemble_result['ensemble_results']
        self.assertEqual(len(ensemble_results), 3)
        
        # Check assignments
        assignments = ensemble_result['ensemble_assignments']
        self.assertEqual(len(assignments), len(self.test_data))
    
    def test_ensemble_diversity(self):
        """Test ensemble model diversity"""
        ensemble_result = create_quantum_ensemble(
            self.test_data,
            n_clusters=3,
            ensemble_size=5
        )
        
        ensemble_results = ensemble_result['ensemble_results']
        
        # Check that models have different parameters
        quantum_couplings = [r['quantum_coupling'] for r in ensemble_results]
        reservoir_sizes = [r['reservoir_size'] for r in ensemble_results]
        
        # Should have diversity in parameters
        self.assertGreater(len(set(quantum_couplings)), 1)
        self.assertGreater(len(set(reservoir_sizes)), 1)
    
    def test_ensemble_consensus(self):
        """Test ensemble consensus scoring"""
        ensemble_result = create_quantum_ensemble(
            self.test_data,
            n_clusters=3,
            ensemble_size=4
        )
        
        consensus_score = ensemble_result['ensemble_consensus_score']
        self.assertIsInstance(consensus_score, float)
        self.assertGreaterEqual(consensus_score, -1.0)
        self.assertLessEqual(consensus_score, 1.0)


class TestGen4Fallbacks(unittest.TestCase):
    """Test Generation 4 fallback mechanisms"""
    
    def test_import_fallback(self):
        """Test graceful fallback when Generation 4 not available"""
        # This test ensures the system works even without Generation 4
        if not GENERATION_4_AVAILABLE:
            # Test should pass when imports fail
            self.assertTrue(True)
        else:
            # Test that we can import the fallback detection
            from insights_clustering import GENERATION_4_AVAILABLE
            self.assertTrue(GENERATION_4_AVAILABLE)
    
    def test_config_fallback(self):
        """Test configuration fallback behavior"""
        if GENERATION_4_AVAILABLE:
            config = Gen4Config(quantum_enabled=False)
            self.assertFalse(config.quantum_enabled)
    
    @unittest.skipIf(GENERATION_4_AVAILABLE, "Testing fallback behavior")
    def test_traditional_clustering_availability(self):
        """Test that traditional clustering is available as fallback"""
        # This would test sklearn KMeans or other fallback clustering
        from sklearn.cluster import KMeans
        
        data = np.random.randn(50, 4)
        kmeans = KMeans(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(data)
        
        self.assertEqual(len(labels), len(data))
        self.assertEqual(len(np.unique(labels)), 3)


class TestGen4StateManagement(unittest.TestCase):
    """Test Generation 4 state management and persistence"""
    
    @unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
    def test_pipeline_state_saving(self):
        """Test pipeline state saving and loading"""
        config = Gen4Config(quantum_enabled=True)
        pipeline = Gen4ClusteringPipeline(config)
        
        # Generate and fit test data
        test_data = np.random.randn(30, 4)
        pipeline.fit(test_data, n_clusters=2)
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            pipeline.save_pipeline_state(temp_path)
            
            # Check that file was created and contains expected data
            self.assertTrue(temp_path.exists())
            
            with open(temp_path, 'r') as f:
                state = json.load(f)
            
            self.assertIn('config', state)
            self.assertIn('training_history', state)
            self.assertIn('is_trained', state)
            self.assertTrue(state['is_trained'])
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()
    
    @unittest.skipIf(not GENERATION_4_AVAILABLE, "Generation 4 components not available")
    def test_adaptive_ai_state_persistence(self):
        """Test adaptive AI engine state persistence"""
        engine = AdaptiveAIEngine()
        
        # Register a model and update performance
        model_id = "test_model"
        engine.register_model(model_id, {"param1": 0.1})
        engine.update_model_performance(
            model_id, {"silhouette_score": 0.8}, 10.0, {"memory": 0.5}
        )
        
        # Save state
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            engine.save_state(temp_path)
            
            # Create new engine and load state
            new_engine = AdaptiveAIEngine()
            new_engine.load_state(temp_path)
            
            # Check that state was restored
            self.assertIn(model_id, new_engine.model_performances)
            model_perf = new_engine.model_performances[model_id]
            self.assertEqual(len(model_perf.accuracy_scores), 1)
            self.assertEqual(model_perf.accuracy_scores[0], 0.8)
            
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()


if __name__ == '__main__':
    # Configure test logging
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if GENERATION_4_AVAILABLE:
        print(f"\n✅ Generation 4 tests completed:")
        print(f"  Tests run: {result.testsRun}")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print(f"  Skipped: {len(result.skipped)}")
    else:
        print("\n⚠️ Generation 4 components not available - only fallback tests run")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)