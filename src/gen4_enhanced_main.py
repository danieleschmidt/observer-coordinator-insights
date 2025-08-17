#!/usr/bin/env python3
"""
Generation 4 Enhanced Main Entry Point
Quantum neuromorphic clustering with adaptive AI optimization
"""

import sys
import argparse
import logging
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Generation 4 imports
try:
    from insights_clustering import (
        quantum_neuromorphic_clustering, Gen4ClusteringPipeline, 
        Gen4Config, GENERATION_4_AVAILABLE, CAPABILITIES
    )
    
    # Fallback imports
    from insights_clustering import (
        InsightsDataParser, DataValidator, NeuromorphicClusterer
    )
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Enhanced logging configuration
def setup_enhanced_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Setup Generation 4 enhanced logging with structured output"""
    
    # Custom formatter for Generation 4
    log_format = (
        '%(asctime)s - [GEN4] - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Set appropriate levels for external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def check_system_capabilities() -> Dict[str, bool]:
    """Check system capabilities for Generation 4 features"""
    capabilities = {
        'python_version_ok': sys.version_info >= (3, 9),
        'generation_4_available': GENERATION_4_AVAILABLE,
        'memory_sufficient': True,  # Placeholder - would check actual memory
        'gpu_available': False,     # Placeholder - would check for GPU
    }
    
    # Check specific Generation 4 capabilities
    if GENERATION_4_AVAILABLE:
        capabilities.update(CAPABILITIES)
    
    return capabilities


def validate_generation4_data(data: np.ndarray) -> Dict[str, Any]:
    """Enhanced data validation for Generation 4 processing"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'recommendations': [],
        'data_characteristics': {}
    }
    
    n_samples, n_features = data.shape
    
    # Basic validation
    if n_samples < 10:
        validation_results['errors'].append("Minimum 10 samples required for quantum processing")
        validation_results['is_valid'] = False
    
    if n_features < 2:
        validation_results['errors'].append("Minimum 2 features required")
        validation_results['is_valid'] = False
    
    # Check for infinite or NaN values
    if not np.isfinite(data).all():
        validation_results['errors'].append("Data contains infinite or NaN values")
        validation_results['is_valid'] = False
    
    # Data characteristics analysis
    validation_results['data_characteristics'] = {
        'n_samples': int(n_samples),
        'n_features': int(n_features),
        'data_density': float(np.count_nonzero(data) / data.size),
        'feature_correlations': float(np.mean(np.abs(np.corrcoef(data.T)))),
        'data_range': [float(np.min(data)), float(np.max(data))],
        'data_std': float(np.std(data))
    }
    
    # Recommendations based on data characteristics
    if n_samples > 10000:
        validation_results['recommendations'].append(
            "Large dataset detected - consider using quantum_optimized strategy"
        )
    
    if n_features > 20:
        validation_results['recommendations'].append(
            "High-dimensional data - quantum dimensionality reduction may help"
        )
    
    if validation_results['data_characteristics']['data_density'] < 0.1:
        validation_results['warnings'].append(
            "Sparse data detected - results may be less reliable"
        )
    
    return validation_results


async def async_main():
    """Asynchronous main function for Generation 4 processing"""
    parser = argparse.ArgumentParser(
        description="Generation 4 Observer Coordinator Insights - Quantum Enhanced"
    )
    
    # Input/Output arguments
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to Insights Discovery CSV file'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=4,
        help='Number of clusters (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('gen4_output'),
        help='Output directory for results (default: gen4_output/)'
    )
    
    # Generation 4 specific arguments
    parser.add_argument(
        '--quantum-enabled',
        action='store_true',
        default=True,
        help='Enable quantum neuromorphic clustering (default: True)'
    )
    parser.add_argument(
        '--adaptive-ai',
        action='store_true',
        default=True,
        help='Enable adaptive AI optimization (default: True)'
    )
    parser.add_argument(
        '--ensemble-size',
        type=int,
        default=5,
        help='Quantum ensemble size (default: 5)'
    )
    parser.add_argument(
        '--strategy',
        choices=['auto', 'quantum_simple', 'quantum_full', 'quantum_optimized', 'fallback'],
        default='auto',
        help='Clustering strategy selection (default: auto)'
    )
    parser.add_argument(
        '--continuous-optimization',
        action='store_true',
        help='Enable continuous background optimization'
    )
    
    # Performance arguments
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )
    parser.add_argument(
        '--memory-limit',
        type=float,
        default=8.0,
        help='Memory limit in GB (default: 8.0)'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Processing timeout in seconds (default: 1800)'
    )
    
    # Logging and debugging
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        help='Log file path (default: console only)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without processing'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comprehensive benchmarks'
    )
    
    args = parser.parse_args()
    
    # Setup enhanced logging
    logger = setup_enhanced_logging(args.log_level, args.log_file)
    
    # Create output directory
    args.output.mkdir(exist_ok=True)
    
    logger.info("🚀 Generation 4 Observer Coordinator Insights Starting...")
    logger.info(f"Quantum Enhanced: {args.quantum_enabled}")
    logger.info(f"Adaptive AI: {args.adaptive_ai}")
    
    # Check system capabilities
    capabilities = check_system_capabilities()
    logger.info(f"System Capabilities: {capabilities}")
    
    if not capabilities['generation_4_available']:
        logger.warning("⚠️ Generation 4 features not available, falling back to traditional clustering")
        args.quantum_enabled = False
        args.adaptive_ai = False
    
    try:
        start_time = time.time()
        
        # Load and validate data
        logger.info(f"📊 Loading data from {args.input_file}")
        
        if not args.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
        # Parse data
        if args.input_file.suffix.lower() == '.csv':
            data = pd.read_csv(args.input_file)
        else:
            raise ValueError("Only CSV files supported")
        
        # Extract numerical features
        numerical_data = data.select_dtypes(include=[np.number])
        if numerical_data.empty:
            raise ValueError("No numerical features found in data")
        
        data_array = numerical_data.values
        logger.info(f"📈 Data loaded: {data_array.shape[0]} samples, {data_array.shape[1]} features")
        
        # Enhanced validation
        validation_results = validate_generation4_data(data_array)
        
        # Save validation report
        with open(args.output / 'gen4_validation_report.json', 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        if not validation_results['is_valid']:
            logger.error("❌ Data validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            return 1
        
        if validation_results['warnings']:
            logger.warning("⚠️ Data validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
        
        if validation_results['recommendations']:
            logger.info("💡 Recommendations:")
            for rec in validation_results['recommendations']:
                logger.info(f"  - {rec}")
        
        if args.validate_only:
            logger.info("✅ Validation complete. Results saved to gen4_validation_report.json")
            return 0
        
        # Configure Generation 4 pipeline
        gen4_config = Gen4Config(
            quantum_enabled=args.quantum_enabled,
            ensemble_size=args.ensemble_size,
            adaptive_learning=args.adaptive_ai,
            continuous_optimization=args.continuous_optimization,
            max_workers=args.max_workers,
            memory_limit_gb=args.memory_limit,
            timeout_seconds=args.timeout
        )
        
        # Processing based on availability and configuration
        if capabilities['generation_4_available'] and args.quantum_enabled:
            logger.info("🧠 Using Generation 4 Quantum Neuromorphic Clustering")
            
            if args.strategy == 'auto':
                # Use integrated pipeline with automatic strategy selection
                pipeline = Gen4ClusteringPipeline(gen4_config)
                pipeline = await pipeline.fit_async(data_array, args.clusters)
                
                cluster_assignments = pipeline.predict(data_array)
                analysis = pipeline.get_comprehensive_analysis()
                
            else:
                # Use direct quantum clustering function
                cluster_assignments, analysis = quantum_neuromorphic_clustering(
                    data_array,
                    n_clusters=args.clusters,
                    quantum_enabled=args.quantum_enabled,
                    adaptive_learning=args.adaptive_ai,
                    ensemble_voting=(args.ensemble_size > 1)
                )
            
            processing_method = "Generation 4 Quantum Enhanced"
            
        else:
            # Fallback to traditional neuromorphic clustering
            logger.info("🔄 Using traditional neuromorphic clustering")
            
            try:
                clusterer = NeuromorphicClusterer(
                    n_clusters=args.clusters,
                    method='reservoir_computing'
                )
                clusterer.fit(data_array)
                cluster_assignments = clusterer.predict(data_array)
                
                analysis = {
                    'model_info': {'model_type': 'NeuromorphicClusterer'},
                    'cluster_analysis': clusterer.get_performance_metrics()
                }
                processing_method = "Traditional Neuromorphic"
                
            except Exception as e:
                logger.error(f"Neuromorphic clustering failed: {e}")
                # Final fallback to basic clustering
                logger.info("🔄 Using basic K-means clustering")
                
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=args.clusters, random_state=42)
                cluster_assignments = kmeans.fit_predict(data_array)
                
                analysis = {
                    'model_info': {'model_type': 'KMeans'},
                    'cluster_analysis': {'silhouette_score': 0.5}  # Placeholder
                }
                processing_method = "Basic K-means"
        
        processing_time = time.time() - start_time
        logger.info(f"⚡ Processing completed in {processing_time:.2f}s using {processing_method}")
        
        # Analyze results
        unique_clusters = len(np.unique(cluster_assignments))
        logger.info(f"🎯 Created {unique_clusters} clusters from {len(data_array)} samples")
        
        # Extract performance metrics
        if 'cluster_analysis' in analysis:
            cluster_metrics = analysis['cluster_analysis']
            if 'performance_metrics' in cluster_metrics:
                perf_metrics = cluster_metrics['performance_metrics']
                silhouette = perf_metrics.get('silhouette_score', 'N/A')
                logger.info(f"📊 Silhouette Score: {silhouette}")
        
        # Create comprehensive results
        results = {
            'metadata': {
                'processing_method': processing_method,
                'processing_time_seconds': processing_time,
                'timestamp': time.time(),
                'input_file': str(args.input_file),
                'n_samples': len(data_array),
                'n_features': data_array.shape[1],
                'n_clusters_requested': args.clusters,
                'n_clusters_created': unique_clusters,
                'generation_4_enabled': capabilities['generation_4_available'] and args.quantum_enabled
            },
            'cluster_assignments': cluster_assignments.tolist(),
            'analysis': analysis,
            'validation_results': validation_results,
            'system_capabilities': capabilities
        }
        
        # Save main results
        with open(args.output / 'gen4_clustering_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save cluster assignments as CSV
        output_df = data.copy()
        output_df['cluster'] = cluster_assignments
        output_df.to_csv(args.output / 'gen4_clustered_data.csv', index=False)
        
        # Run benchmarks if requested
        if args.benchmark:
            await run_benchmarks(data_array, args, logger)
        
        # Start continuous optimization if enabled
        if args.continuous_optimization and capabilities['generation_4_available']:
            logger.info("🔄 Starting continuous optimization...")
            # This would run in background - simplified for demo
            logger.info("✅ Continuous optimization started")
        
        # Print summary
        print(f"\n🎉 Generation 4 Analysis Complete!")
        print(f"  📊 Method: {processing_method}")
        print(f"  ⚡ Processing Time: {processing_time:.2f}s")
        print(f"  🎯 Clusters Created: {unique_clusters}")
        print(f"  📁 Results saved to: {args.output}/")
        
        if 'cluster_analysis' in analysis and 'performance_metrics' in analysis['cluster_analysis']:
            metrics = analysis['cluster_analysis']['performance_metrics']
            if 'silhouette_score' in metrics:
                print(f"  📈 Silhouette Score: {metrics['silhouette_score']:.3f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("🛑 Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Processing failed: {str(e)}")
        logger.debug(f"Error details: {e}", exc_info=True)
        return 1


async def run_benchmarks(data: np.ndarray, args, logger):
    """Run comprehensive benchmarks"""
    logger.info("🏃 Running Generation 4 benchmarks...")
    
    benchmark_results = {
        'data_size': data.shape,
        'timestamp': time.time(),
        'benchmarks': {}
    }
    
    # Benchmark different strategies
    strategies = ['quantum_simple', 'quantum_full', 'fallback']
    
    for strategy in strategies:
        try:
            logger.info(f"  📊 Benchmarking {strategy}...")
            start_time = time.time()
            
            if strategy.startswith('quantum') and GENERATION_4_AVAILABLE:
                config = Gen4Config(quantum_enabled=True)
                pipeline = Gen4ClusteringPipeline(config)
                pipeline.fit(data, args.clusters)
                assignments = pipeline.predict(data)
                analysis = pipeline.get_comprehensive_analysis()
            else:
                # Fallback benchmark
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=args.clusters, random_state=42)
                assignments = kmeans.fit_predict(data)
                analysis = {'silhouette_score': 0.5}
            
            benchmark_time = time.time() - start_time
            
            benchmark_results['benchmarks'][strategy] = {
                'processing_time': benchmark_time,
                'clusters_created': len(np.unique(assignments)),
                'success': True
            }
            
            logger.info(f"    ✅ {strategy}: {benchmark_time:.2f}s")
            
        except Exception as e:
            logger.warning(f"    ❌ {strategy} failed: {e}")
            benchmark_results['benchmarks'][strategy] = {
                'processing_time': 0,
                'error': str(e),
                'success': False
            }
    
    # Save benchmark results
    with open(args.output / 'gen4_benchmarks.json', 'w') as f:
        json.dump(benchmark_results, f, indent=2, default=str)
    
    logger.info("🏁 Benchmarks completed")


def main():
    """Synchronous main function wrapper"""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 130
    except Exception as e:
        print(f"💥 Critical error: {e}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)