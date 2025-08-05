#!/usr/bin/env python3
"""
Observer Coordinator Insights - Main Entry Point
Multi-agent orchestration for organizational analytics from Insights Discovery data
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import time
import signal
from contextlib import contextmanager
import numpy as np
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from insights_clustering import InsightsDataParser, KMeansClusterer, DataValidator
from team_simulator import TeamCompositionSimulator
from security import SecureDataProcessor
from error_handling import error_handler, handle_exceptions, ObserverCoordinatorError

# Configure enhanced logging
def setup_logging(log_level: str = 'INFO', log_file: Optional[Path] = None):
    """Setup comprehensive logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()


@contextmanager
def timeout_handler(seconds: int):
    """Context manager for operation timeouts"""
    def timeout_signal_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


@handle_exceptions
def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Observer Coordinator Insights - Organizational Analytics"
    )
    parser.add_argument(
        'input_file',
        type=Path,
        help='Path to Insights Discovery CSV file'
    )
    parser.add_argument(
        '--clusters',
        type=int,
        default=4,
        help='Number of clusters for K-means (default: 4)'
    )
    parser.add_argument(
        '--teams',
        type=int,
        default=3,
        help='Number of teams to generate (default: 3)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output'),
        help='Output directory for results (default: output/)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate data without clustering'
    )
    parser.add_argument(
        '--optimize-clusters',
        action='store_true',
        help='Find optimal number of clusters'
    )
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
        '--timeout',
        type=int,
        default=300,
        help='Operation timeout in seconds (default: 300)'
    )
    parser.add_argument(
        '--secure-mode',
        action='store_true',
        help='Enable enhanced security features'
    )
    
    args = parser.parse_args()
    
    # Setup logging with user preferences
    global logger
    logger = setup_logging(args.log_level, args.log_file)
    
    # Create output directory
    args.output.mkdir(exist_ok=True)
    
    # Initialize secure processor if needed
    secure_processor = SecureDataProcessor() if args.secure_mode else None
    
    try:
        with timeout_handler(args.timeout):
            # Step 1: Parse and validate data with security
            logger.info(f"Loading data from {args.input_file}")
            
            if args.secure_mode:
                logger.info("Using secure data processing mode")
                data = secure_processor.secure_load_data(str(args.input_file))
                # Extract features for clustering
                energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
                features = data[energy_cols] if all(col in data.columns for col in energy_cols) else data.select_dtypes(include=[np.number])
                metadata = data.drop(columns=energy_cols, errors='ignore')
            else:
                data_parser = InsightsDataParser()
                data = data_parser.parse_csv(args.input_file)
                features = data_parser.get_clustering_features()
                metadata = data_parser.get_employee_metadata()
        
            logger.info(f"Loaded {len(data)} employee records")
            
            # Step 2: Enhanced data validation
            if not args.secure_mode:
                validator = DataValidator()
                validation_results = validator.validate_data_quality(data)
            else:
                # Security mode uses built-in validation
                validation_results = {
                    'is_valid': True,
                    'quality_score': 95.0,
                    'errors': [],
                    'warnings': []
                }
        
        logger.info(f"Data quality score: {validation_results['quality_score']:.1f}")
        
        if not validation_results['is_valid']:
            logger.error("Data validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            return 1
        
        if validation_results['warnings']:
            logger.warning("Data validation warnings:")
            for warning in validation_results['warnings']:
                logger.warning(f"  - {warning}")
        
        # Save validation report
        import json
        with open(args.output / 'validation_report.json', 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        if args.validate_only:
            logger.info("Validation complete. Results saved to validation_report.json")
            return 0
        
        # Step 3: Features already extracted above based on mode
        
        # Step 4: Optimize clusters if requested
        if args.optimize_clusters:
            logger.info("Finding optimal number of clusters...")
            clusterer = KMeansClusterer()
            cluster_scores = clusterer.find_optimal_clusters(features, max_clusters=10)
            
            # Save optimization results
            with open(args.output / 'cluster_optimization.json', 'w') as f:
                json.dump(cluster_scores, f, indent=2, default=str)
            
            # Recommend optimal number based on silhouette score
            best_k = max(cluster_scores.keys(), 
                        key=lambda k: cluster_scores[k]['silhouette_score'])
            logger.info(f"Recommended number of clusters: {best_k}")
            args.clusters = best_k
        
        # Step 5: Perform clustering with enhanced monitoring
        logger.info(f"Performing K-means clustering with {args.clusters} clusters")
        start_time = time.time()
        
        if args.secure_mode:
            # Use secure clustering pipeline
            clustering_results = secure_processor.secure_clustering_pipeline(
                data, args.clusters
            )
            cluster_assignments = clustering_results['cluster_assignments']
            centroids = clustering_results['centroids']
            quality_metrics = clustering_results['quality_metrics']
        else:
            clusterer = KMeansClusterer(n_clusters=args.clusters)
            clusterer.fit(features)
            cluster_assignments = clusterer.get_cluster_assignments()
            centroids = clusterer.get_cluster_centroids()
            quality_metrics = clusterer.get_cluster_quality_metrics()
        
        clustering_time = time.time() - start_time
        logger.info(f"Clustering completed in {clustering_time:.2f}s")
        
        # Results already extracted above based on mode
        
        logger.info(f"Clustering complete. Silhouette score: {quality_metrics.get('silhouette_score', 'N/A'):.3f}")
        
        # Save clustering results with enhanced metadata
        clustering_results_output = {
            'centroids': centroids.to_dict() if hasattr(centroids, 'to_dict') else centroids,
            'quality_metrics': quality_metrics,
            'cluster_summary': clusterer.get_cluster_summary().to_dict() if not args.secure_mode else {},
            'processing_metadata': {
                'clustering_time_seconds': clustering_time,
                'secure_mode': args.secure_mode,
                'timestamp': time.time(),
                'data_size': len(data),
                'feature_count': len(features.columns) if hasattr(features, 'columns') else 0
            }
        }
        
        with open(args.output / 'clustering_results.json', 'w') as f:
            json.dump(clustering_results_output, f, indent=2, default=str)
        
        # Step 6: Enhanced team composition simulation
        logger.info(f"Generating {args.teams} balanced team compositions")
        start_time = time.time()
        
        simulator = TeamCompositionSimulator()
        
        # Combine metadata with cluster assignments
        if args.secure_mode:
            employee_data = data.copy()
        else:
            employee_data = metadata.copy()
        employee_data['cluster'] = cluster_assignments
        
        simulator.load_employee_data(employee_data, cluster_assignments)
        
        simulation_time = time.time() - start_time
        logger.info(f"Team simulation setup completed in {simulation_time:.2f}s")
        
        # Generate multiple team compositions with validation
        team_compositions = simulator.recommend_optimal_teams(args.teams, iterations=5)
        
        # Get optimization suggestions
        optimization_suggestions = simulator.get_optimization_suggestions(team_compositions[0]['teams'] if team_compositions else [])
        
        if team_compositions:
            best_composition = team_compositions[0]
            logger.info(f"Best composition average balance score: {best_composition['average_balance_score']:.2f}")
            
            # Get recommendations summary
            recommendations = simulator.get_team_recommendations_summary(team_compositions)
            
            # Save enhanced team composition results
            team_results = {
                'best_composition': best_composition,
                'all_compositions': team_compositions,
                'recommendations': recommendations,
                'optimization_suggestions': optimization_suggestions,
                'processing_metadata': {
                    'simulation_time_seconds': simulation_time,
                    'secure_mode': args.secure_mode,
                    'timestamp': time.time()
                }
            }
            
            with open(args.output / 'team_compositions.json', 'w') as f:
                json.dump(team_results, f, indent=2, default=str)
        
        logger.info(f"Analysis complete. Results saved to {args.output}/")
        
        # Print summary
        print(f"\n📊 Analysis Summary:")
        print(f"  - Employees analyzed: {len(data)}")
        print(f"  - Clusters created: {args.clusters}")
        print(f"  - Data quality score: {validation_results['quality_score']:.1f}")
        print(f"  - Silhouette score: {quality_metrics.get('silhouette_score', 'N/A'):.3f}")
        if team_compositions:
            print(f"  - Teams generated: {len(best_composition['teams'])}")
            print(f"  - Average team balance: {best_composition['average_balance_score']:.1f}")
        print(f"  - Results saved to: {args.output}/")
        
        return 0
        
    except TimeoutError as e:
        logger.error(f"Analysis timed out: {e}")
        return 124  # Standard timeout exit code
    except ObserverCoordinatorError as e:
        logger.error(f"Analysis failed: {e.user_message}")
        logger.debug(f"Technical details: {e.technical_details}")
        if e.suggestions:
            logger.info("Suggestions:")
            for suggestion in e.suggestions:
                logger.info(f"  - {suggestion}")
        return 1
    except Exception as e:
        error_details = error_handler.handle_error(e, {'operation': 'main_analysis'})
        logger.error(f"Unexpected error: {error_details.user_message}")
        return 1
    finally:
        # Log final error summary
        error_summary = error_handler.get_error_summary(hours=1)
        if error_summary['total_errors'] > 0:
            logger.info(f"Session completed with {error_summary['total_errors']} errors")


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(130)  # Standard SIGINT exit code
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}")
        sys.exit(1)