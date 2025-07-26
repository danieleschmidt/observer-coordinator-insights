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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from insights_clustering import InsightsDataParser, KMeansClusterer, DataValidator
from team_simulator import TeamCompositionSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(exist_ok=True)
    
    try:
        # Step 1: Parse and validate data
        logger.info(f"Loading data from {args.input_file}")
        data_parser = InsightsDataParser()
        data = data_parser.parse_csv(args.input_file)
        
        logger.info(f"Loaded {len(data)} employee records")
        
        # Step 2: Data validation
        validator = DataValidator()
        validation_results = validator.validate_data_quality(data)
        
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
        
        # Step 3: Extract features for clustering
        features = data_parser.get_clustering_features()
        metadata = data_parser.get_employee_metadata()
        
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
        
        # Step 5: Perform clustering
        logger.info(f"Performing K-means clustering with {args.clusters} clusters")
        clusterer = KMeansClusterer(n_clusters=args.clusters)
        clusterer.fit(features)
        
        cluster_assignments = clusterer.get_cluster_assignments()
        centroids = clusterer.get_cluster_centroids()
        quality_metrics = clusterer.get_cluster_quality_metrics()
        
        logger.info(f"Clustering complete. Silhouette score: {quality_metrics.get('silhouette_score', 'N/A'):.3f}")
        
        # Save clustering results
        clustering_results = {
            'centroids': centroids.to_dict(),
            'quality_metrics': quality_metrics,
            'cluster_summary': clusterer.get_cluster_summary().to_dict()
        }
        
        with open(args.output / 'clustering_results.json', 'w') as f:
            json.dump(clustering_results, f, indent=2, default=str)
        
        # Step 6: Team composition simulation
        logger.info(f"Generating {args.teams} balanced team compositions")
        simulator = TeamCompositionSimulator()
        
        # Combine metadata with cluster assignments
        employee_data = metadata.copy()
        employee_data['cluster'] = cluster_assignments
        
        simulator.load_employee_data(employee_data, cluster_assignments)
        
        # Generate multiple team compositions
        team_compositions = simulator.recommend_optimal_teams(args.teams, iterations=5)
        
        if team_compositions:
            best_composition = team_compositions[0]
            logger.info(f"Best composition average balance score: {best_composition['average_balance_score']:.2f}")
            
            # Get recommendations summary
            recommendations = simulator.get_team_recommendations_summary(team_compositions)
            
            # Save team composition results
            team_results = {
                'best_composition': best_composition,
                'all_compositions': team_compositions,
                'recommendations': recommendations
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
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())