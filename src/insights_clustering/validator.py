"""
Data Validation for Insights Discovery Data
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates Insights Discovery data quality and completeness"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """Comprehensive data quality validation"""
        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'quality_score': 0.0,
            'metrics': {}
        }
        
        # Check for duplicate employee IDs
        if data['employee_id'].duplicated().any():
            duplicates = data['employee_id'].duplicated().sum()
            results['errors'].append(f"Found {duplicates} duplicate employee IDs")
            results['is_valid'] = False
        
        # Check energy value ranges
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        for col in energy_cols:
            if col in data.columns:
                out_of_range = (~data[col].between(0, 100)).sum()
                if out_of_range > 0:
                    results['warnings'].append(f"{col}: {out_of_range} values outside 0-100 range")
        
        # Check for missing values
        missing_data = data.isnull().sum()
        if missing_data.any():
            for col, missing_count in missing_data.items():
                if missing_count > 0:
                    pct_missing = (missing_count / len(data)) * 100
                    if pct_missing > 10:
                        results['errors'].append(f"{col}: {pct_missing:.1f}% missing values")
                        results['is_valid'] = False
                    else:
                        results['warnings'].append(f"{col}: {pct_missing:.1f}% missing values")
        
        # Check energy sum consistency
        if all(col in data.columns for col in energy_cols):
            energy_sums = data[energy_cols].sum(axis=1)
            sum_variance = energy_sums.std()
            if sum_variance > 10:  # High variance in energy sums
                results['warnings'].append(f"High variance in energy sums (std: {sum_variance:.2f})")
        
        # Calculate quality score
        quality_score = 100.0
        quality_score -= len(results['errors']) * 20  # Major penalty for errors
        quality_score -= len(results['warnings']) * 5  # Minor penalty for warnings
        quality_score = max(0, quality_score)
        results['quality_score'] = quality_score
        
        # Add metrics
        results['metrics'] = {
            'total_records': len(data),
            'complete_records': len(data.dropna()),
            'duplicate_records': data.duplicated().sum(),
            'data_completeness': (1 - data.isnull().sum().sum() / data.size) * 100
        }
        
        self.validation_results = results
        return results
    
    def get_data_profile(self, data: pd.DataFrame) -> Dict[str, any]:
        """Generate comprehensive data profile"""
        profile = {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric column analysis
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            profile['numeric_summary'][col] = {
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'null_count': data[col].isnull().sum(),
                'unique_count': data[col].nunique()
            }
        
        # Categorical column analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            profile['categorical_summary'][col] = {
                'unique_count': data[col].nunique(),
                'top_values': data[col].value_counts().head(5).to_dict(),
                'null_count': data[col].isnull().sum()
            }
        
        return profile
    
    def suggest_data_improvements(self, data: pd.DataFrame) -> List[str]:
        """Suggest improvements for data quality"""
        suggestions = []
        
        validation = self.validate_data_quality(data)
        
        if validation['quality_score'] < 80:
            suggestions.append("Data quality score is below 80%. Consider data cleaning.")
        
        if validation['metrics']['data_completeness'] < 95:
            suggestions.append("Data completeness is below 95%. Fill or remove incomplete records.")
        
        if validation['metrics']['duplicate_records'] > 0:
            suggestions.append("Remove duplicate records to improve data integrity.")
        
        # Check for outliers in energy values
        energy_cols = ['red_energy', 'blue_energy', 'green_energy', 'yellow_energy']
        for col in energy_cols:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                if outliers > len(data) * 0.05:  # More than 5% outliers
                    suggestions.append(f"Consider reviewing {outliers} outliers in {col}")
        
        return suggestions