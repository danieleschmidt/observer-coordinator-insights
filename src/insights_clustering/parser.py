"""
Insights Discovery CSV Data Parser
Handles parsing and validation of Insights Discovery wheel data
"""

import csv
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class InsightsDataParser:
    """Parses and validates Insights Discovery CSV data"""
    
    REQUIRED_COLUMNS = [
        'employee_id',
        'red_energy',
        'blue_energy', 
        'green_energy',
        'yellow_energy'
    ]
    
    def __init__(self, validate_data: bool = True):
        self.validate_data = validate_data
        self.data: Optional[pd.DataFrame] = None
        
    def parse_csv(self, file_path: Path) -> pd.DataFrame:
        """Parse CSV file with Insights Discovery data"""
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} records from {file_path}")
            
            if self.validate_data:
                self._validate_structure()
                self._clean_data()
                
            return self.data
            
        except Exception as e:
            logger.error(f"Failed to parse CSV file {file_path}: {e}")
            raise
    
    def _validate_structure(self):
        """Validate CSV structure and required columns"""
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for valid energy values (0-100)
        for col in self.REQUIRED_COLUMNS[1:]:  # Skip employee_id
            if not self.data[col].between(0, 100).all():
                logger.warning(f"Column {col} contains values outside 0-100 range")
    
    def _clean_data(self):
        """Clean and normalize data"""
        # Remove rows with missing employee_id
        self.data = self.data.dropna(subset=['employee_id'])
        
        # Fill missing energy values with median
        energy_cols = self.REQUIRED_COLUMNS[1:]
        for col in energy_cols:
            if self.data[col].isnull().any():
                median_val = self.data[col].median()
                null_count = self.data[col].isnull().sum()
                self.data.loc[:, col] = self.data[col].fillna(median_val)
                logger.info(f"Filled {null_count} missing values in {col}")
        
        # Normalize energy values to sum to 100 for each employee
        energy_sum = self.data[energy_cols].sum(axis=1)
        for col in energy_cols:
            self.data[col] = (self.data[col] / energy_sum * 100).round(2)
    
    def get_clustering_features(self) -> pd.DataFrame:
        """Get features ready for clustering (energy values only)"""
        if self.data is None:
            raise ValueError("No data loaded. Call parse_csv first.")
        
        return self.data[self.REQUIRED_COLUMNS[1:]].copy()
    
    def get_employee_metadata(self) -> pd.DataFrame:
        """Get employee metadata (non-clustering features)"""
        if self.data is None:
            raise ValueError("No data loaded. Call parse_csv first.")
        
        meta_cols = [col for col in self.data.columns if col not in self.REQUIRED_COLUMNS[1:]]
        return self.data[meta_cols].copy()