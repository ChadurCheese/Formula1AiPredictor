"""
Data Pipeline Module

Handles loading, cleaning, and caching F1 data from Kaggle datasets.
Entry point for the ML pipeline.

Functions:
    - load_raw_data(): Load CSVs from data/raw/
    - prepare_data(): Clean and validate data
    - cache_data(): Save processed data to pickle
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Load all raw F1 CSVs from data/raw/ directory.
    
    Expected files:
        - races.csv
        - results.csv
        - drivers.csv
        - constructors.csv
        - qualifying.csv
        - pit_stops.csv
        - status.csv (optional)
    
    Returns:
        dict: Dictionary with dataframe for each CSV file
              Keys: 'races', 'results', 'drivers', 'constructors', etc.
    
    Raises:
        FileNotFoundError: If required CSV files are missing
    """
    raw_data_path = Path("data/raw")
    
    required_files = {
        "races": "races.csv",
        "results": "results.csv",
        "drivers": "drivers.csv",
        "constructors": "constructors.csv",
        "qualifying": "qualifying.csv",
    }
    
    data = {}
    
    for key, filename in required_files.items():
        filepath = raw_data_path / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Missing required file: {filepath}")
        
        logger.info(f"Loading {filename}...")
        data[key] = pd.read_csv(filepath)
    
    # Optional files
    optional_files = {
        "pit_stops": "pit_stops.csv",
        "status": "status.csv",
        "circuits": "circuits.csv",
    }
    
    for key, filename in optional_files.items():
        filepath = raw_data_path / filename
        if filepath.exists():
            logger.info(f"Loading {filename}...")
            data[key] = pd.read_csv(filepath)
    
    logger.info(f"Loaded {len(data)} data files")
    return data


def prepare_data(raw_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Clean, validate, and prepare raw data for feature engineering.
    
    Args:
        raw_data: Dictionary of raw dataframes from load_raw_data()
    
    Returns:
        pd.DataFrame: Cleaned and merged dataset ready for features
    """
    # TODO: Implement data cleaning and merging
    # - Merge races with results
    # - Handle missing values
    # - Remove incomplete races
    # - Validate data types
    
    logger.info("Data preparation placeholder - to be implemented")
    pass


def cache_data(data: pd.DataFrame, cache_path: Path = Path("data/cache/features.pkl")) -> None:
    """
    Cache processed data to pickle file for quick loading.
    
    Args:
        data: Processed dataframe
        cache_path: Path to save pickle file
    """
    # TODO: Implement caching
    logger.info(f"Caching data to {cache_path}")
    pass


def load_cached_data(cache_path: Path = Path("data/cache/features.pkl")) -> pd.DataFrame:
    """
    Load previously cached data from pickle file.
    
    Args:
        cache_path: Path to pickle file
    
    Returns:
        pd.DataFrame: Cached dataframe
    """
    # TODO: Implement cache loading
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Load raw data
    try:
        raw_data = load_raw_data()
        print(f"Successfully loaded {len(raw_data)} datasets")
        for key, df in raw_data.items():
            print(f"  {key}: {df.shape[0]} rows, {df.shape[1]} columns")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure CSV files are in data/raw/")
