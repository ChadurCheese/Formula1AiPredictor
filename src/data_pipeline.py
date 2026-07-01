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
    logger.info("Preparing data: merging and cleaning...")
    
    # Extract dataframes
    races = raw_data['races']
    results = raw_data['results']
    drivers = raw_data['drivers']
    constructors = raw_data['constructors']
    circuits = raw_data.get('circuits')
    status = raw_data.get('status')
    qualifying = raw_data.get('qualifying')
    
    # Step 1: Merge results with race info
    merged = results.merge(
        races[['raceId', 'year', 'round', 'circuitId', 'date', 'name']],
        on='raceId',
        how='left'
    ).rename(columns={'name': 'race_name'})
    
    # Step 2: Add driver info
    merged = merged.merge(
        drivers[['driverId', 'surname', 'forename', 'nationality']],
        on='driverId',
        how='left'
    )
    
    # Step 3: Add constructor info
    merged = merged.merge(
        constructors[['constructorId', 'name', 'nationality']],
        on='constructorId',
        how='left',
        suffixes=('_driver', '_constructor')
    )
    
    # Step 4: Add circuit info if available
    if circuits is not None:
        merged = merged.merge(
            circuits[['circuitId', 'name', 'location', 'country']],
            on='circuitId',
            how='left',
            suffixes=('_team', '_circuit')
        )
    
    # Step 5: Add status info if available
    if status is not None:
        merged = merged.merge(
            status,
            on='statusId',
            how='left'
        ).rename(columns={'status': 'race_status'})
    
    # Step 6: Data cleaning
    # A small number of historical races (shared-drive era) have duplicate
    # raceId+driverId rows (e.g. disqualified then re-classified). Keep the
    # first entry so per-driver expanding features aren't double-counted.
    dup_count = merged.duplicated(subset=['raceId', 'driverId']).sum()
    if dup_count:
        logger.info(f"Dropping {dup_count} duplicate raceId+driverId rows")
        merged = merged.drop_duplicates(subset=['raceId', 'driverId'], keep='first')

    # Remove rows with missing critical values
    merged = merged.dropna(subset=['positionOrder', 'year', 'driverId', 'constructorId'])
    
    # Handle grid position nulls (practice starts) - drop them
    merged = merged.dropna(subset=['grid'])
    
    # Convert position to integer
    merged['positionOrder'] = merged['positionOrder'].astype(int)
    merged['grid'] = merged['grid'].astype(int)
    
    # Sort by year, round for temporal consistency
    merged = merged.sort_values(['year', 'round', 'positionOrder']).reset_index(drop=True)
    
    logger.info(f"Data prepared: {merged.shape[0]} rows, {merged.shape[1]} columns")
    logger.info(f"Years covered: {merged['year'].min()} to {merged['year'].max()}")
    logger.info(f"Races: {merged['raceId'].nunique()}, Drivers: {merged['driverId'].nunique()}")
    
    return merged


def cache_data(data: pd.DataFrame, cache_path: Path = Path("data/cache/features.pkl")) -> None:
    """
    Cache processed data to pickle file for quick loading.
    
    Args:
        data: Processed dataframe
        cache_path: Path to save pickle file
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    data.to_pickle(cache_path)
    size_mb = cache_path.stat().st_size / (1024 * 1024)
    logger.info(f"Cached data to {cache_path} ({size_mb:.1f} MB)")


def load_cached_data(cache_path: Path = Path("data/cache/features.pkl")) -> pd.DataFrame:
    """
    Load previously cached data from pickle file.
    
    Args:
        cache_path: Path to pickle file
    
    Returns:
        pd.DataFrame: Cached dataframe
    """
    if not cache_path.exists():
        logger.warning(f"Cache not found: {cache_path}")
        return None
    
    data = pd.read_pickle(cache_path)
    logger.info(f"Loaded cached data from {cache_path} ({data.shape[0]} rows)")
    return data


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
