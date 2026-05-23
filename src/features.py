"""
Feature Engineering Module

Transforms raw F1 data into ML-ready features.

Functions:
    - engineer_driver_features(): Driver stats and form
    - engineer_constructor_features(): Team strength
    - engineer_track_features(): Track-specific features
    - build_feature_matrix(): Combine all features for model
"""

import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def engineer_driver_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate driver-based features from historical results.
    
    Features:
        - avg_points_per_race
        - avg_finish_position
        - dnf_rate (Did Not Finish)
        - consistency (std dev of finishes)
        - form (last 5 races average)
        - races_completed
    
    Args:
        results_df: Results dataframe with driver race history
    
    Returns:
        pd.DataFrame: Driver features indexed by driver_id
    """
    logger.info("Engineering driver features...")
    
    driver_stats = []
    
    for driver_id in results_df['driverId'].unique():
        driver_races = results_df[results_df['driverId'] == driver_id].sort_values('year')
        
        if len(driver_races) < 3:  # Skip drivers with very few races
            continue
        
        # Basic stats
        avg_points = driver_races['points'].mean()
        avg_finish = driver_races['positionOrder'].mean()
        dnf_rate = (driver_races['race_status'] == 'Retired').sum() / len(driver_races) if 'race_status' in driver_races.columns else 0
        consistency = driver_races['positionOrder'].std()
        races_completed = len(driver_races)
        
        # Form: last 5 races average
        last_5 = driver_races.tail(5)
        form_avg = last_5['positionOrder'].mean() if len(last_5) > 0 else avg_finish
        
        driver_stats.append({
            'driverId': driver_id,
            'avg_points_per_race': avg_points,
            'avg_finish_position': avg_finish,
            'dnf_rate': dnf_rate,
            'consistency': consistency,
            'form_last_5_races': form_avg,
            'races_completed': races_completed,
        })
    
    driver_features = pd.DataFrame(driver_stats).set_index('driverId')
    logger.info(f"Engineered features for {len(driver_features)} drivers")
    return driver_features


def engineer_constructor_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate constructor/team-based features.
    
    Features:
        - avg_constructor_position
        - constructor_points_per_race
        - team_consistency
    
    Args:
        results_df: Results dataframe
    
    Returns:
        pd.DataFrame: Constructor features indexed by constructor_id
    """
    logger.info("Engineering constructor features...")
    
    constructor_stats = []
    
    for constructor_id in results_df['constructorId'].unique():
        constructor_races = results_df[results_df['constructorId'] == constructor_id]
        
        if len(constructor_races) < 3:
            continue
        
        avg_pos = constructor_races['positionOrder'].mean()
        avg_points = constructor_races['points'].mean()
        consistency = constructor_races['positionOrder'].std()
        
        constructor_stats.append({
            'constructorId': constructor_id,
            'avg_constructor_position': avg_pos,
            'constructor_points_per_race': avg_points,
            'team_consistency': consistency,
        })
    
    constructor_features = pd.DataFrame(constructor_stats).set_index('constructorId')
    logger.info(f"Engineered features for {len(constructor_features)} constructors")
    return constructor_features


def engineer_track_features(results_df: pd.DataFrame, races_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate track-specific features.
    
    Features:
        - track_avg_finishing_position (for each driver/track combo)
        - track_familiarity (times raced here)
        - track_performance_delta (vs driver average)
    
    Args:
        results_df: Results dataframe
        races_df: Races dataframe with circuit info
    
    Returns:
        pd.DataFrame: Track features indexed by (driverId, circuitId)
    """
    logger.info("Engineering track features...")
    
    # Merge with circuit info if not already present
    if 'circuitId' not in results_df.columns:
        merged = results_df.merge(
            races_df[['raceId', 'circuitId']],
            on='raceId',
            how='left'
        )
    else:
        merged = results_df.copy()
    
    track_stats = []
    
    for (driver_id, circuit_id), group in merged.groupby(['driverId', 'circuitId']):
        if len(group) < 1:
            continue
        
        avg_finish = group['positionOrder'].mean()
        familiarity = len(group)  # Number of times at this track
        driver_overall_avg = merged[merged['driverId'] == driver_id]['positionOrder'].mean()
        perf_delta = driver_overall_avg - avg_finish  # Positive = better at this track
        
        track_stats.append({
            'driverId': driver_id,
            'circuitId': circuit_id,
            'track_avg_finish': avg_finish,
            'track_familiarity': familiarity,
            'track_performance_delta': perf_delta,
        })
    
    track_features = pd.DataFrame(track_stats)
    logger.info(f"Engineered track features for {len(track_features)} driver-circuit combinations")
    return track_features


def engineer_weather_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weather-dependent features.
    
    Features:
        - wet_weather_avg_finish
        - dry_weather_avg_finish
        - weather_preference (wet vs dry performance delta)
    
    Args:
        results_df: Results dataframe with weather info
    
    Returns:
        pd.DataFrame: Weather features
    """
    logger.info("Engineering weather features...")
    
    # For MVP: weather data may not be available in most datasets
    # This is a placeholder for when weather data is integrated
    
    weather_stats = []
    
    # Group by driver
    for driver_id in results_df['driverId'].unique():
        driver_races = results_df[results_df['driverId'] == driver_id]
        
        # For now, assume no weather column - default values
        wet_avg = driver_races['positionOrder'].mean()  # Placeholder
        dry_avg = driver_races['positionOrder'].mean()  # Placeholder
        preference = 0.0  # Neutral
        
        weather_stats.append({
            'driverId': driver_id,
            'wet_weather_avg_finish': wet_avg,
            'dry_weather_avg_finish': dry_avg,
            'weather_preference': preference,
        })
    
    weather_features = pd.DataFrame(weather_stats).set_index('driverId')
    logger.info(f"Weather features: placeholder values (real weather data not in dataset)")
    return weather_features


def build_feature_matrix(
    driver_features: pd.DataFrame,
    constructor_features: pd.DataFrame,
    track_features: pd.DataFrame,
    race_info: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Combine all features into single feature matrix for model.
    
    Args:
        driver_features: Driver-level features
        constructor_features: Constructor-level features
        track_features: Track-specific features
        race_info: Race and result information with target
    
    Returns:
        Tuple[X, metadata, y]: Feature matrix, metadata, target variable
            - X: Feature matrix (races x features)
            - metadata: Race/driver identifiers
            - y: Target (finishing position)
    """
    logger.info("Building feature matrix...")
    
    # Start with race info
    X = race_info.copy()
    
    # Merge driver features
    X = X.merge(
        driver_features,
        left_on='driverId',
        right_index=True,
        how='left'
    )
    
    # Merge constructor features
    X = X.merge(
        constructor_features,
        left_on='constructorId',
        right_index=True,
        how='left'
    )
    
    # Merge track features
    X = X.merge(
        track_features,
        on=['driverId', 'circuitId'],
        how='left'
    )
    
    # Fill missing track features with overall driver average
    for col in ['track_avg_finish', 'track_performance_delta']:
        if col in X.columns:
            X[col] = X.groupby('driverId')[col].transform(
                lambda x: x.fillna(x.mean())
            )
    
    # Extract target and metadata
    y = X['positionOrder'].astype(float)
    metadata = X[['raceId', 'driverId', 'constructorId', 'year', 'round']].copy()
    
    # Select feature columns (exclude identifiers and target)
    feature_cols = [col for col in X.columns if col not in [
        'raceId', 'driverId', 'constructorId', 'year', 'round',
        'positionOrder', 'race_name', 'date', 'circuitId',
        'surname', 'forename', 'nationality_driver',
        'name_constructor', 'nationality_constructor',
        'location_circuit', 'country', 'grid', 'points',
        'statusId', 'race_status'
    ]]
    
    X_features = X[feature_cols].copy()
    
    # Handle NaN values - fill with column mean
    for col in X_features.columns:
        X_features[col] = X_features[col].fillna(X_features[col].mean())
    
    # Remove any remaining NaN
    X_features = X_features.dropna()
    metadata = metadata.loc[X_features.index]
    y = y.loc[X_features.index]
    
    logger.info(f"Feature matrix built: {X_features.shape[0]} samples, {X_features.shape[1]} features")
    logger.info(f"Features: {list(X_features.columns)}")
    
    return X_features, metadata, y


def get_feature_names() -> list:
    """
    Get list of all feature names used in the model.
    
    Returns:
        list: Feature column names
    """
    return [
        "driver_avg_points",
        "driver_avg_finish_pos",
        "driver_dnf_rate",
        "driver_consistency",
        "driver_form_5_races",
        "constructor_avg_pos",
        "constructor_strength",
        "track_familiarity",
        "track_performance_delta",
        "weather_preference",
        # TODO: Add more features as implemented
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Print feature names
    features = get_feature_names()
    print(f"Total features: {len(features)}")
    for i, feat in enumerate(features, 1):
        print(f"  {i}. {feat}")
