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
    # TODO: Implement driver feature engineering
    logger.info("Engineer driver features - to be implemented")
    pass


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
    # TODO: Implement constructor feature engineering
    logger.info("Engineer constructor features - to be implemented")
    pass


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
        pd.DataFrame: Track features
    """
    # TODO: Implement track feature engineering
    logger.info("Engineer track features - to be implemented")
    pass


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
    # TODO: Implement weather feature engineering
    logger.info("Engineer weather features - to be implemented")
    pass


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
        race_info: Race and result information
    
    Returns:
        Tuple[X, metadata, y]: Feature matrix, metadata, target variable
            - X: Feature matrix (races x features)
            - metadata: Race/driver identifiers
            - y: Target (finishing position)
    """
    # TODO: Implement feature matrix building
    # Merge features for each race
    # Handle missing values
    # Normalize/scale features
    logger.info("Build feature matrix - to be implemented")
    pass


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
