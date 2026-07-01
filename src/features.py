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
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def _lap_time_to_seconds(value) -> float:
    """Convert a 'M:SS.mmm' lap time string to seconds, NaN if missing."""
    if not isinstance(value, str) or ':' not in value:
        return np.nan
    minutes, rest = value.split(':', 1)
    try:
        return int(minutes) * 60 + float(rest)
    except ValueError:
        return np.nan


def engineer_qualifying_features(qualifying_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate qualifying-derived features. Qualifying happens before the race,
    so these are safe to use as prediction inputs (no leakage).

    Features:
        - qual_position: Official qualifying position
        - qual_gap_seconds: Best lap time gap to the pole-sitter's best lap
        - teammate_qual_gap_seconds: Best lap time gap to same-race teammate
          (positive = slower than teammate). This isolates driver skill /
          car-side setup from overall car pace, since teammates share
          equipment - a large positive gap suggests the driver is
          underperforming their car that weekend.

    Args:
        qualifying_df: Qualifying session results (qualifyId, raceId, driverId,
            constructorId, q1/q2/q3)

    Returns:
        pd.DataFrame: One row per (raceId, driverId)
    """
    logger.info("Engineering qualifying features...")

    df = qualifying_df.copy()
    for col in ['q1', 'q2', 'q3']:
        df[col] = df[col].apply(_lap_time_to_seconds)
    df['best_lap'] = df[['q1', 'q2', 'q3']].min(axis=1)

    pole_time = df.groupby('raceId')['best_lap'].transform('min')
    df['qual_gap_seconds'] = df['best_lap'] - pole_time

    team_group = df.groupby(['raceId', 'constructorId'])['best_lap']
    team_sum = team_group.transform('sum')
    team_count = team_group.transform('count')
    teammate_avg = (team_sum - df['best_lap']) / (team_count - 1)
    df['teammate_qual_gap_seconds'] = (df['best_lap'] - teammate_avg).where(team_count > 1)

    qual_features = df[
        ['raceId', 'driverId', 'position', 'qual_gap_seconds', 'teammate_qual_gap_seconds']
    ].rename(columns={'position': 'qual_position'})
    logger.info(f"Engineered qualifying features for {len(qual_features)} race entries")
    return qual_features


def engineer_pitstop_features(pit_stops_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pit-crew-speed features from prior races only.

    A race's own pit stop count/duration is a live strategy decision made
    during that race, so it would leak the outcome if used directly. Instead
    this uses the constructor's *average* pit stop duration from prior races
    as a proxy for pit crew quality/speed - a real, pre-existing team
    characteristic that's fair to use as a prediction input.

    Features:
        - avg_pit_stop_duration: Constructor's average pit stop duration (seconds)
          across prior races only

    Args:
        pit_stops_df: Raw pit stop records (raceId, driverId, duration, ...)
        results_df: Results dataframe, used to attach constructorId and year/round

    Returns:
        pd.DataFrame: One row per (raceId, constructorId)
    """
    logger.info("Engineering pit stop features (prior-races-only)...")

    stops = pit_stops_df.merge(
        results_df[['raceId', 'driverId', 'constructorId', 'year', 'round']],
        on=['raceId', 'driverId'],
        how='inner',
    )
    stops['duration'] = pd.to_numeric(stops['duration'], errors='coerce')

    race_level = (
        stops.groupby(['constructorId', 'raceId', 'year', 'round'])
        .agg(avg_duration=('duration', 'mean'))
        .reset_index()
        .sort_values(['constructorId', 'year', 'round'])
    )
    grp = race_level.groupby('constructorId')
    race_level['avg_pit_stop_duration'] = grp['avg_duration'].transform(lambda s: s.shift().expanding().mean())

    pitstop_features = race_level[['raceId', 'constructorId', 'avg_pit_stop_duration']]
    logger.info(f"Engineered pit stop features for {len(pitstop_features)} race entries")
    return pitstop_features


def engineer_circuit_overtaking_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how easy a circuit is to overtake on, using only prior races
    held at that circuit (expanding window with shift(1)).

    Some circuits (e.g. tight street circuits) are notoriously hard to pass
    on, which caps how much a driver can realistically gain even from a
    great race; others are much easier. This is a track-level characteristic
    (not driver-specific), computed from how much the whole field's grid-to-finish
    positions typically move at that circuit.

    Features:
        - circuit_avg_position_change: Average |grid - finish| across the
          whole field in prior races at this circuit. Higher = more
          overtaking/shuffling typically happens there.

    Args:
        results_df: Results dataframe with circuitId, grid, positionOrder

    Returns:
        pd.DataFrame: One row per (raceId, circuitId)
    """
    logger.info("Engineering circuit overtaking features (prior-races-only)...")

    df = results_df.copy()
    df['position_change_abs'] = (df['grid'] - df['positionOrder']).abs()

    race_level = (
        df.groupby(['circuitId', 'raceId', 'year', 'round'])
        .agg(avg_position_change=('position_change_abs', 'mean'))
        .reset_index()
        .sort_values(['circuitId', 'year', 'round'])
    )
    grp = race_level.groupby('circuitId')
    race_level['circuit_avg_position_change'] = grp['avg_position_change'].transform(
        lambda s: s.shift().expanding().mean()
    )

    circuit_features = race_level[['raceId', 'circuitId', 'circuit_avg_position_change']]
    logger.info(f"Engineered circuit overtaking features for {len(circuit_features)} race entries")
    return circuit_features


def engineer_season_form_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate in-season championship momentum using only prior races of the
    same season (expanding window with shift(1)) - no leakage from future
    rounds or other seasons.

    Features:
        - season_points_so_far: Cumulative points earned this season before this race
        - season_position_so_far: Championship standing before this race (rank by points)

    Args:
        results_df: Results dataframe

    Returns:
        pd.DataFrame: One row per (raceId, driverId)
    """
    logger.info("Engineering season form features (prior-races-only)...")

    df = results_df.sort_values(['driverId', 'year', 'round']).copy()
    grp = df.groupby(['driverId', 'year'])
    df['season_points_so_far'] = grp['points'].transform(lambda s: s.shift().cumsum().fillna(0))

    df['season_position_so_far'] = (
        df.groupby(['year', 'round'])['season_points_so_far']
        .rank(ascending=False, method='min')
    )

    season_features = df[['raceId', 'driverId', 'season_points_so_far', 'season_position_so_far']]
    logger.info(f"Engineered season form features for {len(season_features)} race entries")
    return season_features


def engineer_driver_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate driver-based features from historical results.

    All stats are computed as an *expanding window over prior races only*
    (via shift(1) before expanding/rolling), so a race's features never
    include information from that race or any later race. This keeps the
    features valid to use at prediction time and avoids leaking a driver's
    future form into earlier predictions.

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
        pd.DataFrame: Driver features, one row per (raceId, driverId)
    """
    logger.info("Engineering driver features (prior-races-only)...")

    df = results_df.sort_values(['driverId', 'year', 'round']).copy()
    df['is_dnf'] = (df['race_status'] == 'Retired').astype(float) if 'race_status' in df.columns else 0.0

    grp = df.groupby('driverId')
    df['races_completed'] = grp.cumcount()
    df['avg_points_per_race'] = grp['points'].transform(lambda s: s.shift().expanding().mean())
    df['avg_finish_position'] = grp['positionOrder'].transform(lambda s: s.shift().expanding().mean())
    df['consistency'] = grp['positionOrder'].transform(lambda s: s.shift().expanding().std())
    df['dnf_rate'] = grp['is_dnf'].transform(lambda s: s.shift().expanding().mean())
    df['form_last_5_races'] = grp['positionOrder'].transform(lambda s: s.shift().rolling(5, min_periods=1).mean())

    driver_features = df[[
        'raceId', 'driverId', 'avg_points_per_race', 'avg_finish_position',
        'dnf_rate', 'consistency', 'form_last_5_races', 'races_completed',
    ]]
    logger.info(f"Engineered driver features for {len(driver_features)} race entries")
    return driver_features


def engineer_constructor_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate constructor/team-based features using only prior races
    (expanding window with shift(1)) to avoid leaking future team form.

    Features:
        - avg_constructor_position
        - constructor_points_per_race
        - team_consistency

    Args:
        results_df: Results dataframe

    Returns:
        pd.DataFrame: Constructor features, one row per (raceId, constructorId)
    """
    logger.info("Engineering constructor features (prior-races-only)...")

    # Each constructor fields two drivers per race, so first collapse to one
    # row per (constructorId, raceId) - averaging both cars' results - before
    # computing the expanding window. Otherwise the two teammate rows for the
    # same race would merge many-to-many with the driver-level feature matrix.
    race_level = (
        results_df.groupby(['constructorId', 'raceId', 'year', 'round'])
        .agg(positionOrder=('positionOrder', 'mean'), points=('points', 'sum'))
        .reset_index()
        .sort_values(['constructorId', 'year', 'round'])
    )
    grp = race_level.groupby('constructorId')

    race_level['avg_constructor_position'] = grp['positionOrder'].transform(lambda s: s.shift().expanding().mean())
    race_level['constructor_points_per_race'] = grp['points'].transform(lambda s: s.shift().expanding().mean())
    race_level['team_consistency'] = grp['positionOrder'].transform(lambda s: s.shift().expanding().std())

    constructor_features = race_level[[
        'raceId', 'constructorId', 'avg_constructor_position',
        'constructor_points_per_race', 'team_consistency',
    ]]
    logger.info(f"Engineered constructor features for {len(constructor_features)} race entries")
    return constructor_features


def engineer_track_features(results_df: pd.DataFrame, races_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate track-specific features using only prior visits to that
    circuit (expanding window with shift(1)).

    Features:
        - track_avg_finish (driver's avg finish at this circuit, prior visits only)
        - track_familiarity (times raced here before this race)
        - track_performance_delta (track avg vs driver's overall avg so far)

    Args:
        results_df: Results dataframe
        races_df: Races dataframe with circuit info

    Returns:
        pd.DataFrame: Track features, one row per (raceId, driverId, circuitId)
    """
    logger.info("Engineering track features (prior-visits-only)...")

    # Merge with circuit info if not already present
    if 'circuitId' not in results_df.columns:
        merged = results_df.merge(
            races_df[['raceId', 'circuitId']],
            on='raceId',
            how='left'
        )
    else:
        merged = results_df.copy()

    df = merged.sort_values(['driverId', 'circuitId', 'year', 'round']).copy()
    track_grp = df.groupby(['driverId', 'circuitId'])
    df['track_avg_finish'] = track_grp['positionOrder'].transform(lambda s: s.shift().expanding().mean())
    df['track_familiarity'] = track_grp.cumcount()

    # Driver's overall average finish so far (across all circuits), needed to
    # compute track specialization delta without using future races.
    df = df.sort_values(['driverId', 'year', 'round'])
    driver_grp = df.groupby('driverId')
    df['driver_overall_avg_sofar'] = driver_grp['positionOrder'].transform(lambda s: s.shift().expanding().mean())

    df['track_performance_delta'] = df['driver_overall_avg_sofar'] - df['track_avg_finish']

    track_features = df[[
        'raceId', 'driverId', 'circuitId', 'track_avg_finish',
        'track_familiarity', 'track_performance_delta', 'driver_overall_avg_sofar',
    ]]
    logger.info(f"Engineered track features for {len(track_features)} race entries")
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
    race_info: pd.DataFrame,
    qualifying_features: pd.DataFrame = None,
    season_features: pd.DataFrame = None,
    pitstop_features: pd.DataFrame = None,
    circuit_features: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Combine all features into single feature matrix for model.

    Args:
        driver_features: Driver-level features
        constructor_features: Constructor-level features
        track_features: Track-specific features
        race_info: Race and result information with target
        qualifying_features: Optional qualifying-derived features
        season_features: Optional in-season championship momentum features
        pitstop_features: Optional pit-crew-speed features
        circuit_features: Optional circuit overtaking-difficulty features

    Returns:
        Tuple[X, metadata, y]: Feature matrix, metadata, target variable
            - X: Feature matrix (races x features)
            - metadata: Race/driver identifiers
            - y: Target (finishing position)
    """
    logger.info("Building feature matrix...")

    # Start with race info
    X = race_info.copy()

    # Merge driver features (one row per raceId+driverId, prior-races-only)
    X = X.merge(
        driver_features,
        on=['raceId', 'driverId'],
        how='left'
    )

    # Merge constructor features (one row per raceId+constructorId, prior-races-only)
    X = X.merge(
        constructor_features,
        on=['raceId', 'constructorId'],
        how='left'
    )

    # Merge track features (one row per raceId+driverId+circuitId, prior-visits-only)
    X = X.merge(
        track_features,
        on=['raceId', 'driverId', 'circuitId'],
        how='left'
    )

    # Cold start: no prior visit to this circuit yet - fall back to the
    # driver's overall average so far instead of leaving track stats blank
    X['track_avg_finish'] = X['track_avg_finish'].fillna(X['driver_overall_avg_sofar'])
    X['track_performance_delta'] = X['track_performance_delta'].fillna(0)
    X = X.drop(columns=['driver_overall_avg_sofar'])

    # Merge qualifying features (known before the race - no leakage)
    if qualifying_features is not None:
        X = X.merge(qualifying_features, on=['raceId', 'driverId'], how='left')
        # Missing qualifying record (e.g. DNQ) - use grid as a stand-in position
        # and assume an average gap
        X['qual_position'] = X['qual_position'].fillna(X['grid'])
        X['qual_gap_seconds'] = X['qual_gap_seconds'].fillna(X['qual_gap_seconds'].mean())
        X['teammate_qual_gap_seconds'] = X['teammate_qual_gap_seconds'].fillna(0)

    # Merge in-season championship momentum (prior races of same season only)
    if season_features is not None:
        X = X.merge(season_features, on=['raceId', 'driverId'], how='left')
        X['season_points_so_far'] = X['season_points_so_far'].fillna(0)
        X['season_position_so_far'] = X['season_position_so_far'].fillna(X['season_position_so_far'].max())

    # Merge pit-crew-speed features (constructor's prior-race pit stop average)
    if pitstop_features is not None:
        X = X.merge(pitstop_features, on=['raceId', 'constructorId'], how='left')
        X['avg_pit_stop_duration'] = X['avg_pit_stop_duration'].fillna(X['avg_pit_stop_duration'].mean())

    # Merge circuit overtaking-difficulty features (prior races at this circuit)
    if circuit_features is not None:
        X = X.merge(circuit_features, on=['raceId', 'circuitId'], how='left')
        X['circuit_avg_position_change'] = X['circuit_avg_position_change'].fillna(
            X['circuit_avg_position_change'].mean()
        )
    
    # Extract target and metadata
    y = X['positionOrder'].astype(float)
    metadata = X[['raceId', 'driverId', 'constructorId', 'year', 'round']].copy()
    
    # Select feature columns (exclude identifiers, target, and race-outcome leakage)
    exclude_cols = [
        # Identifiers / metadata
        'raceId', 'driverId', 'constructorId', 'year', 'round',
        'race_name', 'date', 'circuitId',
        'surname', 'forename', 'nationality_driver',
        'name_team', 'nationality_constructor',
        'name_circuit', 'location', 'country',
        'statusId', 'race_status', 'resultId', 'number',
        # Target
        'positionOrder',
        # Leakage: these are outcomes of the race being predicted,
        # not known ahead of time
        'position', 'positionText', 'laps', 'time', 'milliseconds',
        'fastestLap', 'rank', 'fastestLapTime', 'fastestLapSpeed', 'points',
    ]
    feature_cols = [col for col in X.columns if col not in exclude_cols]

    X_features = X[feature_cols].copy()

    # Convert to numeric, coercing errors to NaN
    for col in X_features.columns:
        X_features[col] = pd.to_numeric(X_features[col], errors='coerce')

    # Drop columns that are entirely non-numeric/NaN (can't be imputed)
    all_nan_cols = [col for col in X_features.columns if X_features[col].isna().all()]
    if all_nan_cols:
        logger.warning(f"Dropping all-NaN feature columns: {all_nan_cols}")
        X_features = X_features.drop(columns=all_nan_cols)

    # Handle NaN values - fill with column mean (only for numeric columns)
    for col in X_features.columns:
        if X_features[col].dtype in ['float64', 'int64']:
            col_mean = X_features[col].mean()
            if pd.notna(col_mean):
                X_features[col] = X_features[col].fillna(col_mean)
    
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
