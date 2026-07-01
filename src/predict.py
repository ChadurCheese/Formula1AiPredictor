"""
Prediction API Module

High-level interface for generating predictions on races.
Used by Streamlit app to fetch predictions.

Functions:
    - predict_race(): Predict positions for all drivers in a race
    - predict_single_driver(): Predict position for one driver
    - get_available_races(): List races that can be predicted
    - load_predictions_from_cache(): Retrieve cached predictions
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import logging

from src.data_pipeline import load_raw_data, prepare_data
from src.features import (
    engineer_driver_features,
    engineer_constructor_features,
    engineer_track_features,
    engineer_qualifying_features,
    engineer_season_form_features,
    engineer_pitstop_features,
    engineer_circuit_overtaking_features,
    build_feature_matrix,
)
from src.model import load_model
from src.traits import load_traits
from src.explainer import explain_prediction
import shap

logger = logging.getLogger(__name__)

# Module-level cache so the (relatively expensive) data/feature pipeline is
# only built once per process instead of on every prediction call.
_pipeline_cache: Optional[Dict] = None


def _get_pipeline() -> Dict:
    """
    Build (or return cached) prediction pipeline artifacts: model, traits,
    the full feature matrix, and lookup tables for driver/race names.

    Returns:
        dict: {
            'model', 'traits', 'X', 'metadata', 'y',
            'drivers_df', 'races_df', 'feature_names'
        }
    """
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    logger.info("Building prediction pipeline (first call - this may take a moment)...")

    model = load_model()
    traits = load_traits()

    raw_data = load_raw_data()
    prepared = prepare_data(raw_data)

    driver_features = engineer_driver_features(prepared)
    constructor_features = engineer_constructor_features(prepared)
    track_features = engineer_track_features(prepared, raw_data['races'])
    qualifying_features = engineer_qualifying_features(raw_data['qualifying'])
    season_features = engineer_season_form_features(prepared)
    pitstop_features = engineer_pitstop_features(raw_data['pit_stops'], prepared)
    circuit_features = engineer_circuit_overtaking_features(prepared)

    X, metadata, y = build_feature_matrix(
        driver_features,
        constructor_features,
        track_features,
        prepared,
        qualifying_features=qualifying_features,
        season_features=season_features,
        pitstop_features=pitstop_features,
        circuit_features=circuit_features,
    )

    _pipeline_cache = {
        'model': model,
        'traits': traits,
        'X': X,
        'metadata': metadata,
        'y': y,
        'drivers_df': raw_data['drivers'],
        'races_df': raw_data['races'],
        'constructors_df': raw_data['constructors'],
        'feature_names': list(X.columns),
        'shap_explainer': shap.TreeExplainer(model),
    }
    logger.info("Prediction pipeline ready")
    return _pipeline_cache


def get_available_races(years: List[int] = None) -> pd.DataFrame:
    """
    List races that have a full feature row (and therefore can be predicted).

    Args:
        years: Optional list of seasons to restrict to (e.g. [2023, 2024])

    Returns:
        pd.DataFrame: raceId, year, round, race name, date - sorted by date
    """
    pipeline = _get_pipeline()
    races_df = pipeline['races_df']
    available_ids = pipeline['metadata']['raceId'].unique()

    races = races_df[races_df['raceId'].isin(available_ids)][
        ['raceId', 'year', 'round', 'name', 'date']
    ].copy()

    if years:
        races = races[races['year'].isin(years)]

    return races.sort_values(['year', 'round']).reset_index(drop=True)


def predict_race(race_id: int) -> List[Dict]:
    """
    Predict race results for all drivers in a specific race.

    Returns predictions with:
        - Driver name and ID
        - Predicted finishing position
        - Confidence score
        - Trait-based explanation
        - Actual position (if available for historical races)

    Args:
        race_id: Race ID from database

    Returns:
        list: Predictions for each driver, sorted by predicted position
            [{
                'driver_id': 1,
                'driver_name': 'Lewis Hamilton',
                'predicted_position': 2.1,
                'confidence': 0.78,
                'trait_influences': [...],
                'explanation': "...",
                'actual_position': 1,  # if available
            }, ...]
    """
    pipeline = _get_pipeline()
    model = pipeline['model']
    traits = pipeline['traits']
    X = pipeline['X']
    metadata = pipeline['metadata']
    y = pipeline['y']
    drivers_df = pipeline['drivers_df']
    constructors_df = pipeline['constructors_df']
    feature_names = pipeline['feature_names']
    shap_explainer = pipeline['shap_explainer']

    race_rows = metadata[metadata['raceId'] == race_id]
    if race_rows.empty:
        logger.warning(f"No feature data available for race {race_id}")
        return []

    predictions = []
    for idx in race_rows.index:
        driver_id = int(metadata.loc[idx, 'driverId'])
        constructor_id = int(metadata.loc[idx, 'constructorId'])
        X_row = X.loc[idx]

        explanation = explain_prediction(
            model, X_row, traits, driver_id, feature_names, explainer=shap_explainer
        )

        driver_row = drivers_df[drivers_df['driverId'] == driver_id]
        driver_name = (
            f"{driver_row.iloc[0]['forename']} {driver_row.iloc[0]['surname']}"
            if not driver_row.empty else f"Driver {driver_id}"
        )

        constructor_row = constructors_df[constructors_df['constructorId'] == constructor_id]
        constructor_name = (
            constructor_row.iloc[0]['name'] if not constructor_row.empty else "Unknown"
        )

        raw_position = explanation['predicted_position']
        predictions.append({
            'driver_id': driver_id,
            'driver_name': driver_name,
            'constructor_name': constructor_name,
            'predicted_position': raw_position,
            'starting_position': int(X_row['grid']),
            'confidence': explanation['confidence'],
            'trait_influences': explanation['trait_influences'],
            'explanation': explanation['summary'],
            'actual_position': float(y.loc[idx]) if idx in y.index else None,
        })

    # Convert raw regression output into within-race integer ranks (1..N).
    # Finishing position is a permutation, so this is both a fairer accuracy
    # metric (see src.model._rank_within_race) and a cleaner display value
    # ("P3" instead of "P3.3").
    predictions.sort(key=lambda p: p['predicted_position'])
    for rank, prediction in enumerate(predictions, start=1):
        raw_rounded = round(prediction['predicted_position'])
        prediction['explanation'] = prediction['explanation'].replace(
            f"Predicted P{raw_rounded}", f"Predicted P{rank}", 1
        )
        prediction['predicted_position'] = float(rank)

    return predictions


def predict_single_driver(race_id: int, driver_id: int) -> Optional[Dict]:
    """
    Predict finishing position for a single driver in a race.

    Args:
        race_id: Race ID
        driver_id: Driver ID

    Returns:
        dict: Single driver prediction, or None if not found
    """
    race_predictions = predict_race(race_id)
    for prediction in race_predictions:
        if prediction['driver_id'] == driver_id:
            return prediction

    logger.warning(f"Driver {driver_id} not found in race {race_id}")
    return None


def load_predictions_from_cache(
    cache_path: Path = Path("data/cache/predictions_cache.pkl")
) -> Dict:
    """
    Load previously computed predictions from cache.

    Args:
        cache_path: Path to predictions pickle

    Returns:
        dict: Cached predictions by race_id
    """
    if not cache_path.exists():
        logger.warning(f"Prediction cache not found: {cache_path}")
        return {}

    with open(cache_path, 'rb') as f:
        predictions = pickle.load(f)

    logger.info(f"Loaded predictions for {len(predictions)} races from cache")
    return predictions


def save_predictions_to_cache(
    predictions: Dict,
    cache_path: Path = Path("data/cache/predictions_cache.pkl")
) -> None:
    """
    Cache predictions to disk for faster Streamlit loads.

    Args:
        predictions: Dictionary of predictions by race_id
        cache_path: Where to save cache file
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, 'wb') as f:
        pickle.dump(predictions, f)

    logger.info(f"Cached predictions to {cache_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    races = get_available_races(years=[2024])
    print(f"Available 2024 races: {len(races)}")
    print(races.head())

    if not races.empty:
        sample_race_id = int(races.iloc[0]['raceId'])
        predictions = predict_race(sample_race_id)
        print(f"\nPredictions for race {sample_race_id}:")
        for p in predictions[:5]:
            print(f"  P{p['predicted_position']:.1f} - {p['driver_name']} "
                  f"(actual: {p['actual_position']}) - {p['explanation']}")
