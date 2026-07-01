"""
Driver Traits Module

Calculates FM-style driver characteristics from historical data.
Each driver gets 6 trait scores (0-1) that influence prediction explanations.

Traits:
    1. qualifying_specialist: High qualifying, variable race day
    2. wet_weather_master: Better in rain
    3. strong_starter: Overtakes in early laps
    4. tire_management: Consistent across tire strategies
    5. inconsistent: High variability in finishes
    6. track_expert: Strong at familiar circuits

Functions:
    - calculate_driver_traits(): Compute all 6 traits for all drivers
    - trait_qualifying_specialist(): Qualifying vs race performance gap
    - trait_wet_weather_master(): Wet vs dry performance
    - trait_strong_starter(): Grid to finish improvement
    - trait_tire_management(): Consistency metric
    - trait_inconsistent(): Variability metric
    - trait_track_expert(): Track familiarity bonus
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_driver_traits(
    results_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
    drivers_df: pd.DataFrame,
    recent_seasons: int = 2,
    min_recent_races: int = 5,
) -> Dict:
    """
    Calculate all 6 traits for all drivers from historical data.

    Traits reflect *current form*, not a driver's whole career: each driver's
    stats are computed from only their last `recent_seasons` seasons (relative
    to the most recent season in the data). This avoids, e.g., a rookie year
    full of retirements permanently dragging down a driver's consistency
    score years after they've become a front-runner. If a driver doesn't have
    enough races in that recent window (e.g. a rookie), their full career is
    used as a fallback instead of the default neutral score.

    Args:
        results_df: Race results with finishing positions
        qualifying_df: Qualifying session data
        drivers_df: Driver information
        recent_seasons: Number of most recent seasons to use for each driver
        min_recent_races: Minimum races required in the recent window before
            falling back to the driver's full career history

    Returns:
        dict: Traits indexed by driver_id
              {driver_id: {trait_name: score (0-1), ...}, ...}
    """
    driver_traits = {}

    max_year = results_df['year'].max()
    cutoff_year = max_year - recent_seasons + 1
    recent_results = results_df[results_df['year'] >= cutoff_year]

    for driver_id in drivers_df['driverId'].unique():
        try:
            driver_races_recent = recent_results[recent_results['driverId'] == driver_id]
            driver_races_career = results_df[results_df['driverId'] == driver_id]

            # Use recent form if there's enough of it, otherwise fall back to
            # full career (e.g. for a driver who hasn't raced recently)
            driver_races = (
                driver_races_recent if len(driver_races_recent) >= min_recent_races
                else driver_races_career
            )
            driver_quali = qualifying_df[qualifying_df['driverId'] == driver_id]

            if len(driver_races) < 5:  # Skip drivers with < 5 races total
                continue

            traits = {
                "qualifying_specialist": trait_qualifying_specialist(driver_races, driver_quali),
                "wet_weather_master": trait_wet_weather_master(driver_races),
                "strong_starter": trait_strong_starter(driver_races),
                "tire_management": trait_tire_management(driver_races),
                "inconsistent": trait_inconsistent(driver_races),
                "track_expert": trait_track_expert(driver_races),
            }

            driver_traits[driver_id] = traits

        except Exception as e:
            logger.warning(f"Error calculating traits for driver {driver_id}: {e}")
            continue

    logger.info(f"Calculated traits for {len(driver_traits)} drivers")
    return driver_traits


def trait_qualifying_specialist(
    results_df: pd.DataFrame,
    qualifying_df: pd.DataFrame
) -> float:
    """
    Calculate Qualifying Specialist trait.
    
    High value = Driver excels in qualifying but struggles on race day
    Metric: (avg_qualifying_position - avg_race_position) / avg_qualifying_position
    
    Args:
        results_df: Driver's race results
        qualifying_df: Driver's qualifying results
    
    Returns:
        float: Score 0-1 (higher = better at qualifying relative to races)
    """
    # Get valid race positions (no DNF/null)
    valid_races = results_df[results_df['positionOrder'].notna()].copy()
    
    if len(valid_races) < 3:
        return 0.5
    
    # Merge qualifying position with race results
    merged = valid_races.merge(
        qualifying_df[['raceId', 'position']],
        on='raceId',
        how='inner',
        suffixes=('_race', '_qual')
    )
    
    if len(merged) < 3:
        return 0.5
    
    # Calculate position gaps: negative means gained positions, positive means lost
    merged['position_gap'] = merged['positionOrder'] - merged['position_qual']
    
    # Average gap (negative = generally improves from qual to race)
    avg_gap = merged['position_gap'].mean()
    
    # Normalize: if driver loses ~5 positions on avg, score should be low
    # If driver gains positions, score should be high
    # Scale: [-15, +15] -> [0, 1]
    specialist_score = np.clip((-avg_gap + 15) / 30, 0, 1)
    
    return specialist_score


def trait_wet_weather_master(results_df: pd.DataFrame) -> float:
    """
    Calculate Wet Weather Master trait.
    
    High value = Driver performs better in wet weather
    Metric: avg_finish_position(dry_races) / avg_finish_position(wet_races)
    
    Args:
        results_df: Driver's race results (with weather info if available)
    
    Returns:
        float: Score 0-1 (higher = better in wet)
    """
    # Without explicit weather data, use circuit heuristic
    # Circuits known for frequent rain: Monaco, Silverstone, Spa, Hungary
    wet_circuits = {'Monaco', 'Silverstone', 'Spa-Francorchamps', 'Hungaroring'}
    
    valid_races = results_df[results_df['positionOrder'].notna()].copy()
    
    if len(valid_races) < 5:
        return 0.5
    
    # Separate by circuit type (if circuitName available)
    if 'circuitName' in valid_races.columns:
        wet_races = valid_races[valid_races['circuitName'].isin(wet_circuits)]
        dry_races = valid_races[~valid_races['circuitName'].isin(wet_circuits)]
    else:
        # Fallback: use all races, assume roughly equal split
        # Score based on consistency (proxy for wet performance)
        dry_races = valid_races
        wet_races = valid_races.sample(min(len(valid_races) // 3, 5), replace=False) if len(valid_races) > 5 else valid_races.head(2)
    
    if len(wet_races) < 2 or len(dry_races) < 2:
        return 0.5
    
    avg_wet = wet_races['positionOrder'].mean()
    avg_dry = dry_races['positionOrder'].mean()
    
    # If avg_dry > avg_wet (numerically higher = worse position), driver is better in wet
    # Normalize: ratio of 1.2+ = good in wet, ratio of 0.8- = bad in wet
    if avg_wet == 0:
        return 0.5
    
    ratio = avg_dry / avg_wet
    wet_score = np.clip((ratio - 0.8) / 0.6, 0, 1)  # [0.8->0, 1.4->1]
    
    return wet_score


def trait_strong_starter(results_df: pd.DataFrame) -> float:
    """
    Calculate Strong Starter trait.
    
    High value = Driver gains positions in opening laps
    Metric: (grid_position - finish_position) / grid_position
    
    Args:
        results_df: Driver's race results with grid position
    
    Returns:
        float: Score 0-1 (higher = more overtakes at start)
    """
    # Get races with valid grid and finish positions
    valid_races = results_df[
        (results_df['grid'].notna()) & 
        (results_df['positionOrder'].notna()) &
        (results_df['grid'] > 0)
    ].copy()
    
    if len(valid_races) < 3:
        return 0.5
    
    # Calculate position gained (negative = positions lost, positive = gained)
    valid_races['positions_gained'] = valid_races['grid'] - valid_races['positionOrder']
    
    # Average positions gained per race
    avg_gained = valid_races['positions_gained'].mean()
    
    # Normalize: if average gains 3+ positions per race, score = 1
    # If loses 3+ positions, score = 0
    # Scale: [-10, +10] -> [0, 1]
    starter_score = np.clip((avg_gained + 10) / 20, 0, 1)
    
    return starter_score


def _inconsistency_from_std(std_position: float, floor: float = 1.0, ceiling: float = 7.0) -> float:
    """
    Normalize a raw standard deviation of finishing position into a 0-1
    inconsistency score using a fixed absolute scale.

    Uses raw std rather than coefficient of variation (std/mean) because CV
    inflates for drivers with a low mean position: a front-runner averaging
    P1-P2 can have a tiny absolute std (e.g. 3) yet a large CV simply because
    the denominator is small, making dominant drivers look "inconsistent".
    Floor/ceiling (1-7 positions) are calibrated from the observed spread of
    full-time drivers' finishing position std in the dataset.

    Args:
        std_position: Standard deviation of finishing positions
        floor: Std at or below which a driver is considered maximally consistent
        ceiling: Std at or above which a driver is considered maximally inconsistent

    Returns:
        float: Score 0-1 (higher = more inconsistent)
    """
    if pd.isna(std_position):
        return 0.5
    return float(np.clip((std_position - floor) / (ceiling - floor), 0, 1))


def trait_tire_management(results_df: pd.DataFrame) -> float:
    """
    Calculate Tire Management Expert trait.

    High value = Consistent across different tire strategies
    Metric: 1 - (std_dev_finishes / mean_finishes)
    
    Args:
        results_df: Driver's race results
    
    Returns:
        float: Score 0-1 (higher = more consistent)
    """
    valid_races = results_df[results_df['positionOrder'].notna()]

    if len(valid_races) < 5:
        return 0.5

    std_position = valid_races['positionOrder'].std()

    # Higher std = less consistent = lower tire management score
    management_score = 1 - _inconsistency_from_std(std_position)

    return management_score


def trait_inconsistent(results_df: pd.DataFrame) -> float:
    """
    Calculate Inconsistent trait.
    
    High value = Driver is unreliable, high variability
    Metric: std_dev(finish_positions)
    
    Args:
        results_df: Driver's race results
    
    Returns:
        float: Score 0-1 (higher = more inconsistent)
    """
    valid_races = results_df[results_df['positionOrder'].notna()]

    if len(valid_races) < 5:
        return 0.5

    std_position = valid_races['positionOrder'].std()

    return _inconsistency_from_std(std_position)


def trait_track_expert(results_df: pd.DataFrame) -> float:
    """
    Calculate Track Expert trait.
    
    High value = Driver has favorite tracks where they excel
    Metric: (avg_finish_best_track - avg_overall_finish) / avg_overall_finish
    
    Args:
        results_df: Driver's race results with track info
    
    Returns:
        float: Score 0-1 (higher = stronger track specialization)
    """
    valid_races = results_df[results_df['positionOrder'].notna()].copy()
    
    if len(valid_races) < 5:
        return 0.5
    
    overall_avg = valid_races['positionOrder'].mean()
    
    if 'circuitId' not in valid_races.columns:
        return 0.5
    
    # Find circuits where driver has raced 3+ times
    circuit_counts = valid_races['circuitId'].value_counts()
    frequent_circuits = circuit_counts[circuit_counts >= 3].index
    
    if len(frequent_circuits) == 0:
        return 0.5
    
    # Calculate average finish position at each frequent circuit
    circuit_avgs = []
    for circuit in frequent_circuits:
        circuit_races = valid_races[valid_races['circuitId'] == circuit]
        circuit_avg = circuit_races['positionOrder'].mean()
        circuit_avgs.append(circuit_avg)
    
    # Find best performing circuit (lowest avg position)
    best_circuit_avg = min(circuit_avgs)
    
    # Specialization: how much better at best circuit vs overall
    # Positive = better at specialty circuit, negative = no specialization
    # Normalize: if 5 positions better at specialty, score = 1
    if overall_avg == 0:
        return 0.5
    
    improvement = overall_avg - best_circuit_avg
    expert_score = np.clip(improvement / 10, 0, 1)  # 10 position improvement = max score
    
    return expert_score


def save_traits(traits: Dict, output_path: Path = Path("data/cache/driver_traits.json")) -> None:
    """
    Save calculated traits to JSON cache file.
    
    Args:
        traits: Dictionary of driver traits
        output_path: Path to save JSON
    """
    logger.info(f"Saving traits to {output_path}")
    
    # Convert int keys to strings for JSON serialization
    traits_serializable = {str(k): v for k, v in traits.items()}
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(traits_serializable, f, indent=2)
    
    logger.info(f"Saved traits for {len(traits)} drivers")


def load_traits(traits_path: Path = Path("data/cache/driver_traits.json")) -> Dict:
    """
    Load previously calculated traits from JSON cache.
    
    Args:
        traits_path: Path to traits JSON file
    
    Returns:
        dict: Driver traits dictionary
    """
    if not traits_path.exists():
        logger.warning(f"Traits file not found: {traits_path}")
        return {}
    
    with open(traits_path, 'r') as f:
        traits = json.load(f)
    
    # Convert string keys back to integers
    traits = {int(k): v for k, v in traits.items()}
    
    logger.info(f"Loaded traits for {len(traits)} drivers")
    return traits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test: Print trait names
    sample_traits = {
        "1": {
            "qualifying_specialist": 0.75,
            "wet_weather_master": 0.60,
            "strong_starter": 0.70,
            "tire_management": 0.65,
            "inconsistent": 0.30,
            "track_expert": 0.55,
        }
    }
    
    print("Sample Driver Traits:")
    for trait_name, score in sample_traits["1"].items():
        print(f"  {trait_name}: {score:.2f}")
