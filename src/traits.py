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
    drivers_df: pd.DataFrame
) -> Dict:
    """
    Calculate all 6 traits for all drivers from historical data.
    
    Args:
        results_df: Race results with finishing positions
        qualifying_df: Qualifying session data
        drivers_df: Driver information
    
    Returns:
        dict: Traits indexed by driver_id
              {driver_id: {trait_name: score (0-1), ...}, ...}
    """
    driver_traits = {}
    
    for driver_id in drivers_df['driverId'].unique():
        try:
            driver_races = results_df[results_df['driverId'] == driver_id]
            driver_quali = qualifying_df[qualifying_df['driverId'] == driver_id]
            
            if len(driver_races) < 5:  # Skip drivers with < 5 races
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
    
    mean_position = valid_races['positionOrder'].mean()
    std_position = valid_races['positionOrder'].std()
    
    if mean_position == 0:
        return 0.5
    
    # Coefficient of variation
    cv = std_position / mean_position
    
    # Higher CV = less consistent = lower tire management score
    # Normalize: CV of 0.3 = 1.0 (very consistent), CV of 1.5 = 0 (very inconsistent)
    management_score = np.clip(1 - (cv / 1.5), 0, 1)
    
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
    
    mean_position = valid_races['positionOrder'].mean()
    std_position = valid_races['positionOrder'].std()
    
    if mean_position == 0:
        return 0.5
    
    # Coefficient of variation - higher = more inconsistent
    cv = std_position / mean_position
    
    # Normalize: CV of 0.3 = 0 (very consistent), CV of 1.5 = 1 (very inconsistent)
    inconsistent_score = np.clip(cv / 1.5, 0, 1)
    
    return inconsistent_score


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
