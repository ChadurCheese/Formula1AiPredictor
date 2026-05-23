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
    # TODO: Implement
    # Compare qualifying position vs race finish position
    # Normalize to 0-1 scale
    return 0.5


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
    # TODO: Implement
    # Filter for wet weather races
    # Compare wet vs dry performance
    # Normalize to 0-1
    return 0.5


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
    # TODO: Implement
    # Calculate (grid_pos - finish_pos) for each race
    # Normalize and average
    return 0.5


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
    # TODO: Implement
    # Calculate consistency (1 - coefficient_of_variation)
    # Higher consistency = better tire management
    return 0.5


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
    # TODO: Implement
    # Calculate std dev of finishing positions
    # Normalize to 0-1 scale
    return 0.5


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
    # TODO: Implement
    # Find best tracks (where driver has raced multiple times)
    # Compare best track average to overall average
    # Normalize
    return 0.5


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
