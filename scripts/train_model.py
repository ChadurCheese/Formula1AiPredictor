"""
Model Training Script

Full pipeline: Load data → Engineer features → Train model → Evaluate → Save

Run with: python scripts/train_model.py
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_raw_data, prepare_data, load_cached_data
from src.features import (
    engineer_driver_features,
    engineer_constructor_features,
    engineer_track_features,
    engineer_weather_features,
    engineer_qualifying_features,
    engineer_season_form_features,
    engineer_pitstop_features,
    engineer_circuit_overtaking_features,
    build_feature_matrix,
    get_feature_names
)
from src.traits import calculate_driver_traits, save_traits
from src.model import train_model, evaluate_model, save_model
from src.utils import setup_logging

# Setup logging
logger = setup_logging(log_level="INFO")


def main():
    """Execute full training pipeline."""
    
    logger.info("=" * 60)
    logger.info("F1 RACE PREDICTOR - MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load raw data
        logger.info("\n[STEP 1/7] Loading raw data...")
        raw_data = load_raw_data()
        logger.info(f"Loaded {len(raw_data['results'])} race results")
        
        # Step 2: Prepare data
        logger.info("\n[STEP 2/7] Preparing and cleaning data...")
        prepared_data = prepare_data(raw_data)
        logger.info(f"Prepared data shape: {prepared_data.shape}")
        
        # Step 3: Calculate driver traits
        logger.info("\n[STEP 3/7] Calculating driver traits...")
        driver_traits = calculate_driver_traits(
            prepared_data,
            raw_data['qualifying'],
            raw_data['drivers']
        )
        save_traits(driver_traits)
        logger.info(f"Calculated traits for {len(driver_traits)} drivers")
        
        # Step 4: Engineer features
        logger.info("\n[STEP 4/7] Engineering features...")
        driver_features = engineer_driver_features(prepared_data)
        logger.info(f"Driver features: {driver_features.shape[0]} drivers × {driver_features.shape[1]} features")
        
        constructor_features = engineer_constructor_features(prepared_data)
        logger.info(f"Constructor features: {constructor_features.shape[0]} teams × {constructor_features.shape[1]} features")
        
        track_features = engineer_track_features(prepared_data, raw_data['races'])
        logger.info(f"Track features: {len(track_features)} driver-track combinations")
        
        weather_features = engineer_weather_features(prepared_data)
        logger.info(f"Weather features placeholder: {len(weather_features)} entries")

        qualifying_features = engineer_qualifying_features(raw_data['qualifying'])
        logger.info(f"Qualifying features: {len(qualifying_features)} entries")

        season_features = engineer_season_form_features(prepared_data)
        logger.info(f"Season form features: {len(season_features)} entries")

        pitstop_features = engineer_pitstop_features(raw_data['pit_stops'], prepared_data)
        logger.info(f"Pit stop features: {len(pitstop_features)} entries")

        circuit_features = engineer_circuit_overtaking_features(prepared_data)
        logger.info(f"Circuit overtaking features: {len(circuit_features)} entries")

        # Step 5: Build feature matrix
        logger.info("\n[STEP 5/7] Building feature matrix...")
        X, metadata, y = build_feature_matrix(
            driver_features,
            constructor_features,
            track_features,
            prepared_data,
            qualifying_features=qualifying_features,
            season_features=season_features,
            pitstop_features=pitstop_features,
            circuit_features=circuit_features,
        )
        logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        
        # Step 6: Split data into train/val/test by season
        logger.info("\n[STEP 6/7] Splitting data into train/val/test...")
        
        # Extract year from metadata
        years = metadata['year'].values
        
        train_mask = (years >= 2020) & (years <= 2022)
        val_mask = years == 2023
        test_mask = years == 2024
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train set: {len(X_train)} samples (2020-2022)")
        logger.info(f"Val set: {len(X_val)} samples (2023)")
        logger.info(f"Test set: {len(X_test)} samples (2024)")
        
        # Step 7: Train and evaluate model
        logger.info("\n[STEP 7/7] Training model...")
        model = train_model(X_train, y_train, X_val, y_val)
        
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(model, X_val, y_val, race_ids=metadata.loc[X_val.index, 'raceId'])

        logger.info("Evaluating on test set...")
        test_metrics = evaluate_model(model, X_test, y_test, race_ids=metadata.loc[X_test.index, 'raceId'])
        
        # Save model
        logger.info("Saving model...")
        save_model(model, metrics={
            "validation": val_metrics,
            "test": test_metrics
        })
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Test MAE: {test_metrics['mae']:.2f} positions")
        logger.info(f"Test R²: {test_metrics['r2']:.3f}")
        logger.info(f"Within 2 positions: {test_metrics['within_2_positions']:.1f}%")
        logger.info("\n📁 Output files:")
        logger.info("  - Model saved to: models/model.pkl")
        logger.info("  - Traits saved to: data/cache/driver_traits.json")
        logger.info("  - Config saved to: models/model_config.json")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

