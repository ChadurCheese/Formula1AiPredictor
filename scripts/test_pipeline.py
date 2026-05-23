"""
Data Pipeline Test Script

End-to-end test of data loading, preparation, feature engineering,
and feature matrix building.

Run with: python scripts/test_pipeline.py
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_raw_data, prepare_data, cache_data
from src.features import (
    engineer_driver_features,
    engineer_constructor_features,
    engineer_track_features,
    engineer_weather_features,
    build_feature_matrix
)
from src.utils import setup_logging

logger = setup_logging(log_level="INFO")


def main():
    """Execute full pipeline test."""
    
    logger.info("=" * 70)
    logger.info("F1 RACE PREDICTOR - DATA PIPELINE TEST")
    logger.info("=" * 70)
    
    try:
        # ========== STEP 1: Load Raw Data ==========
        logger.info("\n[STEP 1] Loading raw data...")
        raw_data = load_raw_data()
        
        for key, df in raw_data.items():
            logger.info(f"  {key:15} | {df.shape[0]:6} rows, {df.shape[1]:3} columns")
        
        # ========== STEP 2: Prepare Data ==========
        logger.info("\n[STEP 2] Preparing data (merge, clean, validate)...")
        prepared_data = prepare_data(raw_data)
        
        logger.info(f"  Shape: {prepared_data.shape}")
        logger.info(f"  Columns: {prepared_data.shape[1]}")
        logger.info(f"  Years: {prepared_data['year'].min()} - {prepared_data['year'].max()}")
        logger.info(f"  Unique races: {prepared_data['raceId'].nunique()}")
        logger.info(f"  Unique drivers: {prepared_data['driverId'].nunique()}")
        
        # Show sample
        logger.info("\n  Sample data (first 3 rows):")
        cols_to_show = ['year', 'race_name', 'surname', 'name_team', 'grid', 'positionOrder', 'points']
        sample_cols = [c for c in cols_to_show if c in prepared_data.columns]
        for idx, row in prepared_data[sample_cols].head(3).iterrows():
            logger.info(f"    {dict(row)}")
        
        # ========== STEP 3: Cache Prepared Data ==========
        logger.info("\n[STEP 3] Caching prepared data...")
        cache_data(prepared_data)
        
        # ========== STEP 4: Engineer Features ==========
        logger.info("\n[STEP 4] Engineering features...")
        
        # Driver features
        logger.info("  4a. Driver features...")
        driver_feats = engineer_driver_features(prepared_data)
        logger.info(f"      {len(driver_feats)} drivers, {driver_feats.shape[1]} features")
        logger.info(f"      Columns: {list(driver_feats.columns)}")
        
        # Constructor features
        logger.info("  4b. Constructor features...")
        constructor_feats = engineer_constructor_features(prepared_data)
        logger.info(f"      {len(constructor_feats)} constructors, {constructor_feats.shape[1]} features")
        logger.info(f"      Columns: {list(constructor_feats.columns)}")
        
        # Track features
        logger.info("  4c. Track features...")
        track_feats = engineer_track_features(prepared_data, raw_data['races'])
        logger.info(f"      {len(track_feats)} driver-track combos, {track_feats.shape[1]} features")
        logger.info(f"      Columns: {list(track_feats.columns)}")
        
        # Weather features (placeholder for now)
        logger.info("  4d. Weather features (placeholder)...")
        weather_feats = engineer_weather_features(prepared_data)
        logger.info(f"      {len(weather_feats)} drivers, {weather_feats.shape[1]} features")
        
        # ========== STEP 5: Build Feature Matrix ==========
        logger.info("\n[STEP 5] Building feature matrix...")
        
        # Prepare race_info for feature matrix (needs circuitId from races)
        race_info = prepared_data.merge(
            raw_data['races'][['raceId', 'circuitId']],
            on='raceId',
            how='left'
        )
        
        X, metadata, y = build_feature_matrix(
            driver_feats,
            constructor_feats,
            track_feats,
            race_info
        )
        
        logger.info(f"  Feature matrix shape: {X.shape}")
        logger.info(f"  Target shape: {y.shape}")
        logger.info(f"  Metadata shape: {metadata.shape}")
        
        # ========== STEP 6: Analysis ==========
        logger.info("\n[STEP 6] Feature matrix analysis...")
        
        logger.info(f"  Features (total {X.shape[1]}):")
        for col in X.columns:
            mean_val = X[col].mean()
            std_val = X[col].std()
            logger.info(f"    {col:35} | mean={mean_val:8.3f}, std={std_val:8.3f}")
        
        logger.info(f"\n  Target variable (positionOrder):")
        logger.info(f"    Min: {y.min():.1f}, Max: {y.max():.1f}, Mean: {y.mean():.1f}, Std: {y.std():.1f}")
        
        # ========== STEP 7: Train/Val/Test Split ==========
        logger.info("\n[STEP 7] Train/Val/Test split preparation...")
        
        train_years = [2020, 2021, 2022]
        val_years = [2023]
        test_years = [2024]
        
        train_idx = metadata['year'].isin(train_years)
        val_idx = metadata['year'].isin(val_years)
        test_idx = metadata['year'].isin(test_years)
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        logger.info(f"  Training set (2020-2022):   {X_train.shape[0]:5} samples")
        logger.info(f"  Validation set (2023):      {X_val.shape[0]:5} samples")
        logger.info(f"  Test set (2024):            {X_test.shape[0]:5} samples")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ DATA PIPELINE TEST COMPLETE")
        logger.info("=" * 70)
        logger.info("\n📊 Summary:")
        logger.info(f"   - Raw races: {len(raw_data['races'])}")
        logger.info(f"   - Prepared races: {prepared_data['raceId'].nunique()}")
        logger.info(f"   - Feature matrix ready: {X.shape}")
        logger.info(f"   - Ready for model training!")
        logger.info("\nNext step: python scripts/train_model.py")
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
