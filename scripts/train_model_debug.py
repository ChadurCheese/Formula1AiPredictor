"""
Model Training Script with Debug Output
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Starting training pipeline...")

print("Importing modules...")
from src.data_pipeline import load_raw_data, prepare_data, load_cached_data
from src.features import (
    engineer_driver_features,
    engineer_constructor_features,
    engineer_track_features,
    engineer_weather_features,
    build_feature_matrix,
    get_feature_names
)
from src.traits import calculate_driver_traits, save_traits, load_traits
from src.model import train_model, evaluate_model, save_model

print("Modules imported successfully!")

try:
    # Step 1: Load raw data
    print("\n[STEP 1/7] Loading raw data...")
    raw_data = load_raw_data()
    print(f"✓ Loaded {len(raw_data['results'])} race results")
    
    # Step 2: Prepare data
    print("\n[STEP 2/7] Preparing and cleaning data...")
    prepared_data = prepare_data(raw_data)
    print(f"✓ Prepared data: {len(prepared_data)} race results")
    
    # Step 3: Calculate traits
    print("\n[STEP 3/7] Calculating driver traits...")
    traits_df = calculate_driver_traits(raw_data['results'], raw_data['qualifying'], raw_data['drivers'])
    save_traits(traits_df)
    print(f"✓ Calculated traits for {len(traits_df)} drivers")
    
    # Step 4: Engineer features
    print("\n[STEP 4/7] Engineering features...")
    driver_features = engineer_driver_features(prepared_data)
    constructor_features = engineer_constructor_features(prepared_data)
    track_features = engineer_track_features(prepared_data, raw_data['races'])
    weather_features = engineer_weather_features(prepared_data)
    print(f"✓ Driver features: {driver_features.shape}")
    print(f"✓ Constructor features: {constructor_features.shape}")
    print(f"✓ Track features: {track_features.shape}")
    print(f"✓ Weather features: {weather_features.shape}")
    
    # Step 5: Build feature matrix
    print("\n[STEP 5/7] Building feature matrix...")
    X, metadata, y = build_feature_matrix(driver_features, constructor_features, track_features, prepared_data)
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target variable shape: {y.shape}")
    
    # Step 6: Split train/val/test
    print("\n[STEP 6/7] Splitting data...")
    train_mask = prepared_data['year'] <= 2022
    val_mask = prepared_data['year'] == 2023
    test_mask = prepared_data['year'] == 2024
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    print(f"✓ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Step 7: Train and evaluate
    print("\n[STEP 7/7] Training XGBoost model...")
    model = train_model(X_train, y_train, X_val, y_val)
    print("✓ Model trained!")
    
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, X_test, y_test)
    print("\n📊 TEST SET METRICS:")
    print(f"  MAE: {metrics['mae']:.2f}")
    print(f"  MSE: {metrics['mse']:.2f}")
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Within 1 position: {metrics['within_1']:.1f}%")
    print(f"  Within 2 positions: {metrics['within_2']:.1f}%")
    print(f"  Within 3 positions: {metrics['within_3']:.1f}%")
    
    print("\nSaving model...")
    save_model(model, metrics)
    print("✓ Model saved to models/model.pkl")
    
    print("\n" + "="*60)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
