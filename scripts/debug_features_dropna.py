"""
Debug feature matrix dropna issue
"""
import sys
from pathlib import Path
import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.DEBUG)

from src.data_pipeline import load_raw_data, prepare_data
from src.features import engineer_driver_features, engineer_constructor_features, engineer_track_features, engineer_weather_features

raw = load_raw_data()
prep = prepare_data(raw)

# Engineer features
driver_feat = engineer_driver_features(prep)
const_feat = engineer_constructor_features(prep)
track_feat = engineer_track_features(prep, raw['races'])
weather_feat = engineer_weather_features(prep)

print("Starting merge sequence...")
X = prep.copy()

# Merge driver features
X = X.merge(driver_feat, left_on='driverId', right_index=True, how='left')
print(f"After driver merge: {X.shape}")

# Merge constructor features
X = X.merge(const_feat, left_on='constructorId', right_index=True, how='left')
print(f"After constructor merge: {X.shape}")

# Merge track features
X = X.merge(track_feat, on=['driverId', 'circuitId'], how='left')
print(f"After track merge: {X.shape}")

# Extract target and metadata
y = X['positionOrder'].astype(float)
metadata = X[['raceId', 'driverId', 'constructorId', 'year', 'round']].copy()

# Select feature columns
feature_cols = [col for col in X.columns if col not in [
    'raceId', 'driverId', 'constructorId', 'year', 'round',
    'positionOrder', 'race_name', 'date', 'circuitId',
    'surname', 'forename', 'nationality_driver',
    'name_team', 'nationality_constructor',
    'name_circuit', 'location', 'country', 'grid', 'points',
    'statusId', 'race_status',
    # Additional columns to exclude
    'resultId', 'number', 'position', 'positionText', 'laps',
    'time', 'milliseconds', 'fastestLap', 'rank', 'fastestLapTime',
    'fastestLapSpeed'
]]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")

X_features = X[feature_cols].copy()
print(f"X_features shape before processing: {X_features.shape}")
print(f"X_features dtypes:\n{X_features.dtypes}")
print(f"\nX_features samples (first 5 rows, first 5 cols):")
print(X_features.iloc[:5, :5])

# Check for NaNs before coercion
print(f"\nNaN counts before coercion:")
nan_counts = X_features.isna().sum()
print(nan_counts[nan_counts > 0])

# Convert to numeric
for col in X_features.columns:
    X_features[col] = pd.to_numeric(X_features[col], errors='coerce')

print(f"\nX_features dtypes after coercion:")
print(X_features.dtypes)

# Check for NaNs after coercion
print(f"\nNaN counts after coercion:")
nan_counts = X_features.isna().sum()
print(f"Total NaNs per column:")
print(nan_counts)

# Fill NaNs
for col in X_features.columns:
    if X_features[col].dtype in ['float64', 'int64']:
        col_mean = X_features[col].mean()
        if pd.notna(col_mean):
            X_features[col] = X_features[col].fillna(col_mean)

print(f"\nNaN counts after filling:")
print(X_features.isna().sum())

# Final dropna
print(f"\nShape before final dropna: {X_features.shape}")
X_features_final = X_features.dropna()
print(f"Shape after final dropna: {X_features_final.shape}")

if X_features_final.shape[0] == 0:
    print("\n❌ ERROR: All rows were dropped!")
    print("Investigating row-by-row:")
    for idx in range(min(5, len(X_features))):
        row_has_nan = X_features.iloc[idx].isna().any()
        print(f"Row {idx}: has NaN = {row_has_nan}")
        if row_has_nan:
            print(f"  NaN columns: {X_features.iloc[idx][X_features.iloc[idx].isna()].index.tolist()}")
else:
    print(f"\n✓ Feature matrix ready: {X_features_final.shape}")
