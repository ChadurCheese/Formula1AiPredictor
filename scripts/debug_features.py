"""
Debug feature matrix building
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_raw_data, prepare_data
from src.features import engineer_driver_features, engineer_constructor_features, engineer_track_features, engineer_weather_features

raw = load_raw_data()
prep = prepare_data(raw)

print("=" * 60)
print("PREPARED DATA")
print("=" * 60)
print(f"Shape: {prep.shape}")
print(f"Columns: {prep.columns.tolist()}")
print(f"Sample row:")
print(prep.iloc[0][['driverId', 'constructorId', 'circuitId', 'positionOrder']])

# Engineer features
print("\n" + "=" * 60)
print("DRIVER FEATURES")
print("=" * 60)
driver_feat = engineer_driver_features(prep)
print(f"Shape: {driver_feat.shape}")
print(f"Index name: {driver_feat.index.name}")
print(f"Index type: {type(driver_feat.index)}")
print(f"First few index values: {driver_feat.index.tolist()[:5]}")

print("\n" + "=" * 60)
print("CONSTRUCTOR FEATURES")
print("=" * 60)
const_feat = engineer_constructor_features(prep)
print(f"Shape: {const_feat.shape}")
print(f"Index name: {const_feat.index.name}")

print("\n" + "=" * 60)
print("TRACK FEATURES")
print("=" * 60)
track_feat = engineer_track_features(prep, raw['races'])
print(f"Shape: {track_feat.shape}")
print(f"Columns: {track_feat.columns.tolist()}")
print(f"Has driverId: {'driverId' in track_feat.columns}")
print(f"Has circuitId: {'circuitId' in track_feat.columns}")
print(f"Sample rows:")
print(track_feat.head())

print("\n" + "=" * 60)
print("MERGING TEST")
print("=" * 60)

# Test merge
X = prep.copy()
print(f"Before merge: {X.shape}")

# Merge driver features
X = X.merge(driver_feat, left_on='driverId', right_index=True, how='left')
print(f"After driver merge: {X.shape}")

# Merge constructor features  
X = X.merge(const_feat, left_on='constructorId', right_index=True, how='left')
print(f"After constructor merge: {X.shape}")

# Merge track features - this is where it fails
print(f"\nBefore track merge: {X.shape}")
print(f"Checking merge keys:")
print(f"  X driverId sample: {X['driverId'].unique()[:3]}")
print(f"  X circuitId sample: {X['circuitId'].unique()[:3]}")
print(f"  track_feat driverId sample: {track_feat['driverId'].unique()[:3]}")
print(f"  track_feat circuitId sample: {track_feat['circuitId'].unique()[:3]}")

# Check for matching keys
matching_keys = len(X.merge(track_feat, on=['driverId', 'circuitId'], how='inner'))
print(f"  Matching rows on merge: {matching_keys}")

X = X.merge(track_feat, on=['driverId', 'circuitId'], how='left')
print(f"After track merge: {X.shape}")
