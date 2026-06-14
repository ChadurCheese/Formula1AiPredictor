"""
Debug build_feature_matrix directly
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import load_raw_data, prepare_data
from src.features import engineer_driver_features, engineer_constructor_features, engineer_track_features, engineer_weather_features, build_feature_matrix

raw = load_raw_data()
prep = prepare_data(raw)

# Engineer features
driver_feat = engineer_driver_features(prep)
const_feat = engineer_constructor_features(prep)
track_feat = engineer_track_features(prep, raw['races'])

print("About to call build_feature_matrix...")
print(f"  driver_feat shape: {driver_feat.shape}")
print(f"  const_feat shape: {const_feat.shape}")
print(f"  track_feat shape: {track_feat.shape}")
print(f"  prep shape: {prep.shape}")

# Call the function
X, metadata, y = build_feature_matrix(driver_feat, const_feat, track_feat, prep)

print(f"\nAfter build_feature_matrix:")
print(f"  X shape: {X.shape}")
print(f"  metadata shape: {metadata.shape}")
print(f"  y shape: {y.shape}")
print(f"  X type: {type(X)}")
print(f"  y type: {type(y)}")

if X.shape[0] > 0:
    print(f"\n✓ Success!")
    print(f"First row: {X.iloc[0][:5]}")
else:
    print(f"\n❌ ERROR: X has {X.shape[0]} rows")
    print(f"X columns: {X.columns.tolist()}")
