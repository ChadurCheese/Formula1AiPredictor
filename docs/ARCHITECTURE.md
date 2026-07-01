# Architecture

## Overview

The system is a batch-trained regression model (XGBoost) that predicts a
driver's finishing position for a given F1 race, with SHAP-based
explanations and FM-style driver "traits" layered on top for a
human-readable narrative. There is no live/real-time component yet -
everything operates on the historical Kaggle dataset in `data/raw/`.

```
data/raw/*.csv
      |
      v
src/data_pipeline.py  --load_raw_data()--> dict of DataFrames
      |
      --prepare_data()--> merged, cleaned race-result rows (one per driver-race)
      |
      v
src/features.py
      |-- engineer_driver_features()             (expanding window, prior races only)
      |-- engineer_constructor_features()        (expanding window, prior races only)
      |-- engineer_track_features()              (expanding window, prior visits only)
      |-- engineer_qualifying_features()         (from qualifying.csv, pre-race, safe)
      |-- engineer_season_form_features()        (in-season standings so far)
      |-- engineer_pitstop_features()            (constructor pit-crew speed, prior races only)
      |-- engineer_circuit_overtaking_features() (track passing difficulty, prior races only)
      |-- build_feature_matrix()                 (merges all of the above -> X, metadata, y)
      v
src/model.py            --train_model()--> XGBRegressor
      |                 --evaluate_model()--> metrics dict (via within-race rank conversion)
      v
models/model.pkl, models/model_config.json

src/traits.py           --calculate_driver_traits()--> {driver_id: {trait: score}}
      v
data/cache/driver_traits.json

src/explainer.py        --explain_prediction()--> SHAP values + trait narrative
src/predict.py          --predict_race()--> wires model + traits + explainer together
      v
app/  (Streamlit UI, reads from src.predict)
```

## Module responsibilities

- **`src/data_pipeline.py`**: loads the raw Kaggle CSVs and merges them into
  one row per (race, driver) with driver/constructor/circuit names attached.
  Also drops a small number of duplicate `raceId+driverId` rows found in the
  raw data (shared-drive era, disqualification re-entries) - see
  [DATA_SOURCES.md](DATA_SOURCES.md).
- **`src/features.py`**: turns the merged rows into a numeric feature matrix.
  The one hard rule throughout this module is **no leakage**: every
  aggregate (driver form, constructor strength, track history, season
  standings) is computed with `shift(1)` before an expanding/rolling window,
  so a race's features only ever see races that happened strictly before it.
  Qualifying features are the one exception that doesn't need shifting,
  since qualifying genuinely happens before the race it's used to predict.
- **`src/model.py`**: a thin wrapper around `xgboost.XGBRegressor` -
  train/evaluate/save/load plus feature importance extraction. All
  hyperparameters live in `MODEL_CONFIG` in this file, not scattered
  through scripts. `evaluate_model()` can optionally convert raw predictions
  into within-race integer ranks (`_rank_within_race`) before scoring
  "within X positions" accuracy - see the note on this in the Model
  Performance section below.
- **`src/traits.py`**: computes the 6 FM-style traits per driver from the
  last 2 seasons of results (falling back to full career for drivers without
  enough recent races). See [TRAITS_METHODOLOGY.md](TRAITS_METHODOLOGY.md)
  for the reasoning behind the recency window and the scoring formulas.
- **`src/explainer.py`**: builds a SHAP `TreeExplainer` for the trained
  model and turns per-prediction SHAP values plus a driver's traits into a
  short natural-language explanation string.
- **`src/predict.py`**: the single integration point the UI talks to. It
  lazily builds the whole pipeline (model + traits + feature matrix) once
  per process via a module-level cache (`_get_pipeline()`), including a
  single shared `shap.TreeExplainer` - building a fresh explainer per
  prediction was a real performance bug caught while building the
  Historical Analysis page (a season view needs 300+ explanations).

## Streamlit app

`app/app.py` is just a landing page; Streamlit's file-based routing
auto-discovers `app/pages/01_predictions.py`, `02_driver_traits.py`, and
`03_historical_analysis.py`. Shared rendering logic (comparison tables,
trait charts, prediction cards) lives in `app/components/` so the three
pages don't duplicate formatting code.

Each page uses `st.cache_data` around calls into `src.predict`, which is
layered on top of `src.predict`'s own process-level cache - the Streamlit
cache avoids rebuilding dataframes across reruns within a session, and the
module-level cache avoids rebuilding the whole feature pipeline if multiple
pages are visited in the same process.

## Model performance and the rank-conversion step

Finishing position is an ordinal permutation (1..N within a race), not an
independent continuous value, so raw regression output is scored fairly by
converting it into within-race integer ranks before comparing against the
actual position - `src/model.py::_rank_within_race()`. This is applied
consistently in both `evaluate_model()` (for reported metrics) and
`src/predict.py::predict_race()` (for what the UI displays), so the numbers
you see in the app match the numbers in `models/model_config.json`. It also
has a nice side effect: predictions display as clean integers ("P3") instead
of raw regression output ("P3.3").

Current honest, leakage-free test-set (2024) performance: MAE ~3.0 positions,
~50% within 2 positions, ~67% within 3 positions. See
`models/model_config.json` for the exact current numbers and
[DATA_SOURCES.md](DATA_SOURCES.md) for what data is and isn't available to
the model.

## Known limitations

- Training/inference is all offline batch processing against a static
  Kaggle CSV snapshot - no live timing data, no scheduled retraining.
- No weather, tire-compound, or safety-car data is available in this
  dataset, which likely caps how much further pre-race-only features can
  close the gap to the original 65%-within-2 aspiration - see
  [DATA_SOURCES.md](DATA_SOURCES.md).
