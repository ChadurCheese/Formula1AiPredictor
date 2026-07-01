# Data Sources

## Source

[Kaggle: Formula 1 World Championship (1950-2024)](https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2024),
a static CSV snapshot (not a live API). Files live in `data/raw/` and are
loaded by `src/data_pipeline.py::load_raw_data()`.

| File | Required? | Used for |
|---|---|---|
| `races.csv` | yes | Race metadata: year, round, circuit, date, name |
| `results.csv` | yes | Finishing position, grid, points, status per driver per race - the core table everything else joins against |
| `drivers.csv` | yes | Driver names, nationality |
| `constructors.csv` | yes | Constructor (team) names, nationality |
| `qualifying.csv` | yes | Qualifying position and Q1/Q2/Q3 lap times - the only pre-race-day signal used as a feature |
| `pit_stops.csv` | yes | Constructor's average pit stop duration from **prior races only** (proxy for pit crew speed) - the race's own stop count/duration would leak the outcome, so it's never used directly, only as historical team form |
| `status.csv` | optional | Human-readable finish status (`Finished`, `Retired`, `Engine`, etc.), joined onto `results` as `race_status` |
| `circuits.csv` | optional | Circuit name, location, country |

There is **no weather data** in this dataset. `trait_wet_weather_master` in
`src/traits.py` uses a hardcoded list of rain-prone circuits as a proxy -
see [TRAITS_METHODOLOGY.md](TRAITS_METHODOLOGY.md) for the caveat.

## Known data quality issues

- **Duplicate `raceId`+`driverId` rows** (~176 rows, all pre-2020): a small
  number of historical races - mostly the shared-drive era and
  disqualification/re-classification cases - have two `results.csv` rows for
  the same driver in the same race (e.g. "Did not qualify" and "Did not
  prequalify" as separate entries). `src/data_pipeline.py::prepare_data()`
  drops the duplicate, keeping the first entry, so per-driver expanding
  features aren't double-counted. This only affects historical training
  signal from decades-old races; it doesn't touch any 2020+ row used in the
  current train/val/test split.
- **Qualifying times are missing for ~1.5% of entries** (`q1`/`q2`/`q3` are
  blank for drivers who didn't set a time). `engineer_qualifying_features()`
  takes the best of the three sessions per driver; if all three are missing,
  `build_feature_matrix()` falls back to grid position as a stand-in for
  qualifying position and the dataset-wide mean gap for `qual_gap_seconds`.
- **Retired drivers accumulate in `driver_traits.json`**: `calculate_driver_traits()`
  computes traits for any driver with enough career races, including drivers
  who haven't raced in years. The Driver Traits page filters the dropdown to
  drivers active in the last 2 seasons by default, with a checkbox to show
  everyone.

## Time range actually used for modeling

The full dataset covers 1950-2024, but the model is only trained on
2020-2022 (current ground-effect regulation era), validated on 2023, and
tested on 2024 - see `models/model_config.json` for the exact split.
Feature *engineering* (expanding-window stats) still runs over the full
1950-2024 history so that 2020+ races have accurate "prior form" even for
drivers who debuted earlier, but the model itself never trains on
pre-2020 rows directly.
