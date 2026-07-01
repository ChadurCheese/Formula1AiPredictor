# Future Improvements

This is a working MVP (Days 1-10 of the original build plan are complete: data
pipeline, feature engineering, XGBoost model with SHAP explanations, FM-style
driver traits, a 3-page Streamlit app, tests, docs, and a one-command
quickstart script). This doc captures concrete next steps identified during
that build, grounded in the actual gaps found - not a generic wishlist.

## Model accuracy

Current honest baseline: **MAE ~3.0 positions, ~50% within 2 positions**
(2024 test set) - see `models/model_config.json` and
[TRAITS_METHODOLOGY.md](TRAITS_METHODOLOGY.md) /
[ARCHITECTURE.md](ARCHITECTURE.md) for how that number was arrived at.
Hyperparameter tuning is already near its ceiling for the current feature
set. The real next gains likely require data this project doesn't have:

- **Weather data**: no weather column exists in the Kaggle dataset;
  `wet_weather_master` currently uses a hardcoded list of rain-prone
  circuits as a crude proxy. A historical weather API (e.g. Open-Meteo,
  Visual Crossing) joined on race date/location would let this be a real
  feature instead of a guess.
- **Tire compound/strategy data**: not in the current dataset. Would enable
  genuine strategy-aware features (though care is needed - in-race tire
  choices are themselves outcomes of the race, so only pre-race strategy
  *plans* or team-level historical tendencies would be safe to use without
  leakage, following the same pattern as `engineer_pitstop_features`).
- **Safety car / incident history**: circuit-level safety car frequency
  would help explain variance that's currently just noise to the model.
- **Learning-to-rank**: an `xgb.XGBRanker` with `rank:pairwise` was tried
  during Day 8 and roughly matched (didn't beat) regression + within-race
  rank conversion. Worth revisiting with the richer feature set now
  available, or trying `rank:ndcg` with position-based relevance weighting.
- **Hyperparameter search**: current tuning was a manual grid search over a
  handful of values. A proper search (Optuna, or even `RandomizedSearchCV`)
  over a wider space might find a meaningfully better configuration.

## Data pipeline

- **Automated data refresh**: `data/raw/*.csv` is a static Kaggle snapshot
  committed to the repo. A script to pull the latest season's results
  (Kaggle API or the Ergast/F1 API) and append them, then retrain, would
  keep predictions current without manual CSV replacement.
- **Predicting future (not-yet-run) races**: the app currently only
  "predicts" races that have already happened (useful for evaluating the
  model, less useful as a forecasting tool). Once a race weekend's
  qualifying is complete but the race hasn't run, the same feature pipeline
  could produce a genuine pre-race prediction - this would need a small
  change to `src/predict.py` to accept a race with qualifying data but no
  `positionOrder` yet.
- **On-disk prediction caching**: `src/predict.py` already has
  `save_predictions_to_cache()` / `load_predictions_from_cache()` but the
  app never calls them - it relies entirely on the in-memory pipeline cache,
  which is rebuilt from scratch (~10-25s) every time the Streamlit process
  restarts. Wiring up the disk cache would make cold starts much faster.

## Testing & quality

- **App-layer tests**: the 52 existing tests cover `src/` thoroughly and
  `app/*.py` is verified via `streamlit.testing.v1.AppTest` ad hoc during
  development, but there's no permanent AppTest suite in `tests/` - only
  the CI-tracked `pytest` suite runs automatically. Adding AppTest-based
  tests for the three pages would let CI catch UI regressions too.
- **CI smoke test for `run.py`**: `.github/workflows/tests.yml` runs
  `pytest` but doesn't exercise the actual `python run.py` quickstart path.
  A CI job that runs `run.py --skip-install` against the committed data and
  confirms it doesn't error would catch regressions in the setup script
  itself.
- **Static typing**: functions have type hints throughout but nothing
  enforces them. Adding `mypy` (already listed as a natural companion to
  the existing `black`/`pylint` dev dependencies) would catch a class of
  bugs before runtime.

## Deployment

Day 10 deliberately scoped "deployment" as *local* quickstart
(`python run.py`), not hosting. Explicitly deferred:

- **Actual hosting**: Streamlit Community Cloud is the lowest-effort option
  given no backend beyond the Streamlit app itself; a container (Dockerfile
  + any cloud provider) is the more portable option if this grows beyond a
  single Streamlit process.
- **Scheduled retraining**: a GitHub Actions cron job to retrain after each
  race weekend (once automated data refresh above exists) rather than
  requiring a manual `python run.py --retrain`.
- **Model/experiment tracking**: right now, retraining just overwrites
  `models/model.pkl` and `model_config.json` in place, so there's no record
  of how accuracy has changed across changes. Something lightweight like
  MLflow (local file-based backend, no server needed) would preserve that
  history.

## UI / UX

Most of the Day 9 priority list is done (trait colors, error-coded
comparison tables, accuracy captions, radar overlays, team colors,
confidence badges, cross-season leaderboard). Remaining ideas that came up
but weren't pursued:

- **Dark/light theme toggle**: `app/config.py` has a `THEME` constant that
  isn't actually wired to anything - Streamlit's native theming would need
  to be configured via `.streamlit/config.toml` (gitignored) or a
  runtime toggle using `st.session_state`.
- **Driver headshots / team logos**: would need a licensed image source;
  out of scope without one.
- **Export predictions**: a "download as CSV" button on the Predictions
  page for anyone who wants to take the data elsewhere.
