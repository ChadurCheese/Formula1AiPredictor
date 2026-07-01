"""
Tests for prediction API.

These are integration tests against the real trained model and cached data
(models/model.pkl, data/raw/*.csv) since predict.py's whole job is wiring
those real artifacts together. They're skipped if the model hasn't been
trained yet (e.g. a fresh checkout before running scripts/train_model.py).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_PATH = Path(__file__).parent.parent / "models" / "model.pkl"

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="models/model.pkl not found - run scripts/train_model.py first",
)


@pytest.fixture(scope="module")
def sample_race_id():
    from src.predict import get_available_races
    races = get_available_races(years=[2024])
    assert not races.empty, "expected at least one predictable 2024 race"
    return int(races.iloc[0]["raceId"])


class TestPredictRace:
    def test_returns_prediction_per_driver(self, sample_race_id):
        from src.predict import predict_race
        predictions = predict_race(sample_race_id)

        assert len(predictions) > 0
        for p in predictions:
            assert {"driver_id", "driver_name", "predicted_position", "confidence",
                     "trait_influences", "explanation", "actual_position"} <= set(p.keys())

    def test_sorted_by_predicted_position(self, sample_race_id):
        from src.predict import predict_race
        predictions = predict_race(sample_race_id)

        positions = [p["predicted_position"] for p in predictions]
        assert positions == sorted(positions)

    def test_confidence_in_valid_range(self, sample_race_id):
        from src.predict import predict_race
        predictions = predict_race(sample_race_id)

        for p in predictions:
            assert 0 <= p["confidence"] <= 1

    def test_unknown_race_returns_empty_list(self):
        from src.predict import predict_race
        assert predict_race(race_id=-1) == []


class TestPredictSingleDriver:
    def test_matches_entry_in_full_race_prediction(self, sample_race_id):
        from src.predict import predict_race, predict_single_driver
        race_predictions = predict_race(sample_race_id)
        target_driver_id = race_predictions[0]["driver_id"]

        single = predict_single_driver(sample_race_id, target_driver_id)

        assert single is not None
        assert single["driver_id"] == target_driver_id
        assert single["predicted_position"] == race_predictions[0]["predicted_position"]

    def test_unknown_driver_returns_none(self, sample_race_id):
        from src.predict import predict_single_driver
        assert predict_single_driver(sample_race_id, driver_id=-1) is None


class TestPredictionsCache:
    def test_save_and_load_roundtrip(self, tmp_path):
        from src.predict import save_predictions_to_cache, load_predictions_from_cache
        predictions = {1: [{"driver_id": 1, "predicted_position": 3.2}]}
        cache_path = tmp_path / "predictions_cache.pkl"

        save_predictions_to_cache(predictions, cache_path=cache_path)
        loaded = load_predictions_from_cache(cache_path=cache_path)

        assert loaded == predictions

    def test_load_missing_cache_returns_empty_dict(self, tmp_path):
        from src.predict import load_predictions_from_cache
        assert load_predictions_from_cache(cache_path=tmp_path / "missing.pkl") == {}


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
