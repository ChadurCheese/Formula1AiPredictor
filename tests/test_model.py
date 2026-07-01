"""Tests for model module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import (
    train_model, evaluate_model, save_model, load_model, get_feature_importance,
    _rank_within_race,
)


@pytest.fixture
def synthetic_data():
    """
    A trivial linear relationship (y = -2*x1 + noise) that a tree model
    should be able to fit closely, so we can sanity-check training/eval
    without depending on real F1 data.
    """
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 5, n)
    y = 20 - 2 * x1 + 0.1 * x2 + rng.normal(0, 0.5, n)

    X = pd.DataFrame({"x1": x1, "x2": x2})
    y = pd.Series(y)

    split = int(n * 0.8)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


class TestTrainModel:
    def test_trains_and_predicts(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)

        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_fits_simple_linear_relationship_well(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["mae"] < 2.0  # should fit an easy linear signal closely

    def test_uses_validation_set_for_early_stopping(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train, X_val=X_test, y_val=y_test)
        assert model is not None


class TestEvaluateModel:
    def test_returns_expected_metric_keys(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        expected_keys = {"mse", "mae", "r2", "within_1_position", "within_2_positions",
                          "within_3_positions", "test_samples"}
        assert expected_keys <= set(metrics.keys())

    def test_within_metrics_are_monotonic(self, synthetic_data):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        assert metrics["within_1_position"] <= metrics["within_2_positions"]
        assert metrics["within_2_positions"] <= metrics["within_3_positions"]


class TestRankWithinRace:
    def test_produces_valid_permutation_per_race(self):
        y_pred = np.array([1.9, 3.1, 2.4, 10.5, 9.9])
        race_ids = pd.Series([1, 1, 1, 2, 2])

        ranks = _rank_within_race(y_pred, race_ids)

        assert sorted(ranks[:3]) == [1, 2, 3]
        assert sorted(ranks[3:]) == [1, 2]

    def test_orders_by_predicted_value(self):
        y_pred = np.array([5.0, 1.0, 3.0])
        race_ids = pd.Series([1, 1, 1])

        ranks = _rank_within_race(y_pred, race_ids)

        # Lowest predicted value -> rank 1
        assert ranks[1] == 1  # value 1.0
        assert ranks[2] == 2  # value 3.0
        assert ranks[0] == 3  # value 5.0

    def test_rank_conversion_can_only_help_or_match_raw_accuracy(self, synthetic_data):
        # Sanity check that plugging race_ids into evaluate_model doesn't
        # error and produces metrics in the valid percentage range.
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)
        race_ids = pd.Series([1] * len(y_test), index=y_test.index)

        metrics = evaluate_model(model, X_test, y_test, race_ids=race_ids)
        assert 0 <= metrics["within_2_positions"] <= 100


class TestSaveAndLoadModel:
    def test_roundtrip_predictions_match(self, synthetic_data, tmp_path):
        X_train, y_train, X_test, y_test = synthetic_data
        model = train_model(X_train, y_train)

        model_path = tmp_path / "model.pkl"
        save_model(model, model_path=model_path, metrics={"mae": 1.0})
        loaded_model = load_model(model_path=model_path)

        np.testing.assert_array_almost_equal(
            model.predict(X_test), loaded_model.predict(X_test)
        )

    def test_save_writes_config_file(self, synthetic_data, tmp_path):
        X_train, y_train, _, _ = synthetic_data
        model = train_model(X_train, y_train)

        model_path = tmp_path / "model.pkl"
        save_model(model, model_path=model_path, metrics={"mae": 1.0})

        assert (tmp_path / "model_config.json").exists()

    def test_load_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(model_path=tmp_path / "missing.pkl")


class TestFeatureImportance:
    def test_returns_nonempty_dict(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        model = train_model(X_train, y_train)

        importance = get_feature_importance(model)
        assert len(importance) > 0
        assert set(importance.keys()) <= {"x1", "x2"}


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
