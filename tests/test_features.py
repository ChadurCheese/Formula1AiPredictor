"""Tests for features module."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import (
    _lap_time_to_seconds,
    engineer_qualifying_features,
    engineer_driver_features,
    engineer_constructor_features,
    engineer_track_features,
    build_feature_matrix,
)


@pytest.fixture
def results_df():
    """
    Two drivers (1, 2) on the same constructor (10) across three races in
    2024, plus a third driver (3) on a different constructor (20).
    Driver 1 improves each race; driver 2 is flat; driver 3 is a one-off
    to exercise the "different team" and "different circuit" paths.
    """
    return pd.DataFrame([
        # raceId, driverId, constructorId, year, round, positionOrder, points, grid, circuitId, race_status
        [100, 1, 10, 2024, 1, 5, 10, 5, 1, "Finished"],
        [100, 2, 10, 2024, 1, 8, 4, 8, 1, "Finished"],
        [100, 3, 20, 2024, 1, 3, 15, 3, 1, "Finished"],
        [101, 1, 10, 2024, 2, 3, 15, 4, 2, "Finished"],
        [101, 2, 10, 2024, 2, 8, 4, 9, 2, "Retired"],
        [101, 3, 20, 2024, 2, 4, 12, 2, 2, "Finished"],
        [102, 1, 10, 2024, 3, 1, 25, 2, 1, "Finished"],
        [102, 2, 10, 2024, 3, 7, 6, 7, 1, "Finished"],
        [102, 3, 20, 2024, 3, 5, 10, 1, 3, "Finished"],
    ], columns=[
        "raceId", "driverId", "constructorId", "year", "round",
        "positionOrder", "points", "grid", "circuitId", "race_status",
    ])


@pytest.fixture
def races_df():
    return pd.DataFrame([
        [100, 1],
        [101, 2],
        [102, 1],
    ], columns=["raceId", "circuitId"])


class TestLapTimeToSeconds:
    def test_parses_minutes_seconds(self):
        assert _lap_time_to_seconds("1:26.572") == pytest.approx(86.572)

    def test_handles_missing_value(self):
        assert np.isnan(_lap_time_to_seconds(np.nan))

    def test_handles_non_time_string(self):
        assert np.isnan(_lap_time_to_seconds("N/A"))


class TestQualifyingFeatures:
    def test_pole_sitter_has_zero_gap(self):
        qualifying_df = pd.DataFrame([
            [1, 100, 1, 1, "1:25.000", None, None],
            [2, 100, 2, 2, "1:26.500", None, None],
        ], columns=["qualifyId", "raceId", "driverId", "position", "q1", "q2", "q3"])

        result = engineer_qualifying_features(qualifying_df)

        pole = result[result["driverId"] == 1].iloc[0]
        other = result[result["driverId"] == 2].iloc[0]
        assert pole["qual_gap_seconds"] == pytest.approx(0.0)
        assert other["qual_gap_seconds"] == pytest.approx(1.5)
        assert pole["qual_position"] == 1


class TestDriverFeatures:
    def test_first_race_has_no_prior_history(self, results_df):
        features = engineer_driver_features(results_df)
        first_race = features[(features["driverId"] == 1) & (features["raceId"] == 100)].iloc[0]

        assert first_race["races_completed"] == 0
        assert pd.isna(first_race["avg_finish_position"])

    def test_second_race_reflects_only_prior_race(self, results_df):
        features = engineer_driver_features(results_df)
        second_race = features[(features["driverId"] == 1) & (features["raceId"] == 101)].iloc[0]

        # Driver 1 finished P5 in race 100 - that's the only prior data point
        assert second_race["races_completed"] == 1
        assert second_race["avg_finish_position"] == pytest.approx(5.0)

    def test_does_not_leak_future_races(self, results_df):
        features = engineer_driver_features(results_df)
        first_race = features[(features["driverId"] == 1) & (features["raceId"] == 100)].iloc[0]

        # Driver 1's later dominant results (P3, P1) must not appear in the
        # very first race's features
        assert pd.isna(first_race["avg_finish_position"])
        assert first_race["races_completed"] == 0

    def test_dnf_rate_counts_retirements(self, results_df):
        features = engineer_driver_features(results_df)
        third_race = features[(features["driverId"] == 2) & (features["raceId"] == 102)].iloc[0]

        # Driver 2 retired in race 101 (their 2nd race), so by race 102
        # exactly one of their two prior races was a DNF
        assert third_race["dnf_rate"] == pytest.approx(0.5)


class TestConstructorFeatures:
    def test_one_row_per_constructor_race(self, results_df):
        features = engineer_constructor_features(results_df)
        unique_pairs = results_df[["constructorId", "raceId"]].drop_duplicates()

        assert len(features) == len(unique_pairs)

    def test_averages_both_teammates_prior_race(self, results_df):
        features = engineer_constructor_features(results_df)
        # Constructor 10's race 101 stats should reflect race 100's average
        # of driver 1 (P5) and driver 2 (P8) = 6.5
        row = features[(features["constructorId"] == 10) & (features["raceId"] == 101)].iloc[0]
        assert row["avg_constructor_position"] == pytest.approx(6.5)


class TestTrackFeatures:
    def test_first_visit_has_no_history(self, results_df, races_df):
        features = engineer_track_features(results_df, races_df)
        first_visit = features[(features["driverId"] == 1) & (features["raceId"] == 100)].iloc[0]

        assert first_visit["track_familiarity"] == 0
        assert pd.isna(first_visit["track_avg_finish"])

    def test_familiarity_increments_on_revisit(self, results_df, races_df):
        features = engineer_track_features(results_df, races_df)
        # Driver 1 races at circuit 1 in race 100 and again in race 102
        revisit = features[(features["driverId"] == 1) & (features["raceId"] == 102)].iloc[0]

        assert revisit["track_familiarity"] == 1
        assert revisit["track_avg_finish"] == pytest.approx(5.0)  # from race 100


class TestBuildFeatureMatrix:
    def test_excludes_leakage_columns(self, results_df, races_df):
        driver_features = engineer_driver_features(results_df)
        constructor_features = engineer_constructor_features(results_df)
        track_features = engineer_track_features(results_df, races_df)

        X, metadata, y = build_feature_matrix(driver_features, constructor_features, track_features, results_df)

        assert "points" not in X.columns
        assert "positionOrder" not in X.columns

    def test_preserves_row_count(self, results_df, races_df):
        driver_features = engineer_driver_features(results_df)
        constructor_features = engineer_constructor_features(results_df)
        track_features = engineer_track_features(results_df, races_df)

        X, metadata, y = build_feature_matrix(driver_features, constructor_features, track_features, results_df)

        assert len(X) == len(results_df)
        assert len(metadata) == len(results_df)
        assert len(y) == len(results_df)

    def test_no_nan_in_output(self, results_df, races_df):
        driver_features = engineer_driver_features(results_df)
        constructor_features = engineer_constructor_features(results_df)
        track_features = engineer_track_features(results_df, races_df)

        X, _, _ = build_feature_matrix(driver_features, constructor_features, track_features, results_df)

        assert not X.isna().any().any()


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
