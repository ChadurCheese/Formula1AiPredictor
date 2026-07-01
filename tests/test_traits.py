"""Tests for traits module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.traits import (
    _inconsistency_from_std,
    trait_inconsistent,
    trait_tire_management,
    trait_strong_starter,
    calculate_driver_traits,
    save_traits,
    load_traits,
)


def _race_rows(driver_id, year, positions, constructor_id=10):
    """Build a minimal results-style DataFrame for one driver's season."""
    return pd.DataFrame([
        {
            "raceId": 100 * year + i,
            "driverId": driver_id,
            "constructorId": constructor_id,
            "year": year,
            "round": i + 1,
            "positionOrder": pos,
            "grid": pos,
            "circuitId": (i % 3) + 1,
        }
        for i, pos in enumerate(positions)
    ])


class TestInconsistencyFromStd:
    def test_low_std_is_low_inconsistency(self):
        assert _inconsistency_from_std(1.0) == pytest.approx(0.0)

    def test_high_std_is_high_inconsistency(self):
        assert _inconsistency_from_std(7.0) == pytest.approx(1.0)

    def test_clips_beyond_ceiling(self):
        assert _inconsistency_from_std(20.0) == 1.0

    def test_nan_returns_neutral(self):
        assert _inconsistency_from_std(float("nan")) == 0.5


class TestInconsistentTraitNotFooledByLowMean:
    """
    Regression test for the CV-based bug: a dominant front-runner with a low
    mean finishing position but a small absolute std used to score as more
    "inconsistent" than a midfield driver with the same std. Raw-std scoring
    should treat them the same.
    """

    def test_same_std_different_mean_gives_same_score(self):
        # Front-runner: alternates P1/P2/P1/P3 (mean ~1.75, std ~0.96)
        dominant = _race_rows(1, 2023, [1, 2, 1, 3])
        # Midfielder: alternates P11/P12/P11/P13 (same std, much higher mean)
        midfield = _race_rows(2, 2023, [11, 12, 11, 13])

        dominant_score = trait_inconsistent(dominant)
        midfield_score = trait_inconsistent(midfield)

        assert dominant_score == pytest.approx(midfield_score, abs=1e-9)

    def test_tire_management_is_inverse_of_inconsistency(self):
        races = _race_rows(1, 2023, [1, 2, 1, 3, 2])
        assert trait_tire_management(races) == pytest.approx(1 - trait_inconsistent(races))


class TestStrongStarter:
    def test_gaining_positions_scores_above_neutral(self):
        # Always starts P10, finishes P5 -> gains 5 positions every race
        races = _race_rows(1, 2023, [5, 5, 5])
        races["grid"] = 10
        assert trait_strong_starter(races) > 0.5

    def test_losing_positions_scores_below_neutral(self):
        races = _race_rows(1, 2023, [10, 10, 10])
        races["grid"] = 3
        assert trait_strong_starter(races) < 0.5


class TestCalculateDriverTraitsRecency:
    def test_uses_recent_seasons_not_full_career(self):
        # 2015-2018: erratic (wide spread of finishes)
        more_early = pd.concat([
            _race_rows(1, y, [15, 3, 18, 1, 12]) for y in [2015, 2016, 2017, 2018]
        ])
        # 2023-2024: dominant and steady
        recent = pd.concat([
            _race_rows(1, y, [1, 2, 1, 1, 2, 1]) for y in [2023, 2024]
        ])
        results_df = pd.concat([more_early, recent], ignore_index=True)

        drivers_df = pd.DataFrame([{"driverId": 1}])
        qualifying_df = pd.DataFrame(columns=["raceId", "driverId", "position"])

        traits = calculate_driver_traits(results_df, qualifying_df, drivers_df, recent_seasons=2)
        recent_only_score = trait_inconsistent(recent)

        assert traits[1]["inconsistent"] == pytest.approx(recent_only_score)

    def test_falls_back_to_full_career_for_rookie(self):
        # Only raced in 2018 - no data in the "recent" window at all
        results_df = _race_rows(1, 2018, [5, 6, 7, 8, 9])
        drivers_df = pd.DataFrame([{"driverId": 1}])
        qualifying_df = pd.DataFrame(columns=["raceId", "driverId", "position"])

        traits = calculate_driver_traits(
            results_df, qualifying_df, drivers_df, recent_seasons=2, min_recent_races=5
        )

        assert 1 in traits  # should not be skipped just for lacking recent races


class TestSaveLoadTraits:
    def test_roundtrip(self, tmp_path):
        traits = {1: {"qualifying_specialist": 0.5, "inconsistent": 0.3}}
        path = tmp_path / "traits.json"

        save_traits(traits, output_path=path)
        loaded = load_traits(traits_path=path)

        assert loaded == traits

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert load_traits(traits_path=tmp_path / "missing.json") == {}


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
