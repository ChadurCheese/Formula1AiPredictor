"""Tests for the driver-traits UI component (app/components/trait_visualizer.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from components.trait_visualizer import compute_trait_baselines, trait_status, STATUS_COLORS


class TestComputeTraitBaselines:
    def test_computes_mean_and_std(self):
        all_traits = {
            1: {"qualifying_specialist": 0.2},
            2: {"qualifying_specialist": 0.4},
            3: {"qualifying_specialist": 0.6},
        }
        baselines = compute_trait_baselines(all_traits)
        assert baselines["qualifying_specialist"]["mean"] == pytest.approx(0.4)
        assert baselines["qualifying_specialist"]["std"] > 0

    def test_degenerate_std_falls_back_to_default(self):
        # Every driver has the identical score - std would be 0/NaN
        all_traits = {1: {"track_expert": 0.5}, 2: {"track_expert": 0.5}}
        baselines = compute_trait_baselines(all_traits)
        assert baselines["track_expert"]["std"] == 0.15

    def test_empty_input_returns_empty_dict(self):
        assert compute_trait_baselines({}) == {}


class TestTraitStatus:
    def test_average_score_is_average_tier(self):
        baseline = {"mean": 0.5, "std": 0.2}
        label, color = trait_status("qualifying_specialist", 0.5, baseline)
        assert label == "Average"
        assert color == STATUS_COLORS["Average"]

    def test_far_below_mean_is_bad(self):
        baseline = {"mean": 0.5, "std": 0.1}
        label, _ = trait_status("qualifying_specialist", 0.1, baseline)
        assert label == "Bad"

    def test_far_above_mean_is_superb(self):
        baseline = {"mean": 0.5, "std": 0.1}
        label, _ = trait_status("qualifying_specialist", 0.9, baseline)
        assert label == "Superb"

    def test_moderately_above_mean_is_above_average(self):
        baseline = {"mean": 0.5, "std": 0.1}
        label, _ = trait_status("qualifying_specialist", 0.64, baseline)
        assert label == "Above Average"

    def test_inconsistent_trait_is_inverted(self):
        # Same z-score magnitude, but "inconsistent" is bad when HIGH, so a
        # high score here should classify as "Bad", not "Superb".
        baseline = {"mean": 0.5, "std": 0.1}
        label, _ = trait_status("inconsistent", 0.9, baseline)
        assert label == "Bad"

        label, _ = trait_status("inconsistent", 0.1, baseline)
        assert label == "Superb"

    def test_missing_baseline_falls_back_to_neutral(self):
        label, color = trait_status("qualifying_specialist", 0.5, None)
        assert label == "Average"

    def test_all_four_tiers_are_reachable(self):
        baseline = {"mean": 0.5, "std": 0.1}
        scores = {
            "Bad": 0.2,
            "Average": 0.5,
            "Above Average": 0.64,
            "Superb": 0.9,
        }
        for expected_label, score in scores.items():
            label, _ = trait_status("qualifying_specialist", score, baseline)
            assert label == expected_label


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
