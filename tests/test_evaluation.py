"""Scoring must not confuse travel length, reset state, or lateral misses with arrival."""

import math

import numpy as np
import pytest

from mars_project.evaluation import classify_sample, paired_start, route_rotation, summarize


def test_goal_is_a_circle_and_fall_precedes_success():
    def score(xy, height=0.9, up=1, seconds=10, inside=True):
        return classify_sample(xy, [5, 0], height, up, seconds, inside)

    assert score([4.5, 0]) == "success"
    assert score([5, 0.51]) is None
    assert score([0, 5]) is None
    assert score([5, 0], height=0.54) == "fall"
    assert score([5, 0], up=0.49) == "fall"
    assert score([5, 0], inside=False) == "out_of_bounds"
    assert score([4.5, 0], seconds=30) == "success"
    assert score([4.49, 0], seconds=30) == "timeout"
    assert score([math.nan, 0]) == "invalid_state"


def test_seeded_starts_are_paired_and_route_rotation_preserves_distances():
    assert paired_start(42) == paired_start(42)
    assert paired_start(42) != paired_start(43)
    delta = np.array([0.03, -0.02])
    rotation = route_rotation(-math.pi / 2)
    assert np.allclose((rotation @ delta) @ rotation, delta)
    assert np.linalg.norm(rotation @ delta) == pytest.approx(np.linalg.norm(delta))


def test_summary_separates_failures_from_arrival_times():
    results = [
        dict(outcome="success", duration_s=10, max_abs_lateral_m=0.1),
        dict(outcome="fall", duration_s=2, max_abs_lateral_m=0.2),
    ]
    summary = summarize(results)
    assert summary["success_rate"] == 0.5
    assert summary["successful_arrival_mean_s"] == 10
    assert summary["outcomes"]["fall"] == 1
    assert summary["success_rate_wilson_95"][0] < 0.5 < summary["success_rate_wilson_95"][1]
