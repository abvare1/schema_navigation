"""Tests for the statistical and metric helpers.

These functions produce every number quoted in the README, so their edge
cases are pinned here: the Wilson interval must stay inside [0, 1] (in
particular it must never report a negative-zero lower bound), bootstrap
intervals must be reproducible and contain the observed mean, Spearman must
degrade to NaN instead of crashing, and the metric aggregation must never
drop failed runs.
"""

import math

import numpy as np
import pytest

from analysis import metrics as M
from analysis import statistics as S


# ---------------------------------------------------------------------------
# Wilson score interval
# ---------------------------------------------------------------------------
def test_wilson_zero_successes_lower_bound_is_not_negative_zero():
    """Regression test: at 0/30 the floating-point evaluation could yield a
    lower bound of -0.0 (IEEE negative zero), which leaked into the summary
    CSVs as '-0.0'.  The clamped implementation must return positive 0.0."""
    lo, hi = S.wilson_ci(0, 30)
    assert lo == 0.0
    assert math.copysign(1.0, lo) == 1.0          # positive zero, not -0.0
    assert round(lo, 3) == 0.0
    assert str(round(lo, 3)) != "-0.0"
    assert 0.0 < hi < 1.0


def test_wilson_all_successes_upper_bound_is_one():
    lo, hi = S.wilson_ci(30, 30)
    assert hi == 1.0
    assert 0.0 < lo < 1.0


def test_wilson_matches_documented_worked_example():
    # docs/mathematical_background.md section 8: 20/30 -> [0.488, 0.808]
    lo, hi = S.wilson_ci(20, 30)
    assert lo == pytest.approx(0.488, abs=1e-3)
    assert hi == pytest.approx(0.808, abs=1e-3)


def test_wilson_always_inside_unit_interval_and_contains_phat():
    for total in (1, 2, 5, 30, 100):
        for k in range(total + 1):
            lo, hi = S.wilson_ci(k, total)
            p = k / total
            assert 0.0 <= lo <= p <= hi <= 1.0


def test_wilson_invalid_inputs():
    lo, hi = S.wilson_ci(0, 0)
    assert math.isnan(lo) and math.isnan(hi)
    with pytest.raises(ValueError):
        S.wilson_ci(5, 10, ci_level=0.90)


# ---------------------------------------------------------------------------
# Percentile bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_ci_is_deterministic_and_contains_mean():
    values = [10.0, 12.0, 9.0, 40.0, 55.0, 8.0, 12.0, 300.0]
    lo1, hi1 = S.bootstrap_ci_mean(values)
    lo2, hi2 = S.bootstrap_ci_mean(values)
    assert (lo1, hi1) == (lo2, hi2)               # fixed seed -> identical
    observed = float(np.mean(values))
    assert lo1 <= observed <= hi1
    # a heavily skewed sample must give an asymmetric (honest) interval
    assert observed - lo1 != hi1 - observed


def test_bootstrap_ci_edge_cases():
    lo, hi = S.bootstrap_ci_mean([])
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = S.bootstrap_ci_mean([42.0])
    assert lo == pytest.approx(42.0) and hi == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Spearman rank correlation
# ---------------------------------------------------------------------------
def test_spearman_perfect_and_inverse():
    x = [1, 2, 3, 4, 5]
    assert S.spearman_rho(x, [10, 20, 30, 40, 50]) == pytest.approx(1.0)
    assert S.spearman_rho(x, [50, 40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_handles_ties_with_average_ranks():
    # ranks of x = [1, 2.5, 2.5, 4]; rho = 4.5 / sqrt(4.5 * 5) ~= 0.9487
    rho = S.spearman_rho([1, 2, 2, 3], [10, 20, 30, 40])
    assert rho == pytest.approx(0.9486832980505138)


def test_spearman_degrades_to_nan():
    assert math.isnan(S.spearman_rho([1], [2]))                    # too short
    assert math.isnan(S.spearman_rho([1, 2], [1]))                 # length mismatch
    assert math.isnan(S.spearman_rho([7, 7, 7], [1, 2, 3]))        # constant x
    assert math.isnan(S.spearman_rho([1, 2, 3], [5, 5, 5]))        # constant y


# ---------------------------------------------------------------------------
# Metric aggregation (analysis/metrics.py)
# ---------------------------------------------------------------------------
def _log(steps, completed, completion_step=None, reveals=None, bfs=0):
    return {
        "steps": steps,
        "completed": completed,
        "completion_step": completion_step,
        "reward": 0.0,
        "first_reveals": reveals or {},
        "bfs_calls": bfs,
    }


def test_summarize_run_failure_is_recorded_not_dropped():
    logs = [_log(400, False), _log(400, False)]
    s = M.summarize_run(logs)
    assert s["success"] is False
    assert s["first_cycle_steps"] is None
    assert s["subsequent_cycle_mean"] is None
    assert s["n_subsequent_cycles"] == 0
    assert s["total_steps_used"] == 800
    assert s["distinct_observed_before_first_cycle"] is None
    assert s["delay_obsD_to_completion"] is None


def test_summarize_run_first_cycle_cumulative_across_episodes():
    # episode 1 fails after 100 steps; episode 2 completes at its step 50
    logs = [_log(100, False), _log(50, True, 50,
                                   reveals={"D": 20}, bfs=3)]
    s = M.summarize_run(logs)
    assert s["success"] is True
    assert s["first_cycle_steps"] == 150          # 100 + 50, cumulative
    assert s["bfs_calls_at_first_cycle"] == 3
    assert s["delay_obsD_to_completion"] == 150 - (100 + 20)
    assert s["distinct_observed_before_first_cycle"] == 1


def test_summarize_run_distinct_window_is_respected():
    # D revealed only AFTER the first cycle completed: must not be counted
    logs = [_log(30, True, 30, reveals={"A": 5}),      # cycle 1 at abs 30
            _log(40, True, 40, reveals={"D": 10})]     # cycle 2 at abs 70
    s = M.summarize_run(logs)
    assert s["first_cycle_steps"] == 30
    assert s["distinct_observed_before_first_cycle"] == 1   # A only
    assert s["delay_obsD_to_completion"] is None            # D unknown by then
    assert s["n_subsequent_cycles"] == 1
    assert s["subsequent_cycle_mean"] == 40.0


def test_summarize_run_reveals_carry_across_episodes():
    # reveals persist: D first seen in episode 1 still counts later
    logs = [_log(10, False, reveals={"D": 4}),
            _log(60, True, 60)]
    s = M.summarize_run(logs)
    assert s["delay_obsD_to_completion"] == (10 + 60) - 4
    assert s["distinct_observed_before_first_cycle"] == 1


def test_learning_curve_rows_censor_failures_at_cap():
    logs = [_log(37, True, 37), _log(400, False)]
    rows = M.learning_curve_rows("agent", 0, logs, max_steps=400)
    assert rows[0]["steps_censored"] == 37
    assert rows[0]["completed"] is True
    assert rows[1]["steps_censored"] == 400       # right-censored at the cap
    assert rows[1]["completed"] is False
    assert [r["episode"] for r in rows] == [1, 2]


def test_mean_steps_per_completed_cycle():
    logs = [_log(400, False), _log(30, True, 30), _log(50, True, 50),
            _log(70, True, 70)]
    assert M.mean_steps_per_completed_cycle(logs) == pytest.approx(50.0)
    assert M.mean_steps_per_completed_cycle(logs, last_n=2) == pytest.approx(60.0)
    assert M.mean_steps_per_completed_cycle([_log(400, False)]) is None
