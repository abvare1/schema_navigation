"""Reproducibility tests: fixed seeds must fully determine outcomes."""

import numpy as np
import pandas as pd
import pytest

import config as C
from environment.grid_world import GridWorld, generate_layout
from experiments import main_transfer
from experiments import run_utils as R


def test_same_seed_same_layout_and_same_training_trajectory():
    # identical layouts
    l1, l2 = generate_layout(99), generate_layout(99)
    assert l1["obstacles"] == l2["obstacles"]
    assert l1["targets"] == l2["targets"]

    # identical Q-learning trajectories: two fresh agents, same env, same rng
    task = C.DEFAULT_TASK
    env = GridWorld(l1, task)
    a1 = R.fresh_q_agent(task, epsilon_start=1.0)
    a2 = R.fresh_q_agent(task, epsilon_start=1.0)
    logs1, logs2 = [], []
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    for _ep in range(5):
        logs1.append(R.run_q_episode(env, a1, rng1, 300, training=True))
        logs2.append(R.run_q_episode(GridWorld(l2, task), a2, rng2, 300,
                                     training=True))
    assert np.array_equal(a1.q, a2.q)
    assert logs1 == logs2


def test_same_seed_same_schema_pipeline():
    layout = generate_layout(42)
    task = C.DEFAULT_TASK
    s1, logs1, _ = R.train_schema_source(layout, task, 6, 2, 400, 5)
    s2, logs2, _ = R.train_schema_source(layout, task, 6, 2, 400, 5)
    assert s1.counts == s2.counts
    assert s1.memory == s2.memory
    assert s1.schema == s2.schema
    assert logs1 == logs2


def test_mini_experiment_runs_twice_identically():
    """The whole Experiment A pipeline, in miniature, must be a pure function
    of the seeds (this is the project's core reproducibility guarantee)."""
    params = R.effective_params(smoke=True, n_runs=2)
    per_run_a, _curves_a, summary_a = main_transfer.run_experiment(
        params, results_dir="/tmp/schema_nav_test_out_a")
    per_run_b, _curves_b, summary_b = main_transfer.run_experiment(
        params, results_dir="/tmp/schema_nav_test_out_b")
    # Wall-clock columns are machine-dependent by nature; every
    # environment-step metric must be EXACTLY identical.
    timing_cols = ["q_source_wall_s", "schema_source_wall_s"]
    pd.testing.assert_frame_equal(per_run_a.drop(columns=timing_cols),
                                  per_run_b.drop(columns=timing_cols))
    pd.testing.assert_frame_equal(summary_a, summary_b)


def test_smoke_pipeline_all_experiments_run():
    """Smoke mode must execute every experiment end to end without errors."""
    from experiments import (ablation, graph_similarity, incongruent_transfer,
                             obstacle_info_control)
    params = R.effective_params(smoke=True, n_runs=2)
    for module in (main_transfer, ablation, incongruent_transfer,
                   graph_similarity, obstacle_info_control):
        per_run, curves, summary = module.run_experiment(
            params, results_dir="/tmp/schema_nav_test_out_all")
        assert len(per_run) > 0
        assert len(summary) > 0
        assert len(curves) > 0


def test_mini_control_experiment_runs_twice_identically():
    """Experiment E (obstacle-information control) must also be a pure
    function of the seeds."""
    from experiments import obstacle_info_control
    params = R.effective_params(smoke=True, n_runs=2)
    per_run_a, _curves_a, summary_a = obstacle_info_control.run_experiment(
        params, results_dir="/tmp/schema_nav_test_out_e1")
    per_run_b, _curves_b, summary_b = obstacle_info_control.run_experiment(
        params, results_dir="/tmp/schema_nav_test_out_e2")
    timing_cols = ["q_source_wall_s", "q_source_masked_wall_s"]
    pd.testing.assert_frame_equal(per_run_a.drop(columns=timing_cols),
                                  per_run_b.drop(columns=timing_cols))
    pd.testing.assert_frame_equal(summary_a, summary_b)


def test_control_uses_same_source_table_as_original_protocol():
    """q_transferred_masked must start from the EXACT Q-table the original
    Experiment A protocol produces (same seeds, unmasked source training).
    This is what makes the expA-vs-expE comparison a clean single-change
    contrast."""
    layout = R.generate_layout(C.SEED_SOURCE_BASE + 0)
    agent_seed = C.SEED_AGENT_BASE + 0
    a1, _logs1, _ = R.train_q_source(layout, C.DEFAULT_TASK,
                                     20, 400, agent_seed)
    a2, _logs2, _ = R.train_q_source(layout, C.DEFAULT_TASK,
                                     20, 400, agent_seed)
    assert np.array_equal(a1.q, a2.q)
