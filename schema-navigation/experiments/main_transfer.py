"""Experiment A: main transfer experiment.

Design (full details in docs/methodology.md):
  * N_RUNS independent runs (30 in full mode).  Run r uses
      - a source environment with layout seed SEED_SOURCE_BASE + r,
      - a NEW target environment with layout seed SEED_TARGET_BASE + r,
      - the SAME abstract task (DEFAULT_SEQUENCE cycle) in both,
      - one shared agent exploration seed SEED_AGENT_BASE + r for all agents.
  * Agents compared (all under the SAME target-env interaction budget):
      q_fresh        -- tabular Q-learning, empty Q-table in the new env.
                        By definition it does not use source experience, so
                        it receives no source training (its "prior" is nil;
                        this is the point of the baseline).
      q_transferred  -- same learner, initialized with the FULL Q-table
                        trained in the source environment (epsilon restarts
                        lower because it is not naive; see methodology).
      schema         -- schema-based agent; carries ONLY the learned task
                        schema into the new env; spatial memory starts empty.
  * oracle_planner -- upper-bound reference, reported separately.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C
from analysis import metrics as M
from analysis import plots as P
from analysis import statistics as S
from environment.grid_world import generate_layout
from experiments import run_utils as R

AGENTS = ["q_fresh", "q_transferred", "schema"]


def run_three_agent_run(run_idx, params, target_task, source_task=None):
    """Train the agents in the source env and evaluate all three in one
    target environment.  Shared by Experiment A (congruent task, source and
    target use the same task graph) and Experiment C (incongruent transfer:
    SOURCE task is always the default cycle; only the TARGET task changes).
    Returns (rows, curve_rows, extras)."""
    if source_task is None:
        source_task = target_task
    src_layout = generate_layout(C.SEED_SOURCE_BASE + run_idx)
    tgt_layout = generate_layout(C.SEED_TARGET_BASE + run_idx)
    agent_seed = C.SEED_AGENT_BASE + run_idx

    # ---- source-environment training (always the SOURCE task) ------------
    # Q agent that will be transferred (q_fresh needs no source training).
    q_src, q_src_logs, q_wall = R.train_q_source(
        src_layout, source_task,
        params["source_q_episodes"], params["source_max_steps"], agent_seed)
    # Schema agent (learns the task schema from reward feedback).
    schema_src, schema_logs, schema_wall = R.train_schema_source(
        src_layout, source_task,
        params["schema_explore_episodes"],
        params["schema_consolidate_episodes"],
        params["source_max_steps"], agent_seed)
    source_quality = {
        "q_source_final_mean_cycle_steps": M.mean_steps_per_completed_cycle(
            q_src_logs, last_n=20),
        "schema_source_final_mean_cycle_steps": M.mean_steps_per_completed_cycle(
            schema_logs, last_n=5),
    }

    # ---- target-environment evaluation -----------------------------------
    target_env = R.make_env(C.SEED_TARGET_BASE + run_idx, target_task)
    budget = params["target_episodes"], params["target_max_steps"]

    a_fresh = R.fresh_q_agent(target_task, epsilon_start=C.Q_EPS_START_FRESH)
    a_trans = R.fresh_q_agent(target_task,
                              epsilon_start=C.Q_EPS_START_TRANSFERRED,
                              q_init=q_src.q)   # FULL previous Q-table
    logs = {
        "q_fresh": R.run_target_q(a_fresh, target_env, agent_seed, *budget),
        "q_transferred": R.run_target_q(a_trans, target_env, agent_seed, *budget),
    }
    _, logs["schema"] = R.run_target_schema(
        schema_src.schema, target_env, agent_seed, *budget)

    planner_logs = R.run_planner_reference(
        tgt_layout, target_task,
        params["planner_episodes"], params["target_max_steps"])
    planner_steps = [l["completion_step"] for l in planner_logs
                     if l["completed"]]

    # ---- per-run records ---------------------------------------------------
    rows, curve_rows = [], []
    for agent in AGENTS:
        s = M.summarize_run(logs[agent], reveal_id="D")
        rows.append({
            "run": run_idx,
            "agent": agent,
            "success": s["success"],
            "first_cycle_steps": s["first_cycle_steps"],
            "subsequent_cycle_mean": s["subsequent_cycle_mean"],
            "n_subsequent_cycles": s["n_subsequent_cycles"],
            "distinct_observed_before_first_cycle":
                s["distinct_observed_before_first_cycle"],
            "delay_obsD_to_completion": s["delay_obsD_to_completion"],
            "bfs_calls_at_first_cycle": s["bfs_calls_at_first_cycle"],
            "total_steps_used": s["total_steps_used"],
            "q_source_wall_s": q_wall,
            "schema_source_wall_s": schema_wall,
            **source_quality,
        })
        curve_rows.extend(M.learning_curve_rows(
            agent, run_idx, logs[agent], params["target_max_steps"]))

    extras = {
        "planner_mean_steps_per_cycle": (
            float(np.mean(planner_steps)) if planner_steps else None),
    }
    return rows, curve_rows, extras


def build_summary(per_run_df):
    rows = []
    for agent in AGENTS:
        sub = per_run_df[per_run_df["agent"] == agent]
        n = len(sub)
        k = int(sub["success"].sum())
        wlo, whi = S.wilson_ci(k, n)
        steps = sub.loc[sub["success"], "first_cycle_steps"].astype(float)
        lo, hi = S.bootstrap_ci_mean(steps) if len(steps) >= 2 else (np.nan, np.nan)
        rows.append({
            "agent": agent,
            "n_runs": n,
            "successes": k,
            "success_rate": k / n,
            "wilson_lo": round(wlo, 3),
            "wilson_hi": round(whi, 3),
            "mean_first_cycle_steps": round(S.mean(steps), 1),
            "ci_lo": round(lo, 1),
            "ci_hi": round(hi, 1),
            "median_first_cycle_steps": round(S.median(steps), 1),
            "mean_subsequent_cycle_steps": round(
                sub["subsequent_cycle_mean"].astype(float).mean(), 1),
            "mean_distinct_observed_before_first_cycle": round(
                sub["distinct_observed_before_first_cycle"].astype(float).mean(), 2),
            "mean_delay_obsD_to_completion": round(
                sub["delay_obsD_to_completion"].astype(float).mean(), 1),
            "mean_bfs_calls_first_cycle": round(
                sub["bfs_calls_at_first_cycle"].astype(float).mean(), 1),
            "mean_total_steps_used": round(
                sub["total_steps_used"].astype(float).mean(), 0),
        })
    return pd.DataFrame(rows)


def run_experiment(params, results_dir):
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    all_rows, all_curves = [], []
    planner_ref = []
    print(f"Experiment A (main transfer): {params['n_runs']} runs, "
          f"task sequence {C.DEFAULT_SEQUENCE}"
          f"{' [SMOKE MODE]' if params['smoke'] else ''}")
    for r in range(params["n_runs"]):
        rows, curves, extras = run_three_agent_run(r, params, C.DEFAULT_TASK)
        all_rows.extend(rows)
        all_curves.extend(curves)
        if extras["planner_mean_steps_per_cycle"] is not None:
            planner_ref.append({"run": r,
                                "planner_mean_steps_per_cycle":
                                    extras["planner_mean_steps_per_cycle"]})
        done = {row["agent"]: row["first_cycle_steps"] for row in rows}
        print(f"  run {r + 1}/{params['n_runs']}: "
              + ", ".join(f"{a}={done[a] if done[a] is not None else 'fail'}"
                          for a in AGENTS))

    per_run = pd.DataFrame(all_rows)
    curves = pd.DataFrame(all_curves)
    summary = build_summary(per_run)

    planner_df = pd.DataFrame(
        planner_ref, columns=["run", "planner_mean_steps_per_cycle"])
    planner_mean = (float(planner_df["planner_mean_steps_per_cycle"].mean())
                    if len(planner_df) else np.nan)
    print("\nSummary (main comparison; planner is a separate reference):")
    print(summary.to_string(index=False))
    print(f"\nOracle planner reference: mean {planner_mean:.1f} "
          "env steps per cycle (NOT part of the fair comparison)")

    tag = "smoke_" if params["smoke"] else ""
    per_run.to_csv(os.path.join(results_dir, "tables", f"{tag}expA_per_run.csv"),
                   index=False)
    curves.to_csv(os.path.join(results_dir, "tables", f"{tag}expA_curves.csv"),
                  index=False)
    summary.to_csv(os.path.join(results_dir, "tables", f"{tag}expA_summary.csv"),
                   index=False)
    # Planner reference numbers are part of the reproducible outputs (they
    # were previously only printed to the terminal).
    planner_df.assign(experiment="A").to_csv(
        os.path.join(results_dir, "tables", f"{tag}planner_reference_expA.csv"),
        index=False)
    P.plot_learning_curves(
        curves, os.path.join(results_dir, "figures",
                             f"{tag}expA_learning_curves.png"),
        title=("Experiment A: adaptation in a new environment "
               f"({params['n_runs']} runs)"
               + (" - SMOKE" if params["smoke"] else "")),
        max_steps=params["target_max_steps"], agents=AGENTS)
    P.plot_first_cycle_bars(
        summary, os.path.join(results_dir, "figures",
                              f"{tag}expA_first_cycle.png"),
        title="Experiment A: steps to first complete cycle in a new layout",
        reference_steps=planner_mean, order=AGENTS)
    return per_run, curves, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="tiny budgets for a fast pipeline check")
    parser.add_argument("--runs", type=int, default=None,
                        help="override the number of runs")
    args = parser.parse_args()
    params = R.effective_params(smoke=args.smoke, n_runs=args.runs)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Smoke outputs go to results/smoke/ so they can never be confused with
    # the reported full-mode results.
    results_dir = os.path.join(base, "results",
                               "smoke" if args.smoke else "")
    run_experiment(params, results_dir)


if __name__ == "__main__":
    main()
