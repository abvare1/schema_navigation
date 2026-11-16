"""Experiment B: ablation studies of the schema-based agent.

Two ablations, each removing exactly ONE component and leaving everything
else untouched (same training procedure, same seeds, same budgets):

  schema_no_memory      -- use_spatial_memory=False.  Discovered target
      locations are never stored; the agent keeps its learned task schema
      (so it still knows WHICH identity is acceptable next) but can only
      react to targets visible at its current cell.  It cannot plan with BFS
      because planning needs remembered positions.

  schema_no_task_state  -- use_task_state=False.  The FSM/schema component is
      removed: the agent never learns or uses the target sequence.  It keeps
      spatial memory, so in the new environment it remembers where targets
      are but visits them in a RANDOM order, which rarely matches the
      required order.  Note this ablation has nothing to transfer at all:
      its only transferable component WAS the schema.  It therefore receives
      no source training (documented in docs/methodology.md).

  schema_full           -- the intact agent, re-evaluated under identical
      conditions for direct comparison.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C
from agents.schema_agent import SchemaAgent
from analysis import metrics as M
from analysis import plots as P
from analysis import statistics as S
from experiments import run_utils as R

AGENTS = ["schema_full", "schema_no_memory", "schema_no_task_state"]


def run_ablation_run(run_idx, params):
    src_layout = R.generate_layout(C.SEED_SOURCE_BASE + run_idx)
    tgt_layout = R.generate_layout(C.SEED_TARGET_BASE + run_idx)
    agent_seed = C.SEED_AGENT_BASE + run_idx
    task = C.DEFAULT_TASK

    # One source training session serves schema_full and schema_no_memory:
    # both carry the same learned schema; they differ ONLY in the memory flag.
    schema_src, _src_logs, schema_wall = R.train_schema_source(
        src_layout, task,
        params["schema_explore_episodes"],
        params["schema_consolidate_episodes"],
        params["source_max_steps"], agent_seed)

    target_env = R.make_env(C.SEED_TARGET_BASE + run_idx, task)
    episodes, max_steps = params["target_episodes"], params["target_max_steps"]
    rng = np.random.default_rng(agent_seed)

    agents = {
        "schema_full": R.make_target_schema_agent(schema_src.schema, C.GRID_SIZE),
        "schema_no_memory": SchemaAgent(C.GRID_SIZE,
                                        schema=schema_src.schema,
                                        use_spatial_memory=False,
                                        use_task_state=True),
        # No source training: without the task state there is nothing that
        # could transfer from the source environment.
        "schema_no_task_state": SchemaAgent(C.GRID_SIZE,
                                            schema=None,
                                            use_spatial_memory=True,
                                            use_task_state=False),
    }
    logs = {}
    for name, agent in agents.items():
        agent.set_obstacles(target_env.obstacles, target_env.size)
        logs[name] = [R.run_schema_episode(target_env, agent, rng, max_steps)
                      for _ in range(episodes)]

    rows, curve_rows = [], []
    for name in AGENTS:
        s = M.summarize_run(logs[name], reveal_id="D")
        rows.append({
            "run": run_idx,
            "agent": name,
            "success": s["success"],
            "first_cycle_steps": s["first_cycle_steps"],
            "subsequent_cycle_mean": s["subsequent_cycle_mean"],
            "distinct_observed_before_first_cycle":
                s["distinct_observed_before_first_cycle"],
            "delay_obsD_to_completion": s["delay_obsD_to_completion"],
            "bfs_calls_at_first_cycle": s["bfs_calls_at_first_cycle"],
            "total_steps_used": s["total_steps_used"],
            "schema_source_wall_s": schema_wall,
        })
        curve_rows.extend(M.learning_curve_rows(
            name, run_idx, logs[name], max_steps))
    return rows, curve_rows


def build_summary(per_run_df):
    rows = []
    for agent in AGENTS:
        sub = per_run_df[per_run_df["agent"] == agent]
        n, k = len(sub), int(sub["success"].sum())
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
            "mean_bfs_calls_first_cycle": round(
                sub["bfs_calls_at_first_cycle"].astype(float).mean(), 1),
        })
    return pd.DataFrame(rows)


def run_experiment(params, results_dir):
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    all_rows, all_curves = [], []
    print(f"Experiment B (ablations): {params['n_runs']} runs"
          f"{' [SMOKE MODE]' if params['smoke'] else ''}")
    for r in range(params["n_runs"]):
        rows, curves = run_ablation_run(r, params)
        all_rows.extend(rows)
        all_curves.extend(curves)
        done = {row["agent"]: row["first_cycle_steps"] for row in rows}
        print(f"  run {r + 1}/{params['n_runs']}: "
              + ", ".join(f"{a}={done[a] if done[a] is not None else 'fail'}"
                          for a in AGENTS))

    per_run = pd.DataFrame(all_rows)
    curves = pd.DataFrame(all_curves)
    summary = build_summary(per_run)
    print("\nSummary:")
    print(summary.to_string(index=False))

    tag = "smoke_" if params["smoke"] else ""
    per_run.to_csv(os.path.join(results_dir, "tables", f"{tag}expB_per_run.csv"),
                   index=False)
    curves.to_csv(os.path.join(results_dir, "tables", f"{tag}expB_curves.csv"),
                  index=False)
    summary.to_csv(os.path.join(results_dir, "tables", f"{tag}expB_summary.csv"),
                   index=False)
    P.plot_learning_curves(
        curves, os.path.join(results_dir, "figures",
                             f"{tag}expB_ablation_curves.png"),
        title=("Experiment B: schema-agent ablations in a new environment "
               f"({params['n_runs']} runs)"
               + (" - SMOKE" if params["smoke"] else "")),
        max_steps=params["target_max_steps"], agents=AGENTS)
    P.plot_first_cycle_bars(
        summary, os.path.join(results_dir, "figures", f"{tag}expB_ablation.png"),
        title="Experiment B: removing one component at a time",
        order=AGENTS)
    return per_run, curves, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs", type=int, default=None)
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
