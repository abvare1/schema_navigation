"""Experiment C: schema-incongruent transfer.

The source task is the default cycle  A -> B -> C -> D -> A  (identical to
Experiment A, same seeds).  The TARGET task uses the same four identities in
a DIFFERENT valid order:  A -> C -> B -> D -> A.

What each agent experiences in the new environment:
  q_fresh        -- unaffected by the change (it never uses source data);
                    serves as the learning-from-scratch reference.
  q_transferred  -- its transferred Q-table encodes the OLD spatial policy;
                    the new stage/reward mapping partially contradicts it.
  schema         -- carries a FROZEN learned schema of the WRONG order.  The
                    schema is deliberately not updated during the target
                    phase: the question is what happens to transfer when the
                    abstract structure itself changes.  (The consequences of
                    this design choice are discussed in docs/limitations.md.)

Target layouts use the SAME seeds as Experiment A (only the task changes),
so Experiments A and C are directly comparable.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C
from analysis import plots as P
from analysis import statistics as S
from experiments import main_transfer as A
from experiments import run_utils as R

AGENTS = A.AGENTS   # q_fresh, q_transferred, schema


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
        })
    return pd.DataFrame(rows)


def run_experiment(params, results_dir):
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    all_rows, all_curves = [], []
    planner_ref = []
    print(f"Experiment C (incongruent transfer): {params['n_runs']} runs, "
          f"source {C.DEFAULT_SEQUENCE} -> target {C.INCONGRUENT_SEQUENCE}"
          f"{' [SMOKE MODE]' if params['smoke'] else ''}")
    for r in range(params["n_runs"]):
        rows, curves, extras = A.run_three_agent_run(
            r, params, C.INCONGRUENT_TASK, source_task=C.DEFAULT_TASK)
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

    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nOracle planner reference on the incongruent task: "
          f"mean {planner_mean:.1f} env steps per cycle")

    tag = "smoke_" if params["smoke"] else ""
    per_run.to_csv(os.path.join(results_dir, "tables", f"{tag}expC_per_run.csv"),
                   index=False)
    curves.to_csv(os.path.join(results_dir, "tables", f"{tag}expC_curves.csv"),
                  index=False)
    summary.to_csv(os.path.join(results_dir, "tables", f"{tag}expC_summary.csv"),
                   index=False)
    planner_df.assign(experiment="C").to_csv(
        os.path.join(results_dir, "tables", f"{tag}planner_reference_expC.csv"),
        index=False)
    P.plot_learning_curves(
        curves, os.path.join(results_dir, "figures",
                             f"{tag}expC_incongruent_curves.png"),
        title=("Experiment C: new environment AND new task order "
               f"({C.INCONGRUENT_SEQUENCE}, {params['n_runs']} runs)"
               + (" - SMOKE" if params["smoke"] else "")),
        max_steps=params["target_max_steps"], agents=AGENTS)
    P.plot_first_cycle_bars(
        summary, os.path.join(results_dir, "figures",
                              f"{tag}expC_incongruent.png"),
        title="Experiment C: transfer when the task structure changes",
        reference_steps=planner_mean, order=AGENTS)
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
