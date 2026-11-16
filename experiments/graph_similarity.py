"""Experiment D: task-structure similarity extension.

Target sequences are represented as DIRECTED TASK GRAPHS over the four
target identities (plus the pseudo-node START).  Four target task types are
compared against the cycle source task:

    cycle          A->B->C->D->A   (the source structure itself)
    chain          A->B->C->D
    branch         A->{B,C}->D
    reverse_cycle  A->D->C->B->A

Similarity measure: EDGE-JACCARD between the directed edge sets (including
START edges):
    J(G1, G2) = |E1 & E2| / |E1 union E2|
It is simple, explicit and easy to explain, but crude: it ignores edge
ORDER/position in the traversal, treats a missing edge like a redirected
one, and is sensitive to the small node set.  Limitations are discussed in
docs/limitations.md.

Protocol: for each of N_RUNS independent runs, the schema agent is trained
on the cycle source task (same seeds as the other experiments) and evaluated
in a NEW environment under each target task type.  The SAME target layout
seed is used for all four task types within a run, so task types are paired.
Only the schema agent and the planner reference are run here: the question
is whether STRUCTURE similarity predicts the transfer of the structure-carrying
agent.  With only 4 task types the correlation is descriptive, not a
statistical test -- this is stated honestly in the summary.
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C
from analysis import metrics as M
from analysis import plots as P
from analysis import statistics as S
from experiments import run_utils as R

TARGET_TASKS = ["cycle", "chain", "branch", "reverse_cycle"]


def edge_set(task_graph):
    """Directed edge set of a task graph, START edges included."""
    edges = set()
    for node, successors in task_graph["successors"].items():
        for succ in successors:
            edges.add((node, succ))
    return edges


def edge_jaccard(task_a, task_b):
    """|E_A & E_B| / |E_A union E_B| over directed edges (0 when both empty)."""
    ea, eb = edge_set(task_a), edge_set(task_b)
    if not ea and not eb:
        return 1.0
    return len(ea & eb) / len(ea | eb)


def run_similarity_run(run_idx, params):
    src_layout = R.generate_layout(C.SEED_SOURCE_BASE + run_idx)
    agent_seed = C.SEED_AGENT_BASE + run_idx

    # Train the schema agent once on the cycle source task.
    schema_src, _src_logs, _wall = R.train_schema_source(
        src_layout, C.DEFAULT_TASK,
        params["schema_explore_episodes"],
        params["schema_consolidate_episodes"],
        params["source_max_steps"], agent_seed)

    rows, curve_rows = [], []
    for task_name in TARGET_TASKS:
        task = C.TASK_GRAPH_LIBRARY[task_name]
        # SAME layout across task types within a run (paired comparison).
        target_env = R.make_env(C.SEED_EXP_D_TARGET_BASE + run_idx, task)
        episodes, max_steps = (params["target_episodes"],
                               params["target_max_steps"])
        rng = np.random.default_rng(agent_seed)
        agent = R.make_target_schema_agent(schema_src.schema, C.GRID_SIZE)
        agent.set_obstacles(target_env.obstacles, target_env.size)
        logs = [R.run_schema_episode(target_env, agent, rng, max_steps)
                for _ in range(episodes)]
        planner_logs = R.run_planner_reference(
            target_env.layout, task,
            params["planner_episodes"], max_steps)
        planner_steps = [l["completion_step"] for l in planner_logs
                         if l["completed"]]

        s = M.summarize_run(logs, reveal_id="D")
        total_budget = episodes * max_steps
        rows.append({
            "run": run_idx,
            "target_task": task_name,
            "similarity_to_source": round(
                edge_jaccard(C.DEFAULT_TASK, task), 4),
            "success": s["success"],
            "first_cycle_steps": s["first_cycle_steps"],
            # Budget-censored value: the cycle length if completed, otherwise
            # the total interaction budget.  Standard right-censoring; it
            # keeps summary statistics defined even at 0% success.
            "first_cycle_steps_censored": (
                s["first_cycle_steps"] if s["success"] else total_budget),
            "subsequent_cycle_mean": s["subsequent_cycle_mean"],
            "distinct_observed_before_first_cycle":
                s["distinct_observed_before_first_cycle"],
            "total_steps_used": s["total_steps_used"],
            "planner_mean_steps_per_cycle": (
                float(np.mean(planner_steps)) if planner_steps else None),
        })
        curve_rows.extend(M.learning_curve_rows(
            f"schema@{task_name}", run_idx, logs, max_steps))
    return rows, curve_rows


def build_summary(per_run_df):
    rows = []
    for task_name in TARGET_TASKS:
        sub = per_run_df[per_run_df["target_task"] == task_name]
        n, k = len(sub), int(sub["success"].sum())
        wlo, whi = S.wilson_ci(k, n)
        steps = sub.loc[sub["success"], "first_cycle_steps"].astype(float)
        rows.append({
            "target_task": task_name,
            "similarity_to_source": sub["similarity_to_source"].iloc[0],
            "n_runs": n,
            "successes": k,
            "success_rate": k / n,
            "wilson_lo": round(wlo, 3),
            "wilson_hi": round(whi, 3),
            "median_steps_success": round(S.median(steps), 1),
            "mean_steps_success": round(S.mean(steps), 1),
            "median_steps_censored": round(
                S.median(sub["first_cycle_steps_censored"].astype(float)), 1),
            "mean_planner_steps": round(
                sub["planner_mean_steps_per_cycle"].astype(float).mean(), 1),
        })
    summary = pd.DataFrame(rows)
    # Descriptive rank correlation across the (only!) 4 task types, using the
    # budget-censored median so that 0%-success task types are included.
    summary["spearman_similarity_vs_steps"] = S.spearman_rho(
        summary["similarity_to_source"], summary["median_steps_censored"])
    summary["spearman_similarity_vs_success"] = S.spearman_rho(
        summary["similarity_to_source"], summary["success_rate"])
    return summary


def run_experiment(params, results_dir):
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    all_rows, all_curves = [], []
    print(f"Experiment D (task-structure similarity): {params['n_runs']} runs "
          f"x {len(TARGET_TASKS)} target task types"
          f"{' [SMOKE MODE]' if params['smoke'] else ''}")
    for r in range(params["n_runs"]):
        rows, curves = run_similarity_run(r, params)
        all_rows.extend(rows)
        all_curves.extend(curves)
        done = {row["target_task"]: row["first_cycle_steps"] for row in rows}
        print(f"  run {r + 1}/{params['n_runs']}: "
              + ", ".join(f"{t}={done[t] if done[t] is not None else 'fail'}"
                          for t in TARGET_TASKS))

    per_run = pd.DataFrame(all_rows)
    curves = pd.DataFrame(all_curves)
    summary = build_summary(per_run)
    print("\nSummary (Spearman correlations are DESCRIPTIVE: only 4 task "
          "types, so no p-values are reported):")
    print(summary.to_string(index=False))

    tag = "smoke_" if params["smoke"] else ""
    per_run.to_csv(os.path.join(results_dir, "tables", f"{tag}expD_per_run.csv"),
                   index=False)
    curves.to_csv(os.path.join(results_dir, "tables", f"{tag}expD_curves.csv"),
                  index=False)
    summary.to_csv(os.path.join(results_dir, "tables", f"{tag}expD_summary.csv"),
                   index=False)
    P.plot_similarity(
        summary, os.path.join(results_dir, "figures", f"{tag}expD_similarity.png"),
        title=("Experiment D: does task-graph similarity predict transfer? "
               f"({params['n_runs']} runs)"
               + (" - SMOKE" if params["smoke"] else "")))
    return per_run, curves, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--runs", type=int, default=None)
    args = parser.parse_args()
    params = R.effective_params(smoke=args.smoke, n_runs=args.runs)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Smoke outputs go to results/smoke/ so they can never be confused with
    # the reported full-mode results.  (Experiment D's planner reference is
    # already stored per run in expD_per_run.csv.)
    results_dir = os.path.join(base, "results",
                               "smoke" if args.smoke else "")
    run_experiment(params, results_dir)


if __name__ == "__main__":
    main()
