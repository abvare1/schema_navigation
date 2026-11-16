"""Experiment E: obstacle-information control (NEW protocol, kept separate).

Purpose
    In the original Experiment A the schema agent receives the COMPLETE
    obstacle map of the target environment at wiring time
    (``SchemaAgent.set_obstacles``) and uses it for BFS planning and for
    bump-free random walks.  The Q-learning baselines receive NO obstacle
    information in the original protocol: they can only learn obstacle
    locations implicitly from bump penalties.  Experiment E quantifies how
    much of the schema agent's measured advantage is attributable to that
    information asymmetry, by giving Q-learning agents obstacle information
    and nothing else.

Protocol relationship to Experiment A (kept strictly separate)
    * SAME source/target layout seeds (20000+r / 30000+r), SAME agent seeds
      (50000+r), SAME task (the default cycle), SAME target budget
      (60 episodes x 400 steps), SAME hyperparameters.
    * The source Q-table for ``q_transferred_masked`` is re-trained with the
      UNMASKED source procedure, which (same seeds) reproduces the exact
      table of the original protocol; the only change versus Experiment A is
      therefore the obstacle information available in the target phase.
    * NOTHING from Experiment A's results files is modified or overwritten.
      All outputs use the expE_ prefix and carry explicit protocol columns.
      Do NOT pool expA and expE rows: they come from different protocols.

The four control agents (all tabular Q-learning, identical hyperparameters):

    q_fresh_masked          obstacle information: the complete obstacle map,
                            provided at target-environment initialization
                            (the same moment and content the schema agent
                            gets).  Used ONLY to mask out moves into
                            obstacles / off-grid at action-selection time
                            (exploration samples uniformly among admissible
                            moves; greedy is argmax over admissible moves).
                            The agent never bumps, but it has no planning
                            (no BFS) and no target knowledge.
                            Source training: none (fresh table), as in Exp A.

    q_transferred_masked    same as q_fresh_masked, but initialised with a
                            COPY of the exact unmasked source Q-table of the
                            original protocol (same seeds) and epsilon 0.3,
                            as in Exp A.  Isolates: original transfer +
                            obstacle information in the target phase.

    q_transferred_maskcons  same masking, but its source Q-table is
                            re-trained WITH the mask active in the source
                            environment (a mask-consistent transfer
                            lifecycle).  Isolates: whether the masking
                            advantage depends on when the table was learned.

    q_fresh_bumpmem         NO obstacle information at start.  Blocked moves
                            are learned exclusively from the agent's own
                            bump events (perceivable by touch) and are then
                            avoided, so no bump is ever repeated.  Measures
                            the cost of DISCOVERING the obstacle layout.

What this control does and does not equalise
    It gives Q-learning the same obstacle INFORMATION the schema agent has
    (full_map variants) in the weakest possible form: action masking.  It
    does NOT give Q-learning model-based planning (BFS), which remains part
    of the bundled architectural difference documented in
    docs/limitations.md.  The gap between the masked Q agents and the schema
    agent in Experiment A is therefore an upper bound on the combined effect
    of planning + task schema + spatial memory.

Outputs (all under results/tables, never mixed with expA files)
    expE_per_run.csv, expE_summary.csv, expE_curves.csv,
    expE_vs_expA_comparison.csv (explicit protocol columns; juxtaposition
    for reading convenience only), figures/expE_obstacle_control.png
"""

import argparse
import os

import numpy as np
import pandas as pd

import config as C
from agents.q_learning_masked import ObstacleAwareQAgent
from analysis import metrics as M
from analysis import plots as P
from analysis import statistics as S
from environment.grid_world import GridWorld, generate_layout
from experiments import run_utils as R

AGENTS = ["q_fresh_masked", "q_transferred_masked",
          "q_transferred_maskcons", "q_fresh_bumpmem"]

# Protocol labels recorded verbatim in every output row.
OBSTACLE_INFO = {
    "q_fresh_masked": ("complete obstacle map given at target-env init "
                       "(action masking only; no planning)"),
    "q_transferred_masked": ("complete obstacle map given at target-env init "
                             "(action masking only; no planning)"),
    "q_transferred_maskcons": ("complete obstacle map given at target-env "
                               "init (action masking only; no planning)"),
    "q_fresh_bumpmem": ("none at start; blocked moves learned from own bump "
                        "events, then avoided"),
}
SOURCE_TRAINING = {
    "q_fresh_masked": "none (fresh table, as in Experiment A)",
    "q_transferred_masked": ("unmasked source training identical to "
                             "Experiment A (reused table, same seeds)"),
    "q_transferred_maskcons": ("mask-consistent source training (masking "
                               "also active in the source environment)"),
    "q_fresh_bumpmem": "none (fresh table, as in Experiment A)",
}


# ---------------------------------------------------------------------------
# Episode runner with bump feedback (needed only by the bump-memory agent)
# ---------------------------------------------------------------------------
def run_q_episode_with_bump_feedback(env, agent, rng, max_steps, training=True):
    """One Q-learning episode; identical to run_utils.run_q_episode except
    that the agent is told, after every step, whether its move bumped (so a
    bump_memory agent can record the blocked move).  For full_map agents the
    call is a no-op, and the episode is behaviourally identical to the plain
    runner."""
    log = R._new_log()
    obs = env.reset()
    R._track_reveals(log, obs, 0)
    for t in range(1, max_steps + 1):
        state = agent.encode_state(obs)
        action = agent.act(obs, rng, greedy=not training)
        prev_pos = obs["pos"]
        obs, reward, done, info = env.step(action)
        agent.observe_step(prev_pos, action, info["bump"])
        if training:
            agent.update(state, action, reward, agent.encode_state(obs), done)
        log["steps"] = t
        log["reward"] += reward
        R._track_reveals(log, obs, t)
        if done:
            log["completed"] = True
            log["completion_step"] = t
            break
    if training:
        agent.end_episode()
    log["bfs_calls"] = 0
    return log


# ---------------------------------------------------------------------------
# Source training
# ---------------------------------------------------------------------------
def _train_q_source_masked(layout, task, episodes, max_steps, agent_seed):
    """Train a mask-consistent source Q-table: same environment, seeds and
    hyperparameters as the unmasked source training, but with the complete
    obstacle map active from the start (action masking in every episode)."""
    import time
    env = GridWorld(layout, task)
    rng = np.random.default_rng(agent_seed)
    agent = ObstacleAwareQAgent(
        grid_size=C.GRID_SIZE,
        n_stages=task["completion_length"],
        alpha=C.Q_ALPHA,
        gamma=C.Q_GAMMA,
        epsilon_start=C.Q_EPS_START_FRESH,
        epsilon_min=C.Q_EPS_MIN,
        epsilon_decay=C.Q_EPS_DECAY,
        obstacle_mode="full_map",
    )
    agent.set_obstacles(env.obstacles, env.size)
    logs = []
    t0 = time.perf_counter()
    for _ep in range(episodes):
        logs.append(run_q_episode_with_bump_feedback(env, agent, rng,
                                                     max_steps, training=True))
    wall = time.perf_counter() - t0
    return agent, logs, wall


# ---------------------------------------------------------------------------
# One control run
# ---------------------------------------------------------------------------
def run_control_run(run_idx, params):
    """One (source, target) pair under the control protocol."""
    src_layout = generate_layout(C.SEED_SOURCE_BASE + run_idx)
    agent_seed = C.SEED_AGENT_BASE + run_idx
    task = C.DEFAULT_TASK

    # Unmasked source training, identical to Experiment A (same seeds ->
    # identical source Q-table; used by q_transferred_masked).
    q_src, _logs, q_wall = R.train_q_source(
        src_layout, task,
        params["source_q_episodes"], params["source_max_steps"], agent_seed)
    # Mask-consistent source training for q_transferred_maskcons.
    q_src_masked, _logs_m, q_wall_masked = _train_q_source_masked(
        src_layout, task,
        params["source_q_episodes"], params["source_max_steps"], agent_seed)

    target_env = R.make_env(C.SEED_TARGET_BASE + run_idx, task)
    episodes, max_steps = (params["target_episodes"],
                           params["target_max_steps"])

    def make_agent(name):
        common = dict(
            grid_size=C.GRID_SIZE,
            n_stages=task["completion_length"],
            alpha=C.Q_ALPHA,
            gamma=C.Q_GAMMA,
            epsilon_min=C.Q_EPS_MIN,
            epsilon_decay=C.Q_EPS_DECAY,
        )
        if name == "q_fresh_masked":
            agent = ObstacleAwareQAgent(
                epsilon_start=C.Q_EPS_START_FRESH,
                obstacle_mode="full_map", **common)
        elif name == "q_transferred_masked":
            agent = ObstacleAwareQAgent(
                epsilon_start=C.Q_EPS_START_TRANSFERRED,
                obstacle_mode="full_map", q_init=q_src.q, **common)
        elif name == "q_transferred_maskcons":
            agent = ObstacleAwareQAgent(
                epsilon_start=C.Q_EPS_START_TRANSFERRED,
                obstacle_mode="full_map", q_init=q_src_masked.q, **common)
        elif name == "q_fresh_bumpmem":
            agent = ObstacleAwareQAgent(
                epsilon_start=C.Q_EPS_START_FRESH,
                obstacle_mode="bump_memory", **common)
        else:
            raise ValueError(name)
        return agent

    agents = {name: make_agent(name) for name in AGENTS}
    # Obstacle provisioning: ONLY the full_map agents receive the map, and
    # at exactly the same wiring point as the schema agent in Experiment A.
    for name, agent in agents.items():
        if agent.obstacle_mode == "full_map":
            agent.set_obstacles(target_env.obstacles, target_env.size)

    logs = {}
    for name, agent in agents.items():
        rng = np.random.default_rng(C.SEED_AGENT_BASE + run_idx)
        logs[name] = [run_q_episode_with_bump_feedback(target_env, agent, rng,
                                                       max_steps,
                                                       training=True)
                      for _ in range(episodes)]

    rows, curve_rows = [], []
    for name in AGENTS:
        s = M.summarize_run(logs[name], reveal_id="D")
        rows.append({
            "run": run_idx,
            "agent": name,
            "obstacle_info": OBSTACLE_INFO[name],
            "source_training": SOURCE_TRAINING[name],
            "success": s["success"],
            "first_cycle_steps": s["first_cycle_steps"],
            "subsequent_cycle_mean": s["subsequent_cycle_mean"],
            "n_subsequent_cycles": s["n_subsequent_cycles"],
            "distinct_observed_before_first_cycle":
                s["distinct_observed_before_first_cycle"],
            "delay_obsD_to_completion": s["delay_obsD_to_completion"],
            "total_steps_used": s["total_steps_used"],
            "q_source_wall_s": q_wall,
            "q_source_masked_wall_s": q_wall_masked,
        })
        curve_rows.extend(M.learning_curve_rows(
            name, run_idx, logs[name], max_steps))
    return rows, curve_rows


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
            "obstacle_info": OBSTACLE_INFO[agent],
            "source_training": SOURCE_TRAINING[agent],
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
            "mean_total_steps_used": round(
                sub["total_steps_used"].astype(float).mean(), 0),
        })
    return pd.DataFrame(rows)


def build_comparison(results_dir, expE_summary):
    """Juxtapose the ORIGINAL Experiment A summary with the control summary.

    Every row carries its protocol label; the two protocols must never be
    pooled into one estimate (see the module docstring and README).  The
    file is only written when a full-mode Experiment A summary exists.
    """
    expA_path = os.path.join(results_dir, "tables", "expA_summary.csv")
    if not os.path.exists(expA_path):
        print("  (no full-mode expA_summary.csv found -- "
              "expE_vs_expA_comparison.csv not written)")
        return None
    expA = pd.read_csv(expA_path)
    expA_info = {
        "q_fresh": "no obstacle information (original protocol)",
        "q_transferred": "no obstacle information (original protocol)",
        "schema": ("complete obstacle map at init, used for BFS planning "
                   "and bump-free walks (original protocol)"),
    }
    expA_src = {
        "q_fresh": "none (fresh table)",
        "q_transferred": "unmasked source training (same seeds)",
        "schema": "schema learning from reward feedback (60+5 episodes)",
    }
    rows = []
    for _, r in expA.iterrows():
        rows.append({
            "agent": r["agent"],
            "protocol": "original Experiment A",
            "obstacle_info": expA_info[r["agent"]],
            "source_training": expA_src[r["agent"]],
            "n_runs": r["n_runs"],
            "success_rate": r["success_rate"],
            "mean_first_cycle_steps_successful_runs_only":
                r["mean_first_cycle_steps"],
        })
    for _, r in expE_summary.iterrows():
        rows.append({
            "agent": r["agent"],
            "protocol": "control Experiment E (obstacle-information control)",
            "obstacle_info": r["obstacle_info"],
            "source_training": r["source_training"],
            "n_runs": r["n_runs"],
            "success_rate": r["success_rate"],
            "mean_first_cycle_steps_successful_runs_only":
                r["mean_first_cycle_steps"],
        })
    return pd.DataFrame(rows)


def run_experiment(params, results_dir):
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    all_rows, all_curves = [], []
    print(f"Experiment E (obstacle-information control): "
          f"{params['n_runs']} runs, 4 control agents"
          f"{' [SMOKE MODE]' if params['smoke'] else ''}")
    for r in range(params["n_runs"]):
        rows, curves = run_control_run(r, params)
        all_rows.extend(rows)
        all_curves.extend(curves)
        done = {row["agent"]: row["first_cycle_steps"] for row in rows}
        print(f"  run {r + 1}/{params['n_runs']}: "
              + ", ".join(f"{a}={done[a] if done[a] is not None else 'fail'}"
                          for a in AGENTS))

    per_run = pd.DataFrame(all_rows)
    curves = pd.DataFrame(all_curves)
    summary = build_summary(per_run)
    print("\nSummary (control agents only; each row states the information "
          "its agent received):")
    print(summary.to_string(index=False))

    tag = "smoke_" if params["smoke"] else ""
    per_run.to_csv(os.path.join(results_dir, "tables", f"{tag}expE_per_run.csv"),
                   index=False)
    curves.to_csv(os.path.join(results_dir, "tables", f"{tag}expE_curves.csv"),
                  index=False)
    summary.to_csv(os.path.join(results_dir, "tables", f"{tag}expE_summary.csv"),
                   index=False)
    if not params["smoke"]:
        comparison = build_comparison(results_dir, summary)
        if comparison is not None:
            comparison.to_csv(
                os.path.join(results_dir, "tables",
                             "expE_vs_expA_comparison.csv"), index=False)
            print("  wrote expE_vs_expA_comparison.csv (protocol-labelled "
                  "juxtaposition of Exp A and Exp E; do not pool rows)")
    P.plot_first_cycle_bars(
        summary, os.path.join(results_dir, "figures",
                              f"{tag}expE_obstacle_control.png"),
        title="Experiment E: obstacle-information control "
              "(Q-learning variants)",
        order=AGENTS)
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
    # full experimental results.
    results_dir = os.path.join(base, "results",
                               "smoke" if args.smoke else "")
    run_experiment(params, results_dir)


if __name__ == "__main__":
    main()
