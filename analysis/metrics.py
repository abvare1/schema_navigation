"""Metric computation for the schema-navigation study.

An "episode log" is a plain dict produced by the runners in
experiments/run_utils.py with the keys:
    steps            -- env steps taken in this episode
    completed        -- True if a full task cycle was completed
    completion_step  -- step index (within the episode) of cycle completion
    reward           -- total reward collected
    first_reveals    -- {target identity: step index of first perception}
    bfs_calls        -- cumulative planning calls at the end of the episode

Metric definitions (see README section 7 and docs/methodology.md):

1. steps_to_first_cycle
       Cumulative environment steps, counted from the very first step in the
       new environment across consecutive episodes, until the first episode
       in which a complete target cycle is finished.  If no cycle is ever
       completed within the budget, the run is recorded as a FAILURE (never
       silently dropped).

2. subsequent_cycle_steps
       Per-episode step counts of complete cycles AFTER the first one.
       (One episode contains at most one cycle because the episode ends when
       the cycle completes.)

3. success_rate
       Fraction of independent runs in which the first cycle was completed
       within the total interaction budget.

4. learning curves
       Per-episode step counts (censored at the per-episode step cap when the
       episode ends without completing a cycle).

5. distinct_targets_observed_before_first_cycle
       Number of distinct target identities the agent had perceived at least
       once before the moment the first cycle completed.

6. delay_obsD_to_completion
       (step of first cycle completion) - (step of first perception of the
       reveal identity, default "D").  Operational definition only: it
       measures how much work remained after "D" was first perceived.  It is
       NOT by itself evidence of one-shot learning.

7. performance under task-structure change
       Computed by experiments/incongruent_transfer.py and
       experiments/graph_similarity.py using the same metrics as above.
"""

import numpy as np

import config as C


def summarize_run(episode_logs, reveal_id="D"):
    """Compute all per-run summary metrics from a list of episode logs.

    Returns a plain dict; ``None`` values mean "not applicable / not
    achieved" and are recorded explicitly rather than dropped.
    """
    first_cycle_steps = None
    subsequent = []
    distinct_before = None
    delay = None
    bfs_at_first_cycle = None
    total_steps = 0
    first_reveal_abs = {}

    for log in episode_logs:
        for tid, step in log["first_reveals"].items():
            first_reveal_abs.setdefault(tid, total_steps + step)
        if log["completed"]:
            completion_abs = total_steps + log["completion_step"]
            if first_cycle_steps is None:
                first_cycle_steps = completion_abs
                bfs_at_first_cycle = log["bfs_calls"]
                distinct_before = sum(
                    1 for t in first_reveal_abs.values() if t <= completion_abs)
                if reveal_id in first_reveal_abs:
                    delay = completion_abs - first_reveal_abs[reveal_id]
            else:
                subsequent.append(log["completion_step"])
        total_steps += log["steps"]

    return {
        "success": first_cycle_steps is not None,
        "first_cycle_steps": first_cycle_steps,
        "subsequent_cycle_mean": (
            float(np.mean(subsequent)) if subsequent else None),
        "n_subsequent_cycles": len(subsequent),
        "distinct_observed_before_first_cycle": distinct_before,
        f"delay_obs{reveal_id}_to_completion": delay,
        "bfs_calls_at_first_cycle": bfs_at_first_cycle,
        "total_steps_used": total_steps,
        "n_episodes": len(episode_logs),
    }


def learning_curve_rows(agent, run_id, episode_logs, max_steps):
    """Rows for the (censored) learning-curve table.

    Curve value = steps taken if the episode completed a cycle, otherwise the
    per-episode cap (right-censoring at the budget).  Both the raw value and
    the completion flag are kept so no information is hidden.
    """
    rows = []
    for i, log in enumerate(episode_logs, start=1):
        rows.append({
            "agent": agent,
            "run": run_id,
            "episode": i,
            "steps": log["steps"],
            "completed": log["completed"],
            "steps_censored": (log["steps"] if log["completed"]
                               else max_steps),
        })
    return rows


def mean_steps_per_completed_cycle(episode_logs, last_n=None):
    """Mean length of completed cycles, optionally over the last n episodes.
    Used for source-environment training summaries."""
    lengths = [log["completion_step"] for log in episode_logs
               if log["completed"]]
    if last_n is not None:
        lengths = lengths[-last_n:]
    if not lengths:
        return None
    return float(np.mean(lengths))
