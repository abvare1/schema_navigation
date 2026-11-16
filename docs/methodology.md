# Methodology — full experimental protocol

This document specifies exactly how the study is run, so that a reader can
re-run it and predict every number before seeing it.

## 1. Random seeds

| Quantity | Rule |
|---|---|
| Source environment layout (run r) | seed `20000 + r` (`SEED_SOURCE_BASE`) |
| Target environment layout (run r) | seed `30000 + r` (`SEED_TARGET_BASE`) |
| Experiment D target layout (run r) | seed `40000 + r` (`SEED_EXP_D_TARGET_BASE`), identical across the four task types within a run |
| Agent exploration RNG (run r) | seed `50000 + r` (`SEED_AGENT_BASE`), the **same** seed for every agent in the run |
| Bootstrap resampling | seed 12345, 10,000 resamples |
| Validation layouts | seeds 7, 77, 777 |

Layout generation is a deterministic function of the seed
(`environment.grid_world.generate_layout`), so all agents within a run face
byte-identical environments. `r = 0..29` (30 runs) in full mode; `--smoke`
reduces budgets and uses 3 runs (pipeline checks only, never reported).

## 2. Environment generation

For each seed, `generate_layout` samples 6 obstacle cells (never on the start
cell), then places A, B, C, D on free cells with pairwise Manhattan separation
>= 2 (so perception is never ambiguous), and validates: no overlaps with the
start or obstacles, every target reachable from the start via free cells, and
a connected free component of at least 30 cells. Violations raise
`ValueError` with a precise message; generation retries with the same RNG
stream until valid (10000 attempts max, never hit in practice). The starting
cell is fixed at (3,3) for all layouts.

## 3. Observation, transition and reward rules

* State observed by the agent: `(pos, k, last_node, visible)` where `k` is the
  number of correct collections this episode and `last_node` the last
  correctly collected identity ("START" initially). Both are derivable from
  the agent's own reward history (count your +10 events), so they carry no
  hidden information; they are provided in the observation for convenience.
* Actions 0-3 = up/down/left/right, deterministic. A move into a wall or
  obstacle is a bump: position unchanged, step penalty only, and **no**
  target-entry event fires.
* Reward per step: `REWARD_STEP (-0.04)` plus `+10` if the newly entered cell
  holds a target that the task graph expects now, or `-5` if it holds any
  other target. Reward is therefore always `observation + own action` ->
  the MDP is fully observable.
* Episode ends when `k` reaches the task's completion length (5 for the cycle
  A→B→C→D→A; 4 for the chain; 3 for the branch graph).
* Perception: after every step and at reset, the environment reports identity
  and position of all targets within Manhattan distance <= 1. This is the
  only location-information channel for ALL agents.

## 4. Training / validation / test separation

* **Source (training) environments**: layouts 20000+r. All transferable
  knowledge is acquired here: the Q-table for `q_transferred`, the task schema
  for the schema agent.
* **Validation environments**: layouts 7, 77, 777. Used only for manual
  sanity checks while developing (learning works, metrics behave). No
  hyperparameter was numerically tuned anywhere; the values in `config.py`
  were chosen a priori from standard practice (alpha=0.2, gamma=0.95,
  epsilon 1.0→0.05 with 0.98 per-episode decay) and then sanity-checked on
  the validation layouts. Nothing was tuned on the test environments.
* **Test (target) environments**: layouts 30000+r (40000+r for Exp. D).
  Reported results come exclusively from these.

## 5. Interaction budgets

Budget is measured in **environment interaction steps** and is identical for
every compared agent:

| Phase | Budget |
|---|---|
| Q-agent source training | 200 episodes, 1500-step cap each |
| Schema source training | 60 random-walk episodes + 5 consolidated episodes, 1500-step cap each |
| Target-environment evaluation (all agents) | 60 episodes, 400-step cap each (total 24,000 steps) |
| Oracle planner reference | 10 episodes, 400-step cap each (not compared) |

The target budget binds per episode: an episode ends at 400 steps or when the
cycle completes, whichever comes first. All agents get the same number of
episodes with the same cap; a failing agent simply exhausts the cap. Runs that
never complete the first cycle are recorded as failures (success flag False)
and are never dropped from any statistic.

**Source budgets differ across agent families, and target budgets are what is
held equal.** In the target environment every compared agent gets exactly the
same budget (60 x 400). In the SOURCE environment the families receive
different amounts of experience: the transferred Q agent trains for 200
episodes, the schema agent for 65 (60 random-walk + 5 consolidated), and
`q_fresh` for none (that is its defining property). The study therefore
compares transfer-ready agents after family-typical source training -- it is
NOT a compute-matched or experience-matched comparison of the source phase
(see `docs/limitations.md` section 4).

**Computation is not equal across agents and is not claimed to be.** The
schema agent runs one BFS per navigation step (reported as
`bfs_calls_at_first_cycle`, mean 17.7 calls before its first cycle in
Experiment A) and its source-training wall time is about 2x the Q agent's.
Q-learning performs no planning calls. Wall-clock times are recorded per run
(`*_wall_s` columns in `expA_per_run.csv`) and are machine-dependent. All
scientific claims rest exclusively on environment-step counts, which are
computed identically for all agents.

## 6. Fairness constraints and information asymmetries

1. Identical environments across agents (same layout seeds, same task graph).
2. Identical interaction budgets in the target environment.
3. What the schema agent receives versus what it must learn:
   * **GIVEN at wiring time (disclosed asymmetry): the COMPLETE obstacle map**
     of the current environment, via `SchemaAgent.set_obstacles`, called by
     the experiment scripts (`run_utils.train_schema_source`,
     `run_target_schema`, `ablation.py`, `graph_similarity.py`). The map is
     provided in full at initialization -- the agent does NOT have to
     discover it by bumping. It is identical for every schema-type agent
     within a run and contains no target positions. In the ORIGINAL
     experiments the Q-learning baselines receive NO equivalent
     information: they can only learn obstacle locations implicitly from
     bump penalties. **Experiment E**
     (`experiments/obstacle_info_control.py`) is the dedicated control that
     quantifies this asymmetry (see section 11).
   * LEARNED from perception only: target identities and positions (the
     spatial memory is written exclusively from `obs["visible"]`;
     unit-tested: memory == revealed set). This remains the only
     target-location channel for ALL agents.
   * LEARNED from reward feedback only: the task schema (unit-tested:
     counts only update on positive rewards).
   * BFS uses only the obstacle map and contains no target information;
     BFS goal cells are always discovered positions, never true positions
     of undiscovered targets. Note BFS treats target cells as ordinary
     traversable cells, so shortest paths may cross known-but-unacceptable
     targets (section 9b).
4. `q_fresh` by definition uses no source experience, so it receives no source
   training; this is the baseline's defining property, not an asymmetry.
5. `q_transferred` copies the FULL Q-table (all cells x stages x actions) from
   the source-trained agent and never silently resets it (unit-tested). Its
   initial epsilon is 0.3 rather than 1.0: a transferred agent should not act
   as if it were naive, and with epsilon = 1.0 the transferred table would
   never drive actions at all. This choice is documented as a design decision
   (not tuned); sensitivity to it is noted in `docs/limitations.md`.
6. No seed selection, no run pruning: all 30 runs of every agent are reported.
7. The main comparison (Experiment A) compares ARCHITECTURES as bundles: the
   schema agent differs from the Q agents in its given obstacle map + BFS
   planning + task-state FSM + spatial memory together. Experiment A is
   therefore NOT an isolated test of "abstract task structure"; Experiment B
   isolates two components within the schema family, and Experiment E
   isolates the obstacle-information component relative to the Q baselines.

## 7. Evaluation procedure

In the target environment each agent plays its 60 episodes with learning
enabled (epsilon-greedy for Q agents; the schema policy is deterministic given
its memory except during exploration). Episode logs record: steps, completed
flag, completion step, reward, first-perception step of every target identity,
and cumulative BFS calls. `analysis/metrics.summarize_run` converts logs into
per-run metrics; experiment scripts aggregate into `*_summary.csv` tables and
figures.

## 8. Statistical methods

* **Success rates**: Wilson score interval (z = 1.96) — well-behaved near 0%
  and 100%, where the normal approximation fails.
* **Step counts**: mean and median over runs, with a 95% percentile-bootstrap
  confidence interval for the mean (10,000 resamples of the 30 run values,
  fixed seed 12345). Bootstrap is used because step-to-first-cycle
  distributions are right-skewed and multi-modal (fast schema completions vs
  cap-limited Q-learning episodes), so normality cannot be assumed.
  **Conditional on success:** means of "steps to first cycle" are computed
  over SUCCESSFUL runs only (failures have no defined cycle time and are
  never dropped from the success-rate column instead). They are therefore
  conditional summaries, NOT overall performance: an agent with a low
  success rate can have a small successful-run mean. Experiment D
  additionally reports budget-censored medians (failures counted at the
  24,000-step budget), a standard right-censoring convention, so all four
  task types are comparable.
* Failures are never imputed as successes; censored values are labelled as
  such wherever used.
* Spearman rank correlations in Experiment D are descriptive: with only 4
  task types there is no meaningful significance test, and none is claimed.

## 9a. Verified mechanism behind Experiment C's 0%

A step-by-step trace (reproducible with the seeds in section 1) shows the
failure mode of the frozen wrong schema: after collecting A, the schema
expects B but the environment expects C. The agent navigates to B, is punished
(-5), stands on B, its planning finds "already at the expected target", falls
back to a random step that keeps it in B's neighbourhood, and its avoidance
rule ("never enter a known target that is not currently acceptable") makes it
step *around* C forever. It can therefore never collect the identity the
environment wants. This is a genuine property of the agent design (a strong
wrong prior + consistent avoidance), not a bug: the same avoidance rule is
what makes it efficient when the schema is correct.

Verified against representative traces (runs 0-2 of the full protocol): after
A is collected, over 98% of all remaining steps stay within Manhattan
distance 2 of B; C is never entered once in ~24,000 post-A steps per run; the
only other entries are occasional wrong entries of D (e.g. 119 in run 1),
none of which the true incongruent task accepts at that stage. All 30 runs
fail in the summary table with this same signature.

## 9b. Verified mechanism behind Experiment D's branch-task successes

An earlier version of this document attributed the branch task's 16.7%
success rate to "the rare fallback case where every legal move is forbidden
and the agent must enter a forbidden cell". **That explanation was wrong.**
A trace of the five successful branch runs (runs 16, 20, 25, 26, 28) shows a
different and simpler mechanism:

* Every one of the five successes contains exactly one entry of a target the
  agent's frozen schema did NOT expect, and it happened while the agent was
  in BFS **navigation** mode (planning toward a schema-expected target),
  never in the random-walk fallback. No success involved the fallback.
* BFS treats target cells as ordinary traversable cells (it blocks only
  obstacle cells), so a shortest path toward the schema-expected target can
  pass *through* the cell of another target. In the branch task the frozen
  schema expects C after B; the shortest path from B toward C sometimes runs
  through D's cell. D is exactly what the TRUE branch task accepts after B
  (edge B->D), so the crossing entry is rewarded (+10), advances the true
  task and -- via the reward event -- the agent's own FSM, and with
  `completion_length = 3` the task completes.
* Run 28 shows the second variant: while navigating toward B after A, the
  BFS path crossed C, which the true branch task accepts from A
  (A -> {B, C}); frozen (C->D) and true (C->D) expectations then agreed, and
  normal navigation finished the task.
* The wrong-target penalty (-5) is what makes crossings survivable for the
  AGENT's plan: its FSM advances only on positive rewards, so entering a
  frozen-unacceptable target costs steps and reward but never corrupts the
  plan.

Regression tests pin this behaviour: `tests/test_environment.py` function
`test_bfs_paths_treat_target_cells_as_traversable` (BFS semantics) and
`tests/test_agents.py` function
`test_bfs_navigation_may_cross_known_unacceptable_targets` (agent follows
the crossing path). The avoidance rule deliberately
remains scoped to the random-walk fallback; changing the policy to forbid
BFS crossings would alter the original experimental protocol rather than fix
a bug, and is therefore NOT done.

## 10. Hyperparameters (fixed a priori)

| Parameter | Value | Where |
|---|---|---|
| Q learning rate alpha | 0.2 | config.Q_ALPHA |
| Discount gamma | 0.95 | config.Q_GAMMA |
| Epsilon start (fresh / transferred) | 1.0 / 0.3 | config.Q_EPS_START_* |
| Epsilon floor / decay | 0.05 / 0.98 per episode | config.Q_EPS_MIN, Q_EPS_DECAY |
| Schema edge threshold | count >= 3 AND >= 50% of row max | config.SCHEMA_MIN_COUNT |
| Step / correct / wrong rewards | -0.04 / +10 / -5 | config.REWARD_* |
| Grid, obstacles, start | 7, 6, (3,3) | config.* |

None of these was optimized on test data; see section 4.

## 11. Experiment E: obstacle-information control (separate protocol)

Experiment E (`experiments/obstacle_info_control.py`) quantifies how much of
the schema agent's measured advantage over the Q baselines is attributable to
the obstacle-information asymmetry disclosed in section 6. It is a CONTROL
experiment: nothing in Experiments A-D is modified, and all outputs carry the
`expE_` prefix in `results/tables/` and `results/figures/`.

**What is held identical to Experiment A.** Source/target layout seeds
(20000+r / 30000+r), agent seeds (50000+r), the default cycle task, target
budget (60 episodes x 400 steps), and all hyperparameters.

**What differs, per control agent** (each output row records its protocol in
the `obstacle_info` and `source_training` columns):

| Agent | Obstacle information | Source training |
|---|---|---|
| `q_fresh_masked` | complete map given at target-env initialization; used ONLY to mask moves into obstacles/off-grid at action selection (exploration samples uniformly among admissible moves; greedy = argmax over admissible moves). Never bumps. No planning. | none (fresh table) |
| `q_transferred_masked` | same masking as above | unmasked source training, identical to Experiment A (same seeds reproduce the exact original table; verified by `tests/test_reproducibility.py::test_control_uses_same_source_table_as_original_protocol`); epsilon 0.3 |
| `q_transferred_maskcons` | same masking as above | re-trained WITH the mask active in the source environment (mask-consistent lifecycle) |
| `q_fresh_bumpmem` | NONE at start; the set of blocked (position, action) pairs grows only from the agent's own bump events (perceivable by touch), reported to the agent after each step by the episode runner; no bump is ever repeated | none (fresh table) |

**What the control does and does not equalise.** It gives Q-learning the same
obstacle INFORMATION the schema agent has, in the weakest possible form
(action masking). It does NOT give Q-learning model-based planning (BFS),
the task-state FSM, or the schema agent's spatial-memory wiring; those
remain part of the bundled architectural difference (section 6, item 7). The
gap between the masked Q agents and the schema agent in Experiment A is
therefore an upper bound on the combined effect of planning + task schema +
spatial memory.

**Outputs.** `expE_per_run.csv`, `expE_summary.csv`, `expE_curves.csv`,
`expE_obstacle_control.png`, and
`expE_vs_expA_comparison.csv` -- the latter juxtaposes the ORIGINAL
Experiment A summary with the control summary for convenience; every row is
labelled with its protocol, and rows from the two protocols must NEVER be
pooled into a single estimate. Smoke-mode outputs of ALL experiments are
written to `results/smoke/` and are never reported.
