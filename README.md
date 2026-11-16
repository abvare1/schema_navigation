# Schema-Based One-Shot Transfer in Sequential Spatial Navigation

A minimal, reproducible computational study comparing three learning agents in a
7x7 Grid World where targets must be visited in a fixed order.

**What "one-shot" means here (and what it does not mean).** It refers to one
narrow operational fact: after a schema agent perceives a target *once*, its
location is retained and subsequently navigated to optimally (spatial memory
is written from a single perception event). It does NOT mean one-shot task
learning: the task schema itself is learned from ~65 source-environment
episodes and transferred, and the agent still needs ~215 steps of wandering
to discover all four targets before its first complete cycle in a new
environment. See `docs/limitations.md` section 2.

**Research question.** Does separating abstract task structure (the sequence of
target identities) from spatial knowledge (the locations of targets) improve
transfer to a new spatial environment, compared with conventional tabular
Q-learning? And what happens to that advantage when the task structure itself
changes?

**Scope caveat (important).** The schema agent differs from the Q-learning
baselines in several bundled ways: it is *given the complete obstacle map* of
each environment (the Q baselines are not), it plans with BFS, it maintains a
task-state FSM, and it keeps spatial memory. The main comparison (Experiment
A) therefore compares architectures as bundles, not an isolated test of
abstract task structure. Experiment B isolates two components within the
schema family; Experiment E isolates the obstacle-information component
(`experiments/obstacle_info_control.py`).

This project is **inspired by** research on schema learning and rapid
generalization in spatial navigation (e.g. Tse et al., 2007; Behrens et al.,
2018 — see References). It is **not** a reproduction of any published
experimental protocol; the Grid World, tasks and agents are independent,
deliberately simplified designs. All simplifications and their consequences are
documented in `docs/limitations.md`.

---

## 1. Headline results (full experiment mode, 30 independent runs)

All numbers are **environment interaction steps in a NEW environment** (target
layouts never seen during training). "First cycle" = completing the whole
sequence A→B→C→D→A once. Budget: 60 episodes x 400 steps in the new
environment. Full protocol: `docs/methodology.md`.

**Step means below are over SUCCESSFUL runs only** — a conditional summary,
not overall performance (failures have no cycle time; they are counted in the
success-rate column instead). An agent with a low success rate can have a
small successful-run mean. Experiment D additionally reports budget-censored
medians so that 0%-success task types remain comparable.

### Experiment A — main transfer experiment (new target locations, same task)

| Agent | Success rate (Wilson 95% CI) | Mean steps to first cycle, successful runs only (bootstrap 95% CI) | Mean steps of later cycles |
|---|---|---|---|
| Q-learning from scratch | 100% [88.6, 100] | 2095.9 [1658.8, 2557.3] | 94.3 |
| Q-learning + transferred Q-table | 66.7% [48.8, 80.8] | 11792.3 [8519.7, 15052.5] | 86.2 |
| Schema-based agent | 100% [88.6, 100] | **215.3 [164.0, 269.9]** | **24.9** |
| Oracle planner (reference, not compared) | — | 24.9 | — |

Findings: the schema agent completes its first cycle about **10x faster** than
Q-learning from scratch, and its steady-state cycle length (24.9 steps) matches
the oracle planner (24.9). Transferring the raw spatial Q-table causes
**negative transfer**: only 66.7% of runs succeed within the same budget that
the from-scratch agent masters 100% of the time, and successful transferred
runs are 5.6x slower. Because the schema agent is given the obstacle map and
plans with BFS, part of this gap is bundled architectural difference, not pure
task-structure transfer — Experiment E below quantifies the obstacle-map part.

### Experiment B — ablations of the schema agent

| Variant | Success rate | Mean steps to first cycle, successful runs only | Mean later-cycle steps |
|---|---|---|---|
| Schema agent (full) | 100% | 215.3 | 24.9 |
| Without spatial memory | 100% | 599.3 | 243.0 |
| Without task state | **0%** [0, 11.4] | — | — |

Removing the task-state representation is catastrophic (the agent visits known
targets in random order and almost never matches the required sequence).
Removing spatial memory still succeeds in this small grid, but is ~2.8x slower
on the first cycle and ~10x slower afterwards (it cannot plan with BFS and must
re-explore every cycle).

### Experiment C — schema-incongruent transfer (new locations AND new task order)

Source task A→B→C→D→A; target task A→C→B→D→A. The schema agent carries its
**frozen** learned schema.

| Agent | Success rate | Mean steps to first cycle, successful runs only |
|---|---|---|
| Q-learning from scratch | 100% | 2006.9 |
| Q-learning + transferred Q-table | 60% | 11575.1 |
| Schema-based agent (frozen wrong schema) | **0%** [0, 11.4] | — |

The schema agent does not merely lose its advantage — it fails completely, and
is now *worse* than learning from scratch. Trace analysis
(`docs/methodology.md`, section 9a) shows why: its confident wrong predictions,
combined with its avoidance rule "never step onto a known target that is not
currently acceptable" (which applies to its random-walk exploration), make it
orbit a target it expects but the environment rejects — traced runs spend
over 98% of their post-A steps within distance 2 of B and never enter C —
while systematically avoiding the target the environment actually wants. A
strong but wrong prior is worse than no prior. (An adaptive variant that
detects and repairs the mismatch is listed as future work in
`docs/limitations.md`.)

### Experiment D — does task-graph similarity predict transfer?

Target task graphs (edge-Jaccard similarity to the source cycle in
parentheses): cycle (1.00), chain A→B→C→D (0.80), branch A→{B,C}→D (0.43),
reverse cycle A→D→C→B→A (0.11). Schema-agent success rates: 100%, 100%,
16.7%, 0%. Spearman rank correlation between similarity and success rate:
**0.95**; between similarity and budget-censored median steps: **−0.74**.
With only 4 task types these correlations are descriptive, not inferential.

Mechanism note: the branch-task successes come from BFS shortest paths that
pass *through* a known-but-currently-unacceptable target which the true task
happens to accept at that point (BFS treats target cells as ordinary
traversable cells; the avoidance rule covers only random-walk exploration).
This is pinned by regression tests and traced run-by-run in
`docs/methodology.md`, section 9b.

### Experiment E — obstacle-information control (separate protocol, new)

The schema agent is *given* the complete obstacle map of each environment; in
the original experiments the Q baselines receive **no** obstacle information.
Experiment E asks what that asymmetry is worth, by giving Q-learning agents
obstacle information and nothing else. Same layouts, seeds, task and budget
as Experiment A; results in separate `expE_*` files; every output row states
its protocol. Each row below is a DIFFERENT agent under a DIFFERENT
information condition — nothing here is pooled with Experiment A.

| Control agent | Obstacle information | Success rate | Mean steps to first cycle, successful runs only |
|---|---|---|---|
| Q fresh + full map (masked) | complete map at init; used only to mask invalid moves (never bumps, no planning) | 100% [88.6, 100] | 1077.0 [721.9, 1476.5] |
| Q fresh + bump-learned memory | no map; blocked moves learned from own bumps, then avoided | 100% [88.6, 100] | 1460.7 [1047.8, 1936.0] |
| Q transferred + full map (masked) | as masked; source Q-table reused unchanged from the original protocol | 16.7% [7.3, 33.6] | 6010.8 [2058.6, 10480.2] |
| Q transferred + mask-consistent source | as masked; source table re-trained with the mask | 26.7% [14.2, 44.4] | 6496.9 [2750.3, 11013.5] |

What this control shows, and what it does not:

* **Obstacle information explains part — not most — of the schema agent's
  first-cycle advantage.** Masking halves the from-scratch agent's cost
  (2095.9 → 1077.0), and even bump-learned memory captures most of that
  benefit (1460.7). The schema agent (215.3) remains ~5x faster than the
  best masked Q agent, so planning, task structure and spatial memory
  account for the larger share of the gap.
* **Negative Q-table transfer is robust to obstacle-information
  equalisation — it gets worse.** The transferred agent with the map succeeds
  only 16.7% of the time (vs 66.7% without it); the mask-consistent lifecycle
  recovers only to 26.7%. A plausible reading (interpretation, not a traced
  mechanism): bump-free movement lets the agent exploit its confident wrong
  spatial prior more consistently, with fewer of the accidents that
  previously broke it out of that policy. The negative-transfer finding does
  not hinge on the missing obstacle map.
* The control does NOT give Q-learning BFS planning, so the remaining
  schema-vs-masked-Q gap still bundles planning with structure and memory.

### Interpretation in one paragraph

Abstract task structure transfers well **when it is correct**: the schema agent
reaches near-optimal navigation after discovering each target just once — and
this advantage is only partly explained by its privileged obstacle map
(Experiment E). Raw spatial value transfer does not transfer at all — it
hurts, and the harm survives obstacle-information equalisation. But the same
mechanism that makes correct structure powerful makes wrong structure
crippling: with a frozen, wrong schema the agent is trapped by its own
predictions. The structure must be right, and nothing in this minimal design
lets the agent notice when it is not.

---

## 2. Motivation

Animals and humans often learn the *structure* of a new environment very fast:
after discovering where one reward is, they exploit knowledge about regularities
("schemas") to place other rewards with few observations. In machine learning
terms, one hypothesis is that what transfers between related tasks is not a
full state-action value function (which entangles structure with geography) but
a compact, abstract description of the task. This project builds the smallest
setting we could think of in which that hypothesis can be tested quantitatively,
fairly, and reproducibly:

* abstract structure = the ordered sequence of target identities (a finite-state
  machine);
* spatial knowledge = the mapping identity → location, learned by observation;
* the comparison baseline = tabular Q-learning over (position, task stage),
  both from scratch and with a transferred Q-table.

## 3. Environment

A 7x7 Grid World (`environment/grid_world.py`):

* one agent, four targets **A, B, C, D**, 6 obstacle cells, fixed start (3,3);
* actions: up/down/left/right (deterministic; bumping a wall/obstacle keeps
  the agent in place and fires no target event);
* rewards (configurable in `config.py`): −0.04 per step, +10 for entering the
  cell of the target the task currently expects, −5 for entering any other
  target's cell;
* an episode ends when the task's full sequence has been traversed once
  (5 correct collections for the cycle task);
* **perception rule**: targets are invisible from afar; after every move (and
  at spawn) the agent perceives the identity and position of every target
  within Manhattan distance ≤ 1 of itself. The environment itself never
  reveals anything else — this is the only channel through which any agent
  can learn a target location.
* **disclosed asymmetry (obstacle maps)**: the experiment wiring additionally
  hands the schema agent the COMPLETE obstacle map of its current environment
  at initialization (`SchemaAgent.set_obstacles`); the agent does not have to
  discover it by bumping. In the original experiments the Q-learning
  baselines receive NO equivalent information. Experiment E measures what
  this is worth (see section 1).

Layouts are generated procedurally from integer seeds with hard validity
constraints (no overlaps, pairwise target separation ≥ 2, every target
reachable, ≥ 30 free reachable cells). Invalid layouts raise `ValueError`.

## 4. Agents (`agents/`)

1. **`q_learning.py` — TabularQAgent.** State = (cell, progress counter k).
   Epsilon-greedy exploration with per-episode decay; standard Q-learning
   update `Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))` with the
   future term zeroed at episode end. `q_fresh` starts from an empty table in
   every new environment; `q_transferred` starts from a **copy** of the full
   Q-table trained in the source environment (lower initial epsilon, 0.3, so
   the transferred values actually drive behaviour — see `docs/methodology.md`).
2. **`schema_agent.py` — SchemaAgent.** Two levels:
   * *task schema*: a finite-state machine `last collected identity →
     acceptable next identities`, **learned from reward feedback** (counting
     which identity yields +10 right after which other identity, then
     thresholding) — never given for free;
   * *spatial memory*: identity → position, written **only** from perception,
     and **empty in every new environment**.
   Navigation: BFS shortest path to the nearest known acceptable target;
   otherwise a uniform random walk that never deliberately enters a known
   target that is currently unacceptable. Note the avoidance rule applies
   ONLY to the random walk: BFS treats target cells as ordinary traversable
   cells, so a shortest path may pass through a known-but-unacceptable
   target (regression-tested; this is what produces the branch-task
   successes in Experiment D — see `docs/methodology.md` section 9b).
3. **`planner.py` — OraclePlanner (upper-bound reference only).** Knows the
   true target positions and the true task graph; walks BFS paths. Explicitly
   excluded from the fair comparison.
4. **`q_learning_masked.py` — ObstacleAwareQAgent (Experiment E only).**
   Tabular Q-learning with obstacle information, in two variants: `full_map`
   (complete map given at wiring time, used ONLY to mask invalid moves) and
   `bump_memory` (nothing given; blocked moves learned from the agent's own
   bumps, then avoided). Not used by the original experiments.

Fairness properties (unit-tested in `tests/test_agents.py`): the schema agent's
memory contains exactly the targets the environment revealed; its schema is
derived only from its own reward events; its BFS uses only the obstacle map.
**Disclosed information asymmetry (NOT equalised in the original
experiments): the schema agent receives the complete obstacle map at
initialization, the Q-learning baselines receive no obstacle information.**
The target-location channel (perception) is identical for all agents.
Experiment E (`experiments/obstacle_info_control.py`) is the control that
quantifies the asymmetry.

## 5. Mathematical background

See `docs/mathematical_background.md` for MDPs, the Q-learning equation,
epsilon-greedy exploration, finite-state task schemas, BFS, transfer/negative
transfer, and confidence intervals — each with an equation, an intuition and a
tiny worked example.

## 6. Experimental design

30 independent runs (seeds). Each run trains all agents in one **source**
environment (seed 20000+r) and evaluates them in one **new** environment
(seed 30000+r; the same layout seeds are shared by every agent). Separate
validation layouts (seeds 7, 77, 777) exist for hyperparameter sanity checks;
the test environments were never used for tuning. Hyperparameters were fixed a
priori (see `docs/methodology.md`). Statistics: Wilson score intervals for
success rates, percentile-bootstrap intervals (10,000 resamples, fixed seed)
for step counts, explicit recording of every failed run.

**Source budgets differ across agent families** (transferred Q agent: 200
source episodes; schema agent: 65; `q_fresh`: none by definition). What is
held equal is the TARGET-environment budget (60 x 400 for every compared
agent). This is a comparison of transfer-ready architectures, not a
compute- or experience-matched source-training comparison.

Five experiments (`experiments/`): **A** main transfer, **B** ablations
(`schema_no_memory`, `schema_no_task_state`), **C** incongruent transfer,
**D** task-graph similarity (cycle / chain / branch / reverse-cycle), and
**E** obstacle-information control (`obstacle_info_control.py`; separate
protocol, separate output files, does not modify A-D).

## 7. Metrics (`analysis/metrics.py`)

1. steps until the first complete target cycle in the new environment
   (cumulative across episodes; failures recorded explicitly, never dropped);
2. steps of subsequent complete cycles;
3. success rate within the fixed budget (Wilson CI);
4. per-episode learning curves (censored at the per-episode cap);
5. number of distinct targets observed before the first completed cycle;
6. delay between first perceiving D and completing the cycle (operational
   definition — *not* claimed to be "one-shot learning" evidence by itself);
7. performance under task-structure change (Experiments C and D).

Means of step-based metrics are computed over successful runs only and are
labelled as such everywhere (see section 1); failures are recorded in the
success-rate column, never dropped.

## 8. Installation

```bash
cd schema-navigation
python -m venv venv && source venv/bin/activate   # optional
pip install -r requirements.txt                   # numpy, pandas, matplotlib, pytest
```

Tested versions (the exact environment the reported numbers were produced
with):

| Component | Tested version |
|---|---|
| Python | 3.12.16 |
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| matplotlib | 3.9.2 |
| pytest | 9.0.2 |

`requirements.txt` keeps lower bounds only; any reasonably recent versions
should work, but exact reproduction of the reported numbers was verified on
the versions above.

License: MIT (see `LICENSE`).

## 9. Running

```bash
python -m pytest tests/ -q                        # all unit + reproducibility tests
python -m experiments.main_transfer --smoke       # fast end-to-end pipeline check
python -m experiments.main_transfer               # Experiment A (30 runs)
python -m experiments.ablation                    # Experiment B (30 runs)
python -m experiments.incongruent_transfer        # Experiment C (30 runs)
python -m experiments.graph_similarity            # Experiment D (30 runs)
python -m experiments.obstacle_info_control       # Experiment E (30 runs, control)
```

Every script accepts `--smoke` (tiny budgets, 3 runs, for testing the plumbing
only) and `--runs N` (override the number of runs). Full mode (no flags) uses
30 runs and is the only mode that produces the reported numbers. Full mode
takes roughly 1-2 minutes per original experiment and ~4 minutes for the
control on a laptop.

**Smoke outputs are written to `results/smoke/`** and full-mode outputs to
`results/` — smoke output can never be confused with reported results, and
`results/smoke/` is git-ignored.

## 10. Reproducing

Everything is seeded: layout seeds, agent exploration seeds and the bootstrap
seed are fixed in `config.py`. `tests/test_reproducibility.py` re-runs the
whole Experiment A pipeline (and the Experiment E control) twice and asserts
bit-identical step metrics. Rerunning the five full-mode commands regenerates
every CSV and figure in `results/` deterministically (wall-clock-time columns
are machine-dependent by nature and excluded from the equality check).

## 11. Expected output files

```
results/
  tables/  expA_per_run.csv expA_curves.csv expA_summary.csv
           expB_per_run.csv expB_curves.csv expB_summary.csv
           expC_per_run.csv expC_curves.csv expC_summary.csv
           expD_per_run.csv expD_curves.csv expD_summary.csv   (planner per run)
           expE_per_run.csv expE_curves.csv expE_summary.csv   (control)
           expE_vs_expA_comparison.csv                         (protocol-labelled)
           planner_reference_expA.csv planner_reference_expC.csv
  figures/ expA_learning_curves.png expA_first_cycle.png
           expB_ablation_curves.png expB_ablation.png
           expC_incongruent_curves.png expC_incongruent.png
           expD_similarity.png expE_obstacle_control.png
  smoke/   (git-ignored; smoke-mode outputs of all experiments)
```

Experiment E files are written by the control script only and never touch the
expA-expD files; `expE_vs_expA_comparison.csv` juxtaposes the two protocols
with an explicit `protocol` column — its rows must not be pooled.

## 12. Limitations

Honest discussion in `docs/limitations.md`: single simplified grid world;
deterministic transitions; small state space where random exploration is cheap;
tabular Q-learning scale limits; the schema agent's BFS makes its *computation*
more expensive than Q-learning even though *interaction steps* are comparable;
the disclosed obstacle-map asymmetry and what Experiment E does (and does not)
equalise; the bundled architectural differences behind the main comparison;
source-training budgets that differ across agent families; the frozen-schema
design choice behind Experiment C's catastrophic result; the crude edge-Jaccard
similarity measure; and the gap to biological navigation and to published
protocols.

## 13. References

* Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An
  Introduction* (2nd ed.). MIT Press. — Q-learning, epsilon-greedy, MDPs.
* Tse, D. et al. (2007). Schemas and memory consolidation. *Science*,
  315(5818), 1450–1453. — rapid schema-based learning (biological motivation).
* Behrens, T. E. J. et al. (2018). What is a cognitive map? Organizing
  knowledge for cortico-hippocampal interaction. *Neuron*, 100(2), 490–509.
* Wilson, E. B. (1927). Probable inference, the law of succession, and
  statistical inference. *JASA*, 22, 209–212. — the Wilson score interval.
* Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1), 1–26. — bootstrap confidence intervals.

These works motivate the study; none of the experimental protocols here is
taken from them.

## 14. Suggested reading order (for a newcomer)

1. `README.md` (this file) — the story and the results.
2. `config.py` — every knob in one place.
3. `environment/grid_world.py` — the world and its rules.
4. `docs/mathematical_background.md` — the equations behind the agents.
5. `agents/q_learning.py`, then `agents/schema_agent.py`, then
   `agents/planner.py`.
6. `experiments/run_utils.py` — how episodes are run and logged.
7. `experiments/main_transfer.py` — one full experiment end to end.
8. `docs/methodology.md` — seeds, budgets, fairness constraints.
9. `tests/` — executable proof of the fairness and reproducibility claims.
10. The remaining experiments, then `docs/limitations.md` — what the results
    do and do not show.

## Project structure (and deliberate additions)

```
schema-navigation/
├── README.md                this file
├── LICENSE                  MIT
├── .gitignore
├── requirements.txt
├── config.py                every environment/experiment parameter
├── conftest.py              (added) empty; makes pytest find the package root
├── environment/             grid_world.py  (+ __init__.py)
├── agents/                  q_learning.py  schema_agent.py  planner.py
│                            q_learning_masked.py (Experiment E)
├── experiments/             run_utils.py (added)  main_transfer.py  ablation.py
│                            incongruent_transfer.py  graph_similarity.py
│                            obstacle_info_control.py (Experiment E)
├── analysis/                metrics.py  statistics.py  plots.py
├── results/                 tables/ (CSV)  figures/ (PNG)   [generated]
│                            smoke/  [generated, git-ignored]
├── tests/                   test_environment.py test_agents.py
│                            test_statistics_metrics.py test_reproducibility.py
└── docs/                    methodology.md  mathematical_background.md  limitations.md
```

Additions to the originally requested structure, with reasons:
`experiments/run_utils.py` (episode runners shared verbatim by all four
original experiments — four copies would diverge); `conftest.py` (standard
pytest mechanism for import paths); `experiments/obstacle_info_control.py`
and `agents/q_learning_masked.py` (Experiment E, the obstacle-information
control requested after the fairness audit); `tests/test_statistics_metrics.py`
(pin the statistical helpers).
