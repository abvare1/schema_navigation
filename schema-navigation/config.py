"""Central configuration for the schema-navigation project.

Every important environment and experiment parameter lives in this one file so
that the whole study can be inspected and modified from a single place.

Design notes
------------
* Reward values, hyperparameters and budgets are all chosen a priori and
  documented in docs/methodology.md.  They were NOT tuned on the final test
  environments (only sanity-checked on separate validation layouts).
* "Budget" always means *environment interaction steps*, never planning
  operations.  Planning cost (BFS calls, wall-clock time) is reported
  separately by the experiment scripts.
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
GRID_SIZE = 7                 # 7x7 grid
N_OBSTACLES = 6               # number of obstacle cells in generated layouts
START_POSITION = (3, 3)       # fixed start cell for all layouts
TARGET_IDS = ["A", "B", "C", "D"]

# Reward structure (see docs/mathematical_background.md for the rationale)
REWARD_STEP = -0.04           # small per-step penalty to discourage wandering
REWARD_CORRECT = 10.0         # entering the cell of the *expected* target
REWARD_WRONG = -5.0           # entering the cell of any non-expected target

MIN_TARGET_SEPARATION = 2     # Manhattan distance required between any 2 targets
MIN_REACHABLE_CELLS = 30      # free connected component must contain >= 30 cells

# ---------------------------------------------------------------------------
# Task definitions (directed task graphs over target identities)
# ---------------------------------------------------------------------------
def sequence_to_task_graph(sequence):
    """Convert a target sequence, e.g. ["A","B","C","D","A"], to a task graph.

    A task graph is a plain dict with two entries:
      successors:       node -> list of target identities collectable next.
                        "START" is a pseudo-node used before any collection.
      completion_length number of correct collections that finish one cycle.
    A cyclic sequence "A->B->C->D->A" therefore has completion_length 5.
    """
    successors = {"START": [sequence[0]]}
    for current, nxt in zip(sequence, sequence[1:]):
        successors.setdefault(current, [])
        if nxt not in successors[current]:
            successors[current].append(nxt)
    for tid in TARGET_IDS:
        successors.setdefault(tid, [])
    return {"successors": successors, "completion_length": len(sequence)}


# Default (congruent) task used for training and for Experiment A / B
DEFAULT_SEQUENCE = ["A", "B", "C", "D", "A"]
DEFAULT_TASK = sequence_to_task_graph(DEFAULT_SEQUENCE)

# Schema-incongruent task used for Experiment C (same identities, new order)
INCONGRUENT_SEQUENCE = ["A", "C", "B", "D", "A"]
INCONGRUENT_TASK = sequence_to_task_graph(INCONGRUENT_SEQUENCE)

# Task graph library for Experiment D (structural-similarity extension).
# All four graphs share the entry edge START->A; they differ afterwards.
TASK_GRAPH_LIBRARY = {
    "cycle": DEFAULT_TASK,   # A->B->C->D->A (the source task itself)
    "chain": {
        "successors": {"START": ["A"], "A": ["B"], "B": ["C"],
                       "C": ["D"], "D": []},
        "completion_length": 4,
    },
    "branch": {
        "successors": {"START": ["A"], "A": ["B", "C"], "B": ["D"],
                       "C": ["D"], "D": []},
        "completion_length": 3,
    },
    "reverse_cycle": {
        "successors": {"START": ["A"], "A": ["D"], "D": ["C"],
                       "C": ["B"], "B": ["A"]},
        "completion_length": 5,
    },
}

# ---------------------------------------------------------------------------
# Q-learning hyperparameters (chosen a priori; see docs/methodology.md)
# ---------------------------------------------------------------------------
Q_ALPHA = 0.2                 # learning rate
Q_GAMMA = 0.95                # discount factor
Q_EPS_START_FRESH = 1.0       # initial epsilon for an agent with no experience
Q_EPS_START_TRANSFERRED = 0.3 # initial epsilon for the Q-table-transfer agent
Q_EPS_MIN = 0.05
Q_EPS_DECAY = 0.98            # multiplicative decay, applied once per episode

# ---------------------------------------------------------------------------
# Interaction budgets (full mode; unit = environment steps)
# ---------------------------------------------------------------------------
SOURCE_Q_EPISODES = 200       # episodes to train a Q agent in the source env
SOURCE_MAX_STEPS = 1500       # per-episode step cap during source training
SCHEMA_EXPLORE_EPISODES = 60  # random-walk episodes used to learn the schema
SCHEMA_CONSOLIDATE_EPISODES = 5  # BFS-navigation episodes after schema derives
TARGET_EPISODES = 60          # episodes allowed in the new (target) environment
TARGET_MAX_STEPS = 400        # per-episode step cap in the target environment
PLANNER_EPISODES = 10         # oracle upper-bound reference episodes

N_RUNS = 30                   # independent (source, target) run pairs (>= 30)
SCHEMA_MIN_COUNT = 3          # min observations before an edge enters the schema
BOOTSTRAP_RESAMPLES = 10000

# ---------------------------------------------------------------------------
# Seeds (see docs/methodology.md for the full seed scheme)
# ---------------------------------------------------------------------------
SEED_SOURCE_BASE = 20000      # source layout seed = SEED_SOURCE_BASE + run
SEED_TARGET_BASE = 30000      # target layout seed = SEED_TARGET_BASE + run
SEED_EXP_D_TARGET_BASE = 40000  # Experiment D target layout seed = base + run
SEED_AGENT_BASE = 50000       # agent exploration rng seed = base + run
SEED_BOOTSTRAP = 12345        # fixed seed for bootstrap confidence intervals
VALIDATION_SEEDS = [7, 77, 777]  # validation layouts (hyperparameter sanity only)

# ---------------------------------------------------------------------------
# Smoke mode: a fast end-to-end pipeline check with far fewer seeds/episodes.
# Full mode must use N_RUNS >= 30 (see README).
# ---------------------------------------------------------------------------
SMOKE_OVERRIDES = {
    "source_q_episodes": 30,
    "source_max_steps": 800,
    "schema_explore_episodes": 15,
    "schema_consolidate_episodes": 2,
    "target_episodes": 8,
    "target_max_steps": 300,
    "planner_episodes": 3,
    "n_runs": 3,
}
