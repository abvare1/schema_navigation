"""Shared episode runners and training helpers used by all experiments.

NOTE: this module was added to the required project structure (the spec lists
only the four experiment scripts).  Reason: the episode loops, source
training and metric bookkeeping are identical across experiments, and
duplicating them four times would invite copy-paste divergence.  Everything
here is plain functions, no frameworks.

Why one runner per agent family
    Q agents need the classic (s, a, r, s') loop; schema-type agents choose
    actions through their own two-level policy and learn from reward events;
    the planner just follows BFS.  The runners all return the SAME episode
    log format so analysis/metrics.py can treat every agent identically.
"""

import numpy as np

import config as C
from agents.planner import OraclePlanner
from agents.q_learning import TabularQAgent
from agents.schema_agent import SchemaAgent, make_target_schema_agent
from environment.grid_world import GridWorld, generate_layout


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------
def effective_params(smoke=False, n_runs=None):
    """Collect the experiment parameters for this invocation.

    ``smoke=True`` applies the SMOKE_OVERRIDES from config.py: a fast
    end-to-end pipeline check, NOT a scientific result.  Full mode always
    uses the (>= 30 seeds) defaults from config.py.
    """
    params = {
        "source_q_episodes": C.SOURCE_Q_EPISODES,
        "source_max_steps": C.SOURCE_MAX_STEPS,
        "schema_explore_episodes": C.SCHEMA_EXPLORE_EPISODES,
        "schema_consolidate_episodes": C.SCHEMA_CONSOLIDATE_EPISODES,
        "target_episodes": C.TARGET_EPISODES,
        "target_max_steps": C.TARGET_MAX_STEPS,
        "planner_episodes": C.PLANNER_EPISODES,
        "n_runs": C.N_RUNS,
        "smoke": bool(smoke),
    }
    if smoke:
        params.update(C.SMOKE_OVERRIDES)
        params["smoke"] = True
    if n_runs is not None:
        params["n_runs"] = n_runs
    return params


def make_env(layout_seed, task_graph):
    """Deterministic environment from a layout seed (same seed -> same layout
    for every agent)."""
    return GridWorld(generate_layout(layout_seed), task_graph)


def fresh_q_agent(task, epsilon_start, q_init=None):
    return TabularQAgent(
        grid_size=C.GRID_SIZE,
        n_stages=task["completion_length"],
        alpha=C.Q_ALPHA,
        gamma=C.Q_GAMMA,
        epsilon_start=epsilon_start,
        epsilon_min=C.Q_EPS_MIN,
        epsilon_decay=C.Q_EPS_DECAY,
        q_init=q_init,
    )


# ---------------------------------------------------------------------------
# Episode runners (one per agent family, identical log format)
# ---------------------------------------------------------------------------
def _new_log():
    return {
        "steps": 0,
        "completed": False,
        "completion_step": None,
        "reward": 0.0,
        "first_reveals": {},
        "bfs_calls": 0,
    }


def _track_reveals(log, obs, step):
    for tid in obs["visible"]:
        log["first_reveals"].setdefault(tid, step)


def run_q_episode(env, agent, rng, max_steps, training=True):
    """Run one episode with a TabularQAgent.

    ``training=True``: epsilon-greedy actions + Q updates + epsilon decay.
    ``training=False``: pure greedy evaluation (used by tests).
    """
    log = _new_log()
    obs = env.reset()
    _track_reveals(log, obs, 0)
    for t in range(1, max_steps + 1):
        state = agent.encode_state(obs)
        action = agent.act(obs, rng, greedy=not training)
        obs, reward, done, _info = env.step(action)
        if training:
            agent.update(state, action, reward, agent.encode_state(obs), done)
        log["steps"] = t
        log["reward"] += reward
        _track_reveals(log, obs, t)
        if done:
            log["completed"] = True
            log["completion_step"] = t
            break
    if training:
        agent.end_episode()
    log["bfs_calls"] = 0   # tabular Q-learning performs no planning calls
    return log


def run_schema_episode(env, agent, rng, max_steps):
    """Run one episode with a SchemaAgent (or an ablation variant).

    Perception is written to spatial memory after every step; positive reward
    events drive FSM tracking and (while learning) successor counting.
    Spatial memory persists ACROSS episodes within one environment but is
    reset when the agent moves to a new environment (see make_target_schema_agent).
    """
    log = _new_log()
    bfs_before = agent.bfs_calls
    obs = env.reset()
    agent.reset_episode_state()
    agent.update_perception(obs)
    _track_reveals(log, obs, 0)
    for t in range(1, max_steps + 1):
        action = agent.choose_action(obs, rng)
        obs, reward, done, _info = env.step(action)
        agent.update_perception(obs)
        if reward > 0:
            agent.record_success(obs)
        log["steps"] = t
        log["reward"] += reward
        _track_reveals(log, obs, t)
        if done:
            log["completed"] = True
            log["completion_step"] = t
            break
    log["bfs_calls"] = agent.bfs_calls - bfs_before
    return log


def run_planned_episode(env, planner, max_steps):
    """One greedy episode with the oracle planner (upper-bound reference)."""
    log = _new_log()
    bfs_before = planner.bfs_calls
    obs = env.reset()
    planner.reset_episode_state()
    _track_reveals(log, obs, 0)
    for t in range(1, max_steps + 1):
        action = planner.choose_action(obs)
        obs, reward, done, _info = env.step(action)
        if reward > 0:
            planner.record_success(obs)
        log["steps"] = t
        log["reward"] += reward
        _track_reveals(log, obs, t)
        if done:
            log["completed"] = True
            log["completion_step"] = t
            break
    log["bfs_calls"] = planner.bfs_calls - bfs_before
    return log


# ---------------------------------------------------------------------------
# Source-environment training
# ---------------------------------------------------------------------------
def train_q_source(layout, task, episodes, max_steps, agent_seed):
    """Train a fresh tabular Q agent in the source environment.

    Returns (agent, episode_logs, wall_seconds).  The wall-clock time is
    reported separately from env-step budgets (planning/learning compute is
    NOT comparable across agent families and is never counted as steps).
    """
    import time
    env = GridWorld(layout, task)
    rng = np.random.default_rng(agent_seed)
    agent = fresh_q_agent(task, epsilon_start=C.Q_EPS_START_FRESH)
    logs = []
    t0 = time.perf_counter()
    for _ep in range(episodes):
        logs.append(run_q_episode(env, agent, rng, max_steps, training=True))
    wall = time.perf_counter() - t0
    return agent, logs, wall


def train_schema_source(layout, task, explore_episodes, consolidate_episodes,
                        max_steps, agent_seed):
    """Train the schema agent in the source environment.

    Phase 1 (explore): the schema is empty, so the policy is a uniform random
        walk over valid moves; every positive-reward event feeds the
        successor counts, and perception feeds the (source-only) memory.
    Phase 2 (consolidate): after ``derive_schema`` the agent navigates with
        BFS; a few episodes measure its source performance with planning.

    Obstacle information: the agent is GIVEN the complete obstacle map of
    the source environment at wiring time (``set_obstacles`` below) -- it
    does not have to learn it.  In the original experiments the Q agents
    receive no equivalent map (see docs/methodology.md section 6 and
    experiments/obstacle_info_control.py).

    Returns (agent, logs, wall_seconds).  ONLY the learned schema transfers
    to new environments; the source spatial memory is discarded.
    """
    import time
    env = GridWorld(layout, task)
    rng = np.random.default_rng(agent_seed)
    agent = SchemaAgent(C.GRID_SIZE)
    agent.set_obstacles(env.obstacles, env.size)
    logs = []
    t0 = time.perf_counter()
    for _ep in range(explore_episodes):
        logs.append(run_schema_episode(env, agent, rng, max_steps))
    agent.derive_schema(min_count=C.SCHEMA_MIN_COUNT)
    for _ep in range(consolidate_episodes):
        logs.append(run_schema_episode(env, agent, rng, max_steps))
    wall = time.perf_counter() - t0
    return agent, logs, wall


def run_target_q(agent, env, agent_seed, episodes, max_steps):
    """Adaptation/evaluation episodes for a Q agent in the target env.
    Learning continues (same budget rules as every other agent)."""
    rng = np.random.default_rng(agent_seed)
    logs = []
    for _ep in range(episodes):
        logs.append(run_q_episode(env, agent, rng, max_steps, training=True))
    return logs


def run_target_schema(source_schema, env, agent_seed, episodes, max_steps):
    """Schema agent in the NEW environment: frozen schema, EMPTY memory.

    Obstacle information: the complete obstacle map of the NEW environment
    is provided at wiring time (below) -- the map is not something the agent
    discovers or transfers; Q-learning baselines in the original experiments
    receive no equivalent (see experiments/obstacle_info_control.py)."""
    rng = np.random.default_rng(agent_seed)
    agent = make_target_schema_agent(source_schema, C.GRID_SIZE)
    agent.set_obstacles(env.obstacles, env.size)
    logs = []
    for _ep in range(episodes):
        logs.append(run_schema_episode(env, agent, rng, max_steps))
    return agent, logs


def run_planner_reference(layout, task, episodes, max_steps):
    """Oracle planner reference episodes (NOT part of the fair comparison)."""
    env = GridWorld(layout, task)
    planner = OraclePlanner(layout, task)
    logs = []
    for _ep in range(episodes):
        logs.append(run_planned_episode(env, planner, max_steps))
    return logs
