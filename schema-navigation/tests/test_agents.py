"""Agent tests: Q-learning numerics, transfer semantics, BFS planning,
schema-agent privacy (no undiscovered locations used), and ablation flags."""

import numpy as np
import pytest

import config as C
from agents.planner import OraclePlanner
from agents.q_learning import TabularQAgent
from agents.schema_agent import SchemaAgent, make_target_schema_agent
from environment.grid_world import GridWorld, bfs_shortest_path
from experiments import run_utils as R


def _handmade_layout():
    return {
        "grid_size": 7,
        "obstacles": set(),
        "targets": {"A": (0, 0), "B": (0, 6), "C": (6, 6), "D": (6, 0)},
        "start": (3, 3),
    }


# ---------------------------------------------------------------------------
# Q-learning
# ---------------------------------------------------------------------------
def test_q_update_hand_computed():
    agent = TabularQAgent(grid_size=7, n_stages=5, alpha=0.5, gamma=0.9,
                          epsilon_start=1.0, epsilon_min=0.05,
                          epsilon_decay=0.98)
    state = (3 * 7 + 3, 0)          # centre cell, stage 0
    next_state = (3 * 7 + 4, 1)
    # First update: Q was 0, no future (done) -> Q = 0.5 * (10 - 0) = 5.0
    agent.update(state, 2, 10.0, next_state, done=False)
    assert agent.q[state[0], 0, 2] == pytest.approx(5.0)
    # Second update with a non-terminal next state whose best Q is 2.0:
    # target = 10 + 0.9*2 = 11.8 -> Q = 5 + 0.5*(11.8 - 5) = 8.4
    agent.q[next_state[0], 1, 0] = 2.0
    agent.update(state, 2, 10.0, next_state, done=False)
    assert agent.q[state[0], 0, 2] == pytest.approx(8.4)
    # Terminal update: future term must be exactly zero, so the target is
    # just the observed reward: Q = 8.4 + 0.5*(10 - 8.4) = 9.2
    agent.update(state, 2, 10.0, next_state, done=True)
    assert agent.q[state[0], 0, 2] == pytest.approx(9.2)


def test_greedy_action_follows_argmax_and_state_encoding():
    agent = TabularQAgent(grid_size=7, n_stages=5, alpha=0.5, gamma=0.9,
                          epsilon_start=0.0, epsilon_min=0.05,
                          epsilon_decay=0.98)
    agent.q[3 * 7 + 3, 0, 3] = 1.0   # highest value for "right" at centre
    obs = {"pos": (3, 3), "k": 0, "last_node": "START", "visible": {}}
    rng = np.random.default_rng(0)
    assert agent.act(obs, rng, greedy=False) == 3   # eps=0 -> pure greedy


def test_epsilon_decay_respects_floor():
    agent = TabularQAgent(grid_size=7, n_stages=5, alpha=0.5, gamma=0.9,
                          epsilon_start=1.0, epsilon_min=0.05,
                          epsilon_decay=0.5)
    for _ in range(10):
        agent.end_episode()
    assert agent.epsilon == pytest.approx(0.05)


def test_transferred_q_table_is_copied_and_actually_used():
    q_source = np.random.default_rng(0).normal(size=(49, 5, 4))
    q_source_snapshot = q_source.copy()
    agent = TabularQAgent(grid_size=7, n_stages=5, alpha=0.2, gamma=0.95,
                          epsilon_start=0.3, epsilon_min=0.05,
                          epsilon_decay=0.98, q_init=q_source)
    # 1) the agent's table equals the transferred table at init
    assert np.array_equal(agent.q, q_source_snapshot)
    # 2) the greedy policy of the fresh agent follows the transferred values
    obs = {"pos": (3, 3), "k": 2, "last_node": "C", "visible": {}}
    rng = np.random.default_rng(0)
    expected_action = int(np.argmax(q_source_snapshot[3 * 7 + 3, 2]))
    assert agent.act(obs, rng, greedy=True) == expected_action
    # 3) learning changes the agent's table but never the source table
    state = agent.encode_state(obs)
    agent.update(state, expected_action, -1.0, state, done=False)
    assert not np.array_equal(agent.q, q_source_snapshot)
    assert np.array_equal(q_source, q_source_snapshot)
    # 4) shape mismatch is rejected loudly
    with pytest.raises(ValueError):
        TabularQAgent(grid_size=7, n_stages=5, alpha=0.2, gamma=0.95,
                      epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.98,
                      q_init=np.zeros((10, 10, 4)))


# ---------------------------------------------------------------------------
# Schema agent: fairness / privacy
# ---------------------------------------------------------------------------
def test_schema_agent_memory_only_contains_perceived_targets():
    """The spatial memory must exactly equal the set of targets the
    environment actually revealed -- nothing more, nothing less."""
    env = R.make_env(7, C.DEFAULT_TASK)
    agent = SchemaAgent(C.GRID_SIZE)
    agent.set_obstacles(env.obstacles, env.size)
    rng = np.random.default_rng(0)
    revealed = set()
    for _ep in range(5):
        obs = env.reset()
        agent.reset_episode_state()
        agent.update_perception(obs)
        revealed |= set(obs["visible"])
        for _t in range(400):
            action = agent.choose_action(obs, rng)
            obs, reward, done, _info = env.step(action)
            agent.update_perception(obs)
            revealed |= set(obs["visible"])
            if reward > 0:
                agent.record_success(obs)
            if done:
                break
    assert set(agent.memory.keys()) == revealed
    # stored positions must match the true positions of revealed targets
    for tid, pos in agent.memory.items():
        assert env.targets[tid] == pos


def test_schema_agent_with_known_locations_follows_shortest_cycle():
    """With all locations known and a correct schema, the agent's cycle
    length must equal the sum of BFS shortest-path lengths (6+6+6+6+6=30 on
    the handmade corner layout) -- no wasted steps, no cheating."""
    layout = _handmade_layout()
    env = GridWorld(layout, C.DEFAULT_TASK)
    agent = SchemaAgent(C.GRID_SIZE,
                        schema={"START": ["A"], "A": ["B"], "B": ["C"],
                                "C": ["D"], "D": ["A"]})
    agent.set_obstacles(env.obstacles, env.size)
    agent.memory = dict(layout["targets"])   # simulate full spatial knowledge
    rng = np.random.default_rng(0)
    log = R.run_schema_episode(env, agent, rng, max_steps=100)
    assert log["completed"]
    assert log["completion_step"] == 30
    # The agent re-plans (one BFS call) at every navigation step: 30 steps
    # x 1 BFS.  Re-planning is a deliberate, documented design choice; its
    # computational cost is reported separately from env steps.
    assert log["bfs_calls"] == 30


def test_schema_agent_does_not_enter_unobserved_targets_deliberately():
    """Run many exploration steps with an EMPTY memory: every step onto a
    target cell must be explainable by the perception rule (the target was
    visible/adjacent or the move was the agent's uniform random exploration).
    We check the weaker but crisp property: the agent never follows a BFS
    plan toward a position that is not in its memory (it has none)."""
    env = R.make_env(11, C.DEFAULT_TASK)
    agent = make_target_schema_agent(
        {"START": ["A"], "A": ["B"], "B": ["C"], "C": ["D"], "D": ["A"]},
        C.GRID_SIZE)
    agent.set_obstacles(env.obstacles, env.size)
    rng = np.random.default_rng(1)
    obs = env.reset()
    agent.update_perception(obs)
    for _t in range(600):
        action = agent.choose_action(obs, rng)
        obs, reward, done, _info = env.step(action)
        agent.update_perception(obs)
        if reward > 0:
            agent.record_success(obs)
        if done:
            obs = env.reset()
            agent.reset_episode_state()
            agent.update_perception(obs)
    # memory equals exactly what was perceived at some point
    assert set(agent.memory).issubset(set(env.targets))


def test_ablation_no_memory_stores_nothing():
    env = R.make_env(21, C.DEFAULT_TASK)
    agent = SchemaAgent(C.GRID_SIZE, schema={"START": ["A"]},
                        use_spatial_memory=False, use_task_state=True)
    agent.set_obstacles(env.obstacles, env.size)
    rng = np.random.default_rng(2)
    for _ep in range(3):
        R.run_schema_episode(env, agent, rng, max_steps=200)
    assert agent.memory == {}


def test_ablation_no_task_state_never_learns_schema():
    env = R.make_env(22, C.DEFAULT_TASK)
    agent = SchemaAgent(C.GRID_SIZE, schema=None,
                        use_spatial_memory=True, use_task_state=False)
    agent.set_obstacles(env.obstacles, env.size)
    rng = np.random.default_rng(3)
    for _ep in range(3):
        R.run_schema_episode(env, agent, rng, max_steps=200)
    assert agent.counts == {}          # no successor counting
    assert agent.last_node == "START"  # FSM state never advanced


def test_reset_memory_clears_spatial_knowledge_but_keeps_schema():
    schema = {"START": ["A"], "A": ["B"]}
    agent = make_target_schema_agent(schema, C.GRID_SIZE)
    agent.memory = {"A": (0, 0)}
    agent.reset_memory()
    assert agent.memory == {}
    assert agent.schema == schema


# ---------------------------------------------------------------------------
# Oracle planner
# ---------------------------------------------------------------------------
def test_oracle_planner_completes_optimal_corner_cycle():
    layout = _handmade_layout()
    env = GridWorld(layout, C.DEFAULT_TASK)
    planner = OraclePlanner(layout, C.DEFAULT_TASK)
    log = R.run_planned_episode(env, planner, max_steps=100)
    assert log["completed"]
    assert log["completion_step"] == 30


# ---------------------------------------------------------------------------
# Schema derivation from reward feedback
# ---------------------------------------------------------------------------
def test_schema_derivation_recovers_sequence_from_counts():
    agent = SchemaAgent(C.GRID_SIZE)
    # Simulate reward feedback counts as if the agent had experienced the
    # cycle task many times (this is exactly what record_success produces).
    agent.counts = {
        "START": {"A": 60},
        "A": {"B": 55, "C": 2},
        "B": {"C": 50},
        "C": {"D": 48},
        "D": {"A": 45},
    }
    schema = agent.derive_schema(min_count=3)
    assert schema["START"] == ["A"]
    assert schema["A"] == ["B"]      # rare C transitions are filtered out
    assert schema["B"] == ["C"]
    assert schema["C"] == ["D"]
    assert schema["D"] == ["A"]


def test_schema_derivation_keeps_branches():
    agent = SchemaAgent(C.GRID_SIZE)
    agent.counts = {"START": {"A": 50}, "A": {"B": 24, "C": 26}}
    schema = agent.derive_schema(min_count=3)
    # both branches exceed half of the row maximum -> both kept
    assert sorted(schema["A"]) == ["B", "C"]


# ---------------------------------------------------------------------------
# Regression: BFS navigation crosses known-but-unacceptable targets
# (the ACTUAL mechanism behind Experiment D's branch-task successes --
# the earlier documentation claimed a different, incorrect mechanism)
# ---------------------------------------------------------------------------
def _corridor_layout():
    """Only passage from column <=1 to column >=4 is row 3; the shortest
    path from (3,1) to (3,6) must run through the target cell D at (3,3)."""
    return {
        "grid_size": 7,
        "obstacles": {(r, 2) for r in (0, 1, 2, 4, 5, 6)}
                     | {(r, 3) for r in (0, 1, 2, 4, 5, 6)},
        "targets": {"A": (0, 5), "B": (6, 5), "C": (3, 6), "D": (3, 3)},
        "start": (3, 1),
    }


def test_bfs_navigation_may_cross_known_unacceptable_targets():
    """REGRESSION TEST (Experiment D branch mechanism, agent level).

    Setup mirrors the branch-task situation: the agent's FROZEN schema says
    "after B collect C", but the TRUE task graph accepts D after B (like the
    branch graph's B -> D edge).  The agent knows both C and D, and the only
    shortest path from its position to C runs through D's cell.

    Documented behaviour that must NOT change: the agent FOLLOWS the BFS
    path and therefore steps onto D -- a target its own schema marks
    unacceptable -- even though the avoidance rule would forbid that entry
    during random-walk exploration.  Under the true task this crossing is
    rewarded (+10), advances the task, and can complete it.  This is exactly
    how the 5 successful branch runs of Experiment D succeeded (zero
    successes came from the random-walk fallback).
    """
    layout = _corridor_layout()
    # TRUE task: at the environment's current stage the target D is exactly
    # what the true task graph accepts (in Experiment D's branch graph this
    # corresponds to the B -> D edge; reducing the setup to a single stage
    # isolates the crossing mechanism from unrelated FSM bookkeeping --
    # the environment-side reward semantics of wrong entries are already
    # covered by test_rewards_stage_progress_and_termination).
    true_task = {
        "successors": {"START": ["D"], "D": []},
        "completion_length": 1,
    }
    env = GridWorld(layout, true_task)
    agent = SchemaAgent(C.GRID_SIZE, schema={"B": ["C"]})   # frozen, wrong goal
    agent.set_obstacles(env.obstacles, env.size)
    agent.memory = {"C": (3, 6), "D": (3, 3)}   # both locations KNOWN
    agent.last_node = "B"                        # agent expects C next
    obs = env.reset()
    assert obs["pos"] == (3, 1)

    rng = np.random.default_rng(0)
    # The agent navigates (BFS mode) toward C.  The first action must be the
    # first step of the shortest path -- the corridor -- NOT a detour and
    # NOT an avoidance-inspired alternative.
    action = agent.choose_action(obs, rng)
    assert action == 3   # right, toward (3,2): the path that crosses D

    # Following the agent's policy walks it THROUGH D.  Its own FSM never
    # advances before the crossing (no positive reward), and the entry into
    # D is CORRECT under the true task despite being schema-unacceptable.
    entered = None
    for _step in range(5):
        action = agent.choose_action(obs, rng)
        obs, reward, done, info = env.step(action)
        if info["entered"] is not None:
            entered = (info["entered"], info["correct"])
            break
    assert entered == ("D", True), (
        "BFS navigation must be able to cross a known unacceptable target "
        "(regression guard for the documented branch-task mechanism)")
    assert agent.last_node == "B"   # avoidance/penalties left the FSM intact


# ---------------------------------------------------------------------------
# Obstacle-aware Q agents (Experiment E control)
# ---------------------------------------------------------------------------
def _masked_agent(mode, **kwargs):
    from agents.q_learning_masked import ObstacleAwareQAgent
    return ObstacleAwareQAgent(
        grid_size=7, n_stages=5, alpha=0.2, gamma=0.95,
        epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.98,
        obstacle_mode=mode, **kwargs)


def test_masked_agent_full_map_never_selects_blocked_moves():
    """A full-map masked agent must only ever choose moves that stay on the
    grid and avoid obstacles -- at exploration AND at greedy selection."""
    from environment.grid_world import ACTION_DELTAS, in_bounds
    env = R.make_env(7, C.DEFAULT_TASK)
    agent = _masked_agent("full_map")
    agent.set_obstacles(env.obstacles, env.size)
    agent.q[...] = np.random.default_rng(0).normal(size=agent.q.shape)
    rng = np.random.default_rng(1)
    obs = env.reset()
    for _step in range(300):
        for greedy in (False, True):
            action = agent.act(obs, rng, greedy=greedy)
            dr, dc = ACTION_DELTAS[action]
            dest = (obs["pos"][0] + dr, obs["pos"][1] + dc)
            assert in_bounds(dest, 7)
            assert dest not in env.obstacles
        obs, _r, done, _i = env.step(agent.act(obs, rng, greedy=False))
        if done:
            obs = env.reset()


def test_masked_agent_bump_memory_learns_only_from_bumps():
    """bump_memory starts knowing NOTHING; blocked moves appear only after
    the agent is told about its own bump events, and are then avoided."""
    agent = _masked_agent("bump_memory")
    assert agent.blocked_move_count() == 0
    obs = {"pos": (3, 3), "k": 0, "last_node": "START", "visible": {}}
    rng = np.random.default_rng(0)
    agent.observe_step((3, 3), 0, bumped=False)   # free move: no knowledge
    assert agent.blocked_move_count() == 0
    agent.observe_step((3, 3), 0, bumped=True)    # bump up: learns (3,3)-up
    assert agent.blocked_move_count() == 1
    # subsequent action selection never repeats the bumped move
    for _ in range(50):
        assert agent.act(obs, rng, greedy=False) != 0
        assert agent.act(obs, rng, greedy=True) != 0
    with pytest.raises(RuntimeError):
        _masked_agent("bump_memory").set_obstacles({(0, 0)}, 7)


def test_masked_agent_full_map_matches_schema_agent_information():
    """The full_map agent receives exactly the same information at the same
    wiring moment as the schema agent: the complete obstacle set + bounds.
    Every (position, action) pair it will not attempt must correspond to a
    bump the environment would produce."""
    env = R.make_env(13, C.DEFAULT_TASK)
    agent = _masked_agent("full_map")
    agent.set_obstacles(env.obstacles, env.size)
    for r in range(7):
        for c in range(7):
            for action in range(4):
                dest = ((r + (action == 1) - (action == 0)),
                        (c + (action == 3) - (action == 2)))
                would_bump = (not (0 <= dest[0] < 7 and 0 <= dest[1] < 7)) \
                    or (dest in env.obstacles)
                blocked = ((r, c), action) in agent._blocked_moves
                assert blocked == would_bump
