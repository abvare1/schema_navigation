"""Schema-based agent: a two-level architecture with

  * a HIGH-level *task schema*: a finite-state machine over target
    identities, learned from reward feedback in the source environment.  It
    maps "last collected identity" -> list of identities that may be
    collected next (e.g. START->A, A->B, B->C, C->D, D->A for the cycle);
  * a LOW-level *spatial memory*: a mapping identity -> position that is
    filled purely by perception (targets become visible at distance <= 1)
    and must start EMPTY in every new environment.

Navigation policy
    If the schema says which identities are acceptable next and at least one
    of them has a known location, the agent plans with BFS (shortest path on
    free cells) and follows the shortest such path.  Otherwise it explores
    with a uniform random walk that never deliberately steps onto a cell it
    knows holds a target that is NOT currently acceptable.

    IMPORTANT (documented behaviour, regression-tested in
    tests/test_agents.py): the avoidance rule applies ONLY to the
    random-walk fallback.  BFS treats target cells as ordinary traversable
    cells, so a shortest path toward an acceptable target MAY pass through
    the cell of a known but currently unacceptable target.  The agent then
    enters that target and receives the wrong-target penalty (-5), which
    does NOT corrupt its FSM (the FSM advances only on positive rewards).
    This mechanism is what produces the branch-task successes in Experiment
    D (docs/methodology.md, section 9b).

What the agent receives, and what it must learn (fairness disclosure)
    * GIVEN at wiring time: the COMPLETE obstacle map of the environment it
      inhabits, via ``set_obstacles`` -- called by the experiment scripts,
      outside the perception channel.  In the ORIGINAL experiments the
      Q-learning baselines receive NO equivalent information (they learn
      obstacle locations only implicitly from bump penalties);
      experiments/obstacle_info_control.py (Experiment E) is the control
      that quantifies this asymmetry.  The map contains no target
      positions.
    * LEARNED from perception only: target identities and positions
      (spatial memory, written exclusively from ``obs["visible"]``).
    * LEARNED from reward feedback only: the task schema (successor counts
      updated only on positive rewards).

Ablation switches (Experiment B)
    ``use_spatial_memory=False``  -> discovered locations are never stored;
        the agent can only react to what is visible at its current cell.
    ``use_task_state=False``      -> the schema/FSM is removed; the agent can
        still remember locations but visits known targets in a random order.
"""

import copy

import numpy as np

from environment.grid_world import (
    bfs_shortest_path,
    direction_between,
    valid_actions,
)


class SchemaAgent:
    def __init__(self, grid_size, schema=None,
                 use_spatial_memory=True, use_task_state=True):
        self.grid_size = grid_size
        # schema: dict node -> list of acceptable identities (learned, frozen)
        self.schema = dict(schema) if schema is not None else {}
        self.use_spatial_memory = use_spatial_memory
        self.use_task_state = use_task_state

        # Persistent across episodes *within one environment*:
        self.memory = {}   # identity -> position (discovered by perception)
        self.counts = {}   # node -> {identity: n_times_collected_right_after}
        self.bfs_calls = 0  # planning-cost counter (reported, not env steps)

        # Reset at every episode start:
        self.last_node = "START"

        # Obstacle map of the environment the agent currently inhabits.
        # FAIRNESS DISCLOSURE: this is NOT learned from bumps -- the
        # experiment wiring provides the COMPLETE map at wiring time (see
        # set_obstacles).  It is identical for every schema-type agent and
        # contains no target positions, but the Q-learning baselines of the
        # original experiments do NOT receive it (see Experiment E,
        # experiments/obstacle_info_control.py, for the control that
        # quantifies this asymmetry).
        self._obstacles = set()

    # -- lifecycle ------------------------------------------------------------
    def reset_memory(self):
        """Called when the agent moves to a NEW environment: spatial memory
        starts empty again.  The learned schema is NOT touched."""
        self.memory = {}

    def reset_episode_state(self):
        self.last_node = "START"

    # -- perception and reward feedback ---------------------------------------
    def update_perception(self, obs):
        """Write everything currently visible into spatial memory."""
        if self.use_spatial_memory:
            for tid, pos in obs["visible"].items():
                if tid not in self.memory:
                    self.memory[tid] = pos

    def record_success(self, obs):
        """Called after a step with positive reward (a correct target was
        collected).  Updates the FSM state and, while learning, the
        successor counts from which the schema is later derived."""
        if not self.use_task_state:
            return
        tid = self._identity_at(obs, obs["pos"])
        if tid is None:
            raise RuntimeError("positive reward but no target at agent cell")
        # Count the transition, then advance the FSM state.
        row = self.counts.setdefault(self.last_node, {})
        row[tid] = row.get(tid, 0) + 1
        self.last_node = tid

    def _identity_at(self, obs, pos):
        for tid, tpos in obs["visible"].items():
            if tpos == pos:
                return tid
        return None

    # -- schema derivation ------------------------------------------------------
    def derive_schema(self, min_count):
        """Turn successor counts into the frozen schema.

        For each task node, keep every identity whose transition count is at
        least ``min_count`` AND at least half of the row maximum.  For a pure
        sequence this yields a single successor; for a branching graph it
        yields all frequently observed branches.  Nodes with too little
        evidence get no successor (the agent will explore at that stage).
        """
        schema = {}
        for node, row in self.counts.items():
            best = max(row.values())
            if best < min_count:
                continue
            threshold = max(min_count, 0.5 * best)
            successors = [tid for tid, count in
                          sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))
                          if count >= threshold]
            schema[node] = successors
        self.schema = schema
        return schema

    # -- policy -----------------------------------------------------------------
    def choose_action(self, obs, rng):
        """Pick one action given the current observation and internal state."""
        pos = obs["pos"]
        visible = obs["visible"]
        expected = self.schema.get(self.last_node, []) \
            if self.use_task_state else None

        # Case 1: full agent -- navigate to a known acceptable target.
        if self.use_spatial_memory and self.use_task_state:
            known = [tid for tid in (expected or []) if tid in self.memory]
            if known:
                action = self._action_toward_nearest(pos, known)
                if action is not None:
                    return action
            # Acceptable target not discovered yet: explore, but never walk
            # onto a cell known to hold a target that is not acceptable now.
            forbidden = [tid for tid in visible
                         if expected and tid not in expected]
            return self._random_action(pos, rng, visible, forbidden)

        # Case 2: memory ablation -- react only to the current cell's view.
        if (not self.use_spatial_memory) and self.use_task_state:
            visible_expected = [tid for tid in (expected or [])
                                if tid in visible]
            if visible_expected:
                goal = visible[visible_expected[0]]
                if goal != pos:
                    return direction_between(pos, goal)
            forbidden = [tid for tid in visible
                         if expected and tid not in expected]
            return self._random_action(pos, rng, visible, forbidden)

        # Case 3: task-state ablation -- visit known targets in random order.
        candidates = [tid for tid, p in self.memory.items() if p != pos]
        if candidates:
            tid = candidates[int(rng.integers(len(candidates)))]
            action = self._action_toward(pos, self.memory[tid])
            if action is not None:
                return action
        return self._random_action(pos, rng, visible, forbidden_ids=[])

    # -- movement helpers ---------------------------------------------------------
    def _action_toward(self, pos, goal_pos):
        """First action of a BFS shortest path to ``goal_pos`` (or None)."""
        self.bfs_calls += 1
        path = bfs_shortest_path(self._obstacles, pos, goal_pos, self.grid_size)
        if path:
            return direction_between(pos, path[0])
        return None

    def _action_toward_nearest(self, pos, candidate_ids):
        """BFS to every candidate and follow the shortest path found.
        An empty path (agent already stands on the goal cell) is treated as
        'no navigation possible' -> the caller falls back to exploration."""
        best_action, best_len = None, None
        for tid in candidate_ids:
            self.bfs_calls += 1
            path = bfs_shortest_path(self._obstacles, pos,
                                     self.memory[tid], self.grid_size)
            # `if path` (not `is not None`): an empty path means the agent is
            # already standing on that cell, so there is no move to make.
            if path and (best_len is None or len(path) < best_len):
                best_len = len(path)
                best_action = direction_between(pos, path[0])
        return best_action

    def _random_action(self, pos, rng, visible, forbidden_ids):
        """Uniform random move among valid moves, avoiding destinations that
        hold a *forbidden* (currently unacceptable, visible) target.  If the
        avoidance rule blocks every move, fall back to all valid moves."""
        moves = valid_actions(pos, self._obstacles, self.grid_size)
        forbidden_pos = {visible[tid] for tid in forbidden_ids if tid in visible}
        safe = [a for a in moves
                if self._destination(pos, a) not in forbidden_pos]
        pool = safe if safe else moves
        return int(pool[int(rng.integers(len(pool)))])

    def _destination(self, pos, action):
        from environment.grid_world import ACTION_DELTAS
        dr, dc = ACTION_DELTAS[action]
        return (pos[0] + dr, pos[1] + dc)

    # -- environment wiring -------------------------------------------------------
    def set_obstacles(self, obstacles, grid_size):
        """Provide the COMPLETE obstacle map of the current environment.

        This is privileged wiring, not something the agent earns: the map is
        handed over in full at initialization time by the experiment scripts
        (run_utils.train_schema_source / run_target_schema, ablation.py,
        graph_similarity.py).  An agent COULD in principle perceive the same
        information by bumping, but this agent does not have to.  In the
        original experiments the Q-learning baselines receive no such map;
        Experiment E (obstacle_info_control.py) measures what that
        asymmetry is worth.  The map contains no target positions."""
        self._obstacles = set(obstacles)
        self.grid_size = grid_size


def make_target_schema_agent(source_schema, grid_size):
    """Fresh schema agent for a NEW environment: carries the learned schema,
    but its spatial memory starts empty (that is the whole point of the
    transfer experiment)."""
    agent = SchemaAgent(grid_size, schema=copy.deepcopy(source_schema))
    agent.reset_memory()
    return agent
