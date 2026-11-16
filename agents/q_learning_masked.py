"""Obstacle-aware variants of the tabular Q-learning agent (Experiment E).

Why this module exists
    In the ORIGINAL experiments (A-D) the schema-based agent is given the
    complete obstacle map of its current environment at wiring time
    (``SchemaAgent.set_obstacles``), while the Q-learning baselines receive
    no obstacle information at all and can only learn obstacle locations
    implicitly from bump penalties.  Experiment E is a CONTROL experiment
    that quantifies how much of the schema agent's advantage comes from that
    information asymmetry.  This module defines the two Q-learning agents
    used in the control; it changes nothing about the original agents or
    experiments.

What information each variant receives, and when
    ``obstacle_mode="full_map"``
        The COMPLETE obstacle set of the current environment is provided at
        wiring time -- the same moment and the same content the schema agent
        receives (``set_obstacles``).  The agent uses it ONLY to exclude
        moves into obstacles (and off the grid) from action selection; it
        never bumps.  It does NOT provide planning (no BFS), no target
        knowledge, and no other advantage.
    ``obstacle_mode="bump_memory"``
        NO obstacle information at start.  The agent maintains a set of
        blocked (position, action) pairs that starts empty and grows ONLY
        from its own bump events, which are physically perceivable (a bump
        leaves the agent in place).  After bumping into a cell once it never
        attempts the same move from the same cell again.  This variant
        measures the cost of *discovering* the obstacle layout, not the
        benefit of knowing it in advance.

How the information is used
    Both variants restrict ACTION SELECTION to moves whose destination is
    neither off-grid nor (known to be) an obstacle:
      * exploration samples uniformly among the admissible moves (exactly
        the rule the schema agent's random walk uses);
      * greedy selection takes the argmax over the admissible moves only,
        ties broken by the lowest action index (the parent class convention,
        restricted to the admissible set).
    Q-updates are unchanged: the agent simply never selects an inadmissible
    action, so only admissible (state, action) entries are ever updated.
    Inadmissible entries may keep their initial values (zeros, or
    transferred values) but are never chosen.

Blocked-move bookkeeping
    Blocked moves are stored as (position, action) pairs.  For
    ``full_map`` the complete pair set is derived once from the provided
    obstacle set plus the grid boundary; for ``bump_memory`` pairs are added
    one at a time from bump events reported by the episode runner
    (``observe_step``).  Walls (the grid boundary) and obstacle cells are
    treated uniformly in both variants.
"""

import numpy as np

from agents.q_learning import TabularQAgent
from environment.grid_world import ACTION_DELTAS, in_bounds


class ObstacleAwareQAgent(TabularQAgent):
    """TabularQAgent with obstacle information, in two variants.

    See the module docstring for exactly what each variant knows, when it
    learns it, and how the knowledge is used.  Everything else (state
    encoding, Q-update, epsilon schedule, transfer semantics) is inherited
    unchanged from ``TabularQAgent``.
    """

    def __init__(self, grid_size, n_stages, alpha, gamma,
                 epsilon_start, epsilon_min, epsilon_decay,
                 obstacle_mode="full_map", q_init=None):
        if obstacle_mode not in ("full_map", "bump_memory"):
            raise ValueError(
                f"unknown obstacle_mode {obstacle_mode!r}; "
                "use 'full_map' or 'bump_memory'")
        super().__init__(grid_size, n_stages, alpha, gamma,
                         epsilon_start, epsilon_min, epsilon_decay,
                         q_init=q_init)
        self.obstacle_mode = obstacle_mode
        # (position, action) pairs the agent must not attempt.  Starts empty
        # in bump_memory mode; is filled completely by set_obstacles() in
        # full_map mode.  Walls and obstacles are both encoded here.
        self._blocked_moves = set()
        self._grid_size_known = grid_size

    # -- wiring (full_map mode) -------------------------------------------------
    def set_obstacles(self, obstacles, grid_size):
        """Provide the complete obstacle map (full_map mode only).

        Called by the experiment wiring at the same moment the schema agent
        receives its map.  Derives the full blocked-move set from the map
        and the grid boundary.  In bump_memory mode this method must NOT be
        called with a real map (it would defeat the purpose); it raises.
        """
        if self.obstacle_mode != "full_map":
            raise RuntimeError(
                "set_obstacles() is only valid for obstacle_mode='full_map'; "
                "bump_memory agents must learn blocked moves from bumps")
        self._grid_size_known = grid_size
        blocked = set()
        for r in range(grid_size):
            for c in range(grid_size):
                for action, (dr, dc) in ACTION_DELTAS.items():
                    dest = (r + dr, c + dc)
                    if (not in_bounds(dest, grid_size)) or (dest in obstacles):
                        blocked.add(((r, c), action))
        self._blocked_moves = blocked

    # -- bump feedback (bump_memory mode) ---------------------------------------
    def observe_step(self, prev_pos, action, bumped):
        """Called by the episode runner after every environment step.

        In bump_memory mode a bump adds the attempted (position, action) pair
        to the blocked-move set: the agent felt the failed move, so it can
        avoid repeating it.  In full_map mode this is a no-op (the map is
        already complete).
        """
        if self.obstacle_mode == "bump_memory" and bumped:
            self._blocked_moves.add((tuple(prev_pos), int(action)))

    # -- action selection ---------------------------------------------------------
    def _admissible(self, pos):
        """Moves that do not lead off-grid or into a KNOWN obstacle."""
        admissible = [a for a in range(self.n_actions)
                      if (tuple(pos), a) not in self._blocked_moves]
        # A fully blocked position cannot occur for reachable cells, but keep
        # a safe fallback (attempt all moves) so the agent can never deadlock.
        return admissible if admissible else list(range(self.n_actions))

    def act(self, obs, rng, greedy=False):
        """Epsilon-greedy over admissible moves only (see module docstring)."""
        pos = obs["pos"]
        admissible = self._admissible(pos)
        if not greedy and rng.random() < self.epsilon:
            return int(admissible[int(rng.integers(len(admissible)))])
        cell, k = self.encode_state(obs)
        q_values = self.q[cell, k]
        best = max(admissible, key=lambda a: (q_values[a], -a))
        return int(best)

    # -- helpers used by tests ------------------------------------------------
    def blocked_move_count(self):
        return len(self._blocked_moves)
