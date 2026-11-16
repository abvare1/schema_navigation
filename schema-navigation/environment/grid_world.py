"""Grid world environment for sequential multi-target navigation tasks.

Observation and transition rules (the full prose version is in
docs/methodology.md):

State
    Agent position (row, col) on a size x size grid, a progress counter ``k``
    (number of correct targets collected so far in this episode) and the task
    node ``last_node`` (the identity of the last correctly collected target,
    or the pseudo-node "START" at episode start).

Actions
    0 = up, 1 = down, 2 = left, 3 = right.  Transitions are deterministic.
    Moving into a wall or obstacle cell is a "bump": the agent stays in place
    and no target-entry event fires (standing on a target does not re-collect
    it; only entering the cell does).

Rewards
    Every step costs REWARD_STEP.  Entering a target cell gives an extra
    REWARD_CORRECT if that target's identity is a successor of the current
    task node (i.e. the task expects it now), otherwise an extra REWARD_WRONG.
    The episode ends (done = True) as soon as ``k`` reaches the task's
    completion_length.

Perception (identifying targets)
    Targets have no visible markers from afar.  After every move -- and at
    episode reset -- the environment reveals the identity and position of
    every target within Manhattan distance <= 1 of the agent (this includes
    the agent's own cell).  The revealed set is returned in the observation
    under the key ``visible``.  The environment itself never reveals
    anything else about the layout (no obstacle map, no distant targets).
    Note: obstacle maps are additionally handed to model-based agents by
    the EXPERIMENT WIRING (e.g. SchemaAgent.set_obstacles), outside the
    observation channel documented here -- see docs/methodology.md section
    6 for what each agent receives and docs/limitations.md for the
    consequences.

Layouts are plain dicts:
    {"grid_size": int, "obstacles": set[(r, c)], "targets": {id: (r, c)},
     "start": (r, c)}
"""

from collections import deque

import numpy as np

import config as C

# Action encoding used everywhere in the project.
ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


# ---------------------------------------------------------------------------
# Small grid helpers (shared by agents, planner and tests)
# ---------------------------------------------------------------------------
def manhattan(a, b):
    """Manhattan distance between two (row, col) cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(cell, size=C.GRID_SIZE):
    r, c = cell
    return 0 <= r < size and 0 <= c < size


def valid_actions(pos, obstacles, size=C.GRID_SIZE):
    """Return the list of action indices that move to an in-bounds,
    non-obstacle cell.  Bumping into obstacles is legal but never useful, so
    agents that sample uniformly use this helper to avoid wasted bumps.
    NOTE: using this helper requires an obstacle set; in the experiments
    only agents that are GIVEN the map (schema-type agents via
    ``set_obstacles``, and the Experiment E masked Q agents) can call it.
    The original Q-learning baselines receive no obstacle information and
    therefore do sample from all four actions."""
    result = []
    for action, (dr, dc) in ACTION_DELTAS.items():
        dest = (pos[0] + dr, pos[1] + dc)
        if in_bounds(dest, size) and dest not in obstacles:
            result.append(action)
    return result


def direction_between(src, dst):
    """Action index that moves from ``src`` to the adjacent cell ``dst``."""
    delta = (dst[0] - src[0], dst[1] - src[1])
    for action, d in ACTION_DELTAS.items():
        if d == delta:
            return action
    raise ValueError(f"cells {src} and {dst} are not adjacent")


def bfs_shortest_path(obstacles, start, goal, size=C.GRID_SIZE):
    """Breadth-first search on the free cells of the grid.

    Returns the list of cells from start to goal EXCLUDING start and INCLUDING
    goal ([] if start == goal), or None if goal is unreachable.  Only obstacle
    cells are blocked; target cells are ordinary traversable cells.
    """
    if start == goal:
        return []
    visited = {start}
    parent = {}
    frontier = deque([start])
    while frontier:
        cell = frontier.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (cell[0] + dr, cell[1] + dc)
            if not in_bounds(nxt, size) or nxt in obstacles or nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = cell
            if nxt == goal:
                # Reconstruct the path by walking the parent links backwards,
                # then drop the start cell (the returned path starts at the
                # first MOVE, i.e. it excludes the start cell).
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path[1:]
            frontier.append(nxt)
    return None


# ---------------------------------------------------------------------------
# Layout generation and validation
# ---------------------------------------------------------------------------
def generate_layout(seed, size=C.GRID_SIZE, n_obstacles=C.N_OBSTACLES,
                    target_ids=None, start=C.START_POSITION):
    """Generate a random but valid layout from an integer seed.

    Validity constraints (checked by ``validate_layout`` at the end):
      * obstacles and targets never overlap each other or the start cell;
      * targets are pairwise separated by >= MIN_TARGET_SEPARATION steps so
        that perception is never ambiguous;
      * every target is reachable from the start through free cells;
      * the free connected component containing the start is large enough.

    The function is fully deterministic: the same seed always yields the same
    layout, which is what makes identical environments across agents possible.
    """
    if target_ids is None:
        target_ids = C.TARGET_IDS
    rng = np.random.default_rng(seed)
    all_cells = [(r, c) for r in range(size) for c in range(size)]

    for _attempt in range(10000):
        # 1) sample obstacle cells (never on the start cell)
        candidates = [cell for cell in all_cells if cell != start]
        obstacle_idx = rng.choice(len(candidates), size=n_obstacles, replace=False)
        obstacles = {candidates[i] for i in obstacle_idx}

        # 2) place targets one by one on free cells with pairwise separation
        remaining = [cell for cell in candidates
                     if cell not in obstacles]
        shuffled = [remaining[i] for i in rng.permutation(len(remaining))]
        targets = {}
        for cell in shuffled:
            if len(targets) == len(target_ids):
                break
            if all(manhattan(cell, p) >= C.MIN_TARGET_SEPARATION
                   for p in targets.values()):
                targets[target_ids[len(targets)]] = cell
        if len(targets) != len(target_ids):
            continue

        # 3) reachability: BFS from start over free cells
        free = [cell for cell in all_cells
                if cell != start and cell not in obstacles]
        reachable = _reachable_cells(start, obstacles, size)
        if not all(t in reachable for t in targets.values()):
            continue
        if len(reachable) < C.MIN_REACHABLE_CELLS:
            continue

        layout = {
            "grid_size": size,
            "obstacles": obstacles,
            "targets": targets,
            "start": start,
        }
        validate_layout(layout)   # final safety net; raises on any violation
        return layout

    raise RuntimeError(
        f"could not generate a valid layout for seed {seed} after 10000 tries")


def _reachable_cells(start, obstacles, size):
    """Set of cells reachable from ``start`` moving through free cells."""
    seen = {start}
    frontier = deque([start])
    while frontier:
        cell = frontier.popleft()
        for dr, dc in ACTION_DELTAS.values():
            nxt = (cell[0] + dr, cell[1] + dc)
            if (in_bounds(nxt, size) and nxt not in obstacles
                    and nxt not in seen):
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def validate_layout(layout):
    """Raise ValueError with a clear message if the layout is invalid."""
    size = layout["grid_size"]
    obstacles = layout["obstacles"]
    targets = layout["targets"]
    start = layout["start"]

    for name, cell in [("start", start)] + list(targets.items()):
        if not in_bounds(cell, size):
            raise ValueError(f"{name} cell {cell} is outside the grid")
    if start in obstacles:
        raise ValueError("start cell overlaps an obstacle")
    for tid, cell in targets.items():
        if cell in obstacles:
            raise ValueError(f"target {tid} at {cell} overlaps an obstacle")
        if cell == start:
            raise ValueError(f"target {tid} at {cell} overlaps the start cell")
    ids = list(targets.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if targets[ids[i]] == targets[ids[j]]:
                raise ValueError(f"targets {ids[i]} and {ids[j]} overlap")
            if manhattan(targets[ids[i]], targets[ids[j]]) < C.MIN_TARGET_SEPARATION:
                raise ValueError(
                    f"targets {ids[i]} and {ids[j]} are closer than "
                    f"MIN_TARGET_SEPARATION={C.MIN_TARGET_SEPARATION}")
    reachable = _reachable_cells(start, obstacles, size)
    for tid, cell in targets.items():
        if cell not in reachable:
            raise ValueError(f"target {tid} at {cell} is unreachable")
    if len(reachable) < C.MIN_REACHABLE_CELLS:
        raise ValueError(
            f"only {len(reachable)} free reachable cells "
            f"(minimum is {C.MIN_REACHABLE_CELLS})")


# ---------------------------------------------------------------------------
# The environment itself
# ---------------------------------------------------------------------------
class GridWorld:
    """Sequential multi-target grid world (see module docstring for rules)."""

    def __init__(self, layout, task_graph):
        validate_layout(layout)
        self.layout = layout
        self.task_graph = task_graph
        self.size = layout["grid_size"]
        self.obstacles = set(layout["obstacles"])
        self.targets = dict(layout["targets"])
        self.start = layout["start"]
        self.target_at = {pos: tid for tid, pos in self.targets.items()}

        # The task graph must only reference targets that exist in the layout.
        for node in task_graph["successors"]:
            if node != "START" and node not in self.targets:
                raise ValueError(f"task graph references unknown target {node}")

        # Episode state
        self._pos = None
        self._k = None
        self._last_node = None
        self._done = None

    # -- observation --------------------------------------------------------
    def _visible(self):
        """Targets the agent can currently identify (distance <= 1)."""
        return {tid: pos for tid, pos in self.targets.items()
                if manhattan(pos, self._pos) <= 1}

    def _observation(self):
        return {
            "pos": self._pos,
            "k": self._k,                 # correct collections so far
            "last_node": self._last_node,  # last correctly collected identity
            "visible": self._visible(),    # identity -> position, dist <= 1
            "done": self._done,
        }

    # -- episode lifecycle ----------------------------------------------------
    def reset(self):
        self._pos = self.start
        self._k = 0
        self._last_node = "START"
        self._done = False
        return self._observation()

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode")
        if action not in ACTION_DELTAS:
            raise ValueError(f"unknown action {action}")

        dr, dc = ACTION_DELTAS[action]
        nxt = (self._pos[0] + dr, self._pos[1] + dc)
        bump = (not in_bounds(nxt, self.size)) or (nxt in self.obstacles)
        if not bump:
            self._pos = nxt

        reward = C.REWARD_STEP
        entered = self.target_at.get(self._pos) if not bump else None
        # Target-entry events only fire when the agent actually MOVES into a
        # cell.  A bump (agent stays put) never re-triggers the target it is
        # already standing on.
        correct = False
        if entered is not None:
            if entered in self.task_graph["successors"][self._last_node]:
                reward += C.REWARD_CORRECT
                correct = True
                self._k += 1
                self._last_node = entered
                if self._k >= self.task_graph["completion_length"]:
                    self._done = True
            else:
                reward += C.REWARD_WRONG

        obs = self._observation()
        info = {"bump": bump, "entered": entered, "correct": correct}
        return obs, reward, self._done, info

    # -- debugging helper -----------------------------------------------------
    def render(self):
        """Return a small ASCII picture of the current state (for humans)."""
        rows = []
        for r in range(self.size):
            row = []
            for c in range(self.size):
                cell = (r, c)
                if cell == self._pos:
                    ch = "@"
                elif cell in self.obstacles:
                    ch = "#"
                elif cell in self.target_at:
                    ch = self.target_at[cell]
                elif cell == self.start:
                    ch = "S"
                else:
                    ch = "."
                row.append(ch)
            rows.append(" ".join(row))
        return "\n".join(rows)
