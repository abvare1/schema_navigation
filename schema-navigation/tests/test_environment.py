"""Environment tests: layout validity, determinism, rewards, perception, BFS."""

import pytest

import config as C
from environment.grid_world import (
    ACTION_DELTAS,
    GridWorld,
    bfs_shortest_path,
    generate_layout,
    manhattan,
    validate_layout,
)


def test_generate_layout_is_deterministic():
    a = generate_layout(123)
    b = generate_layout(123)
    assert a["obstacles"] == b["obstacles"]
    assert a["targets"] == b["targets"]
    assert a["start"] == b["start"]


def test_generated_layouts_are_valid_and_targets_reachable():
    for seed in range(20):
        layout = generate_layout(seed)
        # validate_layout raises on any violation; targets checked inside.
        validate_layout(layout)
        # explicit pairwise separation check
        positions = list(layout["targets"].values())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                assert manhattan(positions[i], positions[j]) >= 2


def test_validate_layout_rejects_target_on_obstacle():
    layout = generate_layout(0)
    bad = dict(layout)
    bad["targets"] = dict(layout["targets"])
    obstacle = sorted(layout["obstacles"])[0]
    bad["targets"]["A"] = obstacle
    with pytest.raises(ValueError):
        validate_layout(bad)


def test_validate_layout_rejects_overlapping_targets():
    layout = generate_layout(1)
    bad = dict(layout)
    bad["targets"] = dict(layout["targets"])
    bad["targets"]["B"] = bad["targets"]["A"]
    with pytest.raises(ValueError):
        validate_layout(bad)


def test_validate_layout_rejects_unreachable_target():
    layout = generate_layout(2)
    bad = dict(layout)
    bad["obstacles"] = set(layout["obstacles"])
    size = layout["grid_size"]
    target_cell = layout["targets"]["A"]
    # Wall off the target on all four sides (it becomes unreachable).
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        wall_cell = (target_cell[0] + dr, target_cell[1] + dc)
        if 0 <= wall_cell[0] < size and 0 <= wall_cell[1] < size:
            bad["obstacles"].add(wall_cell)
    bad["obstacles"].discard(target_cell)
    with pytest.raises(ValueError):
        validate_layout(bad)


def _handmade_layout():
    """Empty 7x7 grid, targets at the four corners, start in the centre."""
    return {
        "grid_size": 7,
        "obstacles": set(),
        "targets": {"A": (0, 0), "B": (0, 6), "C": (6, 6), "D": (6, 0)},
        "start": (3, 3),
    }


def test_rewards_stage_progress_and_termination():
    env = GridWorld(_handmade_layout(), C.DEFAULT_TASK)
    obs = env.reset()
    assert obs["k"] == 0 and obs["last_node"] == "START"

    # Walk to A (0,0) from (3,3): 3 up + 3 left = 6 steps, +10 on arrival.
    total = 0.0
    for _ in range(5):
        obs, r, done, _ = env.step(0)   # up
        total += r
    for _ in range(6):
        obs, r, done, _ = env.step(2)   # left
        total += r
    assert abs(total - (11 * C.REWARD_STEP + C.REWARD_CORRECT)) < 1e-9
    assert obs["k"] == 1 and obs["last_node"] == "A"
    assert not done

    # Enter C (6,6) while B is expected: wrong-target penalty, k unchanged.
    for _ in range(9):
        obs, r, done, _ = env.step(1)
    for _ in range(6):
        obs, r, done, _ = env.step(3)
    assert obs["k"] == 1
    assert obs["last_node"] == "A"

    # Now complete the rest of the cycle B -> C -> D -> A and check done.
    def go_to(target):
        """Walk from the agent's current position to ``target`` and return
        the observation after the last step."""
        pos = obs["pos"]
        dr = target[0] - pos[0]
        dc = target[1] - pos[1]
        result = None
        for _ in range(abs(dr)):
            result = env.step(1 if dr > 0 else 0)
        for _ in range(abs(dc)):
            result = env.step(3 if dc > 0 else 2)
        return result[0]

    obs = go_to((0, 6))   # B
    assert obs["k"] == 2 and obs["last_node"] == "B"
    obs = go_to((6, 6))   # C
    assert obs["k"] == 3
    obs = go_to((6, 0))   # D
    assert obs["k"] == 4
    obs = go_to((0, 0))   # A again: cycle complete
    assert obs["done"] is True
    # Stepping further is illegal.
    with pytest.raises(RuntimeError):
        env.step(0)


def test_perception_reveals_only_nearby_targets():
    env = GridWorld(_handmade_layout(), C.DEFAULT_TASK)
    obs = env.reset()
    # Start (3,3) is far from all corner targets: nothing visible.
    assert obs["visible"] == {}
    # Move one step towards A; still not adjacent, still nothing visible.
    obs, _, _, _ = env.step(0)
    assert obs["visible"] == {}
    # Walk until adjacent to A (3,3) -> (0,0): after 3 ups agent is at (0,3),
    # still distance 3; after 3 more lefts agent is at (0,0) ON the target.
    for _ in range(3):
        obs, _, _, _ = env.step(0)
    for _ in range(3):
        obs, _, _, _ = env.step(2)
    assert "A" in obs["visible"] and obs["visible"]["A"] == (0, 0)


def test_bump_leaves_agent_in_place():
    layout = _handmade_layout()
    layout["obstacles"] = {(2, 3)}
    env = GridWorld(layout, C.DEFAULT_TASK)
    obs = env.reset()
    obs, r, _, info = env.step(0)   # up into the obstacle at (2,3)
    assert info["bump"] is True
    assert obs["pos"] == (3, 3)
    assert abs(r - C.REWARD_STEP) < 1e-9


def _corridor_layout():
    """7x7 layout where the ONLY passage from column <=1 to column >=4 is
    row 3 (columns 2 and 3 are blocked everywhere else).  Any shortest path
    from (3, 1) to (3, 6) must therefore run straight through row 3 and
    through the target cell D at (3, 3)."""
    return {
        "grid_size": 7,
        "obstacles": {(r, 2) for r in (0, 1, 2, 4, 5, 6)}
                     | {(r, 3) for r in (0, 1, 2, 4, 5, 6)},
        "targets": {"A": (0, 5), "B": (6, 5), "C": (3, 6), "D": (3, 3)},
        "start": (3, 1),
    }


def test_bfs_paths_treat_target_cells_as_traversable():
    """REGRESSION TEST for the Experiment D branch-task mechanism.

    BFS treats target cells as ORDINARY FREE CELLS: it blocks only obstacle
    cells.  A shortest path to a goal may therefore pass THROUGH the cell of
    another target -- even a target the navigating agent currently considers
    unacceptable.  (The schema agent's "avoid unacceptable targets" rule
    applies only to its random-walk fallback, never to BFS navigation; see
    tests/test_agents.py and docs/methodology.md section 9b.)"""
    layout = _corridor_layout()
    path = bfs_shortest_path(layout["obstacles"], (3, 1), (3, 6), 7)
    assert path is not None
    assert len(path) == 5                      # (3,2)...(3,6), no detour exists
    assert path[0] == (3, 2)                   # first move into the corridor
    # The path runs straight through the target cell D at (3, 3): BFS does
    # NOT route around it, and never has -- this behaviour is what made the
    # branch task succeed in Experiment D, so it must not "silently" change.
    assert (3, 3) in path
    assert path[-1] == (3, 6)


def test_bfs_shortest_paths():
    obstacles = set()
    # Empty grid: BFS length equals Manhattan distance.
    path = bfs_shortest_path(obstacles, (3, 3), (0, 0))
    assert len(path) == 6 and path[-1] == (0, 0)
    assert path[0] in [(2, 3), (3, 2)]
    assert bfs_shortest_path(obstacles, (1, 1), (1, 1)) == []

    # One wall with a single gap, forcing a detour: column 3 blocked on
    # rows 0..5, open at row 6.  Shortest path (3,1) -> (3,5) goes through
    # the gap and has length 10 (Manhattan distance would be 4).
    obstacles = {(i, 3) for i in range(6)}
    path = bfs_shortest_path(obstacles, (3, 1), (3, 5))
    assert path is not None
    assert len(path) == 10
    # verify it is a valid collision-free path
    cell = (3, 1)
    for step in path:
        dr = step[0] - cell[0]
        dc = step[1] - cell[1]
        assert (dr, dc) in set(ACTION_DELTAS.values())
        assert step not in obstacles
        cell = step
    assert cell == (3, 5)

    # Unreachable goal
    obstacles = {(i, 3) for i in range(7)}
    assert bfs_shortest_path(obstacles, (3, 0), (3, 6)) is None
