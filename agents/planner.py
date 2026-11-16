"""Oracle planner: an UPPER-BOUND reference agent, explicitly labelled.

This agent has complete knowledge of the environment: the true target
positions, the obstacle map and the true task graph.  It walks a BFS
shortest path to the nearest acceptable target at every stage and therefore
completes a cycle in (almost) the minimum possible number of steps.

It is NOT part of the fair comparison.  It exists only to show how much room
remains between the learning agents and a fully informed planner.  Its
per-cycle step counts are reported separately in every experiment summary.
"""

from environment.grid_world import bfs_shortest_path, direction_between


class OraclePlanner:
    def __init__(self, layout, task_graph):
        self.targets = dict(layout["targets"])
        self.obstacles = set(layout["obstacles"])
        self.grid_size = layout["grid_size"]
        self.successors = task_graph["successors"]
        self.bfs_calls = 0
        self.last_node = "START"

    def reset_episode_state(self):
        self.last_node = "START"

    def record_success(self, obs):
        """The planner knows the true task graph, so it can update its FSM
        state from the identity at its own cell after a correct collection."""
        for tid, pos in self._visible_at(obs).items():
            if pos == obs["pos"]:
                self.last_node = tid
                return

    @staticmethod
    def _visible_at(obs):
        # The planner receives the same observation as every other agent;
        # it just additionally knows the full layout by construction.
        return obs["visible"]

    def choose_action(self, obs):
        pos = obs["pos"]
        best_action, best_len = None, None
        for tid in self.successors[self.last_node]:
            self.bfs_calls += 1
            path = bfs_shortest_path(self.obstacles, pos,
                                     self.targets[tid], self.grid_size)
            if path is None:
                raise RuntimeError(
                    f"oracle planner: target {tid} unreachable -- "
                    "this should never happen in a validated layout")
            if not path:
                # path == [] would mean the planner stands on an expected
                # target without having collected it -- impossible with the
                # true task graph, so treat it as a hard error.
                raise RuntimeError(
                    f"oracle planner: standing on unreachable/uncollectable "
                    f"target {tid} -- this should never happen")
            if best_len is None or len(path) < best_len:
                best_len = len(path)
                best_action = direction_between(pos, path[0])
        return best_action
