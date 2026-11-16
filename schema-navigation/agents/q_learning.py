"""Tabular Q-learning agent.

State representation
    (cell index, progress counter k) where cell index = row * grid_size + col
    and k is the number of correct targets collected so far in the episode.
    k is part of the observation returned by the environment; it is
    information the agent could also derive by counting its own positive
    rewards, so it is not hidden information (see docs/methodology.md).

Q-learning update (Sutton & Barto, 2018, ch. 6)
    Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') * (1 - done) - Q(s, a)]

    When the episode ends (done=True) the future term is zero, because no
    further reward can follow the terminal transition.

Action selection
    epsilon-greedy: with probability epsilon pick a uniformly random action,
    otherwise pick the action with the highest Q-value at the current state
    (ties broken by lowest action index -- deterministic and simple).

Transfer
    ``q_init`` lets a new agent start from a copy of a previously learned
    Q-table.  The copy semantics matter: the source table itself must remain
    untouched (this is verified by tests/test_agents.py).
"""

import numpy as np


class TabularQAgent:
    def __init__(self, grid_size, n_stages, alpha, gamma,
                 epsilon_start, epsilon_min, epsilon_decay, q_init=None):
        self.grid_size = grid_size
        self.n_stages = n_stages
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_actions = 4

        n_cells = grid_size * grid_size
        if q_init is None:
            self.q = np.zeros((n_cells, n_stages, self.n_actions))
        else:
            if q_init.shape != (n_cells, n_stages, self.n_actions):
                raise ValueError(
                    f"q_init has shape {q_init.shape} but the agent needs "
                    f"{(n_cells, n_stages, self.n_actions)}")
            # Copy: transferring must never silently alias the source table.
            self.q = q_init.copy()

    # -- state encoding -------------------------------------------------------
    def encode_state(self, obs):
        """Map an observation to the Q-table indices (cell, k).

        k is clamped to n_stages - 1: at the very last step of a completed
        cycle the environment reports k == completion_length, which would be
        an out-of-range index; that state is terminal and never acts again.
        """
        r, c = obs["pos"]
        cell = r * self.grid_size + c
        k = min(int(obs["k"]), self.n_stages - 1)
        return (cell, k)

    # -- action selection -------------------------------------------------------
    def act(self, obs, rng, greedy=False):
        """Pick an action.  ``greedy=True`` disables exploration (evaluation)."""
        cell, k = self.encode_state(obs)
        if not greedy and rng.random() < self.epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.q[cell, k]))

    # -- learning ---------------------------------------------------------------
    def update(self, state, action, reward, next_state, done):
        """One Q-learning update.  ``state``/``next_state`` are (cell, k)
        tuples as produced by ``encode_state``."""
        cell, k = state
        ncell, nk = next_state
        best_next = 0.0 if done else float(np.max(self.q[ncell, nk]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[cell, k, action]
        self.q[cell, k, action] += self.alpha * td_error

    def end_episode(self):
        """Called once per episode; decays the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
