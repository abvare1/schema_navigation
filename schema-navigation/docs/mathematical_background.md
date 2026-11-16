# Mathematical background

Every equation used in the project, with an intuition and a tiny example.
Notation: `s` state, `a` action, `r` reward, `s'` next state, `gamma` discount
factor, `alpha` learning rate.

## 1. Markov Decision Processes (MDPs)

An MDP is a tuple `(S, A, P, R, gamma)`:

* `S` — set of states. Here: `(row, col, k)` — agent cell plus how many
  correct targets have been collected this episode (k = 0..4 for the cycle
  task, so 7x7x5 = 245 states).
* `A` — actions: up/down/left/right (4 actions).
* `P(s'|s,a)` — transition probabilities. Our grid is deterministic: the
  action moves the agent one cell, or keeps it in place on a bump.
* `R(s,a)` — expected reward: step penalty, plus +10/-5 when entering a
  target cell (see `docs/methodology.md` section 3).
* `gamma` in [0,1) — how much future rewards are worth now.

*Intuition.* "Markov" means the next state and reward depend only on the
current state and action — the full history adds nothing. Our environment is
Markov because knowing the cell and the task progress k is enough to predict
every consequence of every action.

## 2. Returns and the value of acting

The **return** from a state is the discounted sum of future rewards:
`G_t = r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + ...`

*Example.* With gamma = 0.95, a reward of +10 arriving 6 steps from now is
worth `0.95^6 * 10 ≈ 7.35` today. Step penalties of -0.04 arriving forever
sum to `-0.04/(1-0.95) = -0.8`, which is small relative to +10: the agent
prefers the target even if it takes a detour, but mildly prefers short
detours.

## 3. Q-learning

Q-learning (Watkins, 1989; Sutton & Barto, 2018, ch. 6) learns `Q(s,a)`, an
estimate of the return after taking `a` in `s` and acting (greedily) forever
after. After each transition `(s, a, r, s')`:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
```

The bracket is the **TD error**: (what we just observed) − (what we believed).
When `s'` is terminal, `max_a' Q(s',a')` is replaced by 0 because no reward
follows the terminal transition.

*Example (from `tests/test_agents.py`).* alpha = 0.5, gamma = 0.9, all Q = 0.
Take action "left" at the centre cell, receive r = 10, next state not terminal
with max Q(s',·) = 2:

```
Q(s,left) = 0 + 0.5 * (10 + 0.9*2 - 0) = 5.9
```

A second identical transition then gives
`5.9 + 0.5*(10 + 1.8 - 5.9) = 8.85`: each visit nudges the estimate towards
the observed target. Q-learning is *off-policy*: the `max` bootstrap uses the
greedy future, regardless of how the agent actually behaved. Convergence to
`Q*` is guaranteed (under standard conditions) provided every state-action
pair keeps being sampled — which is what epsilon-greedy exploration ensures.

## 4. Epsilon-greedy exploration

With probability epsilon take a uniformly random action; otherwise take
`argmax_a Q(s,a)`:

```
a = random action      with probability epsilon
a = argmax_a Q(s,a)    with probability 1 - epsilon
```

We decay epsilon geometrically per episode: `epsilon <- max(0.05, 0.98 *
epsilon)`, so early episodes explore broadly (epsilon = 1.0 is fully random)
and later episodes exploit what was learned. *Why decay?* Random walks are how
the agent first stumbles onto targets; once Q-values are informative, random
moves mostly waste steps and trigger wrong-target penalties.

## 5. Finite-state task schemas

A task schema is a deterministic finite-state machine over target identities.
States are identities (plus the pseudo-state START); the transition function
says which identity may be collected next:

```
START -> A -> B -> C -> D -> A      (the cycle task)
```

Formally the schema is a directed graph `G = (V, E)`, `V ⊆ {START,A,B,C,D}`;
collecting identity `x` while the FSM state is `v` is "correct" iff
`(v, x) ∈ E`, and the FSM then moves to `x`. The schema agent learns `E` from
feedback: every time a collection yields +10 right after the last +10 came
from identity `v`, increment `count[v][x]`; after training,
`(v,x) ∈ E_learned` iff `count[v][x] >= 3` and `count[v][x] >= 0.5 * max_x
count[v][x]`. For the cycle task this recovers exactly one successor per node
(verified in `tests/test_agents.py`).

*Why is this "abstract"?* It contains **no spatial information whatsoever** —
only order. It therefore transfers to any layout whose targets implement the
same order, which is precisely what Experiment A tests.

## 6. Breadth-First Search (BFS)

BFS finds a shortest path in an unweighted graph. Starting from the agent's
cell, it visits cells in order of increasing distance (a FIFO queue), records
each cell's parent, and stops at the goal; the path is recovered by following
parents backwards. On a 7x7 grid there are at most 49 cells, so BFS is exact
and trivial in cost; it is optimal because every step costs exactly 1.

*Example.* From (3,3) to (0,0) on an empty grid BFS returns any 6-step path,
e.g. up,up,up,left,left,left; with a wall at (2,3) it returns a different
6-step route around the wall — never a longer one
(`tests/test_environment.py` checks exact lengths and validity).

The schema agent and the planner both use BFS **on the obstacle map only**;
for the schema agent the goal is a *discovered* target position, never a true
position of an undiscovered target. Two precise consequences worth noting:
(a) the obstacle map is GIVEN to the schema agent at wiring time, not learned
by bumping (the Q baselines of the original experiments receive no map --
see `docs/methodology.md` section 6 and Experiment E); (b) BFS blocks only
obstacle cells, so a shortest path may pass *through* a target cell -- even a
target the agent currently considers unacceptable (`docs/methodology.md`
section 9b, regression-tested).

## 7. Transfer learning and negative transfer

Transfer learning reuses knowledge from a source task to speed up a target
task. Positive transfer = faster/better learning on the target; **negative
transfer** = slower/worse, because the prior knowledge points the wrong way.

In our setting:

* The schema transfers the *order* of targets, which is unchanged when only
  target locations move — the classic "level" at which reuse pays off.
* The Q-table entangles order with geography: `Q((r,c),k)` is large for cells
  near the *source* locations of the stage-k target. In a new layout those
  cells are often empty or, worse, lie on the cell of a different target, so
  greedy actions are confidently wrong and the agent must first *unlearn*
  before it can learn. That is negative transfer, and it is what Experiment A
  measures (66.7% success vs 100% from scratch within identical budgets).

*Experiment C* shows the mirror image for schemas: a schema that is wrong
about the *order* traps the agent (0% success). The lesson of both: transfer
helps exactly when the transferred representation matches what actually
stayed invariant.

## 8. Confidence intervals

A 95% confidence interval is a range that would contain the true quantity in
about 95% of repeated experiments.

**Wilson score interval (for success rates).** With k successes in n runs,
p̂ = k/n, z = 1.96:

```
centre = (p̂ + z²/(2n)) / (1 + z²/n)
half   = z * sqrt(p̂(1-p̂)/n + z²/(4n²)) / (1 + z²/n)
CI = centre ± half
```

Unlike the textbook `p̂ ± 1.96*sqrt(p̂(1-p̂)/n)`, the Wilson interval never
leaves [0,1] and stays sensible at 0% or 100% success — important here, since
several agents score exactly 0% or 100%.
*Example.* 20 successes in 30 runs: p̂ = 0.667, Wilson CI = [0.488, 0.808]
(the plain normal interval would give [0.498, 0.835], slightly optimistic).

**Percentile bootstrap (for means of skewed step counts).** Resample the n =
30 observed values with replacement 10,000 times, compute each resample's
mean, and take the 2.5th and 97.5th percentiles of those means. This makes no
normality assumption — crucial because "steps to first cycle" mixes small
schema completions with budget-capped failures and is strongly right-skewed.
The bootstrap uses a fixed seed (12345) so intervals are exactly reproducible.
*Example.* The schema agent's 30 first-cycle values in Experiment A give
mean 215.3 with bootstrap CI [164.0, 269.9]; the interval is asymmetric,
which is exactly the information a symmetric +/-1.96*SE interval would hide.

**Censoring.** When a run fails, its "steps to first cycle" is undefined; we
report the failure separately (success rate) and, where a single comparable
number is needed (Experiment D), use the budget-censored value (24,000
steps), labelled as censored wherever it appears.
