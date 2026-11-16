# Limitations

An honest list of what this study does and does not show.

## 1. Environment scale and determinism

The world is a deterministic 7x7 grid with 4 targets and 6 obstacles. At this
scale, random exploration is cheap: a uniform random walk stumbles onto any
given cell within a few hundred steps, which is why even the reactive
no-memory ablation reaches 100% success in Experiment B. In larger or
noisier worlds, exploration becomes the bottleneck, the schema agent's
random-walk fallback would weaken, and Q-learning's tabular memory
(`|S| x |A|` cells) would become intractable. Nothing here was designed to
extrapolate to those regimes.

Transitions are noiseless and rewards are clean (+10 / -5 / -0.04). With
slippery moves or stochastic rewards, the schema-learning threshold
("count >= 3 and >= 50% of row max") would need a probabilistic treatment,
and the Q-transfer comparison could change.

## 2. Difference from biological navigation and published protocols

The project is *inspired by* schema-learning findings (e.g. Tse et al., 2007;
Behrens et al., 2018), but it implements no biological mechanism, no
hippocampal replay, no incremental consolidation, and no event protocol from
any published study. "One-shot" in the title refers to a narrow operational
fact — after perceiving a target once, the agent navigates to it optimally —
and NOT to the psychological claim that a single exposure produces lasting
memory. In particular, the schema agent must still discover all four targets
by wandering (metric 5: it has observed all 4 identities before its first
completed cycle); it never knows an unobserved target's location.

## 3. Tabular Q-learning limits

The Q agents are deliberately basic: no function approximation, no replay, no
optimistic initialization. Their state representation (cell, progress counter)
is itself a small head start: it factorizes the task by progress, which the
schema agent also uses (via its FSM). Results such as "Q-learning needs ~2000
steps to its first cycle" are statements about *this* learner on *this* task
family, not about Q-learning in general. With 30 runs the CIs are tight for
these agents, but the absolute numbers would move with different
hyperparameters — which were fixed a priori and not tuned (see
`docs/methodology.md`).

## 4. Planning cost asymmetry and obstacle-information asymmetry

**Computation.** The schema agent runs one BFS per navigation step (mean 17.7
BFS calls before its first cycle in Experiment A) while Q-learning runs none.
We report interaction steps (equal budgets) and computation separately
(`bfs_calls_at_first_cycle`, `*_wall_s` columns) and claim nothing about
computational equality. A fair "same compute" comparison would need, e.g., a
matching amount of planning/model learning for the Q agents — that variant
does not exist here. Conversely, one could argue BFS on a known map is a
computationally cheap classical tool, and granting it is the *point* of the
schema design (structure + model-based navigation); the reader should simply
be aware that the comparison is about knowledge representation, not about
computation.

**Obstacle information.** The schema agent is given the COMPLETE obstacle map
of each environment at wiring time; in the original experiments the Q agents
receive none (they learn obstacle locations only implicitly from bump
penalties). Experiment E (`experiments/obstacle_info_control.py`, separate
protocol and separate output files) quantifies what this asymmetry is worth:
giving Q-learning the map as pure action masking roughly halves the
from-scratch agent's first-cycle cost (2095.9 -> 1077.0 steps), while bump-
learned memory achieves most of that (1460.7). Two conclusions follow: (i)
obstacle information explains a real but MINORITY part of the schema agent's
advantage (the masked Q agent is still ~5x slower than the schema agent);
(ii) negative Q-table transfer is robust to -- indeed amplified by --
obstacle-information equalisation (transferred success falls from 66.7% to
16.7% with the map; a mask-consistent source lifecycle recovers only to
26.7%). What Experiment E still does NOT equalise: BFS planning, the task
FSM and the spatial-memory wiring remain bundled with the schema agent, so
the residual schema-vs-masked-Q gap is not attributable to structure alone.

**Source training experience.** The transferred Q agent trains for 200 source
episodes and the schema agent for 65; `q_fresh` gets none by definition.
Target budgets are equal, source budgets are not: the study compares
transfer-ready architectures after family-typical source training, not
experience-matched training (see `docs/methodology.md` section 5).

## 5. The schema representation is minimal and brittle

* The learned schema is a thresholded count matrix. It needs >= 3 observations
  of every sequence transition; with fewer source episodes it degrades
  gracefully (missing successors trigger exploration) but may under-connect.
* The FSM state is updated from reward events. When the environment's true
  sequence diverges from the learned schema (Experiment C), the frozen schema
  makes the agent *actively* fail: its avoidance rule refuses to enter
  env-expected targets and its planning drags it to schema-expected ones
  (mechanism trace in `docs/methodology.md` section 9a). This brittleness is a
  real property of confident wrong priors, but an adaptive agent that detects
  persistent prediction/reward mismatch and re-estimates its successor counts
  would plausibly recover; that variant is future work, not part of these
  results.
* Tasks are single ordered sequences or tiny DAGs over 4 identities. Richer
  structure (context-dependent switching, partial observability of the stage,
  parallel tasks) is out of scope.

## 6. The similarity measure (Experiment D)

Edge-Jaccard over directed edges is simple and transparent but crude: it
ignores where an edge sits in the traversal (an extra branch anywhere counts
the same), it treats a redirected edge and a missing edge identically, it is
sensitive to the size of the node set, and START edges inflate similarity for
all tasks sharing the same entry point. With only 4 task types, the reported
Spearman correlations (0.95 with success rate, -0.74 with censored steps) are
descriptive illustrations, not statistical evidence; a proper test would need
a family of many task graphs sampled systematically.

## 7. Statistical caveats

30 runs give Wilson intervals that are still wide at extreme success rates
(e.g. 100% -> [88.6, 100]). Bootstrap intervals for means are sensitive to
censoring choices; we report medians and censored medians alongside means so
the reader can see the shape. The budget (60 x 400 steps) was fixed a priori;
a larger budget would raise every success rate (the 66.7% and 60% figures for
the transferred Q agent are budget-relative, not asymptotic).

## 8. Design decisions a reviewer might question

* The transferred Q agent starts with epsilon = 0.3 (not 1.0). With 1.0 the
  transferred table would not influence early actions at all; with 0.3 it
  does. This favours *using* the transfer, which is what makes the negative
  transfer finding meaningful — but a different schedule could shift the
  numbers.
* The avoidance rule ("never enter a known unacceptable target") is shared by
  the schema agent and its memory ablation but has no counterpart in the Q
  agents (they must learn avoidance from penalties). It is scoped to the
  agents' RANDOM-WALK exploration only: BFS navigation treats target cells as
  ordinary traversable cells and can cross known-but-unacceptable targets
  (regression-tested; this is what produces the branch-task successes of
  Experiment D -- `docs/methodology.md` section 9b). Removing the rule would
  soften Experiment C's 0% (the agent would random-walk into env-expected
  targets eventually) but also weaken Experiment A's first-cycle advantage.
  We chose the rule because it follows directly from the agent's own
  knowledge, and we documented its consequences rather than hiding them.
* The `q_fresh` baseline receives no source training by definition; giving it
  source training and then discarding the Q-table would be identical in
  effect and 2x slower to run.

## 9. What we claim, precisely

Within this environment family and these agents: (i) the schema ARCHITECTURE
-- a bundle of given obstacle map, BFS planning, task-state FSM and spatial
memory -- transfers to new layouts roughly an order of magnitude faster than
tabular Q-learning; the obstacle-map component of that bundle accounts for a
minority of the first-cycle advantage (Experiment E), while BFS planning and
the task FSM remain bundled (Experiment B separates the FSM and memory
components within the schema family); (ii) transferring spatial Q-values
across layout changes produces negative transfer, robust to
obstacle-information equalisation and slightly amplified by it (Experiment
E); (iii) removing the task-state component is catastrophic while removing
spatial memory merely slows the agent; (iv) a frozen, wrong abstract
structure is worse than no structure; (v) transfer success tracks task-graph
similarity in our small graph family, descriptively. Beyond this setting,
all bets are explicitly off.
