"""Plotting for the schema-navigation study.

All figures are saved to results/figures/ as PNG.  The style is deliberately
plain matplotlib (no seaborn) so a student can read and adapt every line.
"""

import matplotlib

matplotlib.use("Agg")   # headless rendering: no display needed

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import config as C
from analysis.statistics import bootstrap_ci_mean

# Colour-blind friendly palette (Okabe-Ito, subset)
COLORS = {
    "q_fresh": "#0072B2",
    "q_transferred": "#E69F00",
    "schema": "#009E73",
    "schema_full": "#009E73",
    "schema_no_memory": "#CC79A7",
    "schema_no_task_state": "#56B4E9",
    "oracle_planner": "#999999",
    # Experiment E control agents (kept visually distinct from the original
    # agents; sky blue for the fresh masked agent, grey for bump memory).
    "q_fresh_masked": "#56B4E9",
    "q_transferred_masked": "#E69F00",
    "q_transferred_maskcons": "#CC79A7",
    "q_fresh_bumpmem": "#999999",
}
LABELS = {
    "q_fresh": "Q-learning from scratch",
    "q_transferred": "Q-learning + transferred Q-table",
    "schema": "Schema-based agent",
    "schema_full": "Schema-based agent (full)",
    "schema_no_memory": "Schema agent, no spatial memory",
    "schema_no_task_state": "Schema agent, no task state",
    "oracle_planner": "Oracle planner (reference)",
    # Experiment E control agents.  Labels contain explicit line breaks (the
    # tick-label logic below respects them) so that four long agent names
    # never collide horizontally.
    "q_fresh_masked": "Q fresh\n+ full map (masked)",
    "q_transferred_masked": "Q transferred\n+ full map (masked)",
    "q_transferred_maskcons": "Q transferred\n+ mask-consistent source",
    "q_fresh_bumpmem": "Q fresh\n+ bump-learned memory",
}


def agent_color(agent):
    return COLORS.get(agent, "#333333")


def agent_label(agent):
    return LABELS.get(agent, agent)


def _finalize(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved figure: {path}")


def plot_learning_curves(curves_df, out_path, title,
                         max_steps=None, agents=None):
    """Mean per-episode steps (censored at the per-episode cap) with a 95%
    bootstrap CI band across independent runs, one line per agent."""
    if agents is None:
        agents = sorted(curves_df["agent"].unique())
    if max_steps is None:
        max_steps = curves_df["steps_censored"].max()

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for agent in agents:
        sub = curves_df[curves_df["agent"] == agent]
        episodes = sorted(sub["episode"].unique())
        # matrix: rows = runs, cols = episodes (aligned by episode number)
        matrix = (sub.pivot_table(index="run", columns="episode",
                                  values="steps_censored")
                  .reindex(columns=episodes).to_numpy())
        means = np.nanmean(matrix, axis=0)
        # bootstrap over runs for each episode point
        rng = np.random.default_rng(C.SEED_BOOTSTRAP)
        n_runs = matrix.shape[0]
        idx = rng.integers(0, n_runs, size=(C.BOOTSTRAP_RESAMPLES, n_runs))
        boot = matrix[idx].mean(axis=1)           # resamples x episodes
        lo = np.nanpercentile(boot, 2.5, axis=0)
        hi = np.nanpercentile(boot, 97.5, axis=0)
        x = np.asarray(episodes)
        ax.plot(x, means, color=agent_color(agent), label=agent_label(agent))
        ax.fill_between(x, lo, hi, color=agent_color(agent), alpha=0.15)

    ax.axhline(max_steps, color="#BBBBBB", linestyle="--", linewidth=1)
    # Label at the LEFT edge: the legend sits at the top right, and the
    # right-hand end of the cap line is where the slowest agents' curves
    # approach the cap, so a right-aligned label collided with them.
    ax.text(0.01, max_steps, "per-episode cap", ha="left", va="bottom",
            fontsize=8, color="#888888", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Episode in the new environment")
    ax.set_ylabel("Env steps to complete one cycle\n(censored at cap)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _finalize(fig, out_path)


def plot_first_cycle_bars(summary_df, out_path, title, reference_steps=None,
                          reference_label="Oracle planner (reference)",
                          order=None):
    """Bar chart of mean steps to first complete cycle (successful runs only)
    with 95% bootstrap CI error bars; success rate annotated on each bar."""
    if order is None:
        order = list(summary_df["agent"])
    # Slightly wider canvas: four-bar charts (Experiment E) otherwise touch
    # at two-line tick labels.
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    xs, heights, lowers, uppers, rate_texts = [], [], [], [], []
    for i, agent in enumerate(order):
        row = summary_df[summary_df["agent"] == agent]
        if row.empty:
            continue
        row = row.iloc[0]
        if row["successes"] == 0:
            # no successful run at all: draw at the cap with 0 height marker
            xs.append(i)
            heights.append(0.0)
            lowers.append(0.0)
            uppers.append(0.0)
            rate_texts.append("0% success")
            continue
        xs.append(i)
        heights.append(row["mean_first_cycle_steps"])
        lowers.append(row["mean_first_cycle_steps"] - row["ci_lo"])
        uppers.append(row["ci_hi"] - row["mean_first_cycle_steps"])
        rate_texts.append(f"{row['success_rate']:.0%} success")

    bars = ax.bar(xs, heights, color=[agent_color(a) for a in order],
                  width=0.6)
    ax.errorbar(xs, heights, yerr=[lowers, uppers], fmt="none",
                ecolor="#333333", capsize=4, lw=1)
    for x, h, text in zip(xs, heights, rate_texts):
        ax.text(x, h, "  " + text, ha="left", va="bottom", fontsize=8)
    if reference_steps is not None and np.isfinite(reference_steps):
        # The reference line is shown as a legend entry (not as free text on
        # the line): free text at the line's height collided with the bars
        # and with the line itself in Experiment A.
        ax.axhline(reference_steps, color="#999999", linestyle="--", lw=1)
        ax.legend(handles=[Line2D([0], [0], color="#999999", ls="--", lw=1,
                                  label=f"{reference_label}: "
                                        f"{reference_steps:.0f} steps")],
                  loc="upper left", frameon=False, fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [agent_label(a) if "\n" in agent_label(a)
         else agent_label(a).replace(" (", "\n(") for a in order],
        fontsize=8)
    ax.set_ylabel("Env steps to first complete cycle\n"
                  "(mean over successful runs, 95% bootstrap CI)")
    ax.set_title(title)
    _finalize(fig, out_path)


def plot_similarity(similarity_df, out_path, title):
    """Two panels: (left) success rate and median steps per target graph type;
    (right) structural similarity vs median steps among successful runs."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.5, 4.2), constrained_layout=True)

    types = list(similarity_df["target_task"])
    x = np.arange(len(types))
    ax1.bar(x - 0.2, similarity_df["success_rate"], width=0.4,
            color="#0072B2", label="success rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Success rate within budget")
    ax1.set_xticks(x)
    ax1.set_xticklabels(types, fontsize=8)
    ax1b = ax1.twinx()
    med = similarity_df["median_steps_success"].astype(float)
    ax1b.bar(x + 0.2, med, width=0.4, color="#E69F00",
             label="median steps (successful runs)")
    ax1b.set_ylabel("Median steps to first cycle")
    ax1.legend(loc="upper left", fontsize=7)
    ax1b.legend(loc="upper right", fontsize=7)
    ax1.set_title("Transfer performance per target task graph")
    # Honest marker for task types with 0% success: their "steps among
    # successful runs" is undefined, so the orange bar is absent there.
    # Drawn mid-panel on the ax1 scale so it cannot collide with the x tick
    # labels at the bottom.
    for xi, val in zip(x, med):
        if not np.isfinite(val):
            ax1.text(xi + 0.2, 0.45, "n/a\n(0% success)", ha="center",
                     va="center", fontsize=7, color="#888888")

    sim = similarity_df["similarity_to_source"].astype(float)
    med = similarity_df["median_steps_censored"].astype(float)
    ax2.scatter(sim, med, color="#009E73", s=60, zorder=3)
    for xi, yi, name in zip(sim, med, types):
        ax2.annotate(name, (xi, yi), textcoords="offset points",
                     xytext=(6, 4), fontsize=8)
    rho = similarity_df["spearman_similarity_vs_steps"].iloc[0]
    ax2.set_xlabel("Edge-Jaccard similarity to the source task graph")
    ax2.set_ylabel("Median steps to first cycle\n"
                   "(failures censored at total budget)")
    # Short title (a longer one was clipped by the panel edge) with the
    # correlation and the sample size in a bottom-left box: the top of this
    # panel holds the censored points of the 0%-success task types.
    ax2.set_title("Similarity vs transfer")
    ax2.text(0.03, 0.30, f"Spearman rho = {rho:.2f}\nn = {len(types)} task "
             "types (descriptive, not a test)", ha="left", va="top",
             fontsize=8, transform=ax2.transAxes,
             bbox=dict(facecolor="white", edgecolor="#CCCCCC", alpha=0.85))
    _finalize(fig, out_path)
