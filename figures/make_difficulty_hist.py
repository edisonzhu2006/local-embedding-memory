"""Vertical histogram: how many of 5 models pass each question.

For each of the 19 questions, count strict-correct models (partials
round down). Bin counts at 0, 1, 2, 3, 4, 5. Annotate the long left
tail of "everyone fails" + "only one model" questions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

ACCENT  = "#B85C2E"   # highlighted bin (long tail) — paper headline orange
NEUTRAL = "#7DA3B4"   # other bins — paper slate

INK     = "#1f1f1f"
MUTED   = "#6b6b6b"
GRID    = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# Pass count per question, recomputed for N=8 models (4 embedders + 4 agents).
# Scoring: strict (partials round down).  Source: chart_scorecard.py grade matrix.
# 0 models: (none)
# 1 model:  L4_3, L4_4, L5_3
# 2 models: L2_3, L4_1, L4_2, L5_2, L5_4
# 3 models: L3_1, L3_2
# 4 models: L3_4
# 5 models: L3_3, L5_1
# 6 models: L2_1, L2_2
# 7 models: L1_3, L1_4
# 8 models: L1_1, L1_2
counts = [0, 3, 5, 2, 1, 2, 2, 2, 2]
bins   = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Highlight bins 0–2 — "the long tail open VLMs struggle on".
colors = [ACCENT if b <= 2 else NEUTRAL for b in bins]

fig_w, fig_h = 9.0, 7.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.16, right=0.95)

bar_w = 0.72
ax.bar(bins, counts, bar_w, color=colors, edgecolor="none", zorder=3)

# Count labels above bars
for b, c in zip(bins, counts):
    ax.text(b, c + 0.18, str(c),
            ha="center", va="bottom",
            fontsize=12, color=INK, fontweight="bold")

# Bracket / annotation over bins 0-2 (the long tail)
ax.annotate("",
            xy=(-0.36, 6.2), xytext=(2.36, 6.2),
            arrowprops=dict(arrowstyle="-", color=ACCENT, lw=1.4))
ax.text(1.0, 6.4,
        "8 / 19 questions answered by ≤ 2 of 8 models",
        ha="center", va="bottom",
        fontsize=10.5, color=ACCENT, fontweight="bold")

ax.set_xticks(bins)
ax.set_xticklabels([str(b) for b in bins], fontsize=12, color=INK)
ax.set_xlabel("Number of models passing (out of 8)",
              fontsize=12, color=INK, labelpad=10)
ax.set_ylabel("Number of questions",
              fontsize=12, color=INK, labelpad=10)
ax.set_ylim(0, 7.5)
ax.set_yticks([0, 1, 2, 3, 4, 5, 6])

ax.text(0, 1.13, "How hard was each question?",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.05,
        "Across 4 embedders + 4 agents, 8/19 questions are answered by\n"
        "≤ 2 models — the hard tail no model in our zoo cleanly solves.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=11)
ax.tick_params(axis="y", length=4, labelsize=10, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

out = OUT / "chart_difficulty_hist.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
