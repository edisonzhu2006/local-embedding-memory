"""Per-model L1→L5 ladder: where each model's score actually comes from.

Five thin vertical columns (one per model), each segmented by level.
Segment height = strict-correct count at that level. Embedders show
heavy bottom (L1) and grey upward; RAVEN spreads across all levels.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# 5-level warm sequential palette: cream → deep rust.
# L1 (easy) light, L5 (hard) dark.
LEVEL_COLORS = ["#FCE4C9", "#F2BE85", "#E89854", "#C76E2C", "#8E4A1C"]
# Label color per segment — dark on light, white on dark.
LABEL_COLORS = ["#5C3A1A", "#5C3A1A", "white", "white", "white"]
NEUTRAL      = "#E5E5E5"

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

# Strict-correct counts per level (partials → 0). Computed from scorecard.py.
# Order of levels: L1, L2, L3, L4, L5.  Sums match strict totals.
# (model, [L1, L2, L3, L4, L5])
data = [
    ("Gemma 4\n31B",        [4, 3, 4, 2, 2]),  # 15
    ("Gemma 3\n27B",        [4, 3, 2, 2, 2]),  # 13
    ("Qwen2.5-VL\n32B",     [4, 2, 4, 1, 2]),  # 13
    ("Qwen3-VL\n32B",       [4, 2, 3, 0, 3]),  # 12
    ("QQMM",                [3, 2, 1, 0, 1]),  # 7
    ("ViT-H-14",            [4, 1, 0, 1, 0]),  # 6
    ("SigLIP",              [4, 2, 0, 0, 0]),  # 6
    ("ViT-L-14",            [4, 1, 0, 0, 0]),  # 5
]

names   = [d[0] for d in data]
counts  = np.array([d[1] for d in data])      # shape (5 models, 5 levels)
totals  = counts.sum(axis=1)
n_models = len(data)
n_levels = 5

x = np.arange(n_models)
bar_w = 0.55

fig_w, fig_h = 11.0, 9.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.80, bottom=0.16, left=0.12, right=0.95)

# Stack levels bottom → top
bottoms = np.zeros(n_models)
for li in range(n_levels):
    seg = counts[:, li]
    ax.bar(x, seg, bar_w, bottom=bottoms,
           color=LEVEL_COLORS[li], edgecolor="white", linewidth=1.2,
           label=f"L{li+1}", zorder=3)
    # In-segment count labels (only if segment >= 2)
    for xi, s, b in zip(x, seg, bottoms):
        if s >= 2:
            ax.text(xi, b + s/2, str(s),
                    ha="center", va="center",
                    fontsize=11, color=LABEL_COLORS[li], fontweight="bold")
    bottoms += seg

# Total at top of each bar
for xi, t in zip(x, totals):
    ax.text(xi, t + 0.4, f"{t} / 19",
            ha="center", va="bottom",
            fontsize=12, color=INK, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=11, color=INK, linespacing=1.3)
ax.set_ylabel("Strict-correct questions  (out of 19)",
              fontsize=12, color=INK, labelpad=10)
ax.set_ylim(0, 18)
ax.set_yticks([0, 4, 8, 12, 16])

ax.text(0, 1.10, "Where each model's score comes from",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.03,
        "Embedders score almost entirely on L1 direct localization.\n"
        "All four agents reach into L3-L5; the four embedders do not.",
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

leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
                frameon=False, fontsize=11, ncol=5,
                handlelength=1.0, handletextpad=0.5, columnspacing=1.4,
                title="Level", title_fontsize=11)
for txt in leg.get_texts():
    txt.set_color(INK)
leg.get_title().set_color(INK)

out = OUT / "chart_level_ladder.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
