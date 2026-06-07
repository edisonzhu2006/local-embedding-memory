"""Horizontal stacked breakdown bars: correct / incorrect out of 19 per model.

Partials round down to incorrect. One row per model, sorted best-first.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

GREEN  = "#59A14F"   # correct
GREY   = "#D6D6D6"   # incorrect (incl. former partials)

INK    = "#1f1f1f"
MUTED  = "#6b6b6b"
GRID   = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# Partials round down to incorrect, so wrong = old_partial + old_wrong.
# (model, correct, incorrect) — sorted descending by correct.
rows = [
    ("RAVEN (Gemma 4 31B)", 15,  4),
    ("QQMM",                 7, 12),
    ("ViT-H-14",             6, 13),
    ("SigLIP",               6, 13),
    ("ViT-L-14",             5, 14),
]

TOTAL = 19

# Top-of-list = top of plot, so reverse for matplotlib (y=0 is bottom).
rows = list(reversed(rows))

names     = [r[0] for r in rows]
correct   = np.array([r[1] for r in rows])
incorrect = np.array([r[2] for r in rows])

y = np.arange(len(rows))
bar_h = 0.62

fig_w, fig_h = 11.0, 5.6
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.74, bottom=0.18, left=0.20, right=0.94)

ax.barh(y, correct, bar_h, color=GREEN, edgecolor="none",
        label="Correct", zorder=3)
ax.barh(y, incorrect, bar_h, left=correct, color=GREY,
        edgecolor="none", label="Incorrect", zorder=3)

# In-segment count labels
for yi, c in zip(y, correct):
    if c >= 1:
        ax.text(c/2, yi, str(c),
                ha="center", va="center",
                fontsize=14, color="white", fontweight="bold")
for yi, c, w in zip(y, correct, incorrect):
    if w >= 1:
        ax.text(c + w/2, yi, str(w),
                ha="center", va="center",
                fontsize=12, color=MUTED, fontweight="bold")

# Score at right end of each bar
for yi, c in zip(y, correct):
    ax.text(TOTAL + 0.4, yi, f"{c} / {TOTAL}",
            ha="left", va="center",
            fontsize=12, color=INK, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=12, color=INK)
ax.set_xlabel(f"Number of questions  (out of {TOTAL})",
              fontsize=12, color=INK, labelpad=10)
ax.set_xlim(0, TOTAL + 2.6)
ax.set_xticks([0, 5, 10, 15, 19])

ax.text(0, 1.30, "Quality of each model's answers",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.06,
        "Embedder-only systems plateau at 5–7 correct on the same\n"
        "19 questions; adding the RAVEN agent reaches 15.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="y", length=0, labelsize=11)
ax.tick_params(axis="x", length=4, labelsize=10, colors=MUTED)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20),
                frameon=False, fontsize=11, ncol=2,
                handlelength=1.2, handletextpad=0.6, columnspacing=2.0)
for txt in leg.get_texts():
    txt.set_color(INK)

out = OUT / "chart_breakdown.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
