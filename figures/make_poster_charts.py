"""Two poster charts from manual scoring of smoke_v2 (L2_4 excluded, denom=19).

Chart 1: horizontal bar — embedder ranking
Chart 2: dumbbell / slope — per-level lift QQMM → RAVEN
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Tableau 10 (academic, restrained)
T_BLUE   = "#4E79A7"
T_ORANGE = "#F28E2B"
T_RED    = "#E15759"
T_TEAL   = "#76B7B2"
T_GREEN  = "#59A14F"
T_PURPLE = "#B07AA1"

INK   = "#1f1f1f"
MUTED = "#6b6b6b"
GRID  = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# =============== Chart 1: HORIZONTAL bar — embedder ranking ===============
emb = [
    ("QQMM-v2",     8.0, T_ORANGE),   # winner highlighted
    ("ViT-H-14",    6.5, T_BLUE),
    ("SigLIP-384",  6.0, T_BLUE),
    ("ViT-L-14",    5.0, T_BLUE),
]
emb.sort(key=lambda r: r[1])  # ascending so bars stack with winner on top

names  = [e[0] for e in emb]
scores = [e[1] for e in emb]
colors = [e[2] for e in emb]
pcts   = [s / 19 * 100 for s in scores]
alphas = [1.0 if c == T_ORANGE else 0.55 for c in colors]

fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)

y = np.arange(len(names))
bars = ax.barh(y, pcts, color=colors, height=0.62,
               edgecolor="none", zorder=3)
for bar, a in zip(bars, alphas):
    bar.set_alpha(a)

for bar, raw, pct in zip(bars, scores, pcts):
    ax.text(pct + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%   ({raw:g} / 19)",
            va="center", ha="left",
            fontsize=12, color=INK, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=12, color=INK)
ax.set_xlim(0, 55)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel("Manual accuracy (%)", fontsize=11, color=MUTED, labelpad=10)

ax.set_title("Embedder-only retrieval on smoke_v2",
             fontsize=18, fontweight="bold", color=INK,
             pad=42, loc="left")
ax.text(0, 1.12,
        "QQMM-v2 outperforms larger and more recent open embedders.",
        transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=4, labelsize=10, colors=MUTED)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

fig.tight_layout()
out1 = OUT / "chart_embedder_comparison.png"
fig.savefig(out1, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out1}")
plt.close(fig)

# =============== Chart 2: GROUPED BARS — per-level QQMM vs RAVEN ===============
levels = ["L1\nDirect", "L2\nIndirect", "L3\nSpatial",
          "L4\nMulti-step", "L5\nNegation"]
denoms = np.array([4, 3, 4, 4, 4])

qqmm_raw  = np.array([3.5, 2.0, 1.0, 0.5, 1.0])
raven_raw = np.array([4.0, 3.0, 4.0, 2.0, 2.5])
qqmm_pct  = qqmm_raw  / denoms * 100
raven_pct = raven_raw / denoms * 100

x = np.arange(len(levels))
width = 0.36

fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200)
fig.subplots_adjust(top=0.78)

b1 = ax.bar(x - width/2, qqmm_pct,  width,
            label="QQMM (embedder-only) — 42.1%",
            color=T_BLUE, edgecolor="none", zorder=3)
b2 = ax.bar(x + width/2, raven_pct, width,
            label="RAVEN (Gemma 4 31B + QQMM) — 81.6%",
            color=T_ORANGE, edgecolor="none", zorder=3)

for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2.0,
                f"{bar.get_height():.0f}%",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=INK)

ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=11, color=INK)
ax.set_ylabel("Accuracy within level (%)", fontsize=12, color=INK, labelpad=10)
ax.set_ylim(0, 118)
ax.set_yticks([0, 25, 50, 75, 100])

ax.set_title("Where the LLM agent earns its cost",
             fontsize=18, fontweight="bold", color=INK,
             pad=58, loc="left")
ax.text(0, 1.10,
        "The agent recovers spatial, multi-step, and negation reasoning the embedder cannot.",
        transform=ax.transAxes, fontsize=11, color=MUTED, style="italic")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=11)
ax.tick_params(axis="y", length=4, labelsize=10, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

leg = ax.legend(loc="upper right", frameon=False, fontsize=11,
                handlelength=1.2, handletextpad=0.6, borderpad=0.6)
for txt in leg.get_texts():
    txt.set_color(INK)

out2 = OUT / "chart_per_level_qqmm_vs_raven.png"
fig.savefig(out2, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out2}")
plt.close(fig)
