"""Embedding dimension vs retrieval-level accuracy — line chart.

Five models ordered by retriever embedding dim. Single line shows that
accuracy rises only slightly across embedders, then jumps when the
VLM agent is added.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

EMB_COLOR   = "#A6A19A"   # embedder portion — warm taupe-grey (recedes)
AGENT_COLOR = "#B85C2E"   # agent jump — deep rust/terracotta (highlight)
CEIL_FILL   = "#FAF6EE"   # very light cream — embedder ceiling band
CEIL_EDGE   = "#EAE3D5"
CEIL_TEXT   = "#8C7656"

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

# Models ordered by retriever embedding dim (RAVEN reuses QQMM at the end).
labels = ["ViT-L-14", "ViT-H-14", "SigLIP-384", "QQMM-v2", "Gemma 4 + QQMM"]
dims   = ["768",      "1024",     "1152",       "3584",    "+ VLM agent"]
counts = [5,           6,          6,            7,         15]
accs   = [c / 19 * 100 for c in counts]

x = np.arange(len(labels))

fig_w, fig_h = 8.0, 8.0   # 1:1 square
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.12, left=0.14, right=0.94)

# Embedder ceiling band (max strict acc among the four embedders ≈ 36.8%)
ceiling = max(accs[:4])
ax.axhspan(0, ceiling, facecolor=CEIL_FILL, edgecolor=CEIL_EDGE,
           linewidth=0.8, zorder=0)
ax.text(-0.35, 2, "Embedder ceiling",
        ha="left", va="bottom",
        fontsize=10, color=CEIL_TEXT, style="italic", fontweight="bold")

# Line: embedder portion (light slate) + agent jump (deep teal)
ax.plot(x[:4], accs[:4], color=EMB_COLOR, lw=3.0, zorder=3)
ax.plot(x[3:], accs[3:], color=AGENT_COLOR, lw=3.0, zorder=3)

# Per-point labels (model name + accuracy stacked above/below to avoid the line)
def place(i, side):
    xi, yi = x[i], accs[i]
    if side == "above":
        ax.text(xi, yi + 5.0, labels[i],
                ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text(xi, yi + 1.5, f"{accs[i]:.0f}%",
                ha="center", va="bottom",
                fontsize=10, color=MUTED)
    else:  # below
        ax.text(xi, yi - 5.0, labels[i],
                ha="center", va="top",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text(xi, yi - 1.5, f"{accs[i]:.0f}%",
                ha="center", va="top",
                fontsize=10, color=MUTED)

place(0, "below")
place(1, "above")
place(2, "below")
place(3, "below")
# Agent endpoint — call out in agent color
ax.text(x[4], accs[4] + 5.5, "Gemma 4 + QQMM",
        ha="center", va="bottom",
        fontsize=12, color=AGENT_COLOR, fontweight="bold")
ax.text(x[4], accs[4] + 1.5, f"{accs[4]:.0f}%",
        ha="center", va="bottom",
        fontsize=10.5, color=AGENT_COLOR, fontweight="bold")

# X-ticks: dim under each
ax.set_xticks(x)
ax.set_xticklabels(dims, fontsize=10.5, color=INK)
ax.set_xlim(-0.4, len(labels) - 0.6)
ax.set_xlabel("Retriever embedding dimension",
              fontsize=12, color=INK, labelpad=10)

ax.set_ylim(0, 100)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Retrieval-level accuracy  (% of 19 questions)",
              fontsize=12, color=INK, labelpad=10)

# Title + subtitle
ax.text(0, 1.10, "Bigger embeddings don't crack the ceiling",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.03,
        "Retrieval-level accuracy creeps up across embedders sized\n"
        "768-3584, then jumps when the VLM agent is added.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=10.5)
ax.tick_params(axis="y", length=4, labelsize=10, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

out = OUT / "chart_dim_vs_acc.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
