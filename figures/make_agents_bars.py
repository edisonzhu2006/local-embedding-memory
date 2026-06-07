"""Cross-family agent comparison: 4 RAVEN agents vs the best embedder.

Single bar chart with 4 agents (Gemma 3 27B, Gemma 4 31B, Qwen2.5-VL 32B,
Qwen3-VL 32B) plus the best embedder-only baseline (QQMM 7/19) as a
reference line.  All four agents use the same QQMM retriever underneath,
so this isolates the contribution of the VLM choice.
"""

import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK   = "#1f1f1f"
MUTED = "#6b6b6b"
GRID  = "#E5E5E5"

# tab10 family colors, two per family
GEMMA_LIGHT = "#aec7e8"   # tab10 light blue
GEMMA_DARK  = "#1f77b4"   # tab10 blue
QWEN_LIGHT  = "#ffbb78"   # tab10 light orange
QWEN_DARK   = "#ff7f0e"   # tab10 orange

BASELINE = "#7f7f7f"      # tab10 grey

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

agents = [
    ("Gemma 3 27B",     13, GEMMA_LIGHT),
    ("Gemma 4 31B",     15, GEMMA_DARK),
    ("Qwen2.5-VL 32B",  13, QWEN_LIGHT),
    ("Qwen3-VL 32B",    12, QWEN_DARK),
]
N = 19
baseline_score = 7   # QQMM embedder-only

fig_w, fig_h = 9.0, 7.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.10, right=0.96)

xs = list(range(len(agents)))
heights = [a[1] for a in agents]
colors  = [a[2] for a in agents]
labels  = [a[0] for a in agents]

bars = ax.bar(xs, heights, width=0.62, color=colors, edgecolor="none", zorder=3)

# % labels above each bar
for x, h in zip(xs, heights):
    ax.text(x, h + 0.35, f"{h}/{N}\n{int(round(100*h/N))}%",
            ha="center", va="bottom",
            fontsize=11, color=INK, fontweight="bold",
            linespacing=1.20)

# Embedder-only reference line (QQMM = 7/19)
ax.axhline(baseline_score, color=BASELINE, lw=1.4, ls="--", zorder=2)
ax.text(len(agents) - 0.5, baseline_score + 0.18,
        f"QQMM embedder-only baseline ({baseline_score}/{N})",
        ha="right", va="bottom",
        fontsize=10, color=BASELINE, style="italic")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=11, color=INK)
ax.set_ylabel(f"Questions correct (out of {N})", fontsize=12, color=INK, labelpad=10)
ax.set_ylim(0, N + 1.5)
ax.set_yticks([0, 5, 10, 15, N])

ax.text(0, 1.13,
        "All four agents beat the embedder, none beat 80%",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.06,
        "Cross-family RAVEN+QQMM scores, manually graded on the 19-question suite.\n"
        "All four agents share the same retriever — only the VLM differs.",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=10.5, color=MUTED, style="italic", linespacing=1.45)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=4, labelsize=10, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)

out = OUT / "chart_agents_bars.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
