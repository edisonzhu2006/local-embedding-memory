"""Per-level accuracy across all four RAVEN agents + the QQMM baseline.

Grouped bar chart: 5 bars per level (QQMM-only, then 4 agents) showing
within-level accuracy.  Two model families (Gemma blue tones, Qwen
orange tones) so the reader can read family-vs-family at a glance.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# tab10-aligned palette
QQMM_COLOR  = "#7f7f7f"   # tab10 grey — embedder baseline
GEMMA_LIGHT = "#aec7e8"
GEMMA_DARK  = "#1f77b4"
QWEN_LIGHT  = "#ffbb78"
QWEN_DARK   = "#ff7f0e"

INK     = "#1f1f1f"
MUTED   = "#6b6b6b"
GRID    = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

levels  = ["L1\nDirect", "L2\nIndirect", "L3\nSpatial",
           "L4\nMulti-step", "L5\nNegation"]
denoms  = np.array([4, 3, 4, 4, 4])  # questions per level (19 total)

# Per-level "right" counts under the binary correct/wrong rubric of
# §3.5 (no partial credit).  Each row sums to that model's overall
# score reported in tab:dim and the §4.2 results table.
# Order: L1 (4) | L2 (3) | L3 (4) | L4 (4) | L5 (4)
qqmm_raw      = np.array([3.0, 2.0, 1.0, 0.0, 1.0])  # 7/19
gemma3_27_raw = np.array([4.0, 3.0, 2.0, 2.0, 2.0])  # 13/19
gemma4_31_raw = np.array([4.0, 3.0, 4.0, 2.0, 2.0])  # 15/19
qwen25_32_raw = np.array([4.0, 2.0, 4.0, 1.0, 2.0])  # 13/19
qwen3_32_raw  = np.array([4.0, 2.0, 3.0, 0.0, 3.0])  # 12/19

models = [
    ("QQMM",            qqmm_raw,      QQMM_COLOR),
    ("Gemma 3 27B",     gemma3_27_raw, GEMMA_LIGHT),
    ("Gemma 4 31B",     gemma4_31_raw, GEMMA_DARK),
    ("Qwen2.5-VL 32B",  qwen25_32_raw, QWEN_LIGHT),
    ("Qwen3-VL 32B",    qwen3_32_raw,  QWEN_DARK),
]

x = np.arange(len(levels))
n_models = len(models)
bar_w = 0.16
group_w = bar_w * n_models
offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_w

fig_w, fig_h = 13.0, 7.5
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.78, bottom=0.18, left=0.08, right=0.97)

for (name, raw, color), off in zip(models, offsets):
    pct = raw / denoms * 100
    ax.bar(x + off, pct, bar_w,
           color=color, edgecolor="none", zorder=3, label=name)

ax.set_xticks(x)
ax.set_xticklabels(levels, fontsize=12, color=INK, linespacing=1.4)
ax.set_ylabel("Accuracy within level (%)",
              fontsize=12, color=INK, labelpad=10)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])

ax.text(0, 1.18, "Per-level accuracy: where each agent earns its compute",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=18, fontweight="bold", color=INK)
ax.text(0, 1.08,
        "All four agents share the QQMM retriever.  Gemma 4 31B leads on multi-step (L4);\n"
        "Qwen3-VL beats every other agent on negation (L5) but collapses on L4.",
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
                frameon=False, fontsize=10.5, ncol=5,
                handlelength=1.2, handletextpad=0.6,
                columnspacing=1.6)
for txt in leg.get_texts():
    txt.set_color(INK)

out = OUT / "chart_lift.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
