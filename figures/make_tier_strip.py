import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

rows = [
    ("L1 Direct",     "\"Where is the purple chair?\"",       "#2ca02c"),  # green — easy
    ("L2 Indirect",   "\"Where can I sit comfortably?\"",     "#bcbd22"),  # olive
    ("L3 Spatial",    "\"Closest room to the entry?\"",       "#ff7f0e"),  # orange
    ("L4 Multi-step", "\"Where did the agent start?\"",       "#d62728"),  # red — hard
    ("L5 Negation",   "\"Chair NOT in the kitchen?\"",        "#9467bd"),  # purple — distinct
]

fig_w, fig_h = 10.0, 6.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

n = len(rows)
gap = 0.012
row_h = (1.0 - gap * (n - 1)) / n

for i, (label, example, color) in enumerate(rows):
    y = 1.0 - (i + 1) * row_h - i * gap
    box = FancyBboxPatch(
        (0.0, y), 1.0, row_h,
        boxstyle="round,pad=0,rounding_size=0.018",
        linewidth=0,
        facecolor=color,
    )
    ax.add_patch(box)
    cy = y + row_h / 2
    ax.text(
        0.035, cy, label,
        ha="left", va="center",
        fontsize=22, fontweight="bold", color="white",
    )
    ax.text(
        0.38, cy, example,
        ha="left", va="center",
        fontsize=20, color="white",
    )

plt.savefig(
    "/Users/edison.zhu/local-embedding-memory/figures/tier_strip.png",
    dpi=300, bbox_inches="tight", pad_inches=0.05, facecolor="white",
)
print("saved tier_strip.png")
