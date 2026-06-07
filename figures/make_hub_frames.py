"""Hub-frame network diagram: which questions each landmark frame answers.

Replaces the previous bar chart with a bipartite hub-and-spoke layout that
visualises the "landmark map" the agent maintains: three hub frames in
the centre column, each connected to the four distinct questions it served.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent

# tab10-aligned palette, one color per hub
HUB_COLORS = {
    "tv":     "#1f77b4",  # tab10 blue
    "chair":  "#ff7f0e",  # tab10 orange
    "toilet": "#2ca02c",  # tab10 green
}
INK   = "#1f1f1f"
MUTED = "#6b6b6b"
GRID  = "#E5E5E5"
QUERY_BG = "#FAF8F4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

# Each hub: (key, frame_id, label, [questions])
hubs = [
    ("tv", "frame_000753", "TV living room",
     ["L2_3  watch the news",
      "L2_4  brightest area",
      "L3_3  seating that faces TV",
      "L3_4  toilet→TV direction"]),
    ("chair", "frame_000238", "red/purple chair",
     ["L1_1  purple chair",
      "L2_1  sit comfortably",
      "L5_1  chair NOT in kitchen",
      "L5_2  seating closest to bath"]),
    ("toilet", "frame_000628", "toilet",
     ["L1_2  toilet location",
      "L2_2  wash hands",
      "L3_2  kitchen → bathroom",
      "L4_4  agent-time question"]),
]

fig_w, fig_h = 11.0, 12.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.88, bottom=0.04, left=0.02, right=0.98)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title block (top-left)
ax.text(0.02, 0.96, "Landmark frames the agent re-uses",
        ha="left", va="top",
        fontsize=20, fontweight="bold", color=INK)
ax.text(0.02, 0.925,
        "Three frames, each retrieved across four distinct questions —\n"
        "the agent maintains an implicit landmark map.",
        ha="left", va="top",
        fontsize=12, color=MUTED, style="italic", linespacing=1.45)

# Layout: hub column at x=0.30, query column at x=0.78
# Three vertical bands, one per hub.
n_hubs = len(hubs)
band_top = 0.83
band_bot = 0.05
band_h = (band_top - band_bot) / n_hubs
hub_x = 0.30
query_x = 0.62
hub_r = 0.055   # radius (in axes coords) of the hub circle

for i, (key, fid, label, queries) in enumerate(hubs):
    color = HUB_COLORS[key]
    band_center = band_top - band_h * (i + 0.5)

    # Hub circle (filled, with white inner ring for label legibility)
    ax.add_patch(mpatches.Circle((hub_x, band_center), hub_r,
                                 facecolor=color, edgecolor="none",
                                 zorder=4))
    ax.add_patch(mpatches.Circle((hub_x, band_center), hub_r * 0.78,
                                 facecolor="white", edgecolor=color,
                                 lw=1.6, zorder=5))
    # Hub label inside circle
    ax.text(hub_x, band_center + 0.012, fid,
            ha="center", va="center",
            fontsize=9.5, color=INK, fontweight="bold",
            family="monospace", zorder=6)
    ax.text(hub_x, band_center - 0.018, label,
            ha="center", va="center",
            fontsize=8.5, color=color, fontweight="bold",
            zorder=6)

    # Count badge
    ax.text(hub_x - hub_r - 0.018, band_center, "4",
            ha="right", va="center",
            fontsize=22, color=color, fontweight="bold")
    ax.text(hub_x - hub_r - 0.018, band_center - 0.030,
            "questions",
            ha="right", va="center",
            fontsize=8, color=MUTED, style="italic")

    # Query nodes + edges
    n_q = len(queries)
    q_band_top = band_center + band_h * 0.40
    q_band_bot = band_center - band_h * 0.40
    q_ys = np.linspace(q_band_top, q_band_bot, n_q)

    for q, qy in zip(queries, q_ys):
        # Edge from hub to query
        edge_xs = [hub_x + hub_r, query_x - 0.005]
        edge_ys = [band_center, qy]
        # Curved line via simple cubic-like interpolation
        cx_mid = (edge_xs[0] + edge_xs[1]) / 2
        ax.plot(
            [edge_xs[0], cx_mid, edge_xs[1]],
            [edge_ys[0], qy, edge_ys[1]],
            color=color, lw=1.4, alpha=0.55, zorder=2,
            solid_capstyle="round",
        )

        # Query pill
        pill_w = 0.34
        pill_h = 0.044
        pill = mpatches.FancyBboxPatch(
            (query_x, qy - pill_h / 2), pill_w, pill_h,
            boxstyle="round,pad=0.0,rounding_size=0.012",
            linewidth=1.2, edgecolor=color, facecolor=QUERY_BG, zorder=3,
        )
        ax.add_patch(pill)
        ax.text(query_x + 0.012, qy, q,
                ha="left", va="center",
                fontsize=9.5, color=INK, family="monospace", zorder=4)

# Column headers
ax.text(hub_x, band_top + 0.020, "Hub frame",
        ha="center", va="bottom",
        fontsize=11, color=INK, fontweight="bold")
ax.text(query_x + 0.17, band_top + 0.020, "Questions answered",
        ha="center", va="bottom",
        fontsize=11, color=INK, fontweight="bold")

out = OUT / "chart_hub_frames.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
