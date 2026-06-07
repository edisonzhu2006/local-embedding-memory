"""Compact spec table: Model · Dim · Params · Year.

Reference grid for the poster — no accuracy column (the dim chart
already carries that). Compact, edge-to-edge, no margins.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK    = "#1f1f1f"
MUTED  = "#6b6b6b"
GRID   = "#E5E5E5"
HEAD   = "#F4F0E8"   # warm cream header
ACCENT = "#B85C2E"   # rust — agent row highlight

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

# Header + rows. Chronological order, agent at the bottom.
header = ["Model",            "Dim",   "Params",       "Year"]
rows = [
    ("ViT-L-14",              "768",   "304 M",        "2021", False),
    ("ViT-H-14",              "1024",  "632 M",        "2022", False),
    ("SigLIP-384",            "1152",  "400 M",        "2023", False),
    ("QQMM-v2",               "3584",  "7 B",          "2025", False),
    ("Gemma 4 + QQMM",        "3584",  "31 B + 7 B",   "2026", True),
]

n_rows = len(rows)
n_cols = len(header)

# Geometry (inches in figure coords)
fig_w     = 6.6
row_h     = 0.46
header_h  = 0.50
title_h   = 0.45
side_pad  = 0.05
top_pad   = 0.05
bot_pad   = 0.05
fig_h     = top_pad + title_h + header_h + n_rows * row_h + bot_pad

# Column widths (must sum to fig_w - 2*side_pad)
col_w_frac = [0.40, 0.18, 0.27, 0.15]
inner_w = fig_w - 2 * side_pad
col_widths = [inner_w * f for f in col_w_frac]
col_x = [side_pad]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# Title
ax.text(side_pad, fig_h - 0.18, "Models tested",
        ha="left", va="top",
        fontsize=14, fontweight="bold", color=INK)

# Header row background
header_y = fig_h - top_pad - title_h - header_h
ax.add_patch(mpatches.Rectangle(
    (side_pad, header_y), inner_w, header_h,
    linewidth=0, facecolor=HEAD, zorder=1,
))
# Header text
for cx, w, h in zip(col_x, col_widths, header):
    ax.text(cx + 0.18, header_y + header_h / 2, h,
            ha="left", va="center",
            fontsize=11, fontweight="bold", color=INK)

# Data rows
for i, (model, dim, params, year, highlight) in enumerate(rows):
    ry = header_y - (i + 1) * row_h
    if highlight:
        ax.add_patch(mpatches.Rectangle(
            (side_pad, ry), inner_w, row_h,
            linewidth=0, facecolor="#FBF1EC", zorder=1,
        ))
    # Bottom border for each row
    ax.plot([side_pad, side_pad + inner_w],
            [ry, ry],
            color=GRID, lw=0.7, zorder=2)

    cells = [model, dim, params, year]
    for cx, w, val, j in zip(col_x, col_widths, cells, range(n_cols)):
        is_model_col = (j == 0)
        color = ACCENT if (highlight and is_model_col) else INK
        weight = "bold" if (highlight and is_model_col) else "normal"
        ax.text(cx + 0.18, ry + row_h / 2, val,
                ha="left", va="center",
                fontsize=10.5, color=color, fontweight=weight)

# Top border under header
ax.plot([side_pad, side_pad + inner_w],
        [header_y, header_y],
        color=GRID, lw=0.9, zorder=2)
# Header bottom border
ax.plot([side_pad, side_pad + inner_w],
        [header_y + header_h, header_y + header_h],
        color=GRID, lw=0.9, zorder=2)

out = OUT / "chart_spec_table.png"
fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0,
            facecolor="white")
print(f"wrote {out}")
plt.close(fig)
