"""Tall scorecard chart: 19 questions × 5 models, color-coded by manual score.

Naturally portrait orientation (questions stacked vertically).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Tableau-ish palette
GREEN  = "#59A14F"   # correct
YELLOW = "#F1CE63"   # partial
GREY   = "#E5E5E5"   # wrong

INK    = "#1f1f1f"
MUTED  = "#6b6b6b"
DIVIDER = "#bbbbbb"

# Per-question scores. Rows = questions, columns = models.
# 1.0 = correct, 0.5 = partial, 0.0 = wrong.
# L2_4 (brightest area) is omitted from the suite.
questions = [
    ("L1_1", "purple chair"),
    ("L1_2", "toilet"),
    ("L1_3", "white sofa"),
    ("L1_4", "kitchen cabinets"),
    ("L2_1", "sit comfortably"),
    ("L2_2", "wash hands"),
    ("L2_3", "watch the news"),
    ("L3_1", "closest to entry"),
    ("L3_2", "kitchen to bathroom"),
    ("L3_3", "most furniture"),
    ("L3_4", "seating facing TV"),
    ("L4_1", "coffee then sit"),
    ("L4_2", "spill cleanup"),
    ("L4_3", "where agent started"),
    ("L4_4", "toilet to TV direction"),
    ("L5_1", "chair NOT in kitchen"),
    ("L5_2", "seating closest to bath"),
    ("L5_3", "flat surface NOT floor"),
    ("L5_4", "most time spent"),
]

models = [
    "QQMM",
    "ViT-H-14",
    "SigLIP",
    "ViT-L-14",
    "Gemma 3\n27B",
    "Gemma 4\n31B",
    "Qwen2.5-VL\n32B",
    "Qwen3-VL\n32B",
]

# Score matrix [question_idx][model_idx]
# Embedder columns (cols 0-3): from per-question retrieval grading (existing).
# Agent columns (cols 4-7): from outputs/smoke_v2/new_raven_results/*_grades.md
#                           and 20q/gemma4_31b_qqmm/GEMMA4_31B_FINDINGS.md.
# 1.0 = right, 0.5 = partial, 0.0 = wrong / empty.
scores = [
    # QQMM ViT-H SigLIP ViT-L | G3-27B G4-31B Q2.5-32B Q3-32B
    [1.0, 1.0, 1.0, 1.0,        1.0, 1.0, 1.0, 1.0],   # L1_1
    [1.0, 1.0, 1.0, 1.0,        1.0, 1.0, 1.0, 1.0],   # L1_2
    [1.0, 1.0, 1.0, 0.0,        1.0, 1.0, 1.0, 1.0],   # L1_3
    [0.5, 1.0, 1.0, 1.0,        1.0, 1.0, 1.0, 1.0],   # L1_4
    [1.0, 0.5, 0.5, 1.0,        1.0, 1.0, 1.0, 1.0],   # L2_1
    [1.0, 0.0, 1.0, 0.0,        1.0, 1.0, 1.0, 1.0],   # L2_2
    [0.0, 0.0, 0.0, 0.0,        1.0, 1.0, 0.0, 0.0],   # L2_3
    [0.0, 0.0, 0.0, 0.0,        0.0, 1.0, 1.0, 1.0],   # L3_1
    [0.0, 0.0, 0.0, 0.0,        1.0, 1.0, 1.0, 0.0],   # L3_2
    [1.0, 0.0, 0.5, 0.0,        1.0, 1.0, 1.0, 1.0],   # L3_3
    [0.0, 0.0, 0.0, 1.0,        0.0, 1.0, 1.0, 1.0],   # L3_4
    [0.0, 1.0, 0.0, 0.0,        1.0, 0.0, 0.0, 0.0],   # L4_1
    [0.5, 0.0, 0.0, 0.0,        0.0, 1.0, 1.0, 0.0],   # L4_2
    [0.0, 0.0, 0.0, 0.0,        0.0, 1.0, 0.0, 0.0],   # L4_3
    [0.0, 0.0, 0.0, 0.0,        1.0, 0.0, 0.0, 0.0],   # L4_4
    [1.0, 0.0, 0.0, 0.0,        1.0, 1.0, 1.0, 1.0],   # L5_1
    [0.0, 0.0, 0.0, 0.0,        0.0, 0.0, 1.0, 1.0],   # L5_2
    [0.0, 0.0, 0.0, 0.0,        1.0, 0.0, 0.0, 0.0],   # L5_3
    [0.0, 0.0, 0.0, 0.0,        0.0, 1.0, 0.0, 1.0],   # L5_4
]

def color_for(score):
    if score == 1.0: return GREEN
    if score == 0.5: return YELLOW
    return GREY

n_q = len(questions)
n_m = len(models)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

fig_w, fig_h = 11.5, 13.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=0.90, bottom=0.05, left=0.36, right=0.96)

cell_w, cell_h = 1.0, 1.0
gap = 0.18

for qi, ((qid, qtext), row) in enumerate(zip(questions, scores)):
    y = (n_q - 1 - qi) * (cell_h + gap)
    for mi, s in enumerate(row):
        x = mi * (cell_w + gap)
        rect = mpatches.FancyBboxPatch(
            (x, y), cell_w, cell_h,
            boxstyle="round,pad=0,rounding_size=0.10",
            linewidth=0, facecolor=color_for(s), zorder=2,
        )
        ax.add_patch(rect)
        if s == 0.5:
            ax.text(x + cell_w/2, y + cell_h/2, "½",
                    ha="center", va="center",
                    fontsize=14, color=INK, fontweight="bold")

# Question labels (left side) — text on the far left, then ID, then cells
for qi, (qid, qtext) in enumerate(questions):
    y = (n_q - 1 - qi) * (cell_h + gap) + cell_h/2
    excluded = qid == "L2_4"
    label_color = "#bbb" if excluded else INK
    sub_color   = "#ccc" if excluded else MUTED
    # ID column
    ax.text(-0.5, y, qid,
            ha="right", va="center",
            fontsize=10.5, color=label_color, fontweight="bold")
    # description column (further left)
    ax.text(-1.7, y, qtext,
            ha="right", va="center",
            fontsize=10, color=sub_color)

# Model labels (top) — rotated 35° so they don't crash into each other
header_y = n_q * (cell_h + gap) + 0.25
for mi, m in enumerate(models):
    x = mi * (cell_w + gap) + cell_w/2
    ax.text(x, header_y, m.replace("\n", " "),
            ha="left", va="bottom",
            rotation=35, rotation_mode="anchor",
            fontsize=10.5, color=INK, fontweight="bold")

# Level dividers between L1/L2/L3/L4/L5
level_breaks = [4, 8, 12, 16]  # indices where a new level begins
for br in level_breaks:
    y_line = (n_q - br) * (cell_h + gap) - gap/2
    ax.axhline(y_line, color=DIVIDER, lw=0.7, ls="-",
               xmin=0, xmax=1, zorder=1)

# Title
fig.suptitle("Per-question scorecard",
             x=0.0, y=0.965, ha="left",
             fontsize=18, fontweight="bold", color=INK)
fig.text(0.0, 0.928,
         "19 questions (L2_4 excluded)  ·  Green = correct, ½ = partial, grey = wrong",
         ha="left", fontsize=10, color=MUTED, style="italic")

# Legend
legend_elements = [
    mpatches.Patch(facecolor=GREEN,  label="Correct (1.0)"),
    mpatches.Patch(facecolor=YELLOW, label="Partial (0.5)"),
    mpatches.Patch(facecolor=GREY,   label="Wrong (0.0)"),
]
leg = ax.legend(handles=legend_elements,
                loc="upper center", bbox_to_anchor=(0.5, -0.025),
                ncol=3, frameon=False, fontsize=10,
                handlelength=1.2, handletextpad=0.6, columnspacing=2.2)
for txt in leg.get_texts():
    txt.set_color(INK)

# Layout
total_w = n_m * cell_w + (n_m - 1) * gap
total_h = n_q * cell_h + (n_q - 1) * gap
ax.set_xlim(-8.2, total_w + 1.6)
ax.set_ylim(-0.8, total_h + 3.2)
ax.set_aspect("equal")
ax.axis("off")

out = OUT / "chart_scorecard.png"
fig.savefig(out, dpi=240, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
plt.close(fig)
