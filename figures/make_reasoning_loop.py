import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.image import imread

ORANGE       = "#ff7f0e"   # tab10 orange
ORANGE_LIGHT = "#ffd0a3"
BLACK        = "#1A1A1A"
GREY         = "#505050"
BLUE_TINT    = "#d6e4f0"
BLUE_BORDER  = "#1f77b4"   # tab10 blue
GREEN_TINT   = "#d4e8d2"
GREEN_BORDER = "#2ca02c"   # tab10 green
WHITE        = "#FFFFFF"

FRAME_TOILET = "/Users/edison.zhu/local-embedding-memory/extracted_videos/smoke_full/frames/frame_000628.png"
FRAME_TV     = "/Users/edison.zhu/local-embedding-memory/extracted_videos/smoke_full/frames/frame_000753.png"

fig_w, fig_h = 12.0, 10.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

def card(x, y, w, h, fc, ec, lw=1.5, rs=0.012):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={rs}",
        linewidth=lw, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(p)

# ---- 1. Question banner (top) ----
q_y, q_h = 0.88, 0.10
card(0.02, q_y, 0.96, q_h, BLACK, BLACK, lw=0)
ax.text(0.5, q_y + q_h/2 + 0.015, "QUESTION",
        ha="center", va="center", fontsize=12, color=ORANGE_LIGHT,
        fontweight="bold")
ax.text(0.5, q_y + q_h/2 - 0.020,
        "“If I am at the toilet and hear the TV, which direction would I walk?”",
        ha="center", va="center", fontsize=15, color=WHITE,
        fontstyle="italic")

# Two arrows fanning out
ax.annotate("", xy=(0.27, 0.855), xytext=(0.45, 0.875),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))
ax.annotate("", xy=(0.73, 0.855), xytext=(0.55, 0.875),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))

# ---- 2. Two retrieval cards side-by-side ----
def retrieval_card(cx, cy, cw, ch, label_num, query, frame_path, pos_label):
    card(cx, cy, cw, ch, BLUE_TINT, BLUE_BORDER, lw=2)
    ax.text(cx + 0.012, cy + ch - 0.022,
            f"STEP {label_num}  ·  text retrieval",
            ha="left", va="top", fontsize=11, color=BLUE_BORDER, fontweight="bold")
    ax.text(cx + 0.012, cy + ch - 0.055,
            f'retrieve_from_text({query!r})',
            ha="left", va="top", fontsize=12, color=BLACK,
            family="monospace")
    img_pad = 0.014
    img_x0 = cx + img_pad
    img_x1 = cx + cw - img_pad
    img_y1 = cy + ch - 0.090
    img_y0 = cy + 0.054
    try:
        frame = imread(frame_path)
        ax.imshow(frame, extent=[img_x0, img_x1, img_y0, img_y1],
                  aspect="auto", zorder=2)
    except Exception:
        pass
    ax.text(cx + cw/2, cy + 0.025, pos_label,
            ha="center", va="center", fontsize=11, color=BLACK,
            fontweight="bold", family="monospace")

r_y, r_h = 0.48, 0.36
retrieval_card(0.05, r_y, 0.42, r_h, 1, "toilet", FRAME_TOILET,
               "Toilet  →  [-0.23, -2.71, 0.44]")
retrieval_card(0.53, r_y, 0.42, r_h, 2, "TV", FRAME_TV,
               "TV  →  [1.49, 3.48, 0.44]")

# ---- 3. Merge arrows into reasoning card ----
ax.annotate("", xy=(0.5, 0.43), xytext=(0.26, 0.475),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
ax.annotate("", xy=(0.5, 0.43), xytext=(0.74, 0.475),
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.2))
ax.text(0.5, 0.456, "agent composes",
        ha="center", va="center", fontsize=11, color=ORANGE,
        fontweight="bold", fontstyle="italic")

# ---- 4. Reasoning card (separate) ----
syn_y, syn_h = 0.24, 0.14
card(0.02, syn_y, 0.96, syn_h, ORANGE_LIGHT, ORANGE, lw=2)
ax.text(0.04, syn_y + syn_h - 0.022, "REASONING",
        ha="left", va="top", fontsize=11, color=BLACK, fontweight="bold")
ax.text(0.5, syn_y + syn_h/2 - 0.000,
        "From toilet (-0.23, -2.71)  →  toward TV (+1.49, +3.48)",
        ha="center", va="center", fontsize=14, color=BLACK)
ax.text(0.5, syn_y + 0.025,
        "Walk in the +X, +Y direction (north-east).",
        ha="center", va="center", fontsize=12, color=BLACK,
        fontstyle="italic")

# Arrow to answer
ax.annotate("", xy=(0.5, 0.215), xytext=(0.5, 0.235),
            arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2))

# ---- 5. Answer card (separate) ----
a_y, a_h = 0.04, 0.17
card(0.02, a_y, 0.96, a_h, GREEN_TINT, GREEN_BORDER, lw=2.5)
ax.text(0.04, a_y + a_h - 0.022, "ANSWER",
        ha="left", va="top", fontsize=11, color=GREEN_BORDER, fontweight="bold")
ax.text(0.5, a_y + a_h/2 + 0.012,
        "Position: [1.494, 3.479, 0.44]",
        ha="center", va="center", fontsize=17, color=BLACK, fontweight="bold")
ax.text(0.5, a_y + a_h/2 - 0.025,
        "✓  Matches ground truth — agent composed two retrievals.",
        ha="center", va="center", fontsize=12, color=GREEN_BORDER,
        fontweight="bold")

plt.savefig(
    "/Users/edison.zhu/local-embedding-memory/figures/reasoning_loop.png",
    dpi=300, bbox_inches="tight", pad_inches=0.08, facecolor="white",
)
print("saved reasoning_loop.png")
