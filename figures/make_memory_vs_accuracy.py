"""Resident VRAM vs. strict retrieval accuracy across 4 open VLMs.
Shows that bigger models do not buy accuracy on this suite."""
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#E5E5E5"
HEADLINE = "#B85C2E"
DEFAULT = "#7DA3B4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
})

# (model, vram_gb, strict_correct_19, headline)
# Runtime resident VRAM, q4_K_M / Q4_0 builds via Ollama on a 48 GB L40.
points = [
    ("Gemma 4 31B",    20.0, 15, True),
    ("Gemma 3 27B",    16.0, 13, False),
    ("Qwen2.5-VL 32B", 18.8, 13, False),
    ("Qwen3-VL 32B",   21.0, 12, False),
]

fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.14, left=0.13, right=0.96)

# (model, vram, score, headline, label_offset_x, label_offset_y, halign)
positioned = [
    ("Gemma 4 31B",    20.0, 15, True,   0.4,  0.45, "left"),
    ("Gemma 3 27B",    16.0, 13, False,  0.4,  0.45, "left"),
    ("Qwen2.5-VL 32B", 18.8, 13, False,  0.4,  0.45, "left"),
    ("Qwen3-VL 32B",   21.0, 12, False,  0.4,  0.45, "left"),
]
for name, vram, score, headline, dx, dy, ha in positioned:
    color = HEADLINE if headline else DEFAULT
    size = 220 if headline else 140
    ax.scatter(vram, score, s=size, color=color, edgecolor="white",
               linewidth=1.5, zorder=3)
    weight = "bold" if headline else "normal"
    ax.annotate(name, xy=(vram, score), xytext=(vram + dx, score + dy),
                fontsize=10, color=INK, fontweight=weight, ha=ha,
                va="bottom")
    ax.annotate(f"{score}/19", xy=(vram, score),
                xytext=(vram + dx, score + dy - 0.30),
                fontsize=9, color=MUTED, ha=ha, va="top")

ax.set_xlim(14, 24)
ax.set_ylim(10, 17)
ax.set_xlabel("Approx. VRAM (GB)", fontsize=11, color=INK, labelpad=8)
ax.set_ylabel("Retrieval-level accuracy (out of 19)",
              fontsize=11, color=INK, labelpad=8)

ax.text(0, 1.18, "Memory footprint vs. accuracy  (4 open VLMs, same retriever)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=13, fontweight="bold", color=INK)
ax.text(0, 1.14,
        "All four backbones sit in a 16-22 GB band; accuracy ranges\n"
        "from 12 to 15 of 19 with no correlation to memory footprint.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=4, labelsize=9, colors=MUTED)
ax.tick_params(axis="y", length=4, labelsize=9, colors=MUTED)
ax.grid(True, color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)

out = OUT / "chart_memory_vs_accuracy.png"
fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
print(f"wrote {out}")
plt.close(fig)
