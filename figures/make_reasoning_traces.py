"""Reasoning traces: stacked card layout of successful + failed agent runs.

Each card is one example: question · search query · retrieval · why.
Successes get a teal accent stripe; failures get a copper one.
Tall portrait orientation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent

OK_COLOR  = "#2D7A5F"   # success accent
ERR_COLOR = "#B85C2E"   # failure accent
CARD_BG   = "#FAF8F4"   # card background
INK       = "#1f1f1f"
MUTED     = "#6b6b6b"
GRID      = "#E5E5E5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
})

# Each entry: (status, level, question, search/approach, retrieval, why)
# Seven traces spanning every level, with both successes and failures.
traces = [
    ("ok",  "L1_1",
     "Where is the purple chair?",
     "search: \"purple chair\"",
     "frame_238  (top-1 result, cosine 0.41)",
     "Direct keyword match - the easy half of the suite that the\nembedder solves before the VLM even reasons."),

    ("ok",  "L3_2",
     "Where would I go to get from the kitchen to the bathroom?",
     "search: \"kitchen\"  ->  search: \"bathroom\"",
     "two retrievals, agent emits bathroom pose",
     "Compositional reasoning across two retrievals - the agent\nlocalizes both endpoints, then names the destination."),

    ("ok",  "L4_2",
     "If I spill water on the kitchen floor, where would I find\nsomething to clean it up?",
     "search: \"cleaning supplies\"",
     "mop at frame_226  (top-2 result)",
     "Concept-to-object retrieval - mapping \"clean a spill\" to a\nphysical tool that QQMM surfaces."),

    ("ok",  "L4_3",
     "Where did the agent start its exploration?",
     "search: \"start of exploration\"  (failed)\nfallback: retrieve_from_time(\"00:00:00\")",
     "frame_000  (very first frame)",
     "Only query in the suite that triggered a non-text retriever.\nThe agent recognized text-search failure and switched tools."),

    ("err", "L3_1",
     "What room is closest to the entry?",
     "search: \"entryway\"  x6  (same query repeated)",
     "same top-K returned each call, no coordinate emitted",
     "Retrieval loop on out-of-distribution concept - no backoff,\nno query diversification, no tool-pivot."),

    ("err", "L5_3",
     "Find a flat surface that is NOT a floor.",
     "search: \"marble countertop\"",
     "marble floor at frame_343  (misidentified)",
     "Texture-surface confusion - the embedder produces the same\nsignal for marble floors and marble countertops."),

    ("err", "L4_4",
     "If I'm at the toilet and hear the TV, which direction\nwould I walk?",
     "search: \"toilet\"  ->  search: \"TV\"",
     "endpoints retrieved, direction never computed",
     "Retrieval succeeds but the geometric reasoning step is missing -\nthe prompt doesn't prime the agent to compute a direction."),
]

n = len(traces)
fig_w = 7.4
card_h = 1.55
gap    = 0.20
top_pad = 0.45
bot_pad = 0.05
fig_h  = top_pad + bot_pad + n * card_h + (n - 1) * gap

fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
fig.subplots_adjust(top=1.0, bottom=0.0, left=0.0, right=1.0)
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis("off")

# Small title (top-left) — same format as other chart titles
ax.text(0.05, fig_h - 0.18, "How the VLM agent reasons",
        ha="left", va="top",
        fontsize=14, fontweight="bold", color=INK)

# Cards
for i, (status, level, q, search, ret, why) in enumerate(traces):
    accent = OK_COLOR if status == "ok" else ERR_COLOR
    mark   = "✓" if status == "ok" else "✗"

    y_top    = fig_h - top_pad - i * (card_h + gap)
    y_bottom = y_top - card_h

    # Card background
    bg = mpatches.FancyBboxPatch(
        (0.05, y_bottom), fig_w - 0.10, card_h,
        boxstyle="round,pad=0.0,rounding_size=0.10",
        linewidth=0, facecolor=CARD_BG, zorder=1,
    )
    ax.add_patch(bg)
    # Accent stripe (left edge)
    stripe = mpatches.Rectangle(
        (0.05, y_bottom), 0.10, card_h,
        linewidth=0, facecolor=accent, zorder=2,
    )
    ax.add_patch(stripe)

    # Status mark + level + question
    ax.text(0.37, y_top - 0.18, mark,
            ha="left", va="top",
            fontsize=14, color=accent, fontweight="bold",
            family="DejaVu Sans")
    ax.text(0.70, y_top - 0.18, level,
            ha="left", va="top",
            fontsize=10, color=MUTED, fontweight="bold",
            family="monospace")
    ax.text(1.20, y_top - 0.18, q,
            ha="left", va="top",
            fontsize=10.5, color=INK, fontweight="bold",
            linespacing=1.30)

    # Body: search · retrieval · why
    body_top = y_top - 0.62
    ax.text(0.37, body_top,
            f"Approach:   {search}",
            ha="left", va="top",
            fontsize=9, color=INK, family="monospace",
            linespacing=1.30)
    ax.text(0.37, body_top - 0.32,
            f"Retrieved:  {ret}",
            ha="left", va="top",
            fontsize=9, color=INK, family="monospace",
            linespacing=1.30)
    ax.text(0.37, body_top - 0.60, why,
            ha="left", va="top",
            fontsize=8.5, color=MUTED, style="italic",
            linespacing=1.30)

out = OUT / "chart_reasoning_traces.png"
fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0,
            facecolor="white")
print(f"wrote {out}")
plt.close(fig)
