"""Latency + tool-call counts per question, parsed from the existing
Gemma 4 31B + QQMM run log.  Two compact bar charts, vertical layout.
"""

import json, re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUN = ('outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/'
       'remembr+gemma4:31b+hf+QQMM-embed-v2+0+3584__'
       'frames__test_questions_smoke_qa_0.json')

INK = "#1f1f1f"
MUTED = "#6b6b6b"
GRID = "#E5E5E5"
LEVEL_COLORS = {
    'L1': "#FCE4C9",
    'L2': "#F2BE85",
    'L3': "#E89854",
    'L4': "#C76E2C",
    'L5': "#8E4A1C",
}
HIGHLIGHT = "#B85C2E"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
})

# ----- Parse the run log -----
data = json.load(open(RUN))
responses = data['responses']

# Drop L2_4 to match the 19-question suite as reported
keep_ids = [r['id'] for r in responses if r['id'] != 'L2_4']
order = sorted(keep_ids,
               key=lambda x: (int(x[1]), int(x.split('_')[1])))

lat = {r['id']: r['elapsed'] for r in responses}
qtexts = {r['id']: r['question'] for r in responses}

# Tool-call counts via the same chunking strategy
qids_in_order = [r['id'] for r in responses]
logs = data['debug_logs']
current_qid = None
qid_to_logs = {q: [] for q in qids_in_order}
for line in logs:
    s = str(line)
    for qid in qids_in_order:
        if qtexts[qid] in s and 'content=' in s:
            current_qid = qid
            break
    if current_qid:
        qid_to_logs[current_qid].append(s)

tool_kw = {
    'text': ['retrieve_from_text', 'text_search'],
    'time': ['retrieve_from_time', 'time_search'],
    'image': ['retrieve_from_image', 'image_search'],
    'pos':  ['retrieve_from_position', 'position_search'],
}

def count_calls(qid):
    counts = {k: 0 for k in tool_kw}
    for line in qid_to_logs[qid]:
        for k, kws in tool_kw.items():
            for kw in kws:
                counts[k] += line.count(kw)
    return counts

calls = {qid: count_calls(qid) for qid in order}

# ----- Figure 1: tool-call count per question (stacked bar) -----
fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.20, left=0.10, right=0.97)

x = np.arange(len(order))
text_calls = np.array([calls[q]['text'] for q in order])
time_calls = np.array([calls[q]['time'] for q in order])

bar_w = 0.72
ax.bar(x, text_calls, bar_w, color="#7DA3B4", edgecolor="none",
       label="text retrieval", zorder=3)
ax.bar(x, time_calls, bar_w, bottom=text_calls,
       color=HIGHLIGHT, edgecolor="none",
       label="time retrieval", zorder=3)

# Total label above each bar
for xi, q in zip(x, order):
    total = text_calls[xi] + time_calls[xi]
    ax.text(xi, total + 0.4, str(total),
            ha="center", va="bottom",
            fontsize=9, color=INK, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9.5, color=INK)
ax.set_ylabel("Tool calls", fontsize=11, color=INK, labelpad=8)
ax.set_ylim(0, max(text_calls + time_calls) + 3)

ax.text(0, 1.32, "Tool calls per question  (Gemma 4 31B + QQMM)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=14, fontweight="bold", color=INK)
ax.text(0, 1.28,
        "L3_1, L4_1, and L5_4 hit 15-16 calls each, the retrieval\n"
        "loops described in §5.1.  Time retrieval is used 3 times total,\n"
        "all on L4_3.  Image and position retrievers were never invoked.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=9)
ax.tick_params(axis="y", length=4, labelsize=9, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=9)

out = OUT / "chart_toolcalls.png"
fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
print(f"wrote {out}")
plt.close(fig)

# ----- Figure 2: latency per question grouped by level -----
fig, ax = plt.subplots(figsize=(10.0, 5.2), dpi=200)
fig.subplots_adjust(top=0.82, bottom=0.20, left=0.10, right=0.97)

colors = [LEVEL_COLORS[q[:2]] for q in order]
ax.bar(x, [lat[q] for q in order], bar_w, color=colors,
       edgecolor="white", linewidth=1.0, zorder=3)

# Annotate slow ones (>=120s)
for xi, q in zip(x, order):
    t = lat[q]
    if t >= 120:
        ax.text(xi, t + 4, f"{t:.0f}s",
                ha="center", va="bottom",
                fontsize=9, color=INK, fontweight="bold")

# Per-level mean horizontal markers
import statistics
by_level = {}
for q in order:
    by_level.setdefault(q[:2], []).append(lat[q])
xpos = 0
for lvl in ['L1', 'L2', 'L3', 'L4', 'L5']:
    n = len(by_level[lvl])
    m = statistics.mean(by_level[lvl])
    ax.hlines(m, xpos - 0.4, xpos + n - 0.6,
              color=INK, linewidth=1.3, linestyle='--', zorder=4)
    ax.text(xpos + n/2 - 0.5, m + 6, f"{lvl} mean {m:.0f}s",
            ha="center", va="bottom",
            fontsize=8.5, color=INK, fontweight="bold")
    xpos += n

ax.set_xticks(x)
ax.set_xticklabels(order, rotation=45, ha='right', fontsize=9.5, color=INK)
ax.set_ylabel("Wall-clock latency (s)", fontsize=11, color=INK, labelpad=8)
ax.set_ylim(0, max(lat[q] for q in order) + 30)

ax.text(0, 1.22, "Per-question latency  (Gemma 4 31B + QQMM)",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=14, fontweight="bold", color=INK)
ax.text(0, 1.18,
        "L3-L5 questions take roughly 2x as long as L1-L2.  Total run:\n"
        "30.8 minutes for the 19-question suite.  Dashed lines: per-level mean.",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, color=MUTED, style="italic", linespacing=1.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0, labelsize=9)
ax.tick_params(axis="y", length=4, labelsize=9, colors=MUTED)
ax.yaxis.grid(True, color=GRID, lw=0.7, zorder=0)
ax.set_axisbelow(True)

out = OUT / "chart_latency.png"
fig.savefig(out, dpi=240, bbox_inches="tight", pad_inches=0.05,
            facecolor="white")
print(f"wrote {out}")
plt.close(fig)
