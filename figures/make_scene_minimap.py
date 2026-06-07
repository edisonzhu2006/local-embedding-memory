"""Render a poster-quality top-down map of the smoke_v2 Habitat trajectory."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "position_data_habitat_smoke_v2.csv"
QA = ROOT / "extracted_videos" / "smoke_v2" / "qa.json"
OUT_PNG = ROOT / "figures" / "scene_minimap.png"
OUT_PDF = ROOT / "figures" / "scene_minimap.pdf"

ROOMS = [
    ("Entry",   -0.4,  0.6),
    ("Living",   5.6,  0.9),
    ("Kitchen",  7.2, -3.6),
    ("Dining",   2.2, -3.4),
    ("Bath",    -0.6, -2.3),
]

xs, ys = [], []
with CSV.open() as f:
    for row in csv.DictReader(f):
        xs.append(float(row["position_x"]))
        ys.append(float(row["position_y"]))
x = np.asarray(xs)
y = np.asarray(ys)
n_frames = len(x)

with QA.open() as f:
    qa = json.load(f)
anchors = []
for item in qa["data"]:
    for pos in item.get("answer_pos", []):
        anchors.append((pos[0], pos[1]))
anchors = np.array(anchors) if anchors else np.empty((0, 2))

fig, ax = plt.subplots(figsize=(7.0, 6.4), dpi=200)

ax.plot(x, y, color="#3b3b3b", lw=1.6, alpha=0.85, label="Agent path", zorder=2)

t = np.linspace(0, 1, n_frames)
ax.scatter(x, y, c=t, cmap="viridis", s=4, alpha=0.55, zorder=3, label=None)

if len(anchors):
    ax.scatter(anchors[:, 0], anchors[:, 1],
               s=95, c="#ff7f0e", edgecolor="black", lw=0.8,
               zorder=5, label="Question anchors")

ax.scatter([x[0]], [y[0]], s=220, marker="o",
           c="#2ca02c", edgecolor="black", lw=1.2,
           zorder=6, label="Start")
ax.scatter([x[-1]], [y[-1]], s=240, marker="X",
           c="#d62728", edgecolor="black", lw=1.2,
           zorder=6, label="End")

for name, rx, ry in ROOMS:
    ax.text(rx, ry, name, fontsize=14, fontweight="bold",
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#888", lw=0.6, alpha=0.85))

ax.set_xlabel("x (m)", fontsize=12)
ax.set_ylabel("y (m)", fontsize=12)
ax.set_title(f"Habitat smoke_v2 trajectory  ({n_frames} frames)",
             fontsize=14, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, ls="--", lw=0.5, alpha=0.5)
ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

pad = 0.6
ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.min() - pad, y.max() + pad)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"wrote {OUT_PNG}")
print(f"wrote {OUT_PDF}")
print(f"frames={n_frames}, anchors={len(anchors)}")
