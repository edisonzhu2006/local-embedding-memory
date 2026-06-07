"""Render the scale-of-memory strip: N answer-anchored robot-view thumbnails + '...'.

For each ground-truth answer position in the 20-question eval, we find the frame
closest to that position. These are by construction frames that contain the
question's subject (purple chair, toilet, sofa, kitchen, TV, etc.) — so the
strip shows real "things the robot saw" rather than empty hallway shots.
"""
import os, json, math
from PIL import Image, ImageDraw, ImageFont

FRAMES_DIR = "extracted_videos/smoke_full/frames"
FRAMES_JSON = "extracted_videos/smoke_full/frames.json"
QUESTIONS_JSON = "test_questions_smoke_qa.json"
RAVEN_DIR = "outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/retrieved_frames"
OUT_DIR = "figures"
N_ROWS = 2
PER_ROW = 6
N_THUMBS = N_ROWS * PER_ROW
THUMB_H = 90
GAP = 6
ROW_GAP = 8
PAD = 14
TOTAL_FRAMES = len(os.listdir(FRAMES_DIR))

with open(FRAMES_JSON) as f:
    frames_meta = json.load(f)
frames_meta = [m for m in frames_meta
               if m.get("position") and m["position"][0] != "unknown"]

def pos_xy(m_or_p):
    p = m_or_p["position"] if isinstance(m_or_p, dict) else m_or_p
    return float(p[0]), float(p[1])

# Walk all 20 gemma4:31b questions in order; for each, take the highest-rank
# retrieved frame whose *source frame_NNN* hasn't been used yet. Stop at 12.
import re
ALL_QUESTIONS = ["L1_1","L1_2","L1_3","L1_4","L2_1","L2_2","L2_3","L2_4",
                 "L3_1","L3_2","L3_3","L3_4","L4_1","L4_2","L4_3","L4_4",
                 "L5_1","L5_2","L5_3","L5_4"]

MIN_DIST = 1.3  # meters between any two picked frames
# Positions the user has explicitly rejected — anything within MIN_DIST is skipped.
BLACKLIST_XY = [(1.4, 2.2), (8.7, 4.9)]

def parse_info(qid):
    """Return ordered list of (rank, (x,y), source_basename) from info.txt."""
    info_path = os.path.join(RAVEN_DIR, f"Q{qid}", "info.txt")
    if not os.path.exists(info_path):
        return []
    out = []
    for line in open(info_path):
        m = re.search(r"rank (\d+):.*pos=\[([0-9.\-]+),\s*([0-9.\-]+)\].*file=(\S+)", line)
        if m:
            rank = int(m.group(1))
            xy = (float(m.group(2)), float(m.group(3)))
            src = os.path.basename(m.group(4))
            out.append((rank, xy, src))
    out.sort()
    return out

def first_far_enough(qid, used_xys):
    qdir = os.path.join(RAVEN_DIR, f"Q{qid}")
    for rank, xy, src in parse_info(qid):
        if any(math.hypot(xy[0]-u[0], xy[1]-u[1]) < MIN_DIST for u in used_xys):
            continue
        if any(math.hypot(xy[0]-b[0], xy[1]-b[1]) < 1.0 for b in BLACKLIST_XY):
            continue
        for fn in sorted(os.listdir(qdir)):
            if fn.startswith(f"rank{rank}_") and fn.endswith(".png"):
                return os.path.join(qdir, fn), xy
    return None, None

SUBJECT_PATHS = []
used_xys = []
for qid in ALL_QUESTIONS:
    if len(SUBJECT_PATHS) >= 12:
        break
    path, xy = first_far_enough(qid, used_xys)
    if path:
        SUBJECT_PATHS.append(path)
        used_xys.append(xy)

selected = [{"image_file_path": p} for p in SUBJECT_PATHS]
picks = [os.path.basename(p) for p in SUBJECT_PATHS]

CROP_TOP_FRAC = 0.10
CROP_BOTTOM_FRAC = 0.45  # drop ~45% of frame height (mostly floor)

thumbs = []
for path in SUBJECT_PATHS:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    top = int(h * CROP_TOP_FRAC)
    bot = int(h * (1 - CROP_BOTTOM_FRAC))
    im = im.crop((0, top, w, bot))
    cw, ch = im.size
    # Center-crop horizontally to square so the strip thumbs aren't ultra-wide.
    side = min(cw, ch)
    left = (cw - side) // 2
    im = im.crop((left, 0, left + side, side))
    thumbs.append(im.resize((THUMB_H, THUMB_H), Image.LANCZOS))

rows = [thumbs[i * PER_ROW : (i + 1) * PER_ROW] for i in range(N_ROWS)]
row_widths = [sum(t.width for t in row) + GAP * (len(row) - 1) for row in rows]
content_w = max(row_widths)
strip_w = content_w + PAD * 2
strip_h = THUMB_H * N_ROWS + ROW_GAP * (N_ROWS - 1) + PAD * 2 + 28

canvas = Image.new("RGB", (strip_w, strip_h), (255, 255, 255))
draw = ImageDraw.Draw(canvas)

try:
    font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 44)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except Exception:
    font_big = ImageFont.load_default()
    font_small = ImageFont.load_default()

for r, row in enumerate(rows):
    y = PAD + r * (THUMB_H + ROW_GAP)
    x = PAD
    for t in row:
        canvas.paste(t, (x, y))
        draw.rectangle([x, y, x + t.width - 1, y + THUMB_H - 1],
                       outline=(80, 80, 80), width=1)
        x += t.width + GAP
caption = f"{TOTAL_FRAMES:,} frames in this trajectory  -->  thousands to millions over a lifelong deployment"
caption_w = draw.textlength(caption, font=font_small)
if caption_w + PAD * 2 > strip_w:
    new_w = int(caption_w + PAD * 2)
    bigger = Image.new("RGB", (new_w, strip_h), (255, 255, 255))
    bigger.paste(canvas, ((new_w - strip_w) // 2, 0))
    canvas = bigger
    draw = ImageDraw.Draw(canvas)
    strip_w = new_w
caption_y = PAD + THUMB_H * N_ROWS + ROW_GAP * (N_ROWS - 1) + 6
draw.text(((strip_w - caption_w) / 2, caption_y), caption, fill=(60, 60, 60), font=font_small)

png_path = os.path.join(OUT_DIR, "scale_strip.png")
pdf_path = os.path.join(OUT_DIR, "scale_strip.pdf")
canvas.save(png_path, dpi=(300, 300))
canvas.save(pdf_path, "PDF", resolution=300.0)
print(f"wrote {png_path} ({canvas.size[0]}x{canvas.size[1]})")
print(f"wrote {pdf_path}")
print(f"thumbnails: {len(thumbs)}  picked frames: {[p for p in picks]}")
