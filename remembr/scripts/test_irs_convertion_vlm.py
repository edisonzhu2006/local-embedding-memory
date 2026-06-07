#!/usr/bin/env python3
"""
Build per-video JSON files:
  1) Frames listing in your target schema.
  2) QA listing derived from questions.json in the requested format.

Example frame item:
{
    "id": "./data/image_reverse_search_dataset/indoor/_BHE9vqx-qk/frames/frame_001.jpg",
    "position": ["unknown"],
    "time": 1,
    "image_file_path": "./data/image_reverse_search_dataset/indoor/_BHE9vqx-qk/frames/frame_001.jpg",
    "caption": "unknown",
    "vlm_embedding": null
}

Example QA output (per video):
{
  "data": [
    { "id": "1", "question": "Where is this object: <desc> ?", "answer": 221003116 },
    ...
  ]
}

Notes on QA:
- We read "questions (ques, answer_in_filename, category, query_img)" from questions.json.
- For each question, we map its answer filename(s) to a timestamp using "timestamp_to_filename".
- If multiple filenames are valid, we pick the earliest timestamp (microseconds). If none map, answer is null.
- The schema requested a single numeric 'answer', so we return one integer (microseconds) or null.
"""

from __future__ import annotations
import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple, Any

def parse_hms_to_microseconds(hms: str) -> Optional[int]:
    """Parse HH:MM:SS, MM:SS, or SS to seconds. Returns None if invalid."""
    parts = [float(p) for p in hms.split(":")]
    if len(parts) == 1:
        h, m, s = 0, 0, parts[0]
    elif len(parts) == 2:
        h, m, s = 0, parts[0], parts[1]
    elif len(parts) == 3:
        h, m, s = parts
    else:
        raise ValueError("Too many parts")
    if min(h, m, s) < 0:
        raise ValueError("Negative time component")
    return (h*3600 + m*60 + s) * 1e6

def natural_key(s: str):
    """Human-friendly sort key: splits numbers to sort numerically (frame_2 < frame_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def relpath_under(root: str, path: str) -> str:
    """Return POSIX-style relative path from root to path."""
    rel = os.path.relpath(path, root)
    return rel.replace(os.sep, "/")

def build_filename_to_time_map(qjson_path: str) -> Dict[str, Optional[int]]:
    """From questions.json, build mapping: filename -> time_in_seconds (if parsable)."""
    if not os.path.exists(qjson_path):
        return {}
    try:
        with open(qjson_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ts2fn = obj.get("timestamp_to_filename", {})
        out = {}
        for ts, fn in ts2fn.items():
            secs = parse_hms_to_microseconds(ts)
            # Normalize filename to POSIX and basename only for robust matching
            bname = os.path.basename(fn)
            out[bname] = secs
            # Also keep raw provided path as a key in case it's used verbatim
            out[fn.replace("\\", "/")] = secs
        return out
    except Exception as e:
        print(f"[WARN] Failed reading {qjson_path}: {e}", flush=True)
        return {}

def collect_video_dirs(data_root: str, splits: List[str]) -> List[str]:
    found = []
    for split in splits:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            continue
        for vid in os.listdir(split_dir):
            vdir = os.path.join(split_dir, vid)
            if os.path.isdir(vdir):
                frames_dir = os.path.join(vdir, "frames")
                if os.path.isdir(frames_dir):
                    found.append(vdir)
    return sorted(found, key=natural_key)

def pick_answer_time_us(answer_filenames: Any, fname_to_time_s: Dict[str, Optional[int]]) -> Optional[tuple[Optional[int], str]] | Optional[tuple[list[int], list[str]]]:
    """
    Given an answer filename or list of filenames, return the earliest matching time (in microseconds).
    We attempt basename matches first, then literal key matches as provided in questions.json mapping.
    """
    if answer_filenames is None:
        return [], []
    if isinstance(answer_filenames, (str,)):
        candidates = [answer_filenames]
    elif isinstance(answer_filenames, list):
        candidates = answer_filenames
    else:
        return [], []

    secs_list = []
    fn_list = []
    for fn in candidates:
        bname = os.path.basename(fn)
        secs = None
        if bname in fname_to_time_s:
            secs = fname_to_time_s[bname]
        elif fn.replace("\\", "/") in fname_to_time_s:
            secs = fname_to_time_s[fn.replace("\\", "/")]
        if secs is not None:
            secs_list.append(int(secs))
            fn_list.append(fn)
    return secs_list, fn_list

def load_questions(qjson_path: str) -> List[Tuple[str, Any, Optional[str], Optional[str]]]:
    """
    Return list of tuples: (query_text, answer_filenames, category, query_img_filename)
    If questions.json is missing or malformed, return empty list.
    """
    if not os.path.exists(qjson_path):
        return []
    try:
        with open(qjson_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        q_list = obj.get("questions (ques, answer_in_filename)", None)
        # Newer structure includes 4 elements per item
        if q_list is None:
            q_list = obj.get("questions", [])
        out = []
        for entry in q_list:
            # Safely unpack: allow 2..4 elements
            query = entry[0] if len(entry) > 0 else ""
            ans = entry[1] if len(entry) > 1 else None
            cat = entry[2] if len(entry) > 2 else None
            qimg = entry[3] if len(entry) > 3 else None
            out.append((query, ans, cat, qimg))
        return out
    except Exception as e:
        print(f"[WARN] Failed parsing questions from {qjson_path}: {e}", flush=True)
        return []

def main():
    ap = argparse.ArgumentParser(description="Build per-video frames JSON and QA JSON for reverse search benchmark.")
    ap.add_argument("--data_root", required=True, help="Dataset root (e.g., /downloads).")
    ap.add_argument("--out_dir", required=True, help="Where to write per-video JSON files.")
    ap.add_argument("--splits", nargs="+", default=["indoor", "outdoor", "robot_indoor", "harder_indoor"], help="Dataset splits to include (default: indoor outdoor).")
    ap.add_argument("--prefix_path", default="", help="Prefix to prepend to the RELATIVE path under data-root, e.g., ./data/image_reverse_search_dataset")
    ap.add_argument("--frames_subdir", default="frames", help="Subdirectory containing frames (default: frames).")
    ap.add_argument("--frames_json_name", default="{split}_{video_id}_frames.json", help="Output filename for frames JSON.")
    ap.add_argument("--qa_json_name", default="{split}_{video_id}_qa.json", help="Output filename for QA JSON derived from questions.json.")
    ap.add_argument("--fallback_index_time", action="store_true", help="If no time found for a frame, use 1-based index as 'time'.")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    video_dirs = collect_video_dirs(data_root, args.splits)
    if not video_dirs:
        print("[INFO] No video directories with frames found. Check --data-root and --splits.")
        return

    total_items = 0
    outputs = []

    for vdir in video_dirs:
        split = os.path.basename(os.path.dirname(vdir))  # indoor/outdoor
        vid = os.path.basename(vdir)
        frames_dir = os.path.join(vdir, args.frames_subdir)
        qjson_path = os.path.join(vdir, "questions.json")

        # --- Frames JSON ---
        fname_to_time = build_filename_to_time_map(qjson_path)  # filename -> seconds
        frame_files = [f for f in os.listdir(frames_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        frame_files.sort(key=natural_key)

        items = []
        max_t, min_t = 0, float('inf') 
        for idx, fname in enumerate(frame_files, 1):
            abs_path = os.path.join(frames_dir, fname)
            rel = relpath_under(data_root, abs_path)  # e.g., indoor/VID/frames/frame_001.jpg
            export_path = (args.prefix_path.rstrip("/")
                           + ("/" if args.prefix_path else "")
                           + rel)

            t = None
            if fname in fname_to_time:
                t = fname_to_time[fname]
            else:
                rel_posix = rel.replace("\\", "/")
                if rel_posix in fname_to_time:
                    t = fname_to_time[rel_posix]

            if t is None and args.fallback_index_time:
                t = idx  # 1-based index as a rough fallback

            max_t = max(max_t, t) if t is not None else max_t
            min_t = min(min_t, t) if t is not None else min_t

            item = {
                "id": export_path,
                "position": ["unknown"],
                "time": t,
                "start_time": t,
                "end_time": t,
                "image_file_path": export_path,
                "caption": "unknown",
                "vlm_embedding": None,
            }
            items.append(item)

        frames_out_name = args.frames_json_name.format(split=split, video_id=vid)
        frames_out_path = os.path.join(out_dir, frames_out_name)
        with open(frames_out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

        # --- QA JSON ---
        qa_entries_src = load_questions(qjson_path)
        qa_data = []
        for i, (query_text, ans_files, _cat, _qimg) in enumerate(qa_entries_src, 1):
            ans_us, answer_file = pick_answer_time_us(ans_files, fname_to_time)
            qa_item = {
                "id": str(i),
                "question": f"When does the following object or person appear? {query_text}",
                "answer": ans_us,  # integer microseconds or null
                "answer_file": answer_file,  # integer microseconds or null
                "category": _cat if _cat is not None else "unknown",
                "query_img": _qimg if _qimg is not None else "unknown",
            }
            qa_data.append(qa_item)

        qa_obj = {"start_time": min_t, "end_time": max_t, "frame_num": len(items), "data": qa_data}
        qa_out_name = args.qa_json_name.format(split=split, video_id=vid)
        qa_out_path = os.path.join(out_dir, qa_out_name)
        with open(qa_out_path, "w", encoding="utf-8") as f:
            json.dump(qa_obj, f, ensure_ascii=False, indent=2)

        outputs.append((vdir, frames_out_path, qa_out_path, len(items), len(qa_data)))
        total_items += len(items)
        print(f"[OK] {vid} -> frames:{frames_out_path} ({len(items)}), qa:{qa_out_path} ({len(qa_data)})")

    print(f"\n[SUMMARY] Videos processed: {len(outputs)} | Total frames: {total_items}")
    for vdir, frames_path, qa_path, n_frames, n_q in outputs:
        print(f"  - {os.path.basename(vdir)}: {n_frames} frames -> {frames_path}; {n_q} Qs -> {qa_path}")

if __name__ == "__main__":
    main()