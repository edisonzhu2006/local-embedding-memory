#!/usr/bin/env python3
"""
Extract retrieved frames from RAVEN/eval JSON outputs into trial frame directories.

For remembr_static_eval_vlm.py outputs (both embedder-only and RAVEN agent),
the JSON stores predicted positions but not frame file paths. This script
matches positions back to frames.json to find and copy the actual frame images.

Usage:
    # Single output JSON -> trial frames directory
    python generate_trial_frames.py \
        --result_json outputs/old_smoke_546frames/openclip_vitl14_smoke_v3/embedder_only+...json \
        --frames_json extracted_videos/smoke_full/frames.json \
        --out_dir trials/TRIAL-0005_frames

    # Batch mode: process all JSONs in a directory
    python generate_trial_frames.py \
        --result_dir outputs/old_smoke_546frames/openclip_vitl14_smoke_v3/ \
        --frames_json extracted_videos/smoke_full/frames.json \
        --out_dir trials/TRIAL-0005_frames
"""

import os
import sys
import json
import math
import re
import shutil
import argparse
from pathlib import Path


def load_frames_index(frames_json):
    """Load frames.json and build a position -> frame lookup."""
    with open(frames_json) as f:
        frames = json.load(f)

    index = []
    for frame in frames:
        pos = frame.get("position", [])
        if pos and pos[0] != "unknown":
            index.append({
                "path": frame["image_file_path"],
                "position": [float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else 0.44],
                "time": frame.get("time", 0),
                "caption": frame.get("caption", "unknown"),
            })
    return index


def find_frame_by_position(frames_index, target_pos, tolerance=0.001):
    """Find the frame whose position best matches target_pos."""
    if not target_pos or target_pos[0] is None:
        return None

    tx, ty = float(target_pos[0]), float(target_pos[1])
    best_frame = None
    best_dist = float("inf")

    for frame in frames_index:
        fx, fy = frame["position"][0], frame["position"][1]
        dist = math.sqrt((tx - fx) ** 2 + (ty - fy) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_frame = frame

    if best_frame and best_dist < tolerance:
        return best_frame, best_dist

    # Fallback: return closest even if > tolerance, but flag it
    if best_frame:
        return best_frame, best_dist

    return None


def parse_raven_retrievals(debug_logs, question_idx):
    """
    Parse RAVEN debug_logs to extract top-k retrieval results.

    RAVEN debug_logs contain "Image Retrieval Results" blocks with:
    - Frame paths: [IMG]extracted_videos/.../frame_000096.png[/IMG]
    - Positions: "position of [x, y, z]"
    - Similarities: "Cosine similarity between the query and the frame: 0.209"

    Returns list of dicts: {path, position, similarity}
    """
    results = []

    # Find the debug entries for this question (between [RST] markers)
    rst_indices = [i for i, entry in enumerate(debug_logs) if entry == "[RST]"]

    if question_idx >= len(rst_indices):
        return results

    start = rst_indices[question_idx]
    end = rst_indices[question_idx + 1] if question_idx + 1 < len(rst_indices) else len(debug_logs)

    # Collect all text from debug entries
    all_text = ""
    for i in range(start, end):
        entry = debug_logs[i]
        if isinstance(entry, list):
            for text in entry:
                all_text += str(text) + "\n"
        elif isinstance(entry, str):
            all_text += entry + "\n"

    # Parse each retrieval result block: "(N) At time=... position of [...] ... similarity: X ... [IMG]path[/IMG]"
    # Each block starts with (N) and contains position, similarity, and frame path
    block_pattern = re.compile(
        r'\((\d+)\)\s+At time=.*?'
        r'position of \[([0-9.\-]+),\s*([0-9.\-]+)(?:,\s*([0-9.\-]+))?\].*?'
        r'similarity.*?:\s*([0-9.\-]+).*?'
        r'\[IMG\](.*?)\[/IMG\]',
        re.DOTALL
    )

    for match in block_pattern.finditer(all_text):
        rank = int(match.group(1))
        x, y = float(match.group(2)), float(match.group(3))
        z = float(match.group(4)) if match.group(4) else 0.44
        sim = float(match.group(5))
        path = match.group(6).strip()

        results.append({
            "rank": rank,
            "path": path,
            "position": [x, y, z],
            "similarity": sim,
        })

    # Deduplicate by path (RAVEN may call retrieve_from_text multiple times)
    seen = set()
    unique = []
    for r in results:
        if r["path"] not in seen:
            seen.add(r["path"])
            unique.append(r)

    return unique


def parse_embedder_debug_results(debug_logs, question_idx):
    """
    Parse embedder-only debug_logs to extract top-k results.

    Old format: "Result 1: score=0.226"
    New format: "Result 1: score=0.226 frame=extracted_videos/.../frame_000123.png"

    Returns list of dicts: {rank, score, path (or None if old format)}
    """
    results = []
    rst_indices = [i for i, entry in enumerate(debug_logs) if entry == "[RST]"]

    if question_idx >= len(rst_indices):
        return results

    start = rst_indices[question_idx]
    end = rst_indices[question_idx + 1] if question_idx + 1 < len(rst_indices) else len(debug_logs)

    for i in range(start, end):
        entry = debug_logs[i]
        if isinstance(entry, list):
            for text in entry:
                if isinstance(text, str):
                    match = re.match(r'Result (\d+): score=([0-9.\-]+)(?:\s+frame=(.+))?', text)
                    if match:
                        results.append({
                            "rank": int(match.group(1)),
                            "score": float(match.group(2)),
                            "path": match.group(3) if match.group(3) else None,
                        })

    return results


def extract_similarity_from_response(response):
    """Extract similarity score from response text like 'similarity score: 0.226'."""
    text = response.get("text", "") or ""
    match = re.search(r'similarity score:\s*([0-9.\-]+)', text)
    if match:
        return float(match.group(1))
    return None


def copy_frame(frame_info, dest_path, rank, similarity, project_root):
    """Copy a frame image to the trial frames directory with standardized naming."""
    src = os.path.join(project_root, frame_info["path"])
    if not os.path.exists(src):
        print(f"  WARNING: Frame not found: {src}")
        return False

    ext = os.path.splitext(src)[1]
    pos = frame_info["position"]
    sim_str = f"{similarity:.3f}" if similarity is not None else "unknown"
    dst_name = f"rank{rank}_sim{sim_str}_pos{pos[0]:.1f}_{pos[1]:.1f}{ext}"
    shutil.copy2(src, os.path.join(dest_path, dst_name))
    return True


def process_result_json(result_json, frames_json, out_dir, project_root):
    """Process a single result JSON and extract frames."""

    with open(result_json) as f:
        data = json.load(f)

    frames_index = load_frames_index(frames_json)
    responses = data.get("responses", [])
    debug_logs = data.get("debug_logs", [])

    # Detect if this is a RAVEN/agent run or embedder-only
    is_raven = False
    for entry in debug_logs[:20]:
        if isinstance(entry, list):
            for text in entry:
                if isinstance(text, str) and "Image Retrieval" in text:
                    is_raven = True
                    break
        if is_raven:
            break

    # Also check by response length / elapsed time
    if not is_raven and responses:
        avg_elapsed = sum(r.get("elapsed", 0) for r in responses) / len(responses)
        if avg_elapsed > 5.0:  # RAVEN runs take 20-60s per question
            is_raven = True

    agent_type = "raven" if is_raven else "embedder_only"
    print(f"Detected agent type: {agent_type}")
    print(f"Processing {len(responses)} questions...")

    os.makedirs(out_dir, exist_ok=True)
    summary = []

    for q_idx, resp in enumerate(responses):
        qid = resp.get("id", str(q_idx + 1))
        question = resp.get("question", "unknown")
        pred_pos = resp.get("position", resp.get("response", {}).get("position"))

        # Create question directory
        q_dir = os.path.join(out_dir, f"Q{qid}")
        os.makedirs(q_dir, exist_ok=True)

        # Extract similarity from response text
        sim = extract_similarity_from_response(resp)
        if sim is None:
            sim = extract_similarity_from_response(resp.get("response", {}))

        retrieved_frames = []  # list of {rank, path, position, similarity}
        debug_scores = []

        if is_raven:
            # RAVEN: extract full retrieval results (paths, positions, scores) from debug_logs
            raven_results = parse_raven_retrievals(debug_logs, q_idx)
            for rank_idx, rr in enumerate(raven_results[:5]):
                src = os.path.join(project_root, rr["path"])
                if os.path.exists(src):
                    ext = os.path.splitext(src)[1]
                    pos = rr["position"]
                    dst_name = f"rank{rank_idx+1}_sim{rr['similarity']:.3f}_pos{pos[0]:.1f}_{pos[1]:.1f}{ext}"
                    shutil.copy2(src, os.path.join(q_dir, dst_name))
                    retrieved_frames.append(rr)
                else:
                    print(f"  WARNING: Frame not found: {src}")
        else:
            # Embedder-only: extract frame paths from debug_logs
            debug_results = parse_embedder_debug_results(debug_logs, q_idx)
            has_paths = any(r["path"] for r in debug_results)

            if has_paths:
                for dr in debug_results[:5]:
                    if dr["path"]:
                        src = os.path.join(project_root, dr["path"])
                        if os.path.exists(src):
                            ext = os.path.splitext(src)[1]
                            frame_pos = [0, 0, 0]
                            for fi in frames_index:
                                if fi["path"] == dr["path"]:
                                    frame_pos = fi["position"]
                                    break
                            dst_name = f"rank{dr['rank']}_sim{dr['score']:.3f}_pos{frame_pos[0]:.1f}_{frame_pos[1]:.1f}{ext}"
                            shutil.copy2(src, os.path.join(q_dir, dst_name))
                            retrieved_frames.append({
                                "rank": dr["rank"], "path": dr["path"],
                                "position": frame_pos, "similarity": dr["score"],
                            })
                        else:
                            print(f"  WARNING: Frame not found: {src}")
            else:
                # Old format: no frame paths in debug_logs — cannot extract exact frames
                debug_scores = [r["score"] for r in debug_results]

        # --- Write info.txt ---
        with open(os.path.join(q_dir, "info.txt"), "w") as f:
            f.write(f"Question ID: {qid}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Agent: {agent_type}\n")
            f.write(f"Predicted Position: {pred_pos}\n")
            f.write(f"Elapsed: {resp.get('elapsed', 'N/A')}s\n")

            resp_text = resp.get("text") or resp.get("response", {}).get("text", "")
            if resp_text:
                f.write(f"\nResponse Text:\n{resp_text}\n")

            if retrieved_frames:
                f.write(f"\nRetrieved Frames ({len(retrieved_frames)}):\n")
                for rf in retrieved_frames:
                    f.write(f"  rank {rf.get('rank', '?')}: sim={rf.get('similarity', 'N/A')}"
                            f" pos=[{rf['position'][0]:.2f}, {rf['position'][1]:.2f}]"
                            f" file={rf['path']}\n")
            else:
                f.write(f"\nRetrieved Frames: NONE\n")

            if debug_scores:
                f.write(f"\nTop-k Scores (from debug): {debug_scores}\n")
                f.write(f"Note: No frame paths in old-format debug logs. Re-run with updated pipeline to get exact frames.\n")

        num_frames = len(retrieved_frames)
        status = f"{num_frames} frames" if num_frames > 0 else "NO_MATCH"
        print(f"  [{status}] Q{qid}: {question[:60]}...")
        summary.append({
            "id": qid,
            "question": question,
            "position": pred_pos,
            "similarity": sim,
            "num_frames_extracted": num_frames,
            "retrieved_frames": [rf["path"] for rf in retrieved_frames],
        })

    # Write summary JSON
    with open(os.path.join(out_dir, "extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Frames saved to: {out_dir}")
    matched = len([s for s in summary if s["num_frames_extracted"] > 0])
    total_frames = sum(s["num_frames_extracted"] for s in summary)
    print(f"Summary: {matched}/{len(summary)} questions matched, {total_frames} frames extracted")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract retrieved frames from RAVEN/eval JSON into trial frame directories"
    )
    parser.add_argument("--result_json", help="Path to a single result JSON file")
    parser.add_argument("--result_dir", help="Directory containing result JSON file(s)")
    parser.add_argument("--frames_json", required=True,
                        help="Path to frames.json with frame positions and paths")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for extracted frames")
    parser.add_argument("--project_root", default=".",
                        help="Project root for resolving frame image paths (default: .)")
    parser.add_argument("--position_tolerance", type=float, default=0.01,
                        help="Max distance (m) for position matching (default: 0.01)")
    args = parser.parse_args()

    if not args.result_json and not args.result_dir:
        parser.error("Must specify either --result_json or --result_dir")

    json_files = []
    if args.result_json:
        json_files.append(args.result_json)
    elif args.result_dir:
        for f in sorted(os.listdir(args.result_dir)):
            if f.endswith(".json"):
                json_files.append(os.path.join(args.result_dir, f))

    if not json_files:
        print("No JSON files found.")
        sys.exit(1)

    for jf in json_files:
        print(f"\n{'='*60}")
        print(f"Processing: {jf}")
        print(f"{'='*60}")
        process_result_json(jf, args.frames_json, args.out_dir, args.project_root)


if __name__ == "__main__":
    main()
