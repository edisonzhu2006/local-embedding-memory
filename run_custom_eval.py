#!/usr/bin/env python3
"""
Run custom question evaluation across multiple models.
Saves retrieved frames to labeled folders and generates a results report.
"""

import os
import sys
import json
import shutil
import csv
import math
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from remembr.embedder.embedders import VLMEmbeddings
from remembr.memory.memory_factory import MemoryFactory
from remembr.memory.memory import VLMMemoryItem


def load_frames(frames_json):
    with open(frames_json) as f:
        return json.load(f)


def load_questions(qa_json):
    with open(qa_json) as f:
        data = json.load(f)
    return data["data"]


def dist_2d(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def check_correct(pred_pos, gt_positions, threshold=2.0):
    """Check if predicted position is within threshold of any GT position."""
    if not gt_positions:
        return None  # No GT to check against
    min_dist = min(dist_2d(pred_pos, gt) for gt in gt_positions)
    return min_dist <= threshold, min_dist


def run_eval(model_name, oc_model, oc_pretrained, emb_dim, frames_json, qa_json,
             out_dir, device="cuda", top_k=5):
    """Run evaluation for one model. Returns list of result dicts."""

    print(f"\n{'='*60}")
    print(f"Model: {oc_model} ({oc_pretrained}) dim={emb_dim}")
    print(f"{'='*60}")

    # Load data
    frames = load_frames(frames_json)
    questions = load_questions(qa_json)

    # Create embedder
    embedder = VLMEmbeddings(
        device, backend="oc", oc_model=oc_model, oc_pretrained=oc_pretrained,
        hf_model_id="", vlm_layer=-1, batch_size=32, fp_16=False,
        emb_dim=emb_dim, vlm_text_prompts="", vlm_image_prompts="",
        online_model_nickname=""
    )

    # Create memory and insert frames
    memory = MemoryFactory.create_memory(
        backend="faiss", db_collection_name=f"eval_{model_name}",
        storage_path=os.path.join(out_dir, "faiss_tmp"),
        use_vlm_embedding=True, embedder=embedder,
        dim=emb_dim, retriever_k=top_k, respond_with_score=True
    )

    print("Embedding frames...")
    for frame in frames:
        entity = VLMMemoryItem(
            caption=frame.get("caption", "unknown"),
            time=frame.get("time", 0),
            position=frame["position"],
            theta=0.0,
            image_file_path=frame.get("image_file_path", ""),
        )
        emb = embedder.embed_query("[IMG]" + frame["image_file_path"])
        memory.insert(entity, vlm_embedding=emb)

    print(f"Inserted {len(frames)} frames. Running {len(questions)} questions...")

    # Create output dirs
    frames_out_dir = os.path.join(out_dir, model_name, "retrieved_frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]
        gt_positions = q.get("answer_pos", [])
        category = q.get("category", "unknown")

        # Query
        query_emb = embedder.embed_query("[TXT]" + question)
        docs_with_scores = memory.faiss_wrapper.search_by_vlm_embedding(query_emb, k=top_k)

        if not docs_with_scores:
            results.append({
                "id": qid, "question": question, "category": category,
                "pred_pos": None, "similarity": 0, "correct": None,
                "min_dist": None, "retrieved_frame": None
            })
            continue

        best_doc, best_score = docs_with_scores[0]
        pred_pos = best_doc.metadata.get("position", [0, 0, 0])
        retrieved_path = best_doc.metadata.get("image_file_path", "")

        # Check correctness
        correct_result = check_correct(pred_pos, gt_positions)
        is_correct = correct_result[0] if correct_result else None
        min_dist = correct_result[1] if correct_result else None

        # Copy retrieved frame to labeled folder
        q_frame_dir = os.path.join(frames_out_dir, f"{qid}_{category}")
        os.makedirs(q_frame_dir, exist_ok=True)

        for rank, (doc, score) in enumerate(docs_with_scores):
            src_path = doc.metadata.get("image_file_path", "")
            if os.path.exists(src_path):
                ext = os.path.splitext(src_path)[1]
                pos = doc.metadata.get("position", [0, 0, 0])
                dst_name = f"rank{rank+1}_sim{score:.3f}_pos{pos[0]:.1f}_{pos[1]:.1f}{ext}"
                shutil.copy2(src_path, os.path.join(q_frame_dir, dst_name))

        # Write question info
        with open(os.path.join(q_frame_dir, "info.txt"), "w") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Category: {category}\n")
            f.write(f"GT positions: {gt_positions}\n")
            f.write(f"Predicted: {pred_pos} (sim={best_score:.3f})\n")
            f.write(f"Distance to nearest GT: {min_dist:.2f}m\n" if min_dist is not None else "No GT\n")
            f.write(f"Correct (<2m): {is_correct}\n")
            f.write(f"\nTop-{top_k} results:\n")
            for rank, (doc, score) in enumerate(docs_with_scores):
                pos = doc.metadata.get("position", [0, 0, 0])
                f.write(f"  {rank+1}. sim={score:.3f} pos=[{pos[0]:.2f}, {pos[1]:.2f}] file={doc.metadata.get('image_file_path', '')}\n")

        results.append({
            "id": qid, "question": question, "category": category,
            "pred_pos": pred_pos, "similarity": float(best_score),
            "correct": is_correct, "min_dist": float(min_dist) if min_dist else None,
            "retrieved_frame": retrieved_path
        })

        status = "CORRECT" if is_correct else ("WRONG" if is_correct is not None else "NO_GT")
        print(f"  [{status}] {qid} sim={best_score:.3f} dist={min_dist:.1f}m" if min_dist else f"  [{status}] {qid} sim={best_score:.3f}")

    # Save results JSON
    results_path = os.path.join(out_dir, model_name, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Cleanup FAISS temp
    faiss_tmp = os.path.join(out_dir, "faiss_tmp")
    if os.path.exists(faiss_tmp):
        shutil.rmtree(faiss_tmp)

    # Free GPU memory
    del embedder, memory
    torch.cuda.empty_cache()

    return results


def generate_report(all_results, out_dir):
    """Generate markdown report comparing all models."""

    report_lines = ["# Custom Questions Evaluation Report\n"]
    report_lines.append(f"**Dataset**: smoke_full (546 frames)")
    report_lines.append(f"**Correctness threshold**: 2.0m from nearest ground truth\n")

    # Summary table
    report_lines.append("## Summary\n")
    report_lines.append("| Model | Overall | L1 Direct | L2 Indirect | L3 Spatial | L4 MultiStep | L5 Negation | Mean Sim |")
    report_lines.append("|-------|---------|-----------|-------------|------------|--------------|-------------|----------|")

    for model_name, results in all_results.items():
        # Per-level accuracy
        levels = {}
        all_correct = 0
        all_total = 0
        all_sim = []

        for r in results:
            cat = r["category"]
            level = cat.split("_")[0]
            if level not in levels:
                levels[level] = {"correct": 0, "total": 0}
            if r["correct"] is not None:
                levels[level]["total"] += 1
                all_total += 1
                if r["correct"]:
                    levels[level]["correct"] += 1
                    all_correct += 1
            all_sim.append(r["similarity"])

        overall = f"{all_correct}/{all_total}" if all_total > 0 else "N/A"
        mean_sim = f"{np.mean(all_sim):.3f}" if all_sim else "N/A"

        level_strs = []
        for lev in ["L1", "L2", "L3", "L4", "L5"]:
            if lev in levels and levels[lev]["total"] > 0:
                c, t = levels[lev]["correct"], levels[lev]["total"]
                level_strs.append(f"{c}/{t}")
            else:
                level_strs.append("N/A")

        report_lines.append(f"| {model_name} | {overall} | {' | '.join(level_strs)} | {mean_sim} |")

    # Detailed per-question results
    report_lines.append("\n## Per-Question Results\n")

    for model_name, results in all_results.items():
        report_lines.append(f"### {model_name}\n")
        report_lines.append("| ID | Question | Category | Sim | Pred Pos | Dist | Result |")
        report_lines.append("|-----|----------|----------|-----|----------|------|--------|")

        for r in results:
            pos_str = f"[{r['pred_pos'][0]:.1f}, {r['pred_pos'][1]:.1f}]" if r["pred_pos"] else "None"
            dist_str = f"{r['min_dist']:.1f}m" if r["min_dist"] is not None else "N/A"
            status = "CORRECT" if r["correct"] else ("WRONG" if r["correct"] is not None else "NO_GT")
            q_short = r["question"][:50] + "..." if len(r["question"]) > 50 else r["question"]
            report_lines.append(f"| {r['id']} | {q_short} | {r['category']} | {r['similarity']:.3f} | {pos_str} | {dist_str} | {status} |")

        report_lines.append("")

    report_path = os.path.join(out_dir, "EVAL_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_json", default="./extracted_videos/smoke_full/frames.json")
    parser.add_argument("--qa_json", default="./test_questions_smoke_qa.json")
    parser.add_argument("--out_dir", default="./outputs/custom_eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["siglip384", "vitl14"],
                        help="Models to test: siglip384, vitl14, vith14, vitbigg14")
    args = parser.parse_args()

    MODEL_CONFIGS = {
        "siglip384": ("ViT-SO400M-14-SigLIP-384", "webli", 1152),
        "vitl14": ("ViT-L-14", "openai", 768),
        "vith14": ("ViT-H-14", "laion2b_s32b_b79k", 1024),
        "vitbigg14": ("ViT-bigG-14", "laion2b_s39b_b160k", 1280),
        "vitb32": ("ViT-B-32", "openai", 512),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = {}

    for model_key in args.models:
        if model_key not in MODEL_CONFIGS:
            print(f"Unknown model: {model_key}. Options: {list(MODEL_CONFIGS.keys())}")
            continue

        oc_model, oc_pretrained, emb_dim = MODEL_CONFIGS[model_key]
        results = run_eval(
            model_name=model_key,
            oc_model=oc_model, oc_pretrained=oc_pretrained, emb_dim=emb_dim,
            frames_json=args.frames_json, qa_json=args.qa_json,
            out_dir=args.out_dir, device=args.device, top_k=args.top_k
        )
        all_results[model_key] = results

    generate_report(all_results, args.out_dir)


if __name__ == "__main__":
    main()
