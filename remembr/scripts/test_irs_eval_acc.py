
#!/usr/bin/env python3
"""
Evaluate prediction accuracy for time-based Q/A JSON files.

- Prediction files live under a "pred_dir". Each file has a "responses" list where each item
  contains fields including:
      - "id": question id (string or int)
      - "time": predicted HH:MM:SS string (e.g. "19:01:34")

- Ground-truth files live under a "gt_dir". Each GT filename should contain the substring
  extracted from the prediction filename: the token between the first "__" and the next "_frames".
  Example:
      pred filename: remembr+gpt-4o+oc+ViT-B-32+...__indoor_2JYMWPElMHk_frames__._0.json
      token: "indoor_2JYMWPElMHk"
      GT file to use: any file in gt_dir whose name contains "indoor_2JYMWPElMHk"
                      (typically "indoor_2JYMWPElMHk_qa.json")

- GT format contains "data": [{"id": "...", "answer": 94000000, ...}, ...]
  where "answer" is a relative timestamp in microseconds.
  Convert to HH:MM:SS using (as requested) localtime:
      t = answer / 1e6
      hhmmss = time.strftime("%H:%M:%S", time.localtime(t))

- Compare prediction time string vs. GT time string for equality.
  Compute per-file accuracy and overall accuracy.

Outputs:
- A CSV summary at "time_eval_summary.csv"
- Prints a per-file table and overall accuracy.
"""

import argparse
import json
import re, os
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import csv
from collections import defaultdict
IMG_TAG_RE = re.compile(r"\[IMG\](.*?)\[/IMG\]")


def extract_token_from_pred_filename(fname: str) -> Optional[str]:
    """
    Extract the token between the first '__' and the next '_frames' in the filename.
    Returns None if not found.
    """
    # We only look at the stem to make matching simpler
    m = re.search(r'__(.*?)_frames', fname)
    if m:
        return m.group(1)
    return None

def find_gt_file(gt_dir: Path, token: str) -> Optional[Path]:
    """
    Find the GT file in gt_dir whose filename contains the given token.
    Prefer files ending with '_qa.json', but fall back to any .json containing the token.
    """
    candidates: List[Path] = sorted([p for p in gt_dir.glob("*.json") if token in p.name])
    if not candidates:
        return None
    # prefer *_qa.json if available
    qa = [p for p in candidates if p.name.endswith("_qa.json")]
    return qa[0] if qa else candidates[0]

def load_gt_map(gt_path: Path) -> Tuple[Dict[str, list], Dict[str, list]]:
    """
    Load GT JSON and return a dict mapping id (string) -> HH:MM:SS (via localtime).
    """
    with gt_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    gt_map: Dict[str, list] = {}
    gt_map_file: Dict[str, list] = {}
    data = obj.get("data", [])
    frame_num = int(obj["frame_num"]) if "frame_num" in obj else 1e10
    total_random_acc = []
    for item in data:
        qid = str(item.get("id"))
        ans_us = item.get("answer", None)
        if ans_us is None:
            continue
        assert isinstance(ans_us, list) 
        total_random_acc.append(len(ans_us) / frame_num)
        for ans in ans_us:
            seconds = float(ans) / 1e6
            # Per user spec: use localtime
            tstruct = time.localtime(seconds)
            # turn to seconds for easier comparison
            sec = tstruct.tm_hour * 3600 + tstruct.tm_min * 60 + tstruct.tm_sec
            gt_map.setdefault(qid, []).append(sec)
        ans_filename = item.get("answer_file", None)
        if ans_filename is not None:
            assert isinstance(ans_filename, list)
            for fn in ans_filename:
                gt_map_file.setdefault(qid, []).append(fn)
    return gt_map, gt_map_file, total_random_acc



def load_pred_map(pred_path: Path, top_k) -> Tuple[Dict[str, Optional[str]], List[List[str]], Dict[str, List[List[str]]]]:
    """
    Load prediction JSON and return:
      - pred_map: mapping id (string) -> HH:MM:SS string (or None if missing)
      - all_retrievals: list of retrieval calls; each is a list of image basenames
      - qid_to_retrievals: mapping qid -> list of retrieval calls (each call is a list of image basenames)

    'Retrieval calls' are extracted from debug_logs entries containing [IMG]...[/IMG].
    Multiple tool calls are supported; order is preserved.
    """
    with pred_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    # 1) existing time parsing
    pred_map: Dict[str, Optional[str]] = {}
    for resp in obj.get("responses", []):
        qid = str(resp.get("id"))
        t = resp.get("time")
        if isinstance(t, str):
            # normalize to HH:MM:SS if possible
            # accept already formatted "HH:MM:SS", otherwise try best-effort parse
            if re.fullmatch(r"\d{2}:\d{2}:\d{2}", t):
                pred_map[qid] = t
            else:
                # attempt to extract HH:MM:SS from text field or from t itself
                # look for pattern in the "time" string
                m = re.search(r"(\d{2}:\d{2}:\d{2})", t)
                pred_map[qid] = m.group(1) if m else None
        else:
            # try the "response"->"time" field if present
            inner = resp.get("response", {})
            tin = inner.get("time")
            if isinstance(tin, str) and re.fullmatch(r"\d{2}:\d{2}:\d{2}", tin):
                pred_map[qid] = tin
            else:
                pred_map[qid] = None

    # 2) retrieval parsing from debug_logs
    debug_logs = obj.get("debug_logs", [])
    responses = obj.get("responses", [])
    # We align [RST] blocks with 'responses' order.
    # Pattern per block:
    #   "[RST]"
    #   [ "<question...>" ]                     # list containing question line
    #   [ "... tool_calls=[...]", "... [IMG]...[/IMG] ..." ]  # list with tool output(s)
    # We collect every list that contains [IMG]... as one retrieval call.
    all_retrievals: List[List[str]] = []
    qid_to_retrievals: Dict[str, List[List[str]]] = {}

    qidx = -1
    expect_new_question = False
    current_qid: Optional[str] = None

    if not debug_logs:
        return pred_map, all_retrievals, qid_to_retrievals
    for entry in debug_logs:
        if entry == "[RST]":
            expect_new_question = True
            continue

        if isinstance(entry, list):
            # If this list is the "question" list immediately after [RST], advance qid
            if expect_new_question:
                qidx += 1
                expect_new_question = False
                if 0 <= qidx < len(responses):
                    current_qid = str(responses[qidx].get("id"))
                else:
                    current_qid = None  # safety fallback

            # Regardless, scan this list for [IMG]...[/IMG] occurrences
            img_paths: List[str] = []
            # print(entry)
            for s in entry:
                if not isinstance(s, str):
                    continue
                for p in IMG_TAG_RE.findall(s):
                    # Normalize to filename only
                    img_paths.append(os.path.basename(p.strip()))

            # If we found any images in this list, that's ONE retrieval call
            if img_paths:
                all_retrievals.append(img_paths)
                # print(img_paths)
                if current_qid is not None:
                    assert len(img_paths) % top_k == 0, f"Number of images in one retrieval call should be multiple of top_k={top_k}, got {len(img_paths)}"
                    # Split into chunks of top_k
                    for i in range(0, len(img_paths), top_k):
                        img_chunk = img_paths[i:i+top_k]
                        qid_to_retrievals.setdefault(current_qid, []).append(img_chunk)

    return pred_map, all_retrievals, qid_to_retrievals


def load_gt_category_map(gt_path: Path) -> Dict[str, Optional[str]]:
    """
    Load GT JSON and return a dict mapping id (string) -> category (if available).
    Tries common keys: 'category', 'type', 'question_type', 'Question\nCategory'.
    Fallback to 'unknown' if not found.
    """
    with gt_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    cat_map: Dict[str, Optional[str]] = {}
    data = obj.get("data", [])
    for item in data:
        qid = str(item.get("id"))
        cat = (
            item.get("category")
            or item.get("type")
            or item.get("question_type")
            or item.get("Question\nCategory")
        )
        if cat is None:
            cat = "unknown"
        cat_map[qid] = str(cat)
    return cat_map

def compare_times(pred_map: Dict[str, Optional[str]], gt_map: Dict[str, list], gt_map_file=None, qid_to_retrievals=None) -> Tuple[int, int, List[Tuple[str, Optional[str], Optional[str], bool]]]:
    """
    Compare predicted vs. ground-truth times for matching question ids.
    Returns: (num_correct, num_total, rows) where rows are (id, pred, gt, is_correct)
    Only questions present in GT are counted toward total.
    """
    num_correct = 0
    num_total = 0
    rows = []
    # print(gt_map_file)
    # print(qid_to_retrievals)
    num_correct_retrieved_in_1_call = 0
    num_correct_retrieved_in_2_more_call = 0
    num_correct_not_retrieved = 0
    num_incorrect_retrieved_in_1_call = 0
    num_incorrect_retrieved_in_2_more_call = 0
    num_incorrect_not_retrieved = 0
    total_tool_calls = 0
    def PinG(P:list, G:list) :
        for p in P:
            if p in G:
                return True
        return False

    for qid, gt in gt_map.items():
        pred = pred_map.get(qid)
        if isinstance(pred, str):
            ph, pm, ps = map(float, pred.split(":"))
            pred_sec = ph * 3600 + pm * 60 + ps
        elif isinstance(pred, (float, int)):
            pred_sec = float(pred)
        else:
            print(f"Unexpected pred format for id={qid}: {pred}")
            pred_sec = -1.0  # invalid
                    
        if isinstance(gt, (float, int)):
            ok = (abs(pred_sec - float(gt)) < 2.)  
        else:
            ok = False
            for g in gt:
                if abs(pred_sec - g) < 2.:
                    ok = True
                    break
        if gt is not None:
            num_total += 1
            if ok:
                num_correct += 1
            # if not (qid_to_retrievals and gt_map_file and qid in qid_to_retrievals and qid in gt_map_file):
            #     print(f"QID={qid}: pred={pred} ({pred_sec:.1f}s) | gt={gt} | correct={ok}")
            #     print(qid_to_retrievals)
            #     print(gt_map_file)
            #     raise ValueError("Debug")
            if qid_to_retrievals and gt_map_file and qid in qid_to_retrievals and qid in gt_map_file:
                if PinG(qid_to_retrievals[qid][0], gt_map_file[qid]):
                    if ok:
                        num_correct_retrieved_in_1_call += 1
                    else:
                        num_incorrect_retrieved_in_1_call += 1
                else:
                    found_in_later_call = False
                    for call in qid_to_retrievals[qid][1:]:
                        if PinG(call, gt_map_file[qid]):
                            found_in_later_call = True
                            break
                    if found_in_later_call:
                        if ok:
                            num_correct_retrieved_in_2_more_call += 1
                        else:
                            num_incorrect_retrieved_in_2_more_call += 1
                    else:
                        if ok:
                            num_correct_not_retrieved += 1
                        else:
                            num_incorrect_not_retrieved += 1
                            # print(f"Not retrieved: qid={qid}, pred={pred}, gt={gt}, retrievals={qid_to_retrievals[qid]}, gt_files={gt_map_file[qid]}")

            total_tool_calls += len(qid_to_retrievals.get(qid, [])) if qid_to_retrievals else 0
        rows.append((qid, pred, gt, ok, qid_to_retrievals.get(qid, []) if qid_to_retrievals else [], gt_map_file.get(qid, []) if gt_map_file else []))
    return num_correct, num_total, rows, (num_correct_retrieved_in_1_call, num_correct_retrieved_in_2_more_call, num_correct_not_retrieved, num_incorrect_retrieved_in_1_call, num_incorrect_retrieved_in_2_more_call, num_incorrect_not_retrieved), total_tool_calls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="Directory with prediction JSON files")
    ap.add_argument("--gt_dir", required=True, help="Directory with ground-truth JSON files")
    ap.add_argument("--out_csv", default="time_eval_summary.csv", help="Path to write CSV summary")
    ap.add_argument("--category_out_csv", help="Path to write per-category CSV summary")
    # ap.add_argument("--category_out_csv", default="time_eval_category_summary.csv", help="Path to write per-category CSV summary")
    ap.add_argument("--topk", type=int,default=5, help="Top K retrievals to consider (default 5)")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out_csv)
    # import ipdb; ipdb.set_trace()
    all_files = sorted([p for p in pred_dir.glob("*.json")])
    if not all_files:
        print(f"No prediction JSONs found in: {pred_dir}")
        return

    overall_correct = 0
    overall_total = 0
    overall_cat_correct: Dict[str, int] = defaultdict(int)
    overall_cat_total: Dict[str, int] = defaultdict(int)
    per_file_summaries = []  # list of dicts for CSV

    overall_correct_retrieved_in_1_call = 0
    overall_correct_retrieved_in_2_more_call = 0
    overall_correct_not_retrieved = 0
    overall_incorrect_retrieved_in_1_call = 0
    overall_incorrect_retrieved_in_2_more_call = 0
    overall_incorrect_not_retrieved = 0
    overall_tool_calls = 0

    # Header row for CSV
    fieldnames = ["pred_file", "gt_file", "file_correct", "file_total", "file_accuracy", "retrieval_results", "total_tool_calls"]

    if not os.path.exists(out_csv.parent):
        os.makedirs(out_csv.parent, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        overall_random_acc = []
        for pred_path in all_files:
            token = extract_token_from_pred_filename(pred_path.name)
            if not token:
                print(f"[WARN] Could not extract token from '{pred_path.name}', skipping.")
                continue

            gt_path = find_gt_file(gt_dir, token)
            if not gt_path:
                print(f"[WARN] No GT file found in '{gt_dir}' containing token '{token}' for '{pred_path.name}', skipping.")
                continue

            try:
                gt_map, gt_map_file, total_random_acc = load_gt_map(gt_path)
            except Exception as e:
                print(f"[ERROR] Failed to read GT '{gt_path}': {e}")
                continue

            overall_random_acc.extend(total_random_acc)
            
            # try:
            pred_map, all_retrievals, qid_to_retrievals = load_pred_map(pred_path, top_k=args.topk)
                # print(pred_map)
            # except Exception as e:
            #     print(f"[ERROR] Failed to read prediction '{pred_path}': {e}")
            #     continue

            # Attempt to load categories for per-category accuracy
            try:
                gt_cat_map = load_gt_category_map(gt_path)
            except Exception:
                gt_cat_map = {}

            # print(gt_map_file)
            # print(qid_to_retrievals) 
            # {'1': ['frame_010.jpg'], '2': ['frame_015.jpg'], '3': ['frame_014.jpg'], '4': ['frame_011.jpg'], '5': ['frame_005.jpg'], '6': ['frame_009.jpg'], '7': ['frame_008.jpg']}
            # {'1': [['frame_010.jpg', 'frame_012.jpg', 'frame_005.jpg', 'frame_008.jpg', 'frame_011.jpg']], '2': [['frame_015.jpg', 'frame_011.jpg', 'frame_014.jpg', 'frame_001.jpg', 'frame_006.jpg']], '3': [['frame_014.jpg', 'frame_010.jpg', 'frame_001.jpg', 'frame_005.jpg', 'frame_012.jpg']], '4': [['frame_011.jpg', 'frame_008.jpg', 'frame_014.jpg', 'frame_010.jpg', 'frame_012.jpg']], '5': [['frame_005.jpg', 'frame_004.jpg', 'frame_013.jpg', 'frame_009.jpg', 'frame_006.jpg']], '6': [['frame_010.jpg', 'frame_011.jpg', 'frame_014.jpg', 'frame_012.jpg', 'frame_005.jpg'], ['frame_010.jpg', 'frame_012.jpg', 'frame_005.jpg', 'frame_011.jpg', 'frame_015.jpg']], '7': [['frame_001.jpg', 'frame_012.jpg', 'frame_008.jpg', 'frame_011.jpg', 'frame_014.jpg']]}

            file_correct, file_total, rows, retrieval_results, total_tool_calls = compare_times(pred_map, gt_map, gt_map_file, qid_to_retrievals)
            file_acc = (file_correct / file_total) if file_total else 0.0

            num_correct_retrieved_in_1_call, num_correct_retrieved_in_2_more_call, num_correct_not_retrieved, num_incorrect_retrieved_in_1_call, num_incorrect_retrieved_in_2_more_call, num_incorrect_not_retrieved = retrieval_results

            # Print a compact per-file report
            print(f"\n== File: {pred_path.name}")
            print(f"   GT : {gt_path.name}")
            print(f"   Correct: {file_correct} / {file_total}  (acc={file_acc:.3f})")
            # align columns 
            print(f"   Correct retrieved in 1 call: {num_correct_retrieved_in_1_call}")
            print(f"   Correct retrieved in 2 or more calls: {num_correct_retrieved_in_2_more_call}")
            print(f"   Correct not retrieved: {num_correct_not_retrieved}")
            print(f"   Incorrect retrieved in 1 call: {num_incorrect_retrieved_in_1_call}")
            print(f"   Incorrect retrieved in 2 or more calls: {num_incorrect_retrieved_in_2_more_call}")
            print(f"   Incorrect not retrieved: {num_incorrect_not_retrieved}")
            # show a few mismatches for debugging
            mismatches = [(qid, p, g) for (qid, p, g, ok, pr, gt) in rows if not ok]
            if mismatches:
                print("   Examples of mismatches:")
                for qid, p, g in mismatches[:]:
                    print(f"     - id={qid}: pred={p} | gt={g}")

            per_file_summaries.append({
                "pred_file": pred_path.name,
                "gt_file": gt_path.name,
                "file_correct": file_correct,
                "file_total": file_total,
                "file_accuracy": f"{file_acc:.6f}",
                "retrieval_results": retrieval_results,
                "total_tool_calls": total_tool_calls,
            })

            writer.writerow(per_file_summaries[-1])

            overall_correct += file_correct
            overall_total += file_total
            
            overall_correct_retrieved_in_1_call += num_correct_retrieved_in_1_call
            overall_correct_retrieved_in_2_more_call += num_correct_retrieved_in_2_more_call
            overall_correct_not_retrieved += num_correct_not_retrieved
            overall_incorrect_retrieved_in_1_call += num_incorrect_retrieved_in_1_call
            overall_incorrect_retrieved_in_2_more_call += num_incorrect_retrieved_in_2_more_call
            overall_incorrect_not_retrieved += num_incorrect_not_retrieved
            overall_tool_calls += total_tool_calls


            # Update per-category aggregates (overall)
            for (qid, _p, _g, ok, pr, gt) in rows:
                cat = gt_cat_map.get(qid, "unknown")
                overall_cat_total[cat] += 1
                if ok:
                    overall_cat_correct[cat] += 1

    overall_acc = (overall_correct / overall_total) if overall_total else 0.0
    overall_random_acc_value = (sum(overall_random_acc) / len(overall_random_acc)) if overall_random_acc else 0.0
    print(f"\nAverage random baseline accuracy (based on #GT answers / #frames): {overall_random_acc_value:.6f}")
    # Print overall
    print("\n================ OVERALL ================")
    print(f"Total Correct: {overall_correct} / {overall_total}")
    print(f"Overall Accuracy: {overall_acc:.6f}")
    print(f"Overall Correct retrieved in 1 call: {overall_correct_retrieved_in_1_call} / {overall_correct}")
    print(f"Overall Correct retrieved in 2 or more calls: {overall_correct_retrieved_in_2_more_call} / {overall_correct}")
    print(f"Overall Correct not retrieved: {overall_correct_not_retrieved} / {overall_correct}")
    print(f"Overall Incorrect retrieved in 1 call: {overall_incorrect_retrieved_in_1_call} / {overall_total - overall_correct}")
    print(f"Overall Incorrect retrieved in 2 or more calls: {overall_incorrect_retrieved_in_2_more_call} / {overall_total - overall_correct}")
    print(f"Overall Incorrect not retrieved: {overall_incorrect_not_retrieved} / {overall_total - overall_correct}")
    print(f"Average tool calls: {overall_tool_calls} / {(overall_total if overall_total else 1):.3f}")
    
    # Print per-category accuracy
    if overall_cat_total:
        print("\nPer-category accuracy:")
        for cat in sorted(overall_cat_total.keys()):
            c = overall_cat_correct.get(cat, 0)
            t = overall_cat_total[cat]
            acc = (c / t) if t else 0.0
            print(f"  - {cat}: {c} / {t} (acc={acc:.6f})")

    # Write per-category CSV
    cat_out_csv = Path(args.out_csv[:-4] + ".category.csv") if not args.category_out_csv else Path(args.category_out_csv)
    with cat_out_csv.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["category", "correct", "total", "accuracy"])
        writer.writeheader()
        for cat in sorted(overall_cat_total.keys()):
            c = overall_cat_correct.get(cat, 0)
            t = overall_cat_total[cat]
            acc = (c / t) if t else 0.0
            writer.writerow({
                "category": cat,
                "correct": c,
                "total": t,
                "accuracy": f"{acc:.6f}",
            })

    print(f"\nCSV summary written to: {out_csv.resolve()}")
    print(f"Category CSV summary written to: {cat_out_csv.resolve()}")

if __name__ == "__main__":
    main()