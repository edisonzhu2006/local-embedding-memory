# My Contributions

All work below was done against the **DARPA TIAMAT** competition stack, using the **Habitat simulator** as the environment and the `darpa_vlm_prompts` prompt set shipped with the repo. This positions everything inside the DARPA competition evaluation pipeline, not the public RAVEN-QA / FindingDory / NaVQA benchmarks used in the paper.

## Scope

Two parallel threads:

1. **Ran the RAVEN agent on open-source VLMs** (Gemma + Qwen families, served locally via Ollama) where the paper's Table 1 only explored closed models (GPT / Gemini) and a limited open set. This fills out the open-VLM side of the comparison on our own DARPA sim data.
2. **Authored a custom 5-level evaluation suite** on the `smoke_v2` DARPA Habitat trajectory and ran both an **embedder-only baseline** and the full **RAVEN agent** against it.

> Note on the embedder-only baseline: it is a formal baseline in the paper (§5.2, p. 11), and the paper's own conclusion (§5.3, p. 12) is that *"medium-sized open VLMs underperform embedder-only baselines… for offline deployments without Internet access, directly adopting an embedder-only approach may be preferable."* So my embedder-only numbers aren't novel methodology — they're the right comparison floor for the open-VLM RAVEN runs.

## Process / workflow

```
  DARPA Habitat trajectory                       hand-authored eval
  (smoke_v2, 546 frames, 7.5 min)                (20 Q across L1-L5)
            │                                              │
            ▼                                              ▼
   QQMM-embed-v2 → FAISS memory            test_questions_smoke.md
            │                                              │
            └──────────────── RAVEN agent loop ────────────┘
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
         Gemma 3 12B/27B     Qwen2.5-VL 3B/32B     Gemma 4 26B/31B
                (all via local Ollama)
                                    │
                                    ▼
              per-model outputs + retrieved frames →
              qualitative + quantitative write-up
```

Concretely the steps were:

1. **Built the DARPA sim memory.** Used [generate_trial_frames.py](generate_trial_frames.py) on the `smoke_v2` Habitat trajectory, embedded each frame with **QQMM-embed-v2** (3584-dim), and indexed into FAISS alongside `(position, timestamp)` per the RAVEN triplet memory design.
2. **Authored a custom question set** (see next section) targeting specific reasoning failure modes that the standard RAVEN-QA categories collapse together.
3. **Wired open-source VLMs into RAVEN.** Each model was served locally through Ollama and added as a `cfgs/vlms/*.yaml` config, so the same `remembr_static_eval_vlm.py` harness can swap reasoning backbones without code changes.
4. **Ran per-model evaluations** end-to-end: retrieval traces, final answers, and retrieved frames all captured per-question under [outputs/smoke_v2/raven_results/20q/](outputs/smoke_v2/raven_results/20q/).
5. **Embedder-only floor.** Ran CLIP ViT-L-14 and SigLIP-384 on the same 20 questions with no LLM at all, to separate retrieval quality from reasoning quality.
6. **Qualitative readout.** Went through the retrieved-frame folders question-by-question to diagnose *why* a model was right/wrong, not just the coordinates — captured in [RESEARCH_JOURNAL.md](RESEARCH_JOURNAL.md) and [GEMMA4_31B_FINDINGS.md](outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/GEMMA4_31B_FINDINGS.md).

## Custom evaluation set (what I generated and why)

The DARPA smoke trajectory is one 7.5-minute walkthrough of a small residential scene — entry, living room, kitchen, bathroom, dining area. Public benchmarks like NaVQA or RAVEN-QA don't cover this scene, and their question categories (dominant / secondary / reasoning / info-recall / spatial) pool together very different *kinds* of failure. I wanted a more surgical instrument.

So I authored **20 questions across 5 difficulty levels** ([test_questions_smoke.md](test_questions_smoke.md) / [test_questions_smoke_qa.json](test_questions_smoke_qa.json)):

| Level | What it probes | Example | Why it's there |
|---|---|---|---|
| **L1 — Direct** | Can the embedder find a named object at all? | *"Where is the purple chair?"* | Sanity floor — if this fails, retrieval is broken before reasoning even starts. |
| **L2 — Indirect / attribute** | Concept → object mapping (*"watch the news"* → TV) | *"Where can I sit down comfortably?"* | Separates embedders that truly align vision+language from ones that only do literal keyword matching. |
| **L3 — Spatial** | Relations between objects / rooms | *"What room is closest to the entry?"* | Tests whether the agent uses the position retriever or just hopes text retrieval carries spatial meaning. |
| **L4 — Multi-step** | Composed inference, often with a temporal or distance component | *"Where did the agent start its exploration?"* | Specifically designed to force the **time-based retriever** — pure text retrieval can't answer this. |
| **L5 — Negation / disambiguation** | Handling "NOT", "closest to X", "most time" | *"Find a chair that is NOT in the kitchen."* | Tests whether the VLM post-filters retrieval results or just returns top-1. |

Each question has a ground-truth position list (often multiple valid positions) so the evaluator can grade strict (≤2 m), semi-true (2–3.5 m), and wrong (>3.5 m) buckets rather than a single 0/1.

The explicit hypothesis written into [test_questions_smoke.md](test_questions_smoke.md) was:

> Embedder-only retrieval is sufficient for direct object localization but fails on questions requiring reasoning, spatial understanding, or multi-step inference — motivating the full ReMEmbR/RAVEN agent with LLM.

The L1–L5 structure is specifically designed to show the *crossover point* where adding an LLM helps, and by how much — which the aggregated RAVEN-QA categories in the paper don't expose cleanly.

## Open-source VLMs wired in

Each config below was added so `remembr_static_eval_vlm.py` could swap reasoning backbones without code changes. All served locally through Ollama on the DARPA cluster.

| Config | Ollama tag | Purpose |
|---|---|---|
| [cfgs/vlms/gemma3-1b.yaml](cfgs/vlms/gemma3-1b.yaml) | `gemma3:1b` | edge-scale smoke test |
| [cfgs/vlms/gemma3-27b.yaml](cfgs/vlms/gemma3-27b.yaml) | `gemma3:27b` | dense open baseline |
| [cfgs/vlms/qwen25vl-3b.yaml](cfgs/vlms/qwen25vl-3b.yaml) | `qwen2.5vl:3b` | small multimodal model |
| [cfgs/vlms/qwen25vl-32b.yaml](cfgs/vlms/qwen25vl-32b.yaml) | `qwen2.5vl:32b-q4_K_M` | 4-bit quantized Qwen-VL |
| [cfgs/vlms/qwen3vl-32b.yaml](cfgs/vlms/qwen3vl-32b.yaml) | `qwen3-vl:32b-instruct-q4_K_M` | latest open Qwen-VL |

Paired with **QQMM-embed-v2** ([cfgs/embedders/qqmm.yaml](cfgs/embedders/qqmm.yaml)) as the primary embedder and a **CLIP ViT-H-14** ablation ([cfgs/embedders/clip.yaml](cfgs/embedders/clip.yaml)).

## Results on DARPA sim

### RAVEN + Gemma on QQMM (full agent)

| Model | Params | Strict (≤2m) | Semi-true (2–3.5m) | Wrong (>3.5m) |
|---|---|---|---|---|
| Gemma 3 12B | 12B | 2 / 17 | 2 / 17 | 13 / 17 |
| Gemma 4 26B | 26B | 3 / 19 | 3 / 19 | 13 / 19 |
| **Gemma 4 31B** | **31B** | **2 / 17** | **6 / 17** | **9 / 17** |

Strict pin-point accuracy plateaus with scale, but the 31B shifts many answers from "wrong" into the "right room, wrong coordinate" (semi-true) bucket. Qualitative grading of retrieved frames: **13/20 correct, 4/20 partial, 3/20 incorrect**.

### Embedder-only baseline (no LLM)

| Model | Dim | Mean similarity |
|---|---|---|
| ViT-L-14 (OpenAI CLIP) | 768 | 0.205 |
| ViT-SO400M-14-SigLIP-384 | 1152 | 0.108 |

Report in [outputs/custom_eval/EVAL_REPORT.md](outputs/custom_eval/EVAL_REPORT.md). Contrastive CLIP scored ~2× SigLIP on this domain; absolute similarities are below typical benchmarks (~0.3), consistent with the sim-to-real gap between web training data and Habitat renders.

## Qualitative findings (Gemma 4 31B + QQMM on DARPA sim)

- **Frame reuse across related queries.** `frame_000238` (red/purple chair) retrieved for 4 different seating questions; `frame_000628` (toilet) for 4 bathroom questions; `frame_000753` (TV living room) for 4 "watch news / brightest / seating facing TV" questions. Suggests the VLM builds a stable internal map of landmark frames and re-indexes into it rather than treating each query from scratch.
- **Multi-tool strategy — only one question triggered it.** The L4 *"Where did the agent start its exploration?"* was the **only** question where the model pivoted off text retrieval: it first tried `"start of exploration"` (generic results), recognized the failure, and called `retrieve_from_time("00:00:00")`, which landed on `frame_000000`. Evidence that open VLMs *can* use the time retriever — but they don't reach for it unprompted.
- **Functional inference works.** *"Where can I wash my hands?"* → searched `"sink"` and returned the kitchen sink (semantically valid even though GT expected the bathroom). *"Where can I sit down comfortably?"* → searched `"comfortable seating"` rather than just `"chair"`.
- **Negation by reformulation, not filtering.** *"Find a flat surface that is NOT a floor"* → reformulated to `"a flat surface like a table or a counter"`. For *"Find a chair that is NOT in the kitchen"*, the model retrieved chair frames, identified the red chair, and verbally verified it wasn't in the kitchen — but this behavior wasn't reliable across runs.
- **Cleaning-supplies concept retrieval.** For the water-spill question it searched `"cleaning supplies"` and returned a **mop at top-2** (`frame_000226`). Concept → object retrieval via QQMM worked cleanly.
- **Material-vs-surface confusion.** For L5_3, 31B identified "marble countertop" from a frame that actually shows a marble floor — it read the texture correctly but misclassified the surface type.

## Failure modes observed

1. **Retrieval loops.** When the first retrieval is unhelpful, models re-issue the *identical* query (`"coffee machine"` 6×, `"entryway"` 6×) instead of varying strategy. No built-in backoff / diversification.
2. **Right answer, missing coordinates.** L3_1 and L3_3 returned the correct room in prose (*"the entryway"*, *"the living room"*) but no `(x, y, z)`, so the evaluator scores them as wrong.
3. **Position offset vs. correct frames.** The retrieved frames are right, but the reported coordinates drift from ground truth — likely a metadata/extraction issue rather than a reasoning failure.
4. **Texture/surface type confusion.** Surfaces with similar visual textures (marble floor vs. marble countertop) are conflated.

## Takeaways

- On DARPA sim, the retrieval backbone (QQMM-v2 + FAISS) transfers cleanly — the accuracy gap between open-VLM RAVEN and the paper's closed-VLM results is almost entirely on the **reasoning** side, not the retrieval side.
- The paper's claim that open medium-VLMs can underperform embedder-only retrieval (§5.3) is directly testable with my L1–L5 split: it holds on L1 (embedder-only is already near-ceiling) and breaks down on L3–L5 (LLM reasoning is strictly necessary).
- Dense open models (Gemma 4 31B) move answers from "wrong room" to "right room, wrong coordinate" — the next bottleneck is metadata plumbing, not the VLM itself.
- The top fixable agent-side failure is **retrieval loops**: the model needs a backoff policy to diversify queries after `K` identical retrievals, independent of base model. No retraining required.
