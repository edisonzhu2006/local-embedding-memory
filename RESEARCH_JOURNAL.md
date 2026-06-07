# RAVEN Agent Evaluation -- Research Journal

## 2026-04-07: Gemma 4 31B RAVEN Agent Evaluation with QQMM Embedder

### Objective

Evaluate the RAVEN agent (LLM + QQMM retriever) using Gemma 4 models on the Habitat smoke dataset. Compare reasoning quality across model sizes (Gemma 3 12B, Gemma 4 26B, Gemma 4 31B) and assess whether denser VLMs produce better spatial understanding and multi-step reasoning.

### Dataset: `smoke_full`

- **Frames**: 546 egocentric views from Habitat simulator (~7.5 min navigation trajectory)
- **Scene**: Indoor residential -- entry/hallway, living room (TV, sofa, chairs), kitchen (cabinets, sink, counters), bathroom (toilet)
- **Questions**: 20, graded across 5 difficulty levels (L1 direct, L2 indirect, L3 spatial, L4 multi-step, L5 negation)
- **Position data**: Per-frame camera poses from `position_data_habitat_smoke_v2.csv`

### Setup

- **Framework**: `remembr_static_eval_vlm.py` with RAVEN agent (tool-calling LLM + retriever)
- **Embedder**: QQMM-embed-v2 (`youzexue/QQMM-embed-v2`, 3584-dim)
- **Memory backend**: FAISS, top-k=5, cosine similarity with score info
- **LLM**: Gemma 4 31B via Ollama (`gemma4:31b`)
- **GPU**: 2x NVIDIA L40 (Ollama on GPU 0, QQMM on GPU 1)

### Models Compared

| Model | Params | Strict Accuracy (2m) | Semi-True (2--3.5m) | Wrong (>3.5m) |
|-------|--------|---------------------|---------------------|---------------|
| Gemma 3 12B | 12B | 2 / 17 | 2 / 17 | 13 / 17 |
| Gemma 4 26B | 26B | 3 / 19 | 3 / 19 | 13 / 19 |
| **Gemma 4 31B** | **31B** | **2 / 17** | **6 / 17** | **9 / 17** |

Strict 2m accuracy is similar across all models, but Gemma 4 31B has significantly more semi-true results -- it gets to the right area but with a positional offset. Qualitative inspection of retrieved frames shows **13/20 correct, 4/20 partially correct, 3/20 incorrect**.

---

### Key Findings

#### 1. Denser models reason better with larger context

The Gemma 4 31B VLM benefits from a larger effective context window. It retains and cross-references frames from previous retrieval calls within the same question. For multi-tool-call questions, the model synthesizes information across separate retrievals rather than just using the last one.

Frame reuse analysis confirms this -- the same frames are consistently retrieved for semantically related questions:
- `frame_000238` (red/purple chair) retrieved across 4 questions: purple chair, sit down, chair not in kitchen, seating near bathroom
- `frame_000628` (toilet) retrieved across 4 questions: toilet, kitchen-to-bathroom, toilet-to-TV direction, seating near bathroom
- `frame_000753` (living room with TV) retrieved across 4 questions: watch news, brightest area, most furniture, seating facing TV

The model builds a consistent spatial map and reuses the same landmark frames for related queries.

#### 2. Strong multi-step and inference reasoning

Gemma 4 31B handles functional inference questions well:
- **"Where can I wash my hands?"** -- searched for `"sink"`, found the kitchen sink. The reasoning was correct (kitchen sink is a valid place to wash hands), even though the ground-truth expected the bathroom sink. It also answered the kitchen cabinets question the same way, pointing to the same area.
- **"Where can I sit down comfortably?"** -- searched for `"comfortable seating"` (not just "chair"), found the armchair. Demonstrates understanding of the functional intent.
- **"Where can I wash my hands?"** and **"Where are the kitchen cabinets?"** -- both answered correctly pointing to the kitchen area. The reasoning was sound: you can wash your hands near the kitchen sink, which is next to the cabinets.

#### 3. Cleaning supplies retrieval

For **"If I spill water on the kitchen floor, where would I find something to clean it up?"**, the model searched for `"cleaning supplies"` and found a **mop** -- it was the **second-ranked frame** (frame_000226). The model understood the concept of cleaning a spill and visually identified a relevant tool from memory.

#### 4. Temporal reasoning -- exploration start

**"Where did the agent start its exploration?"** was a standout. Gemma 4 31B used a multi-step strategy:
1. First searched by text: `"start of exploration"` (returned generic frames)
2. Recognized the text approach wasn't working
3. Switched to **time-based retrieval**: `retrieve_from_time("00:00:00")`
4. Successfully retrieved frame_000000 -- the very first frame

This is the only question where the model strategically chose a non-text retrieval tool. Previous models (Gemma 3) also got this right, but the reasoning path shows Gemma 4 adapts its tool use when the first approach fails.

#### 5. Environmental inference

Gemma 4 31B made spatial inferences beyond what was explicitly in the frames:
- **TV in the living room**: When asked where to watch the news, it searched for `"television"` and responded *"the television, which is located in the other room"* -- it inferred the room layout.
- **Assumed common-sense spatial relationships**: For "where is the brightest area", it inferred the living room near the sofa and fireplace, even without explicit lighting data.

#### 6. Negation handling

The VLM handles negation constraints:
- **"Find a chair that is NOT in the kitchen"**: Retrieved chair frames, found the red chair, and explicitly verified it was not in the kitchen. Response: *"a red chair that is not in the kitchen."*
- **"Find a flat surface that is NOT a floor"**: Reformulated the query to `"a flat surface like a table or a counter"` -- smart positive rephrasing of a negation.

#### 7. Marble countertop misidentification

For the flat surface question (L5_3), the model identified a **"marble countertop"** from frame_000343/355. The retrieved frames do show a surface with marble-like veining, so the material identification is visually understandable. However, the surface appears to be a marble floor or stair area, not a countertop. The model correctly read the marble texture but misidentified what type of surface it was. This suggests the VLM still struggles with distinguishing surfaces that share visual textures.

---

### Remaining Challenges

1. **Retrieval loops**: Some questions trigger repeated identical queries (e.g., `"coffee machine"` 6 times, `"entryway"` 6 times) when the first retrieval doesn't return useful results. The model doesn't always vary its search strategy.

2. **Text-only answers**: L3_1 ("room closest to entry") and L3_3 ("most furniture") returned correct text answers ("the entryway", "the living room") but without position coordinates, which the evaluation framework expects.

3. **Position offset**: The model often retrieves the correct frames but reports a position that is offset from ground-truth annotations. The qualitative answer is right while the coordinates are wrong -- this is likely a metadata/extraction issue rather than a reasoning failure.

### Output Files

- `outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/` -- Gemma 4 31B results + retrieved frames
- `outputs/smoke_v2/raven_results/20q/gemma4_qqmm/` -- Gemma 4 26B results
- `outputs/smoke_v2/raven_results/20q/gemma12b_qqmm/` -- Gemma 3 12B results
- Full analysis: `outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/GEMMA4_31B_FINDINGS.md`

---

## 2026-03-26: Initial OpenCLIP Embedder-Only Baseline

### Objective

Establish a baseline using pure embedder-only retrieval (no LLM reasoning) with OpenCLIP models on the Habitat smoke dataset.

### Models Tested

| Model | Embedding Dim | Mean Similarity |
|-------|---------------|-----------------|
| ViT-SO400M-14-SigLIP-384 | 1152 | 0.108 |
| ViT-L-14 (OpenAI CLIP) | 768 | 0.205 |

### Key Observations

1. ViT-L-14 (contrastive CLIP) outperformed SigLIP-384 in absolute similarity scores (~2x higher)
2. Direct object queries ("toilet", "hiking boots") localized accurately; reasoning queries performed poorly without LLM
3. Similarity scores were low overall (0.07--0.23) compared to ~0.3 benchmarks, likely due to domain gap between web training images and Habitat sim renders

### Output Files

- `outputs/custom_eval/siglip384/` -- SigLIP-384 with retrieved frame images
- `outputs/custom_eval/vitl14/` -- ViT-L-14 with retrieved frame images
