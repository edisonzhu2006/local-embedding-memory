# Open-Source VLMs in Visuo-Spatial-Temporal Memory: An Evaluation of RAVEN on DARPA TIAMAT Simulation Data

*Independent Work — Draft*
Edison Zhu · Princeton University

---

## Abstract

Retrieval-augmented memory systems for embodied agents, such as **RAVEN** (Hu et al., RSS 2026), store per-frame visual embeddings alongside pose and timestamp in a vector database, then use a vision-language model (VLM) as a tool-using agent to answer queries over long robot trajectories. RAVEN's published evaluation primarily benchmarks closed-source VLMs (Gemini, GPT) and reports that *medium-sized* open VLMs can underperform an embedder-only retrieval floor. In this independent work we (i) extend RAVEN's open-VLM coverage with five locally-served backbones (Gemma 3 12B/27B, Gemma 4 26B/31B, Qwen2.5-VL 3B/32B, Qwen3-VL 32B) inside the DARPA TIAMAT simulation harness, (ii) author a purpose-built 20-question, five-level evaluation suite over a short Habitat trajectory designed to localize *where* along the difficulty curve an LLM stops being optional, and (iii) report quantitative grading (strict ≤2 m, semi-true 2–3.5 m, wrong >3.5 m) and a qualitative diagnostic readout of the agent's retrieval and reasoning behavior. Our results corroborate RAVEN's claim that retrieval is not the bottleneck for open VLMs on this scene, localize the failure to agent-side issues (retrieval loops, missing coordinate plumbing) rather than the VLM itself, and give a practical recommendation for DARPA-style offline deployments.

---

## 1. Introduction

Long-horizon robot deployment requires a memory that is **compact** (can scale to days of experience), **grounded in space and time** (to support navigation), and **information-rich** (to preserve open-vocabulary visual detail). The two dominant families are *explicit* semantic maps (fixed-vocabulary labels over point clouds or grids) and *captioning-based* pipelines (ReMEmbR, Anwar et al.) that translate frames into text before storing embeddings. The former is limited to its vocabulary; the latter introduces a **captioning bottleneck** — textures, layouts, and fine details that cannot be concisely verbalized are lost before retrieval ever begins.

RAVEN (*RetrievAl via Visual Embeddings for Navigation*) bypasses captioning entirely: raw RGB frames are encoded by a multimodal embedder (e.g., QQMM-v2, SigLIP, Seed1.6-Embed) and stored in a vector database as visuo-spatial-temporal triplets \((p_i, t_i, z_i, o_i)\). At query time, a VLM agent iteratively invokes four retrieval primitives — text-based, image-based, time-based, and position-based — until accumulated evidence is sufficient to emit an answer.

The RAVEN paper evaluates primarily closed VLMs (Gemini-2.5-Flash, Gemini-3-Pro, GPT-5.2) on the RAVEN-QA, FindingDory, and NaVQA benchmarks, with partial open-VLM coverage (Gemma3-27B, Qwen3-VL-32B). It reports an important negative result (§5.3): *"medium-sized open VLMs underperform embedder-only baselines"*, and recommends the embedder-only pipeline for offline deployments. This observation is the starting point of our work.

**Contributions.** This independent work makes three scoped contributions:

1. **Open-VLM extension of RAVEN on DARPA TIAMAT simulation data.** We integrate five locally-served open VLMs into the RAVEN harness via Ollama, add per-model YAML configs to the framework, and run each against a Habitat smoke trajectory.
2. **A difficulty-stratified evaluation suite (L1–L5).** We design 20 questions over the scene that specifically separate retrieval-solvable queries from queries that require multi-step, temporal, or negation reasoning, with multi-position ground truth and a three-bucket grading scheme.
3. **Failure-mode diagnosis.** We produce a qualitative readout identifying agent-side failure patterns — most prominently *retrieval loops* where the VLM re-issues identical queries rather than diversifying — that are tractable without retraining.

We emphasize that this work is a **replication-and-extension study**, not a novel method. The retrieval substrate, tool-use loop, and embedder-only baseline are all from the RAVEN paper; our contribution is the open-VLM extension on DARPA sim data and the diagnostic evaluation over it.

---

## 2. Background and Related Work

### 2.1 Memory representations for embodied agents

Classical approaches attach discrete semantic labels to metric maps — point clouds, occupancy grids, or topological graphs — and suffer from closed-vocabulary limitations: out-of-vocabulary objects cannot be retrieved at all. Open-vocabulary methods replace the closed label set with text embeddings derived from captions (e.g., ReMEmbR [Anwar et al.]), but the image-to-text step discards visual detail. A separate line (CLIP-Fields) grounds features in a learned 3D field but requires per-scene optimization, limiting scalability. RNN-based memories struggle with long-term forgetting; retrieval-free long-context VLMs fail to scale past their context window.

### 2.2 RAVEN

RAVEN proposes two design shifts. First, **embeddings, not captions**: modern multimodal encoders are sufficiently aligned with language that raw visual embeddings can be retrieved with text queries directly, skipping the lossy captioning step. Second, **retrieval, not attention**: instead of passing all frames to a long-context VLM, the agent invokes targeted retrieval tools and operates over a *working memory* of retrieved frames, achieving sub-linear retrieval complexity and order-of-magnitude higher memory efficiency.

The RAVEN agent loop maintains context \(R_\tau\), parses the query, and either emits an answer or issues a retrieval call. Retrieval primitives:

- **Text-based.** VLM proposes a query \(\hat{q}\); system computes \(\hat{z}_q = f_\text{enc}(\hat{q})\) and returns top-\(K\) frames by cosine distance.
- **Image-based.** Same, but with an image query.
- **Time-based.** Returns \(K\) *consecutive* frames from a queried timestamp. The paper notes this performs better than nearest-neighbor retrieval in time.
- **Position-based.** Returns \(K\) spatial nearest neighbors to a VLM-proposed \((\hat{x}, \hat{y}, \hat{z})\).

Results: RAVEN with QQMM-v2 outperforms ReMEmbR by up to ~30% on hard queries, maintains quality as memory scales to ~3,000 frames (FindingDory long subset) where VLM-only degrades, and achieves ≥97% success on real Unitree Go1 rollouts. The paper's key concession is that open medium-sized VLMs (Gemma3-27B, Qwen3-VL-32B) can underperform embedder-only retrieval, particularly on Habitat-simulation splits.

### 2.3 Where this work fits

This independent work sits in the RAVEN author-identified gap: *if open-VLM RAVEN is already near the embedder-only floor, what is actually driving the loss?* Is it (a) the retrieval backbone transferring poorly to new simulation data, (b) the VLM's language understanding, (c) the VLM's tool-use strategy, or (d) agent/framework plumbing? Our evaluation is designed to separate these.

---

## 3. Method

### 3.1 Environment and base system

We evaluate inside the **DARPA TIAMAT** competition stack, using the `remembr_static_eval_vlm.py` harness and the `darpa_vlm_prompts` prompt set. All code is a fork of the RAVEN authors' reference implementation; no retrieval, tool, or prompt changes are introduced. Our changes are limited to YAML config additions and eval-harness inputs.

### 3.2 Memory construction

The **smoke_v2** trajectory (546 egocentric Habitat frames, ~7.5 min) covers an indoor residential scene: entry/hallway, living room (TV, sofa, chairs, fireplace), kitchen (cabinets, counters, sink), bathroom (toilet, sink), and a dining transition area. Per-frame poses are in `position_data_habitat_smoke_v2.csv`.

Each frame is embedded with **QQMM-embed-v2** (3584-dim; Hugging Face: `youzexue/QQMM-embed-v2`) and indexed into FAISS as a triplet \((p_i, t_i, z_i, o_i)\) per the RAVEN memory schema. Top-\(K\) is set to 5 throughout, using cosine distance.

### 3.3 Reasoning backbones

Five open-source VLMs are integrated by adding YAML configs that reference Ollama-served model tags. The harness instantiates each as the reasoning LLM inside the RAVEN tool-use loop:

| Size class | Model | Ollama tag |
|---|---|---|
| Small | Gemma 3 1B | `gemma3:1b` |
| | Qwen2.5-VL 3B | `qwen2.5vl:3b` |
| Medium | Gemma 3 12B | `gemma3:12b` |
| Large | Gemma 3 27B | `gemma3:27b` |
| | Gemma 4 26B | `gemma4:26b` |
| | Gemma 4 31B | `gemma4:31b` |
| | Qwen2.5-VL 32B (4-bit) | `qwen2.5vl:32b-q4_K_M` |
| | Qwen3-VL 32B (4-bit) | `qwen3-vl:32b-instruct-q4_K_M` |

Hardware: 2× NVIDIA L40 (Ollama on GPU 0, QQMM embedder on GPU 1).

### 3.4 Evaluation suite (L1–L5)

Public benchmarks (RAVEN-QA, NaVQA, FindingDory) are not available for the DARPA sim trajectory, and the RAVEN-QA category taxonomy (dominant / secondary / reasoning / info-recall / spatial) pools fundamentally different *failure modes*. We author a **20-question, five-level suite** over the smoke_v2 scene designed to isolate specific capabilities.

| Level | Name | Capability probed | Example |
|---|---|---|---|
| L1 | Direct | Object recall from keyword | *"Where is the purple chair?"* |
| L2 | Indirect / attribute | Concept → object mapping | *"Where can I sit down comfortably?"* |
| L3 | Spatial | Relational reasoning over scene | *"What room is closest to the entry?"* |
| L4 | Multi-step | Composed inference (often temporal) | *"Where did the agent start its exploration?"* |
| L5 | Negation / disambiguation | Filter or exclusion over retrieval | *"Find a chair that is NOT in the kitchen."* |

Each level has four questions. Ground truth is multi-position (a question may have 2–3 acceptable \((x, y, z)\) answers) so "right room, wrong exact pose" is gradable distinctly from "wrong room entirely". One question per level is specifically adversarial to a single RAVEN capability — e.g., L4.3 (*"Where did the agent start its exploration?"*) is unanswerable by text retrieval alone and forces the **time-based retriever**; L5.4 (*"Where did the agent spend the most time?"*) forces temporal-statistical reasoning over the trajectory.

**Grading.** Given predicted position \(\hat{p}\) and ground-truth set \(\{p^*_j\}\):
$$d = \min_j \|\hat{p} - p^*_j\|_2$$
- **Strict correct**: \(d \leq 2\) m
- **Semi-true**: \(2 < d \leq 3.5\) m
- **Wrong**: \(d > 3.5\) m

This three-bucket scheme (rather than single 0/1) is critical because dense open VLMs tend to produce the *right room* with a *wrong exact coordinate* — a failure that deserves a different diagnosis than missing the scene entirely.

### 3.5 Baselines

Two baselines, both established by the RAVEN paper:

- **Embedder-only.** CLIP ViT-L-14 and SigLIP-384 with no LLM — top-1 retrieval on the user query embedding. Establishes the retrieval floor.
- **ReMEmbR.** Caption-based baseline using the same 20-question set. (Captions generated by the DARPA-stack captioner shipped in `remembr/captioners/vila_captioner.py`; results are inherited from the RAVEN paper's reported patterns and partially replicated here.)

---

## 4. Experiments

### 4.1 Embedder-only baseline

Mean similarity across 20 queries:

| Model | Dim | Mean similarity |
|---|---|---|
| CLIP ViT-L-14 (OpenAI) | 768 | 0.205 |
| SigLIP-384 (ViT-SO400M-14) | 1152 | 0.108 |

**Observation 1.** Contrastive CLIP achieves ~2× SigLIP's absolute similarity on the DARPA sim data, inverting the relative ordering seen on some web benchmarks. This is consistent with a sim-to-real gap: SigLIP's training distribution is heavier on natural photographs, while contrastive CLIP generalizes more evenly to the rendered Habitat imagery.

**Observation 2.** Absolute similarities (0.07–0.23) sit noticeably below the ~0.3 range typical on real-world benchmarks — a quantitative fingerprint of the sim-to-real distributional shift.

### 4.2 RAVEN + open VLM

Full RAVEN agent runs on QQMM-embed-v2 with three Gemma backbones (counts are over questions where the model returned a gradable coordinate):

| Model | Strict ≤2 m | Semi-true 2–3.5 m | Wrong >3.5 m |
|---|---|---|---|
| Gemma 3 12B | 2 / 17 | 2 / 17 | 13 / 17 |
| Gemma 4 26B | 3 / 19 | 3 / 19 | 13 / 19 |
| **Gemma 4 31B** | **2 / 17** | **6 / 17** | **9 / 17** |

**Observation 3.** Strict accuracy does not improve with scale, but the *semi-true* bucket grows substantially at 31B — the model moves from "wrong room" to "right room, offset pose". This is evidence that the scaling gain for open VLMs in RAVEN is in *coarse spatial grounding*, not fine pose prediction.

**Observation 4.** Independent qualitative grading of retrieved frames (were the correct frames in the top-K retrieval at all, regardless of final coordinate?) gives **13/20 correct, 4/20 partial, 3/20 incorrect** for Gemma 4 31B. This is notably higher than the strict coordinate accuracy — confirming retrieval is not the bottleneck.

### 4.3 Per-level breakdown

Agent behavior by difficulty level (Gemma 4 31B + QQMM):

- **L1 (direct).** Near-ceiling: all four objects are retrieved in top-3. Embedder-only also succeeds on L1, matching RAVEN's §5.3 observation that embedder-only is competitive on easy queries.
- **L2 (indirect).** Strong. The VLM translates *"watch the news"* → `"television"`, *"sit down comfortably"* → `"comfortable seating"`, *"wash my hands"* → `"sink"`. Concept-to-object generalization via QQMM survives the sim-to-real gap.
- **L3 (spatial).** Mixed. The agent produces correct *prose* answers (*"the entryway"*, *"the living room"*) but frequently omits coordinates — a framework plumbing issue (see §5).
- **L4 (multi-step).** The standout result: *"Where did the agent start its exploration?"* is the **only** query in the suite where the model spontaneously pivoted off text retrieval. After `"start of exploration"` returned generic frames, it called `retrieve_from_time("00:00:00")` and landed on `frame_000000`. This is a positive existence proof that open VLMs *can* orchestrate multi-tool strategies; we discuss below why it happens rarely.
- **L5 (negation).** The model reformulates rather than filters: *"flat surface that is NOT a floor"* → `"a flat surface like a table or a counter"`. For *"chair NOT in kitchen"*, the model retrieves chair frames and verbally verifies "not in the kitchen" — behavior that was inconsistent across seeds.

### 4.4 Frame reuse analysis

A striking qualitative finding is that the 31B model consistently re-retrieves the same *landmark* frames for semantically related queries:

| Landmark | Queries it is retrieved for |
|---|---|
| `frame_000238` (red/purple chair) | L1.1, L2.1, L5.1, L5.2 |
| `frame_000628` (toilet) | L1.2, L2.2, L3.2, L4.4 |
| `frame_000753` (TV living room) | L2.3, L2.4, L3.3, L3.4 |

This is interpretable as the VLM maintaining an implicit *landmark map* of the scene: once QQMM localizes a canonical frame for an object concept, the agent re-indexes into it across queries rather than treating each query from scratch. This is consistent with RAVEN's design intent.

---

## 5. Failure-Mode Diagnosis

Localizing *why* the open-VLM RAVEN loses ground to closed models:

1. **Retrieval loops dominate the failure cases.** On questions that fail, the VLM re-issues the *identical* query 4–6 times rather than varying phrasing or switching tools. We observed `"coffee machine"` repeated 6× in one L4 trace and `"entryway"` repeated 6× in another, always with the same top-K frames returned. The tool-use loop has no backoff, no query-diversification prior, and no evidence-accumulation criterion that prevents immediate re-querying.

2. **Prose-without-coordinates.** L3_1 and L3_3 produced correct room-level answers ("the entryway", "the living room") but no \((x, y, z)\). The evaluator — which is strictly coordinate-based — scores these as wrong. This is a *framework* failure, not a VLM failure: a coordinate-extraction fallback (e.g., centroid of retrieved frames' poses) would salvage these cases.

3. **Correct frames, offset coordinates.** The retrieved frames are right but the reported coordinate drifts from GT. This is consistent with a metadata-extraction bug in the answer-formulation path, not reasoning failure.

4. **Texture–surface confusion.** For L5_3 the model correctly read marble texture but mislabeled a marble floor as a countertop. This is an embedder-side ambiguity and cannot be fixed at the agent layer.

5. **Rare multi-tool strategy.** Only 1/20 queries triggered a non-text retriever. The time and position retrievers are under-used, implying the system prompt or tool descriptions do not sufficiently prime the VLM to reach for them when text fails.

---

## 6. Discussion

### 6.1 Why the embedder-only floor is high on L1–L2

The RAVEN paper's negative result — *medium open VLMs ≈ embedder-only* — reproduces on our DARPA sim data for L1 direct queries. The mechanism is visible: QQMM's top-1 retrieval for a keyword query already lands on the correct frame, so a VLM that simply parrots the retrieved frame's location is no better than retrieval-only. **The LLM adds value precisely at L3–L5**, where retrieval alone is insufficient: spatial relationships, multi-step composition, and negation.

### 6.2 What scales with VLM size

Our three Gemma data points suggest strict accuracy is roughly flat (2/17 → 3/19 → 2/17) but *coarse spatial accuracy* (strict + semi-true) improves at 31B (4/17 → 6/19 → 8/17). Paired with the frame-reuse finding, we interpret this as: larger models form and consult a more stable internal landmark map, bringing answers to the right room more often — but the final coordinate plumbing is the bottleneck after that.

### 6.3 Actionable recommendations for the framework

- **Query diversification.** After \(K\) consecutive identical retrievals, force a prompt-level rewrite (e.g., alternate query candidates from the VLM before re-issuing).
- **Coordinate fallback.** When the VLM returns a textual room name with no coordinate, emit the centroid of the retrieved frames' poses. This salvages all observed L3 failures.
- **Tool-use priming.** Add few-shot exemplars in `agent_system_prompt.txt` specifically showing the time and position retrievers being used after an initial text retrieval fails.

None of the above requires fine-tuning the VLM and none touches the retrieval substrate.

### 6.4 Limitations

- **Single scene, 546 frames.** Our eval is one trajectory; the RAVEN paper's scale claims (up to ~3k frames on FindingDory long) are not probed here.
- **Variance not fully quantified.** We report qualitative observations across runs but do not report per-model standard deviations on the DARPA suite (the RAVEN paper reports \(\pm 3\)–\(6\%\) for open VLMs; we expect similar).
- **Sim-to-real gap.** All results are on Habitat-rendered frames; the RAVEN paper's real-robot results (>97% success on Unitree Go1) are separate and not replicated here.
- **Replication, not novel method.** The retrieval substrate, the tool-use loop, the embedder-only baseline, and the L1–L5 taxonomy concept (which echoes RAVEN-QA's category structure) are all established. Our contribution is the open-VLM coverage on new evaluation data and the failure-mode diagnosis.

---

## 7. Future Work

- **Routing fast queries around the VLM agent.** Our per-level breakdown shows embedder-only retrieval is competitive on L1–L2 while L3–L5 is where the VLM agent earns its cost. The natural next step is to route at inference time, but the design choice is non-obvious: **retrieval similarity is not a reliable proxy for answer correctness**. The embedder will confidently retrieve visually similar but semantically wrong frames — a failure mode our DARPA scene exhibits whenever rooms share furniture or lighting — so a similarity-margin gate alone will short-circuit to wrong answers in exactly the cases it should defer. The open question is what gate mechanism is trustworthy enough for offline deployment. Three candidates with different cost–robustness tradeoffs:
    - *Cheap VLM verifier (draft + verify).* Embedder retrieves a top-1 candidate; a single bounded VLM call decides whether it answers the question. Accepted candidates return immediately; rejected ones escalate to the full agent loop. Costs one cheap inference per easy query versus the 4–6 calls plus tool-use overhead of the full loop, and the verifier actually *sees* the candidate so confidently-wrong retrievals get caught.
    - *Multi-signal triangulation.* Combine retrieval margin, a question-shape heuristic (bare entity lookup vs. spatial / multi-step), and an answer-shape parse (does the top-1 frame's metadata fit the expected answer type?). All three must agree before the gate accepts. Preserves a true "no VLM call" fast path, but remains vulnerable to the worst case — a wrong frame from a similar-looking room can pass all three checks.
    - *Learned classifier on the L1–L5 taxonomy.* Stretch goal. Would require a substantially larger labeled set than the 20 questions used here to be defensible, and the same correctness-vs-difficulty conflation issue applies if the classifier only sees the question and not the candidate.
  
  Of these, the cheap-verifier path is the most defensible because it directly addresses the failure mode that dooms similarity-only gates. The others are useful baselines for an ablation.
- **Joint integration with full RAVEN.** The gate, whichever mechanism it ends up being, should be added to the full RAVEN system *jointly* with the §6.3 recommendations (query diversification, coordinate fallback, tool-use priming) rather than as disjoint patches. The changes interact: coordinate fallback shortens the agent path the gate is trying to avoid, query diversification reduces the retrieval-loop failures that would otherwise cause the verifier to reject good candidates, and tool-use priming changes which queries are actually reasoning-required. Evaluating them together on the same suite is the only way to measure their combined lift versus the additive estimate.
- **Variance and scale.** Replicate on multiple TIAMAT trajectories and on a FindingDory-sized memory ($\sim$3k frames) with per-model standard deviations.

---

## 8. Conclusion

We extended the RAVEN evaluation to locally-served open-source VLMs over DARPA TIAMAT simulation data, authored a difficulty-stratified 20-question suite designed to separate retrieval-solvable from reasoning-required queries, and diagnosed open-VLM failure modes as predominantly *agent-side* — retrieval loops, coordinate plumbing gaps, and under-used non-text retrievers — rather than retrieval-substrate or base-VLM failures. The practical implication is that offline, air-gapped deployments of RAVEN on open models can likely close most of the gap to closed-VLM performance without retraining, via three framework-level changes to query diversification, coordinate fallback, and tool-use priming. The embedder-only pipeline remains a competitive fallback for L1 queries, consistent with RAVEN's own recommendation.

---

## References

1. Y. Hu, Z. Zheng, L. Zha, C. Xing, R. Singh, O. Hossain, A. Loquercio, D. Shah. *RAVEN: Long-Horizon Reasoning and Navigation with a Visuo-Spatial-Temporal Memory.* Robotics: Science and Systems, 2026. Code: github.com/zzcnewly/RAVEN. Local copy: [_RSS_2026__RAVEN__Visuo_Spatial_Temporal_Memory_System.pdf](_RSS_2026__RAVEN__Visuo_Spatial_Temporal_Memory_System.pdf).
2. A. Anwar et al. *ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation.* (NaVQA / caption-based baseline used in RAVEN.)
3. K. Yadav et al. *FindingDory: A Benchmark for Memory-Based Navigation.*
4. J. Radford et al. *Learning Transferable Visual Models from Natural Language Supervision.* (CLIP.)
5. X. Zhai et al. *Sigmoid Loss for Language-Image Pre-training.* (SigLIP.)
6. Z. Xue et al. *QQMM: Qwen-based Navigation Memory Model* (QQMM-embed-v2, `youzexue/QQMM-embed-v2`).
7. J. Johnson, M. Douze, H. Jégou. *Billion-Scale Similarity Search with GPUs.* (FAISS.)
8. Habitat AI Challenge / Habitat-Sim.
9. DARPA TIAMAT Competition documentation (internal).

---

## Appendix A. Full Evaluation Question List

See [test_questions_smoke.md](test_questions_smoke.md) for the full 20-question specification with per-question rationale and ground truth positions, and [test_questions_smoke_qa.json](test_questions_smoke_qa.json) for the machine-readable evaluation input.

## Appendix B. Per-Model Output Directories

All runs are reproducible from [outputs/smoke_v2/raven_results/20q/](outputs/smoke_v2/raven_results/20q/):

- [gemma12b_qqmm/](outputs/smoke_v2/raven_results/20q/gemma12b_qqmm/)
- [gemma27b_qqmm/](outputs/smoke_v2/raven_results/20q/gemma27b_qqmm/)
- [gemma27b_vith14/](outputs/smoke_v2/raven_results/20q/gemma27b_vith14/) — CLIP ablation
- [gemma4_26b_qqmm/](outputs/smoke_v2/raven_results/20q/gemma4_26b_qqmm/)
- [gemma4_31b_qqmm/](outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/) — primary result, with [GEMMA4_31B_FINDINGS.md](outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm/GEMMA4_31B_FINDINGS.md)
- [qwen25vl3b_qqmm/](outputs/smoke_v2/raven_results/20q/qwen25vl3b_qqmm/)

Embedder-only baselines: [outputs/custom_eval/](outputs/custom_eval/) with summary [EVAL_REPORT.md](outputs/custom_eval/EVAL_REPORT.md).

## Appendix C. Reproduction Command

```bash
python remembr_static_eval_vlm.py \
  --vlm_config cfgs/vlms/gemma3-27b.yaml \
  --embedder_config cfgs/embedders/qqmm.yaml \
  --agent_config cfgs/agents/raven.yaml \
  --input_folder ./extracted_videos/smoke_full \
  --qa_file ./test_questions_smoke_qa.json \
  --caption_file ./extracted_videos/smoke_full/frames.json \
  --out_dir ./outputs/smoke_v2/raven_results/20q/gemma27b_qqmm \
  --memory_backend faiss --top_k 5 --device cuda
```

Swap `--vlm_config` to any of the five configs in [cfgs/vlms/](cfgs/vlms/).
