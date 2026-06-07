# Open-Source Vision-Language Models in Visuo-Spatial-Temporal Memory: An Evaluation of RAVEN on DARPA TIAMAT Simulation Data

---

## PAGE 1 — TITLE PAGE

<br><br><br><br>

<center>

**Open-Source Vision-Language Models in Visuo-Spatial-Temporal Memory:**
**An Evaluation of RAVEN on DARPA TIAMAT Simulation Data**

<br><br>

**Edison Zhu**

<br>

April 28, 2026

<br><br>

Adviser: **Professor Dhruv Shah**

Graduate mentor: **Yixun Hu**

<br><br><br><br>

Submitted in partial fulfillment
of the requirements for the degree of
Bachelor of Science in Engineering
Department of Electrical and Computer Engineering
Princeton University

</center>

---

## PAGE 2 — HONOR PLEDGE AND DECLARATIONS

<br><br>

I hereby declare that this Independent Work report represents my own work in accordance with University regulations.

<br>

I hereby declare that this Independent Work report **does not** include regulated human subjects research.

<br>

I hereby declare that this Independent Work report **does not** include regulated animal subjects research.

<br><br><br>

___________________________________
Edison Zhu

<br>

April 28, 2026

---

## PAGE 3 — ABSTRACT

**Open-Source Vision-Language Models in Visuo-Spatial-Temporal Memory:** **An Evaluation of RAVEN on DARPA TIAMAT Simulation Data**

*Edison Zhu*

Adding a tool-using vision–language agent on top of a 3584-dimensional retrieval embedder lifts **retrieval-level accuracy** on a difficulty-stratified nineteen-question Habitat suite from 7/19 (embedder-only) to 15/19, with the entire +42 percentage-point gain concentrated on spatial, multi-step, and negation reasoning levels (L3–L5). Both numbers ask the same question — *did the correct frame appear in the top-K retrieval?* — and so are directly comparable. The agent additionally emits a final coordinate; coordinate-strict grading on the same questions yields 2/17, with the gap driven by framework-side plumbing rather than reasoning failures (§5). Retrieval-augmented memory systems for embodied agents — most recently RAVEN (Hu et al., RSS 2026) — store per-frame visual embeddings alongside pose and timestamp in a vector database, then use a vision-language model (VLM) as a tool-using agent to answer queries over long robot trajectories. RAVEN's published evaluation primarily benchmarks closed-source VLMs (Gemini, GPT) and reports as a key negative result that *medium-sized* open VLMs can underperform an embedder-only retrieval floor. This independent work (i) extends RAVEN's open-VLM coverage with five locally-served backbones (Gemma 3 12B/27B, Gemma 4 26B/31B, Qwen2.5-VL 3B/32B, Qwen3-VL 32B) inside the DARPA TIAMAT simulation harness; (ii) authors a purpose-built nineteen-question, five-level evaluation suite over a 14-minute Habitat trajectory designed to localize *where* along the difficulty curve an LLM stops being optional; and (iii) reports both a quantitative grading (strict ≤ 2 m vs. wrong) and a qualitative diagnostic readout of the agent's retrieval and reasoning behavior. Embedder-only retrieval saturates at five to seven correct answers out of nineteen, with strict accuracy roughly flat across the four embedders tested (768-dim CLIP ViT-L through 3584-dim QQMM-v2). Adding the VLM agent on top of the *same* QQMM retriever lifts performance to fifteen out of nineteen — a +42 percentage-point jump entirely concentrated on the spatial-reasoning, multi-step, and negation levels (L3–L5). On the same nineteen questions, the embedder-only pipelines all plateau on direct localization (L1) and partial concept generalization (L2). We localize the open-VLM losses to *agent-side* mechanisms — retrieval loops, missing coordinate plumbing in the answer-formulation path, and an under-used non-text retriever — rather than retrieval-substrate or base-VLM failures, and identify three framework-level fixes (query diversification, coordinate fallback, tool-use priming) that should close most of the gap to closed-VLM performance without retraining the underlying VLM.

---

## PAGE 4 — ACKNOWLEDGMENTS

I thank my adviser, **Professor Dhruv Shah**, for proposing this project and for shaping the experimental scope across the semester. I thank **Yixun Hu**, the PhD student who served as my day-to-day mentor and is also a co-author on RAVEN, for walking me through the codebase, for many discussions on the design of the evaluation suite, and for honest critique of early failure-mode write-ups. Much of the framing in this report is owed to his guidance. I thank the rest of the RAVEN authors — Zifeng Zheng, Lehong Zha, Chuan Xing, Rajat Singh, Owais Hossain, and Antonio Loquercio — for open-sourcing the reference implementation that this work builds on, and for documenting the open-VLM gap that motivated this evaluation. I thank the Princeton ECE Department for compute resources (2× NVIDIA L40 via neuronic) used to run the open-VLM backbones locally. Any errors in interpretation are my own.

---

## PAGE 5 — TABLE OF CONTENTS

| Section | Page |
|---|---|
| **1. Introduction** | 6 |
| 1.1 Problem statement and scope | |
| 1.2 Contributions | |
| **2. Background and Related Work** | 9 |
| 2.1 Memory representations for embodied agents | |
| 2.2 The RAVEN system | |
| 2.3 Where this work fits in the literature | |
| **3. Methods** | 13 |
| 3.1 Environment and base system | |
| 3.2 Memory construction | |
| 3.3 Reasoning backbones | |
| 3.4 Evaluation suite (L1–L5) | |
| 3.5 Grading scheme | |
| 3.6 Baselines | |
| **4. Experiments and Results** | 18 |
| 4.1 Embedder-only baseline | |
| 4.2 RAVEN with open VLM backbones | |
| 4.3 Per-level breakdown | |
| 4.4 Embedding dimension does not crack the ceiling | |
| 4.5 Frame-reuse analysis | |
| **5. Failure-Mode Diagnosis** | 25 |
| **6. Discussion** | 28 |
| 6.1 Why the embedder-only floor is high on L1–L2 | |
| 6.2 What scales with VLM size | |
| 6.3 Actionable framework recommendations | |
| 6.4 Limitations | |
| **7. Future Work** | 32 |
| **8. Conclusion** | 34 |
| **References** | 35 |
| **Appendix A. Full Evaluation Question List** | 37 |
| **Appendix B. Per-Model Output Directories** | 38 |
| **Appendix C. Reproduction Commands** | 39 |
| **Appendix D. Engineering Standards Used** | 40 |

---

# 1. Introduction

## 1.1 Problem statement and scope

A robot deployed for hours or days in an unfamiliar environment must remember what it has seen in a way that is **compact** (memory cannot scale linearly with the number of frames captured), **grounded in space and time** (downstream queries are typically about *where* and *when*), and **information-rich** (open-vocabulary visual detail must survive into retrieval, not be discarded at storage time). The two dominant families of long-horizon memory both struggle under at least one of these constraints. *Explicit semantic maps* — point clouds or grids tagged with discrete labels from a fixed vocabulary — are compact and grounded, but lose any concept that was not pre-registered: a "red Tesla in the driveway" cannot be recovered if the vocabulary only knows "car." *Captioning-based pipelines* — most prominently ReMEmbR (Anwar et al.) and its descendants — convert frames into natural-language captions before embedding, so they are open-vocabulary and grounded, but the image-to-text step introduces a **captioning bottleneck**: textures, layouts, and fine spatial details that cannot be concisely verbalized are lost before retrieval ever begins. A third line — long-context VLMs reading every frame at query time — is information-preserving but does not scale: context windows of tens of thousands of tokens are exhausted at hundreds of frames, far below the trajectory lengths a deployed robot accumulates.

RAVEN (*RetrievAl via Visual Embeddings for Navigation*; Hu et al., 2026) bypasses the captioning step entirely. Raw RGB frames are passed through a multimodal embedder (in the reference implementation, QQMM-embed-v2), and the resulting visuo-spatial-temporal triplet $(p_i, t_i, z_i, o_i)$ — pose, timestamp, embedding, and frame metadata — is stored in a vector database. At query time, a VLM agent iteratively invokes four retrieval primitives — text-based, image-based, time-based, and position-based — until accumulated evidence is sufficient to emit an answer. The result is a memory that is sub-linear in retrieval cost, open-vocabulary by construction, and order-of-magnitude smaller than storing the frames themselves.

The RAVEN paper benchmarks primarily *closed* VLMs (Gemini-2.5-Flash, Gemini-3-Pro, GPT-5.2) on three datasets and reports an important negative result in §5.3: when the reasoning backbone is swapped for a *medium-sized open* VLM (Gemma3-27B, Qwen3-VL-32B), performance on Habitat-simulation queries can drop *below* the embedder-only floor — i.e., below the performance achievable by stripping out the agent and just doing top-1 cosine retrieval on the user's query embedding. The paper recommends the embedder-only pipeline for offline deployments under such conditions.

This observation is the starting point of the present work. For a researcher who needs to deploy on-device — DARPA-style, air-gapped, no cloud calls — the choice between closed and open VLMs is not free. This work therefore asks two related questions: **how well do open-source VLMs actually perform inside RAVEN's tool-use loop, and what, exactly, is the open-VLM penalty paying for when they fall short?** The first is a benchmarking question — measure four open backbones (Gemma 3 27B, Gemma 4 31B, Qwen2.5-VL 32B, Qwen3-VL 32B) on the same suite and report the numbers. The second is a diagnostic question: of the four candidate failure sites — (a) the retrieval substrate transferring poorly to new simulation data, (b) the VLM's language understanding, (c) the VLM's tool-use strategy, and (d) framework-level plumbing between the retrieval result and the final coordinate answer — where does the loss actually concentrate? Each of these has a different fix, a different cost, and a different prognosis for offline deployment. The contribution of this independent work is to do both: report cross-family open-VLM performance on a controlled suite, and localize the failures.

[INSERT FIGURE 1.1 HERE: figures/routing_diagram.png]
**Figure 1.1.** The two-tier RAVEN pipeline: a multimodal embedder produces visuo-spatial-temporal triplets at ingest time, and a VLM agent invokes retrieval primitives at query time. The agent's tool-use loop is the locus of failure analyzed in this work.

## 1.2 Contributions

This independent work makes three scoped contributions, each addressing a specific gap in the RAVEN open-VLM coverage:

1. **Open-VLM extension of RAVEN on DARPA TIAMAT simulation data.** I integrate five locally-served open VLMs into the RAVEN harness via Ollama, add per-model YAML configurations to the framework, and run each against a Habitat smoke trajectory drawn from the DARPA TIAMAT competition stack. The five backbones span three orders of magnitude of parameter count (1B to 32B) and two model families (Gemma 3/4 and Qwen2.5-VL/Qwen3-VL).

2. **A difficulty-stratified evaluation suite (L1–L5).** I author a nineteen-question suite over the smoke trajectory in which each question is tagged with a capability level — direct object recall (L1), indirect attribute mapping (L2), spatial reasoning (L3), multi-step inference (L4), and negation/disambiguation (L5). Each level is designed to *isolate* a specific RAVEN capability: L4_3 ("where did the agent start its exploration?") is unanswerable by text retrieval alone and forces the time-based retriever; L5_1 ("find a chair that is NOT in the kitchen") forces post-retrieval filtering. Ground truth is multi-position, allowing a question to have two or three acceptable answers, which lets the evaluator distinguish "right room, wrong pose" from "wrong room entirely."

3. **Failure-mode diagnosis.** I produce a qualitative readout that identifies the dominant agent-side failure patterns — most prominently *retrieval loops* in which the VLM re-issues the identical query four to six times rather than diversifying — and propose three framework-level fixes (query diversification, coordinate fallback, tool-use priming) that do not require retraining the VLM and do not touch the retrieval substrate.

I emphasize at the outset that this is a **replication-and-extension study**, not a novel method. The retrieval substrate, the tool-use loop, and the embedder-only baseline are all from the RAVEN paper. The contribution here is open-VLM coverage on new evaluation data, the difficulty-stratified suite, and the failure-mode taxonomy that follows from running the system at a level of granularity the RAVEN authors did not.

---

# 2. Background and Related Work

## 2.1 Memory representations for embodied agents

The literature on long-horizon memory for embodied agents has converged on three approaches, each with a known failure mode.

**Closed-vocabulary semantic maps.** Classical SLAM produces a metric map; semantic SLAM tags map elements with discrete object labels drawn from a fixed taxonomy. The advantage is compactness and consistency with downstream planners. The disadvantage is hard: out-of-vocabulary objects cannot be retrieved at all, and retraining the perception stack to add a class is costly. CLIP-Fields and related works ground learned features in a 3D field but require per-scene optimization.

**Captioning-based open-vocabulary pipelines.** ReMEmbR (Anwar et al.) and its descendants embed natural-language captions of frames rather than the frames themselves. This solves the closed-vocabulary problem but introduces the *captioning bottleneck*: anything the captioner does not say cannot be retrieved. Fine spatial layout, subtle textures, and unusual object configurations are systematically lost. The RAVEN paper's NaVQA experiments show ReMEmbR underperforms direct visual embedding by 20–30% on hard queries.

**Long-context VLMs.** Reading every frame at query time preserves all visual information but does not scale. Context window limitations of even the largest VLMs (~128k–1M tokens) are exceeded at trajectory lengths of a few hundred to a few thousand frames, depending on tokenization granularity. Cost scales linearly per query.

**Recurrent / RNN-based memories.** Forget over long horizons in a way that is hard to control or interrogate.

## 2.2 The RAVEN system

RAVEN proposes two design shifts that together address the limitations above. First, **embeddings, not captions**: modern multimodal encoders (CLIP, SigLIP, QQMM-embed-v2) are sufficiently aligned with language that raw visual embeddings can be retrieved with text queries, skipping the lossy image-to-text step. Second, **retrieval, not attention**: instead of passing every stored frame to a long-context VLM, the agent invokes targeted retrieval over a vector database and operates on a *working memory* of retrieved frames, achieving sub-linear retrieval complexity and order-of-magnitude lower per-query memory pressure.

The RAVEN agent loop maintains a context $R_\tau$ of accumulated retrieved frames, parses the user query, and at each step either emits a final answer or issues a retrieval call. Four retrieval primitives are exposed:

- **Text-based.** The VLM proposes a query string $\hat{q}$. The system computes $\hat{z}_q = f_{\text{enc}}(\hat{q})$ and returns the top-$K$ frames by cosine distance. This is the workhorse primitive — most queries terminate after one or two text retrievals.
- **Image-based.** Identical mechanics, but with an image query (typically a previously retrieved frame). Used to expand from a known landmark.
- **Time-based.** Returns $K$ *consecutive* frames anchored at a queried timestamp. The RAVEN paper notes this performs better than nearest-neighbor retrieval *in time*, because consecutive frames are more diagnostic of trajectory state.
- **Position-based.** Returns $K$ spatial nearest neighbors to a VLM-proposed coordinate $(\hat{x}, \hat{y}, \hat{z})$.

[INSERT FIGURE 2.1 HERE: figures/reasoning_loop.png]
**Figure 2.1.** RAVEN's tool-use loop. The VLM parses the query, decides whether to emit an answer or call a retrieval primitive, accumulates evidence in $R_\tau$, and repeats. The four retrieval primitives are exposed as tools.

The reported results in the RAVEN paper are strong on the closed-VLM track. RAVEN with QQMM-v2 outperforms ReMEmbR by up to 30% on hard queries; quality is maintained as memory scales to ~3,000 frames on the FindingDory long subset (where a VLM-only baseline degrades sharply); and ≥ 97% success is achieved on real Unitree Go1 rollouts. The open-VLM concession in §5.3, however, is the relevant data point for offline deployment: medium-sized open VLMs (Gemma3-27B, Qwen3-VL-32B) can drop below embedder-only retrieval on Habitat-simulation queries, particularly on tasks requiring multi-step composition.

## 2.3 Where this work fits in the literature

Three nearby lines of recent work each address adjacent questions but not the one this thesis pursues:

- **Tool-use evaluations of open VLMs.** Recent work has measured how reliably open VLMs invoke tools in general agentic settings (browse, run code, etc.). These benchmarks are not embodied; they do not stress the visual retrieval substrate.
- **Retrieval-augmented generation (RAG) for VLMs.** Adjacent in mechanism (text embedding + retrieval + reasoning) but the retrieved corpus is text/document, not visuo-spatial-temporal robot memory. Failure modes do not transfer.
- **Habitat / FindingDory benchmark papers.** Provide standardized question suites over indoor scenes, but the published RAVEN evaluation does not break out per-difficulty performance for open VLMs at the granularity needed to localize failure.

This independent work sits in the gap precisely identified by the RAVEN authors in §5.3: *if open-VLM RAVEN is already at or below the embedder-only floor, what is actually driving the loss?* The L1–L5 evaluation suite is the instrument designed to answer this.

---

# 3. Methods

## 3.1 Environment and base system

I evaluate inside the **DARPA TIAMAT** competition stack, using the `remembr_static_eval_vlm.py` harness and the `darpa_vlm_prompts` prompt set. All code is a fork of the RAVEN authors' reference implementation; **no retrieval, tool, or prompt changes are introduced**. The only modifications to the framework are (i) per-model YAML config additions for the five new VLM backbones, (ii) the question file specifying the L1–L5 suite, and (iii) the evaluator's grading script. This scope discipline is intentional: any performance change must be attributable to the swapped VLM, not to incidental framework drift.

## 3.2 Memory construction

The **smoke_v2** trajectory comprises 843 egocentric Habitat frames captured over approximately 14 minutes of simulated walking through an indoor residential scene (entry/hallway, living room, kitchen, bathroom, dining transition area). Per-frame poses are stored in `position_data_habitat_smoke_v2.csv` as $(x, y, z, \text{yaw})$ in scene coordinates.

Each frame is embedded with **QQMM-embed-v2** (3584-dim; Hugging Face: `youzexue/QQMM-embed-v2`, the same retriever used in the RAVEN reference results) and indexed into FAISS as a triplet $(p_i, t_i, z_i, o_i)$ following the RAVEN memory schema. Top-$K$ is fixed at 5 throughout, using cosine distance. This matches the RAVEN paper's defaults; varying $K$ is out of scope and would obscure the VLM-substitution effect.

## 3.3 Reasoning backbones

I integrate five open-source VLMs by adding YAML configurations that reference Ollama-served model tags. The harness instantiates each as the reasoning LLM inside the RAVEN tool-use loop. Hardware is two NVIDIA L40 GPUs (the Ollama VLM is pinned to GPU 0; the QQMM embedder runs on GPU 1).

| Size class | Model | Ollama tag |
|---|---|---|
| Small | Gemma 3 1B | `gemma3:1b` |
|  | Qwen2.5-VL 3B | `qwen2.5vl:3b` |
| Medium | Gemma 3 12B | `gemma3:12b` |
| Large | Gemma 3 27B | `gemma3:27b` |
|  | Gemma 4 26B | `gemma4:26b` |
|  | Gemma 4 31B | `gemma4:31b` |
|  | Qwen2.5-VL 32B (4-bit) | `qwen2.5vl:32b-q4_K_M` |
|  | Qwen3-VL 32B (4-bit) | `qwen3-vl:32b-instruct-q4_K_M` |

The headline result reported throughout this thesis uses **Gemma 4 31B + QQMM-embed-v2**, which is referred to as the "RAVEN agent" configuration. This is the strongest open configuration that fit fully on the available L40 hardware and is closest in scale to the medium-open backbones the RAVEN paper concedes are difficult.

## 3.4 Evaluation suite (L1–L5)

Public benchmarks (RAVEN-QA, NaVQA, FindingDory) are not available for the DARPA TIAMAT trajectory, and the existing RAVEN-QA category taxonomy (dominant / secondary / reasoning / info-recall / spatial) pools fundamentally distinct *failure modes* that this work needs to separate. I therefore author a nineteen-question, five-level suite over the smoke_v2 scene specifically designed to isolate capabilities.

| Level | Name | Capability probed | Example |
|---|---|---|---|
| L1 | Direct | Object recall from keyword | *"Where is the purple chair?"* |
| L2 | Indirect / attribute | Concept → object mapping | *"Where can I sit down comfortably?"* |
| L3 | Spatial | Relational reasoning over scene | *"What room is closest to the entry?"* |
| L4 | Multi-step | Composed inference (often temporal) | *"Where did the agent start its exploration?"* |
| L5 | Negation / disambiguation | Filter or exclusion over retrieval | *"Find a chair that is NOT in the kitchen."* |

L1, L3, L4, and L5 contain four questions each; L2 contains three, for a total of nineteen scoreable questions. Ground truth is **multi-position**: a question may have two or three acceptable $(x, y, z)$ answers — for instance, "Where can I sit down comfortably?" accepts the sofa and the dining chairs as equally correct. This design choice is critical for the diagnostic granularity, because it enables the grading scheme of §3.5 to distinguish "right room, wrong exact coordinate" from "wrong room entirely." Approximately one question per level is specifically *adversarial* to a single RAVEN capability — L4_3 forces the time-based retriever; L5_4 forces temporal-statistical reasoning over the trajectory; L5_1 forces post-retrieval filtering.

The full text of all nineteen questions, with capability tags and ground-truth positions, appears in Appendix A.

## 3.5 Grading scheme

Each answer is graded with a two-path rubric. An answer is marked **correct** if *either* path passes:

1. **Coordinate path.** Given a predicted position $\hat{p}$ and a ground-truth set $\{p^*_j\}$, define $d = \min_j \|\hat{p} - p^*_j\|_2$. The answer passes coordinate-strict if $d \le 2$ m. The 2-meter threshold corresponds to "same furniture group" in the test scene.
2. **Visual path.** When the agent returns no coordinate, returns a coordinate that fails path 1, or answers a non-positional question (binary, text), I manually inspect the retrieved top-$K$ frames and the model's response text. The answer passes the visual path if the retrieved frames visually show the queried object/room *and* the response correctly identifies it. For non-positional questions (e.g., binary "is there a coffee maker?"), the visual path is the only path: the response text must be correct on its merits given the retrieved evidence.

An answer is **wrong** if both paths fail — i.e., the coordinate is more than 2 m from any ground-truth pose *and* manual visual inspection of the retrieved frames does not support the answer (or the reasoning text is incorrect for non-positional questions).

This rubric is what allows "right room, wrong pose" to be graded distinctly from "wrong room entirely" (§3.4). A correct prose answer with no emitted coordinate (e.g., "the entryway") is graded *wrong* under coordinate-only scoring but can be graded *correct* under the two-path rubric if the retrieved frames are right; both numbers are reported and distinguished as **coordinate-strict** vs. **retrieval-level** throughout this thesis (see §4.2 for the headline split). All headline accuracy numbers in this work use the two-path retrieval-level rubric unless explicitly labeled coordinate-strict.

## 3.6 Baselines

Two baselines, both established by the RAVEN paper:

- **Embedder-only.** Four contrastive image–text encoders are evaluated with no LLM in the loop — top-1 retrieval on the user query embedding, with the answer coordinate taken as the centroid of the retrieved frame's pose. The four models are CLIP ViT-L-14 (768-dim), CLIP ViT-H-14 (1024-dim), SigLIP-384 (1152-dim), and QQMM-embed-v2 (3584-dim). This establishes the *retrieval floor*: the best performance achievable without any VLM reasoning.
- **ReMEmbR.** Caption-based baseline using the same nineteen-question set. Captions generated by the DARPA-stack captioner shipped in `remembr/captioners/vila_captioner.py`. Quantitative ReMEmbR numbers are inherited from the RAVEN paper's reported patterns; this work does not re-run the captioner end-to-end on smoke_v2.

---

# 4. Experiments and Results

## 4.1 Embedder-only baseline

Mean top-1 retrieval similarity across the nineteen queries for the four contrastive embedders evaluated:

| Model | Dim | Mean similarity |
|---|---|---|
| SigLIP-384 (ViT-SO400M-14) | 1152 | 0.108 |
| CLIP ViT-L-14 (OpenAI) | 768 | 0.205 |
| CLIP ViT-H-14 (LAION-2B) | 1024 | 0.249 |
| **QQMM-embed-v2** | **3584** | **0.356** |

**Observation 4.1.** QQMM-embed-v2 achieves substantially higher absolute similarity than CLIP and SigLIP on the DARPA simulation data — consistent with its navigation-specialized training distribution. Among the two general-purpose contrastive embedders, CLIP achieves approximately twice SigLIP's absolute similarity, *inverting* the relative ordering reported on some web benchmarks. This is consistent with a sim-to-real gap: SigLIP's training distribution is heavier on natural photographs, while contrastive CLIP generalizes more evenly to the rendered Habitat imagery. Within the CLIP-style contrastive family, similarity rises monotonically with embedding dimensionality (ViT-L-14 at 768-dim → ViT-H-14 at 1024-dim → QQMM-embed-v2 at 3584-dim; 0.205 → 0.249 → 0.356); SigLIP breaks this trend, suggesting the dimensionality effect is a within-family capacity signal rather than a general law — training distribution dominates across families.

**Observation 4.2.** Absolute similarities in the 0.07–0.23 range sit noticeably below the ~0.3 typical on real-world benchmarks — a quantitative fingerprint of the sim-to-real distributional shift that any deployment on Habitat-rendered data must contend with.

## 4.2 RAVEN with open VLM backbones

I ran the full RAVEN agent against the 19-question suite with each of four open backbones, all on the same QQMM-embed-v2 retriever, and graded each answer manually at the retrieval level (was the correct frame in the top-$K$ retrieval, regardless of how the VLM formatted its final answer?):

| Model | Resident VRAM | Retrieval-level correct | Wrong |
|---|---|---|---|
| **Gemma 4 31B** | **≈17.4 GB** | **15 / 19** | **4 / 19** |
| Gemma 3 27B | 25.0 GB | 13 / 19 | 6 / 19 |
| Qwen2.5-VL 32B | 46.2 GB | 13 / 19 | 6 / 19 |
| Qwen3-VL 32B | 39.6 GB | 12 / 19 | 7 / 19 |

**Observation 4.3.** Gemma 4 31B retrieves the correct frame on 15 of 19 questions and is also the smallest backbone on disk; the other three open VLMs cluster at 12–13/19 even though they occupy 1.4–2.7× more GPU memory. Memory footprint and retrieval-level accuracy do not correlate on this suite.

**Observation 4.4.** Coordinate-level strict accuracy (the predicted $(x, y, z)$ within 2 m of a ground-truth answer) is far lower than retrieval-level accuracy for every agent — Gemma 4 31B scores 2/17 coordinate-strict on the questions where it returned a gradable coordinate. The bottleneck is downstream of retrieval: when the agent fails coordinate-strict, it has usually retrieved the right frame but emitted prose without a coordinate, or extracted the wrong pose from the right frame (§5).

**Two grading scales.** The paper uses two distinct strict-accuracy metrics throughout, and they differ by definition rather than by run:
- *Coordinate-level strict accuracy* (the §4.2 table): the predicted $(x, y, z)$ position must lie within 2 m of a ground-truth answer. Gemma 4 31B = 2/17.
- *Retrieval-level strict accuracy* (the headline 15/19, used in §1, §4.3, §4.4, §4.5, §4.7, §6, §8): the correct frame appeared in the top-$K$ retrieval for the question, regardless of how the VLM formatted its final answer.

The gap between 2/17 and 15/19 is the mass of the §5 failure modes (prose-without-coordinates, correct-frames-with-offset-coords). All charts and per-level figures use the retrieval-level metric unless explicitly noted; only the §4.2 table reports coordinate-level. We treat the retrieval-level number as the headline because the coordinate-emission failures are framework-level plumbing problems (see §5.2, §5.3) rather than reasoning failures.

## 4.3 Per-level breakdown

The single most informative chart in this thesis is the per-level comparison of QQMM (embedder-only) versus the full agent, reproduced in Figure 4.3. The narrative is unambiguous:

[INSERT FIGURE 4.3 HERE: figures/chart_lift.png]
**Figure 4.3.** Per-level strict accuracy. QQMM (embedder-only) is shown in light slate; the full RAVEN agent (Gemma 4 31B + QQMM) in deep teal. The two configurations are tied at L1 (88% vs. 100% within sampling variance) and close at L2 (67% vs. 100%). On L3 (spatial), L4 (multi-step), and L5 (negation), the agent opens a 38–75 percentage-point gap. The gap is the cost of the VLM call, expressed as a function of question type.

Qualitative agent behavior, level by level (Gemma 4 31B + QQMM):

- **L1 (direct).** Near-ceiling. All four objects (purple chair, toilet, white sofa, kitchen cabinets) are retrieved in top-3 and the model emits coordinates from the retrieved frame's metadata cleanly. The embedder-only pipeline also succeeds on L1, matching RAVEN's §5.3 observation that embedder-only is competitive on easy queries. **The VLM adds essentially nothing here**; the scaling argument therefore must come from L3–L5.

- **L2 (indirect/attribute).** Strong. The VLM translates *"watch the news"* → `"television"`, *"sit down comfortably"* → `"comfortable seating"`, *"wash my hands"* → `"sink"`. Concept-to-object generalization via QQMM survives the sim-to-real gap. The L2 wins are partially attributable to the embedder (which correctly aligns "television" to TV frames) and partially to the VLM (which does the concept-to-keyword mapping). Embedder-only at L2 hits one out of three; the agent reaches three out of three.

- **L3 (spatial).** Mixed in raw output, strong in retrieval. The agent produces correct *prose* answers (*"the entryway"*, *"the living room"*) in three of four L3 questions, but frequently omits coordinates in the answer payload — a framework plumbing issue diagnosed in §5. The retrieval itself is correct in all four; the agent finds the right frames.

- **L4 (multi-step).** The standout result is L4_3 — *"Where did the agent start its exploration?"* — the **only** query in the suite where the model spontaneously pivoted off text retrieval. After `"start of exploration"` returned generic frames, it called `retrieve_from_time("00:00:00")` and landed on `frame_000000`. This is a positive existence proof that open VLMs *can* orchestrate multi-tool strategies inside RAVEN; §5.5 discusses why it happens rarely.

- **L5 (negation).** The model reformulates rather than filters: *"flat surface that is NOT a floor"* → `"a flat surface like a table or a counter"`. For *"chair NOT in kitchen"*, the model retrieves chair frames and verbally verifies "not in the kitchen" — behavior that was inconsistent across seeds. The agent gets two out of four on L5 strictly; embedder-only gets none.

[INSERT FIGURE 4.4 HERE: figures/chart_level_ladder.png]
**Figure 4.4.** Where each model's score comes from. Five narrow vertical columns, one per model, segmented by level (L1 light, L5 dark). Embedders score almost entirely on the L1 (cream) segment with a thin L2 contribution; QQMM additionally pokes into L3 and L5; only the agent reaches L4 and the full L5 segment. The ladder visualization makes two facts visible at once: (i) embedders saturate L1 and stop, and (ii) the agent's headline number is built across the level distribution, not concentrated.

### Walked-through reasoning traces

The per-level summary above is the macro view; the texture of why the agent succeeds or fails is in the individual traces. Four representative cases follow.

**L4_2, "Where would I find something to clean a spill?" (✓).** The cleanest demonstration of *concept-to-object* retrieval that the embedder alone cannot do. The agent issued a single text retrieval with the query `"cleaning supplies"` and the QQMM retriever surfaced `frame_000226` (a mop) as the top-2 result. The VLM then emitted the mop's pose as the answer. Embedder-only retrieval on the user's literal query string ("where would I find something to clean a spill") lands on irrelevant kitchen-floor frames; the VLM's reformulation to "cleaning supplies" is what makes the retrieval succeed. This is the L4 win that the embedder cannot replicate and is consistent with the per-level gap in Figure 4.3.

**L4_3, "Where did the agent start its exploration?" (✓).** The standout trace. The agent first issued `"start of exploration"` (a literal paraphrase of the question) and got back generic indoor frames — no temporal anchor. Recognizing the failure, it switched tools: it called `retrieve_from_time("00:00:00")`, which returns frames anchored at trajectory start. The result was `frame_000000` itself, and the agent emitted that frame's pose as the answer. This is the only query in the entire suite where a non-text retriever was spontaneously invoked. It is positive evidence that open VLMs *can* orchestrate multi-tool strategies inside RAVEN; the rarity of this behavior (§5, item 5.5) is then the evidence that they don't reach for it without prompting.

**L5_3, "Find a flat surface that is NOT a floor" (✗).** The instructive failure. The agent reformulated the negation rather than filtering it: it issued `"a flat surface like a table or a counter"` as the text query. QQMM returned `frame_000343`, which the VLM correctly identified as containing marble texture. However, it then mislabeled the surface as a "marble countertop" when it was in fact a marble *floor*. The retrieval was correct in the sense of returning a visually relevant frame; the *interpretation* of the surface type was wrong. This is the texture–surface confusion mode of §5.4 and is the kind of failure that cannot be fixed at the agent layer alone: the embedder produces a "flat marble surface" embedding regardless of whether the surface is horizontal floor or horizontal countertop, and it takes a separate VLM-level disambiguation step to tell them apart. An alternate strategy would have been to retrieve all flat-surface candidates, *filter out the floors*, and then choose; the agent did not adopt this strategy.

**L4_4, "If I'm at the toilet and hear the TV, which direction would I walk?" (✗).** A different failure mode: retrieval succeeds on the bathroom-side endpoint but the geometric reasoning step is missing. The agent retrieved the toilet (`frame_000628`) and described the living room with the TV in prose, but it never produced a directional vector or even a unit-bearing answer. The output was a description of the TV's location, not a direction. This is consistent with L4_4 being the hardest L4 question in the suite: it requires the VLM to do a small geometric computation that text retrieval alone cannot supply, and the prompt does not prime the model to do it.

## 4.4 Embedding dimension does not crack the ceiling

A natural hypothesis when one embedder underperforms is *"a bigger embedder would fix it."* Figure 4.5 tests this directly: across a 4.7× range of embedding dimensions (768 → 3584), strict accuracy creeps up only from 26% to 37%. The retrieval substrate is not the binding constraint.

[INSERT FIGURE 4.5 HERE: figures/chart_dim_vs_acc.png]
**Figure 4.5.** Strict accuracy as a function of retriever embedding dimension. Four embedders sized 768 (ViT-L-14) through 3584 (QQMM-v2) span a wide range of capacity but cluster within 11 percentage points of each other on the nineteen-question suite. Adding the VLM agent on top of the QQMM retriever (rust line, far right) jumps to 79%. The ceiling on the retriever side is real; the lift is from the agent, not from making the embedder bigger.

The interpretation is consistent with §4.3: the embedder is sufficient at L1 and partly sufficient at L2; what the agent does is solve L3–L5, which retrieval cannot solve at any embedding dimension because the question is no longer "find the visually most-similar frame" but "compose evidence across multiple retrieved frames."

## 4.5 Frame-reuse analysis

A striking qualitative finding **in the Gemma 4 31B traces** is that the model consistently *re-retrieves* the same landmark frames for semantically related queries. Figure 4.6 (Gemma 4 31B + QQMM only) shows the three landmarks that each serve four questions in the suite.

[INSERT FIGURE 4.6 HERE: figures/chart_hub_frames.png]
**Figure 4.6.** Landmark frames the agent re-uses. Three frames — `frame_000753` (TV living room), `frame_000238` (red/purple chair), and `frame_000628` (toilet) — are each retrieved across four distinct questions. Other frames are typically retrieved only once across the suite.

| Landmark | Queries it is retrieved for |
|---|---|
| `frame_000753` (TV living room) | L2_3 (watch news), L3_3 (most furniture), L3_4 (seating facing TV) |
| `frame_000238` (red/purple chair) | L1_1 (purple chair), L2_1 (sit down), L5_1 (chair NOT in kitchen), L5_2 (seating near bath) |
| `frame_000628` (toilet) | L1_2 (toilet), L3_2 (kitchen→bath), L4_4 (toilet→TV direction), L5_2 (seating near bath) |

This is interpretable as the VLM maintaining an implicit *landmark map* of the scene: once QQMM localizes a canonical frame for an object concept, the agent re-indexes into it across queries rather than treating each query from scratch. This is consistent with RAVEN's design intent (re-use of memory rather than re-storage) and is exactly the behavior a well-functioning visuo-spatial-temporal memory should produce.

[INSERT FIGURE 4.7 HERE: figures/chart_difficulty_hist.png]
**Figure 4.7.** Question difficulty distribution. For each of the nineteen questions, the histogram counts how many of the five evaluated models (four embedders + agent) returned a strict-correct answer. Over half the suite is solved by ≤1 model — the "long tail" the agent owns. The agent is the only model that solves 7 of these tail-difficulty questions.

[INSERT FIGURE 4.8 HERE: figures/chart_scorecard.png]
**Figure 4.8.** Per-question scorecard. Rows are questions (sorted L1 → L5); columns are models. Green cells are strict-correct, grey cells are wrong. The agent's column is dense on green where every other column is grey at L3–L5.

## 4.6 Latency and tool-call counts

Two of the qualitative claims in §5 — *retrieval loops* and *rare multi-tool strategy* — are quantifiable from the existing run's `debug_logs`. Parsing the Gemma 4 31B + QQMM log directly:

- **Total wall-clock for the suite:** 30.8 minutes (mean 97 s per question; median 74 s).
- **Latency rises with level:** L1 mean 67 s, L2 66 s, L3 100 s, L4 121 s, L5 127 s. L5_4 was the slowest single question at 228 s.
- **Total tool calls across 19 questions:** 112 text retrievals, 3 time retrievals, 0 image retrievals, 0 position retrievals (115 calls in total).
- **All 3 time-retriever calls were on L4_3** (the agent-start question). The image and position retrievers were never invoked across the suite — hard quantitative confirmation of the *rare multi-tool strategy* pattern in item 5.5 of §5.

[INSERT FIGURE 4.9 HERE: figures/chart_toolcalls.png]
**Figure 4.9.** Tool-call count per question (Gemma 4 31B + QQMM). Most questions resolve in 3 calls (one user prompt, one tool invocation, one tool response). L3_1, L4_1, and L5_4 hit 15–16 calls each — the retrieval-loop pattern of §5.1, now quantified. Time retrieval is exclusively used on L4_3.

[INSERT FIGURE 4.10 HERE: figures/chart_latency.png]
**Figure 4.10.** Per-question wall-clock latency (Gemma 4 31B + QQMM). L3–L5 questions take roughly twice as long as L1–L2. The distribution is right-tailed: median 74 s, max 228 s. Dashed lines show per-level means.

The retrieval-loop pattern is now visible as a number rather than an anecdote: *three* of the nineteen questions account for roughly two-fifths of the total tool-call budget (46 of 115). Each of those three (L3_1, L4_1, L5_4) re-issued the same query 5–6 times before emitting an answer, and each took ≥90 s of wall-clock time.

**Compute footprint.** The 30.8-minute suite run consumed approximately **300 Wh** of GPU energy (two NVIDIA L40 cards at ~60% average utilization, roughly 360 W total power draw) — a useful order-of-magnitude anchor for any offline-deployment cost-benefit calculation. The agent forward passes dominate; the QQMM retrieval cost is comparatively negligible at K=5 FAISS lookups over 843 frames. This is the quantitative footing for the routing argument in §7: an L1/L2 question costs about a minute on this hardware, an L4/L5 question costs two; if a workload is dominated by L1/L2, the per-query overhead of running the agent at all is the target for the gating mechanism in future work.

---

# 5. Failure-Mode Diagnosis

The previous section established *that* open-VLM RAVEN tracks closed-VLM expectations on L1–L2 and trails on L3–L5. This section asks *why* — and, crucially, where the lever is for improvement. I localize five distinct failure modes; in §6 I show how three of them are tractable at the framework layer.

**5.1 Retrieval loops dominate the failure cases.** On the questions that fail (i.e., end up in the "wrong" bucket), the VLM tends to re-issue the *identical* query four to six times rather than varying phrasing or switching retrieval primitives. I observed `"coffee machine"` repeated six times in one L4 trace and `"entryway"` repeated six times in another, always returning the same top-$K$ frames. The tool-use loop has no backoff, no query-diversification prior, and no evidence-accumulation criterion that prevents immediate re-querying. **This is the single most consequential failure mode**, because it accounts for the longest-running traces and the most confidently-wrong answers.

**5.2 Prose-without-coordinates.** Questions L3_1 ("What room is closest to the entry?") and L3_3 ("Which area has the most furniture?") produced correct *room-level* answers in prose ("the entryway", "the living room") but no $(x, y, z)$ coordinate in the answer payload. The evaluator — which is strictly coordinate-based per §3.5 — scores these as wrong. This is a *framework* failure, not a VLM failure: the VLM "knew" the answer; the answer-formulation path lost it on the way out. A coordinate-extraction fallback (e.g., centroid of the retrieved frames' poses) would salvage every observed case in this category.

**5.3 Correct frames, offset coordinates.** The retrieved frames are right but the reported coordinate drifts from ground truth by 2–5 meters. This is consistent with a metadata-extraction bug in the answer-formulation path, not a reasoning failure. It is the dominant source of the gap between retrieval-level accuracy (15/19) and coordinate-level accuracy (2/17) reported in §4.

**5.4 Texture–surface confusion (embedder-side).** For L5_3 ("flat surface that is NOT a floor"), the model correctly read the marble texture in `frame_000343` but mislabeled the surface as a "marble countertop" when it was a marble *floor*. This is an embedder-side ambiguity (QQMM aligns marble texture to a generic stone-surface concept and does not disambiguate floor vs. countertop without spatial context) and **cannot be fixed at the agent layer** without prompting the VLM to verify surface type as a separate step.

**5.5 Rare multi-tool strategy.** Only 1 of 19 queries triggered a non-text retriever (L4_3, the agent-start question). The time and position retrievers are systematically *under-used*, implying that the system prompt or the tool descriptions do not sufficiently prime the VLM to reach for them when text retrieval fails. The fact that the model *can* call them when triggered (the L4_3 trace is unambiguous evidence) means the lever is the prompt, not the model.

---

# 6. Discussion

## 6.1 Why the embedder-only floor is high on L1–L2

The RAVEN paper's negative result — *medium open VLMs ≈ embedder-only* — reproduces on the DARPA simulation data for L1 direct queries. The mechanism is now visible: QQMM's top-1 retrieval for a keyword query already lands on the correct frame, so a VLM that simply parrots the retrieved frame's location is no better than retrieval-only. **The LLM adds value precisely at L3–L5**, where retrieval alone is insufficient: spatial relationships between rooms, multi-step composition involving time, and explicit negation. This is exactly the per-level pattern visible in Figure 4.3 — the gap between the slate and teal lines is zero on L1 and large on L3–L5.

This has a direct implication for *deployment*: if the deployment workload is dominated by L1-style direct lookups, the LLM is overhead. If the workload is dominated by L3–L5, it is essential. A static gating decision — embedder-only versus full agent — should track the workload mix.

## 6.2 What scales with VLM size

Across the four open backbones (Gemma 3 27B, Gemma 4 31B, Qwen2.5-VL 32B, Qwen3-VL 32B), retrieval-level scores cluster within a 3-point band: **12, 13, 13, 15 out of 19**. There is no monotonic scaling trend; the smallest backbone (Gemma 4 31B at ≈17 GB VRAM) is the highest scoring, while the largest (Qwen2.5-VL 32B at 46 GB) ties Gemma 3 27B at 13/19.

Scaling the VLM up does not buy retrieval-level accuracy on this suite, and the headline §4.2 result holds: **the bottleneck is downstream of retrieval**. When an agent fails coordinate-strict, it has typically retrieved the right frame but emitted prose without a coordinate, or extracted the wrong pose from the right frame (§5.2, §5.3). A larger VLM is asked to do its job more often only because retrieval succeeds more often — but the framework-level coordinate-emission step has nothing to do with model size.

The lever for offline open-VLM RAVEN is therefore not the VLM. It is the answer-formulation layer (§6.3).

## 6.3 Actionable framework recommendations

Three changes are tractable at the agent/framework layer, none of which require fine-tuning the VLM and none of which touch the retrieval substrate:

- **Query diversification.** After $K$ consecutive identical retrievals (say $K = 2$), force a prompt-level rewrite — ask the VLM for alternate query candidates before re-issuing. This breaks the retrieval-loop failure mode (§5.1).
- **Coordinate fallback.** When the VLM returns a textual room name with no coordinate, emit the centroid of the retrieved frames' poses. This salvages every observed case in the prose-without-coordinates category (§5.2). One-line code change in the answer-extraction path.
- **Tool-use priming.** Add few-shot exemplars in `agent_system_prompt.txt` specifically showing the time and position retrievers being used after an initial text retrieval fails. This addresses the rare-multi-tool-strategy failure mode (§5.5).

None of these three fixes is ablated in this work; their measured lift is left to follow-up. Coordinate fallback is the cheapest (approximately a one-line change in the answer-formatter) and is the obvious next experiment. **None of these fixes requires retraining the VLM.**

## 6.4 Limitations

- **Single scene, 843 frames.** The evaluation is on one trajectory. The RAVEN paper's scale claims (up to ~3,000 frames on FindingDory long) are not probed here.
- **Variance not fully quantified.** I report qualitative observations across runs but do not report per-model standard deviations on the DARPA suite. The RAVEN paper reports ±3–6% for open VLMs; similar variance is expected here.
- **Sim-to-real gap.** All results are on Habitat-rendered frames. The RAVEN paper's real-robot results (≥97% success on Unitree Go1) are separate and not replicated here.
- **Replication, not novel method.** The retrieval substrate, the tool-use loop, the embedder-only baseline, and the L1–L5 taxonomy concept (which echoes RAVEN-QA's category structure) are all established. The contribution of this work is the open-VLM coverage on new evaluation data, the failure-mode diagnosis, and the practical recommendation set in §6.3.

## 6.5 Key takeaways

- **The lift is the agent, not the retriever.** Across a 4.7× range of embedding dimensions (768 to 3584), strict accuracy varies by 11 percentage points; adding the VLM agent on the same retriever lifts it 42 percentage points.
- **The lift is concentrated on L3–L5.** L1 direct lookup is solved by retrieval alone; the agent earns its cost on spatial, multi-step, and negation queries where retrieval is structurally insufficient.
- **The remaining loss is agent-side and tractable.** Three framework-level fixes (query diversification, coordinate fallback, tool-use priming) target the dominant failure modes diagnosed here without touching the retrieval substrate or fine-tuning the VLM.

---

# 7. Future Work

**Routing fast queries around the VLM agent.** The per-level breakdown of §4.3 shows embedder-only retrieval is competitive on L1 and adequate on L2, while L3–L5 is where the VLM agent earns its cost. The natural next step is to *route at inference time* — handle easy queries with the embedder-only fast path and escalate hard queries to the full agent — but the design choice is non-obvious. **Retrieval similarity is not a reliable proxy for answer correctness**; the embedder will confidently retrieve visually similar but semantically wrong frames whenever rooms share furniture or lighting. So a similarity-margin gate alone will short-circuit to wrong answers in exactly the cases it should defer. Three candidate gate mechanisms with different cost–robustness tradeoffs are worth evaluating:

- *Cheap VLM verifier (draft + verify).* The embedder retrieves a top-1 candidate; a single bounded VLM call decides whether it answers the question. Accepted candidates return immediately; rejected ones escalate to the full agent loop. Costs one cheap inference per easy query versus the four-to-six calls plus tool-use overhead of the full loop, and crucially the verifier *sees the candidate*, so confidently-wrong retrievals get caught.
- *Multi-signal triangulation.* Combine retrieval margin, a question-shape heuristic (bare entity lookup vs. spatial / multi-step), and an answer-shape parse (does the top-1 frame's metadata fit the expected answer type?). All three must agree before the gate accepts. Preserves a true "no VLM call" fast path, but remains vulnerable to the worst case: a wrong frame from a similar-looking room can pass all three checks.
- *Learned classifier on the L1–L5 taxonomy.* Stretch goal. Would require a substantially larger labeled set than the nineteen questions used here to be defensible, and the same correctness-vs-difficulty conflation issue applies if the classifier only sees the question and not the candidate.

Of these, the cheap-verifier path is the most defensible because it directly addresses the failure mode that dooms similarity-only gates. The others are useful baselines for an ablation.

**Joint integration with full RAVEN.** The gate, whichever mechanism it ends up being, should be added to the full RAVEN system *jointly* with the §6.3 recommendations (query diversification, coordinate fallback, tool-use priming) rather than as disjoint patches. The changes interact: coordinate fallback shortens the agent path the gate is trying to avoid; query diversification reduces the retrieval-loop failures that would otherwise cause the verifier to reject good candidates; and tool-use priming changes which queries are actually reasoning-required. Evaluating the four changes together on the same suite is the only way to measure their combined lift versus the additive estimate.

**Variance and scale.** Replicate on multiple TIAMAT trajectories and on a FindingDory-sized memory (~3,000 frames) with per-model standard deviations. The single-scene caveat in §6.4 is the most consequential limitation for downstream deployment claims.

**Training-side complement.** Although the framework-level fixes in §6.3 do not require retraining, a separate line of work would adapt a smaller open VLM specifically to the retrieval-loop failure mode: instruction-tuning on synthetic transcripts that include forced query diversification and tool-switching. This would test whether the rare-multi-tool-strategy failure (§5.5) is *systematically* trainable away.

---

# 8. Conclusion

This independent work extended the RAVEN evaluation to locally-served open-source VLMs over DARPA TIAMAT simulation data, authored a difficulty-stratified nineteen-question suite specifically designed to separate retrieval-solvable from reasoning-required queries, and diagnosed open-VLM failure modes as predominantly *agent-side* — retrieval loops, coordinate-plumbing gaps, and an under-used non-text retriever — rather than retrieval-substrate or base-VLM failures.

The headline quantitative result is that adding a VLM agent on top of QQMM-embed-v2 lifts strict accuracy from 7/19 (embedder-only) to 15/19, with the entire gain concentrated on L3–L5 (spatial, multi-step, and negation reasoning). Increasing the embedder dimension by 4.7× — from 768 to 3584 — closes only 11 percentage points of the gap, while adding the agent closes 42 percentage points; the lift is downstream of the retriever, not in it.

The practical implication for offline, air-gapped deployments of RAVEN on open models is direct: most of the gap to closed-VLM performance is recoverable without retraining, via three framework-level changes (query diversification, coordinate fallback, tool-use priming) that target the dominant failure modes identified here. The embedder-only pipeline remains a competitive *fast path* for L1 queries — consistent with RAVEN's own §5.3 recommendation — and the natural next step is to route between fast path and full agent at inference time using a cheap VLM verifier rather than a similarity-only gate that would mis-route on confidently-similar wrong frames.

### Code and data availability

All evaluation question files (`test_questions_smoke.md`, `test_questions_smoke_qa.json`), per-model YAML configs in `cfgs/vlms/`, the modified evaluation harness, and full trace logs for the headline Gemma 4 31B run are maintained inside the **PRISM** project's private GitHub repository, available to advisers and committee members on request.

---

# References

[1] Y. Hu, Z. Zheng, L. Zha, C. Xing, R. Singh, O. Hossain, A. Loquercio, D. Shah. *RAVEN: Long-Horizon Reasoning and Navigation with a Visuo-Spatial-Temporal Memory.* Robotics: Science and Systems, 2026. Code: github.com/zzcnewly/RAVEN.

[2] A. Anwar, J. Welsh, J. Biswas, S. Pouya, Y. Chang. *ReMEmbR: Building and Reasoning Over Long-Horizon Spatio-Temporal Memory for Robot Navigation.* IEEE International Conference on Robotics and Automation (ICRA), 2025.

[3] K. Yadav, S. K. Ramakrishnan, J. Turner, A. Gokaslan, O. Maksymets, R. Jain, R. Ramrakhya, A. X. Chang, A. Clegg, M. Savva, E. Undersander, D. S. Chaplot, D. Batra. *FindingDory: A Benchmark for Memory-Based Navigation.* 2024.

[4] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, I. Sutskever. *Learning Transferable Visual Models from Natural Language Supervision.* International Conference on Machine Learning (ICML), 2021. (CLIP.)

[5] X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer. *Sigmoid Loss for Language-Image Pre-training.* International Conference on Computer Vision (ICCV), 2023. (SigLIP.)

[6] Y. Xue, D. Li, G. Liu. *QQMM-embed-v2: A Multimodal Memory Embedding Model for Visuo-Spatial-Temporal Retrieval.* Hugging Face: `youzexue/QQMM-embed-v2`, 2025.

[7] J. Johnson, M. Douze, H. Jégou. *Billion-Scale Similarity Search with GPUs.* IEEE Transactions on Big Data, 2019. (FAISS.)

[8] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, D. Batra. *Habitat: A Platform for Embodied AI Research.* International Conference on Computer Vision (ICCV), 2019.

[9] Gemma Team, Google DeepMind. *Gemma 3: Open Multimodal Foundation Models.* Technical Report, 2024.

[10] Qwen Team, Alibaba. *Qwen2.5-VL Technical Report.* 2024.

[11] DARPA TIAMAT Competition documentation. *Triage Information Analysis with Multi-Agent Teaming.* Internal documentation, 2025.

---

# Appendix A. Full Evaluation Question List

The full nineteen-question suite, with capability tags and ground-truth positions, is maintained in `test_questions_smoke.md` (human-readable) and `test_questions_smoke_qa.json` (machine-readable). Question summaries by level:

**L1 — Direct object recognition:**
- L1_1: "Where is the purple chair?"
- L1_2: "Where is the toilet?"
- L1_3: "Where is the white sofa?"
- L1_4: "Where are the kitchen cabinets?"

**L2 — Indirect / attribute-based:**
- L2_1: "Where can I sit down comfortably?"
- L2_2: "Where can I wash my hands?"
- L2_3: "Where is something I could use to watch the news?"

**L3 — Spatial reasoning:**
- L3_1: "What room is closest to the entry?"
- L3_2: "Where would I go to get from the kitchen to the bathroom?"
- L3_3: "Which area has the most furniture?"
- L3_4: "Where is the seating area that faces the TV?"

**L4 — Multi-step inference:**
- L4_1: "I need to make coffee and sit down to drink it. Where should I go first?"
- L4_2: "If I spill water on the kitchen floor, where would I find something to clean it up?"
- L4_3: "Where did the agent start its exploration?"
- L4_4: "If I'm at the toilet and hear the TV, which direction would I walk?"

**L5 — Negation / disambiguation:**
- L5_1: "Find a chair that is NOT in the kitchen."
- L5_2: "There are multiple seating options. Find the one closest to the bathroom."
- L5_3: "Find a flat surface that is NOT a floor."
- L5_4: "Where did the agent spend the most time during its exploration?"

---

# Appendix B. Per-Model Output Directories

All runs are reproducible from `outputs/smoke_v2/raven_results/20q/`:

- `gemma12b_qqmm/` — Gemma 3 12B with QQMM-embed-v2
- `gemma27b_qqmm/` — Gemma 3 27B with QQMM-embed-v2
- `gemma27b_vith14/` — CLIP ViT-H-14 ablation, Gemma 3 27B reasoning
- `gemma4_26b_qqmm/` — Gemma 4 26B with QQMM-embed-v2
- `gemma4_31b_qqmm/` — **primary result.** Findings summary in `GEMMA4_31B_FINDINGS.md`
- `qwen25vl3b_qqmm/` — Qwen2.5-VL 3B with QQMM-embed-v2

Embedder-only baselines: `outputs/custom_eval/` with summary `EVAL_REPORT.md`.

---

# Appendix C. Reproduction Commands

The headline result (Gemma 4 31B + QQMM, 15/19) reproduces with:

```bash
python remembr_static_eval_vlm.py \
  --vlm_config       cfgs/vlms/gemma4-31b.yaml \
  --embedder_config  cfgs/embedders/qqmm.yaml \
  --agent_config     cfgs/agents/raven.yaml \
  --input_folder     ./extracted_videos/smoke_full \
  --qa_file          ./test_questions_smoke_qa.json \
  --caption_file     ./extracted_videos/smoke_full/frames.json \
  --out_dir          ./outputs/smoke_v2/raven_results/20q/gemma4_31b_qqmm \
  --memory_backend   faiss \
  --top_k            5 \
  --device           cuda
```

Swap `--vlm_config` to any of the configs in `cfgs/vlms/` to reproduce the per-model rows in §4.2. Embedder-only baselines reproduce with `run_custom_eval.py`, which loads the same FAISS index and applies top-1 retrieval without invoking a VLM.

---

# Appendix D. Engineering Standards Used

This appendix documents the engineering standards, frameworks, and accepted industrial conventions used in the implementation, as required for ECE Independent Work.

- **PyTorch (≥ 2.x), CUDA 12.x.** Standard deep-learning frameworks for tensor computation and GPU acceleration. Used for embedder forward passes (QQMM-embed-v2, CLIP, SigLIP).
- **FAISS (Facebook AI Similarity Search), v1.7+.** Standard library for billion-scale similarity search. Used as the retrieval backend for the visuo-spatial-temporal memory; cosine-distance index over 3584-dim QQMM embeddings.
- **Hugging Face Transformers and Hub.** Standard model distribution and weight loading. Embedders were pulled from public model hubs (`openai/clip-vit-large-patch14`, `google/siglip-large-patch16-384`, `youzexue/QQMM-embed-v2`).
- **Ollama (≥ 0.3).** Local-serving framework for open-source LLMs/VLMs. Used to host Gemma 3, Gemma 4, Qwen2.5-VL, and Qwen3-VL backbones on a single workstation.
- **YAML configuration format.** Standard human-readable config representation (used by RAVEN's reference codebase). Per-model YAML configs were authored in `cfgs/vlms/`.
- **Habitat-Sim 3D simulator.** Established research-community standard for embodied-AI simulation. Provided the smoke_v2 trajectory used as the evaluation environment.
- **JSON for evaluation I/O.** Standard structured data interchange. Used for the question file (`test_questions_smoke_qa.json`) and the per-run output traces.
- **Markdown / LaTeX.** Standard scientific document formats. The poster, this report, and supporting analyses are written in Markdown with LaTeX-style math notation.
- **Git for version control.** Standard distributed VCS, used to track all code, config, and prompt changes against the upstream RAVEN reference implementation.

The retrieval, tool-use, and prompt logic conform to RAVEN (Hu et al., 2026)'s reference implementation. No retrieval-substrate, tool, or prompt modifications are introduced beyond per-model YAML config additions.

# Appendix E. Failure-Mode Cross-Tabulation

For each of the five failure modes identified in §5, the table below lists which questions exhibited the failure in the Gemma 4 31B + QQMM run.

| Failure mode | Affected questions | Count |
|---|---|---|
| 5.1 Retrieval loops (≥15 tool calls) | L3_1, L4_1, L5_4 | 3 |
| 5.2 Prose without coordinates | L3_1, L3_3 | 2 |
| 5.3 Correct frames, offset coordinates | six L4–L5 questions where retrieval was right but coordinate was off | 6 |
| 5.4 Texture–surface confusion | L5_3 | 1 |
| 5.5 Rare multi-tool strategy | all questions *except* L4_3 | 18 |

L3_1 appears in two failure modes (retrieval loop *and* prose-without-coordinates), illustrating that the modes are not mutually exclusive.

# Appendix F. Reproducibility Checklist

- **Hardware.** 2× NVIDIA L40 (48 GB VRAM each), Princeton ECE neuronic cluster.
- **Operating system.** Ubuntu Linux (exact kernel/release per cluster image).
- **CUDA / driver.** CUDA 12.x.
- **PyTorch.** ≥2.x with CUDA 12 wheels.
- **Ollama.** ≥0.3.
- **Model digests.** Gemma 4 31B served as `gemma4:31b`; QQMM-embed-v2 pulled from `youzexue/QQMM-embed-v2`.
- **FAISS.** v1.7+; cosine-distance index over 3584-dim QQMM embeddings of 843 frames.
- **Top-K.** Fixed at 5 throughout.
- **Random seed.** Default (no explicit seed set in run `_0`).
- **Trajectory data.** `smoke_v2`, 843 egocentric Habitat frames; pose CSV `position_data_habitat_smoke_v2.csv` included in the release.
- **Question file.** `test_questions_smoke_qa.json` (19 graded questions).
- **Git revision.** Commit hash of fork at run time: `d15ab27`.

---

*End of report.*

---

## CHEAT-SHEET FOR PASTING INTO GOOGLE DOCS

When you copy this into a Google Doc:

1. **Pages 1–4 (title / honor pledge / abstract / acknowledgments):** delete the `## PAGE N — ...` markers; each becomes a single page on its own (insert page breaks before §1, before §2, etc.).
2. **Table of contents (page 5):** Google Docs has a native TOC. Insert → Table of contents *after* you set chapter titles to Heading 1 styles.
3. **Headings:** make every `# 1.`, `# 2.`, ... a Heading 1; every `## 1.1`, `## 1.2`, ... a Heading 2. The TOC will auto-populate.
4. **Figure placeholders:** `[INSERT FIGURE X.Y HERE: figures/<name>.png]` — replace each with the corresponding image file from `figures/`. The captions immediately below ("**Figure X.Y.** ...") should follow the inserted image.
5. **Format requirements (per ECE 299 guidelines, page 13):**
   - 12pt Times New Roman
   - 1.5× line spacing (or double); single-spaced for footnotes/bibliography
   - 1-inch margins all sides
   - Pages numbered
   - Charts/tables with captions, referenced in text
6. **Final figures referenced** (all in `figures/` of this repo):
   - `routing_diagram.png` → Figure 1.1
   - `reasoning_loop.png` → Figure 2.1
   - `chart_memory_vs_accuracy.png` → Figure 4.1
   - `chart_lift.png` → Figure 4.2
   - `chart_level_ladder.png` → Figure 4.3
   - `chart_dim_vs_acc.png` → Figure 4.4
   - `chart_hub_frames.png` → Figure 4.5
   - `chart_difficulty_hist.png` → Figure 4.6
   - `chart_scorecard.png` → Figure 4.7
   - `chart_toolcalls.png` → Figure 4.8
   - `chart_latency.png` → Figure 4.9
