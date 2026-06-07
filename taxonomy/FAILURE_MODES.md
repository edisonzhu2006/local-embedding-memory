# Failure Mode Taxonomy for VLM Spatial QA

Version: 1.0 | Last updated: 2026-03-27

## How to Use

In trial report Per-Question tables, assign one failure code per failed question.
If multiple apply, use the PRIMARY cause. Note secondary codes in Notes.
A dash `--` means correct (no failure).

---

### VC -- Visual Confusion

The model retrieves a visually similar but incorrect object or scene.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| VC-1 | Intra-category | Confuses objects in same category | "white sofa" retrieves white chair |
| VC-2 | Spatial aliasing | Correct object type, wrong location | "kitchen cabinets" retrieves hallway cabinets |
| VC-3 | Scene-level match | Correct room/area, wrong object | "watch the news" retrieves living room without TV |

### KD -- Keyword Dominance

Multi-concept queries reduced to the most visually salient keyword.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| KD-1 | Salient keyword capture | One keyword dominates, rest ignored | "toilet and hear the TV" retrieves toilet |
| KD-2 | Modifier ignored | Adjectives/qualifiers dropped | "closest to entry" retrieves any entry-like frame |

### SR -- Spatial Reasoning Failure

Query requires spatial understanding that embeddings cannot provide.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| SR-1 | Proximity | Cannot compute "closest to" / "nearest" | "seating closest to bathroom" |
| SR-2 | Directional | Cannot compute "facing" / "left of" / "toward" | "seating that faces the TV" |
| SR-3 | Path/connectivity | Cannot reason about routes | "get from kitchen to bathroom" |
| SR-4 | Aggregation | Cannot count/aggregate over regions | "area with the most furniture" |

### NG -- Negation/Exclusion Failure

Query contains negation or exclusion that embeddings cannot process.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| NG-1 | Simple negation | "NOT X" treated as "X" | "chair NOT in kitchen" retrieves kitchen chair |
| NG-2 | Set exclusion | Must exclude one member of a set | "flat surface NOT a floor" |

### TR -- Temporal/Trajectory Reasoning

Query requires time, ordering, or trajectory statistics.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| TR-1 | Temporal ordering | Requires knowing frame sequence | "where did agent start?" |
| TR-2 | Dwell time / statistics | Requires aggregation over trajectory | "where did agent spend most time?" |
| TR-3 | Sequence reasoning | Requires before/after understanding | "what did agent see after the kitchen?" |

### RB -- Retrieval Bias

Wrong due to dataset-level biases, not model failure.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| RB-1 | Oversampled region | Region with many frames dominates | Vague queries -> entry [0,0] with 75 frames |
| RB-2 | Undersampled region | Target has too few frames | Object visible in only 1-2 frames |

### LR -- LLM Reasoning Failure (RAVEN/ReMEmbR only)

LLM fails despite having access to relevant information.

| Code | Name | Description | Example |
|------|------|-------------|---------|
| LR-1 | Tool misuse | Wrong tool or wrong parameters | Queries text memory instead of image |
| LR-2 | Hallucination | Answer not grounded in retrieved data | Position doesn't match any frame |
| LR-3 | Reasoning chain error | Logical error in multi-step reasoning | Identifies kitchen but reports living room |
| LR-4 | Context overflow | Truncates/ignores retrieved context | Ignores later items in long results |
| LR-5 | Template/format error | Malformed response, parsing fails | Returns placeholders instead of answers |

---

## Versioning

When adding new codes, append to the relevant category. Do not renumber existing codes -- trial reports reference them.
