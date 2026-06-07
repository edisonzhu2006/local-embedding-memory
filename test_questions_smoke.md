# Custom Test Questions — Smoke Dataset (Habitat Indoor)

## Scene Description
Habitat simulated indoor residential environment. Agent navigates through:
- **Entry/Hallway** (~[0,0]): Shoes, front door area
- **Living Room** (~[4,-0.4] to [7,0.6]): Purple chair, white sofa, TV, fireplace, shelves
- **Kitchen** (~[4,-5] to [5.5,-5.5]): Countertops, cabinets, dining chairs
- **Bathroom** (~[-0.3,-2.9]): Toilet, sink
- **Dining/Transition Area** (~[2,-6] to [3,-5]): Tables, metal chairs

---

## Level 1: Direct Object Recognition (Easy)
Simple, single-object queries. The object is visually distinctive and named directly.

### Q1.1
- **Question**: "Where is the purple chair?"
- **Expected region**: Living room (~[2.1, 1.1])
- **Difficulty**: Easy — unique, visually distinctive object
- **Type**: position

### Q1.2
- **Question**: "Where is the toilet?"
- **Expected region**: Bathroom (~[-0.28, -2.90])
- **Difficulty**: Easy — unique object, clearly visible
- **Type**: position

### Q1.3
- **Question**: "Where is the white sofa?"
- **Expected region**: Living room (~[4.5, -0.6] to [6.0, 0.1])
- **Difficulty**: Easy — large, distinctive furniture
- **Type**: position

### Q1.4
- **Question**: "Where are the kitchen cabinets?"
- **Expected region**: Kitchen (~[4.9, -5.4])
- **Difficulty**: Easy — prominent feature
- **Type**: position

---

## Level 2: Indirect Object / Attribute-Based (Medium)
Requires matching descriptions to objects, not naming them directly.

### Q2.1
- **Question**: "Where can I sit down comfortably?"
- **Expected region**: Living room chairs/sofa (~[2.1, 1.1] or [4.5, -0.6])
- **Difficulty**: Medium — must map "sit comfortably" to chair/sofa
- **Type**: position

### Q2.2
- **Question**: "Where can I wash my hands?"
- **Expected region**: Bathroom sink (~[-0.28, -2.90]) or kitchen sink
- **Difficulty**: Medium — indirect reference to sink
- **Type**: position

### Q2.3
- **Question**: "Where is something I could use to watch the news?"
- **Expected region**: TV area (~[4.3, -0.37] or [7.0, -4.6])
- **Difficulty**: Medium — must infer TV from "watch the news"
- **Type**: position

### Q2.4
- **Question**: "Where is the brightest area in the home?"
- **Expected region**: Varies — areas near windows or well-lit rooms
- **Difficulty**: Medium — abstract attribute (brightness) rather than object
- **Type**: position

---

## Level 3: Spatial Reasoning (Hard)
Requires understanding spatial relationships, relative positions, or scene layout.

### Q3.1
- **Question**: "What room is closest to the entry?"
- **Expected region**: Living room or hallway (~[0-2, 0-1])
- **Difficulty**: Hard — requires spatial proximity reasoning
- **Type**: position

### Q3.2
- **Question**: "Where would I go to get from the kitchen to the bathroom?"
- **Expected region**: Transition area between kitchen and bathroom
- **Difficulty**: Hard — requires understanding of scene graph / connectivity
- **Type**: position

### Q3.3
- **Question**: "Which area has the most furniture?"
- **Expected region**: Living room (~[4-7, -1 to 1])
- **Difficulty**: Hard — requires aggregating over multiple objects
- **Type**: position

### Q3.4
- **Question**: "Where is the seating area that faces the TV?"
- **Expected region**: Sofa/chairs in front of TV
- **Difficulty**: Hard — requires understanding facing direction / arrangement
- **Type**: position

---

## Level 4: Multi-Step Inference (Very Hard)
Requires combining multiple pieces of information, temporal reasoning, or world knowledge.

### Q4.1
- **Question**: "I need to make coffee and sit down to drink it. Where should I go first?"
- **Expected region**: Kitchen (~[4.9, -5.4])
- **Difficulty**: Very Hard — multi-step task planning, first step = kitchen
- **Type**: position

### Q4.2
- **Question**: "If I spill water on the kitchen floor, where would I find something to clean it up?"
- **Expected region**: Bathroom or kitchen cabinets (cleaning supplies)
- **Difficulty**: Very Hard — world knowledge (cleaning supplies location) + spatial
- **Type**: position

### Q4.3
- **Question**: "The agent entered through the front door and walked to the kitchen. Approximately how far did it travel?"
- **Expected region**: N/A
- **Difficulty**: Very Hard — requires computing path distance from position data
- **Type**: duration/distance

### Q4.4
- **Question**: "If I'm at the toilet and hear the TV, which direction would I walk?"
- **Expected region**: Path from bathroom to living room
- **Difficulty**: Very Hard — requires spatial reasoning about two known locations + direction
- **Type**: position

---

## Level 5: Negation / Disambiguation (Expert)
Requires distinguishing between similar objects or handling negation.

### Q5.1
- **Question**: "Find a chair that is NOT in the kitchen."
- **Expected region**: Living room purple chair or sofa (~[2.1, 1.1])
- **Difficulty**: Expert — requires negation (exclude kitchen chairs)
- **Type**: position

### Q5.2
- **Question**: "There are multiple seating options. Find the one closest to the bathroom."
- **Expected region**: Nearest chair/sofa to [-0.28, -2.90]
- **Difficulty**: Expert — disambiguation + spatial distance
- **Type**: position

### Q5.3
- **Question**: "Find a flat surface that is NOT a floor."
- **Expected region**: Kitchen counter, table, or shelf
- **Difficulty**: Expert — negation + abstract matching
- **Type**: position

### Q5.4
- **Question**: "Where did the agent spend the most time during its exploration?"
- **Expected region**: Entry area [0,0] (frames 0-75 all at origin = longest dwell)
- **Difficulty**: Expert — requires temporal/statistical reasoning over trajectory
- **Type**: position

---

## Evaluation Notes

### Expected Performance by Level (Embedder-Only)
- **Level 1**: Should perform well — direct visual matching
- **Level 2**: Moderate — CLIP/SigLIP handles attribute-based queries reasonably
- **Level 3**: Poor — pure embeddings lack spatial reasoning
- **Level 4**: Very poor — no multi-step inference without LLM
- **Level 5**: Very poor — negation and disambiguation are beyond embedding similarity

### This tests the hypothesis:
> Embedder-only retrieval is sufficient for direct object localization but fails on questions requiring reasoning, spatial understanding, or multi-step inference — motivating the full ReMEmbR agent with LLM.

### How to run:
Convert these questions to `qa.json` format and run with:
```bash
python remembr_static_eval_vlm.py \
  --model embedder_only+none+oc+ViT-L-14+openai+768 \
  --input_folder ./extracted_videos/smoke_full \
  --qa_file ./test_questions_smoke_qa.json \
  --caption_file ./extracted_videos/smoke_full/frames.json \
  --out_dir ./outputs/custom_questions_vitl14 \
  --memory_backend faiss --top_k 5 --device cuda
```
