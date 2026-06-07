# Setup Guide

This guide covers environment setup for running the RAVEN/ReMEmbR evaluation framework on a SLURM cluster with GPU access.

---

## 1. Conda Environment

Create and activate the conda environment:

```bash
conda create -n remembr_cluster python=3.10 -y
conda activate remembr_cluster
```

### Install dependencies

```bash
pip install \
  langchain==0.3.27 \
  "langchain-core==0.3.83" \
  langgraph==0.3.34 \
  "langsmith>=0.7.0,<1.0" \
  transformers==4.44.2 \
  "tokenizers>=0.19,<0.20" \
  accelerate==0.34.2 \
  langchain-huggingface \
  langchain-ollama \
  langchain-openai \
  langchain-google-genai \
  langchain-nvidia-ai-endpoints \
  langchain-community \
  langchain-text-splitters \
  open-clip-torch \
  torch torchvision \
  faiss-cpu \
  google-ai-generativelanguage \
  google-cloud-aiplatform \
  qwen-vl-utils \
  volcengine-python-sdk \
  "numpy<2.0"
```

### Version pinning (critical)

| Package | Version | Notes |
|---------|---------|-------|
| `langchain` | `0.3.27` | Requires `langchain-core` 0.3.x -- do NOT upgrade to 1.x |
| `langchain-core` | `0.3.83` | Must stay on 0.3.x branch |
| `langgraph` | `0.3.34` | Do NOT use `>=0.4.0` (requires langchain-core 1.x) |
| `transformers` | `4.44.2` | Needs `tokenizers >=0.19,<0.20` |
| `numpy` | `<2.0` (1.26.4) | numpy 2.x breaks compatibility |

### Common import errors

| Error | Cause | Fix |
|-------|-------|-----|
| `cannot import name '_DirectlyInjectedToolArg'` | langgraph too new | `pip install langgraph==0.3.34` |
| `cannot import name 'PipelinePromptTemplate'` | langchain-core too new | `pip install langchain-core==0.3.83` |
| `No module named 'langgraph._internal'` | Wrong langgraph version | Reinstall with pinned version above |

---

## 2. GPU Sessions (SLURM)

```bash
# 1 GPU (embedder-only or small models)
srun --gres=gpu:1 --mem=32G --time=02:00:00 --pty bash

# 2 GPUs (QQMM + Ollama LLM together need ~35GB VRAM)
srun --gres=gpu:2 --mem=64G --time=02:00:00 --pty bash
```

With 2 GPUs, put Ollama on GPU 0 and the embedder on GPU 1 (`--device cuda:1`).

---

## 3. Environment Variables

Set these before every run:

```bash
conda activate remembr_cluster
export HF_HOME=/scratch/$USER/.cache/huggingface
export TORCH_HOME=/scratch/$USER/.cache/torch
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_MODELS=/scratch/$USER/.ollama/models
export LD_LIBRARY_PATH=/scratch/$USER/ollama-install/lib/ollama:$LD_LIBRARY_PATH
```

---

## 4. Ollama Setup

> **Important**: Scratch storage (`/scratch/`) is per-node and temporary. Each time you get a new compute node via `srun`, you must reinstall Ollama and pull models.

### Quick install on a new node (~5 min)

```bash
# 1. Install zstd if not available
which zstd || conda install -n remembr_cluster -c conda-forge zstd -y

# 2. Download and extract Ollama with GPU support
curl -L "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst" \
  -o /tmp/ollama.tar.zst
mkdir -p /scratch/$USER/ollama-install /scratch/$USER/.ollama/models
tar --use-compress-program=zstd -xf /tmp/ollama.tar.zst -C /scratch/$USER/ollama-install/
rm /tmp/ollama.tar.zst

# 3. Start server on GPU 0
export OLLAMA_MODELS=/scratch/$USER/.ollama/models
export LD_LIBRARY_PATH=/scratch/$USER/ollama-install/lib/ollama:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /scratch/$USER/ollama-install/bin/ollama serve &
sleep 5

# 4. Pull models
/scratch/$USER/ollama-install/bin/ollama pull gemma3:27b
/scratch/$USER/ollama-install/bin/ollama pull gemma3:12b
/scratch/$USER/ollama-install/bin/ollama pull qwen2.5vl:3b
```

### Troubleshooting

| Issue | Fix |
|-------|-----|
| `ollama: command not found` | Use full path: `/scratch/$USER/ollama-install/bin/ollama` |
| `No such file or directory` | Scratch not set up on this node -- run full install above |
| `address already in use` | Ollama already running, just use it |
| `zstd: Cannot exec` | `conda install -c conda-forge zstd -y` |
| `total_vram="0 B"` | Need the full `.tar.zst` tarball (includes GPU runners), not just the binary |

### Ollama model name format

Use colon notation in model strings: `remembr+gemma3:27b+hf+...` (NOT `gemma3-27b`).

---

## 5. QQMM Embedder

- **Model**: `youzexue/QQMM-embed-v2` (3584-dim, from HuggingFace)
- **Config**: `configs/embed/qqmm-embed/mmeb.yaml`
- **VRAM**: QQMM + Ollama LLM together need ~35GB -- use 2 GPUs to avoid slowdown
- **Cache**: Create OpenCLIP cache dir on each new node:
  ```bash
  mkdir -p /scratch/$USER/.cache/openclip
  ```

---

## 6. Storage Layout

This cluster uses two storage tiers:

- **Home directory** (`~/`): Permanent, shared across all nodes, but limited capacity (16 GB quota). Store source code, outputs, evaluation results, and anything you want to keep here.
- **Scratch** (`/scratch/$USER/`): Temporary, high-capacity storage for large files like ML model weights and caches. Scratch is **per-node** (not shared across compute nodes) and is **auto-deleted after inactivity**. Anything on scratch must be treated as disposable -- you will need to re-download models and rebuild caches each time you land on a new node.

**Rule of thumb**: If you can't afford to lose it, put it in your home directory. If it's large and re-downloadable (model weights, caches, Ollama binaries), put it on scratch.

| Location | Contents | Persistence |
|----------|----------|-------------|
| `~/remembr_project/` | Source code, scripts, data, outputs, results | Permanent (home directory) |
| `/scratch/$USER/.cache/huggingface` | HuggingFace model weights | Temporary, per-node |
| `/scratch/$USER/.cache/openclip` | OpenCLIP model cache | Temporary, per-node |
| `/scratch/$USER/ollama-install/` | Ollama binary + GPU runners | Temporary, per-node |
| `/scratch/$USER/.ollama/models/` | Pulled Ollama models | Temporary, per-node |

---

## 7. Data Preparation

### Extract dataset

```bash
cd ~/remembr_project
tar -xzf extracted_videos.tar.gz
```

### Convert to JSON (frames.json + qa.json)

```bash
python remembr_static_convertion_vlm.py \
  --data_root ./extracted_videos \
  --out_dir ./extracted_videos/smoke_v2 \
  --splits smoke_full \
  --no-auto-discover \
  --fallback_index_time \
  --position_csv ./position_data_habitat_smoke_v2.csv
```

### Fix image paths

Paths in `frames.json` must be relative to the project root:
- Correct: `extracted_videos/smoke_full/frames/frame_000000.png`
- Wrong: `smoke_full/frames/frame_000000.png`

---

## 8. Running Evaluations

### Embedder-only (no Ollama needed)

```bash
python remembr_static_eval_vlm.py \
  --model "embedder_only+none+hf+youzexue/QQMM-embed-v2+0+3584" \
  --input_folder ./extracted_videos/smoke_v2 \
  --qa_file ./test_questions_smoke_qa.json \
  --caption_file ./extracted_videos/smoke_v2/frames.json \
  --out_dir ./outputs/smoke_v2/embedder_only_results \
  --memory_backend faiss --top_k 5 --device cuda
```

### RAVEN agent (requires Ollama running)

```bash
python remembr_static_eval_vlm.py \
  --model "remembr+gemma3:27b+hf+youzexue/QQMM-embed-v2+0+3584" \
  --input_folder ./extracted_videos/smoke_v2 \
  --qa_file ./test_questions_smoke_qa.json \
  --caption_file ./extracted_videos/smoke_v2/frames.json \
  --out_dir ./outputs/smoke_v2/raven_results \
  --input_prompt_folder prompts/irs_vlm_prompts \
  --memory_backend faiss --top_k 5 --device cuda:1 \
  --add_score_info --debug
```

### Model string format

```
framework+base_llm+backend+model_id+vlm_layer+emb_dim
```

Examples:
- `remembr+gemma4:31b+hf+youzexue/QQMM-embed-v2+0+3584` -- Gemma 4 31B with QQMM
- `remembr+gemma3:27b+hf+youzexue/QQMM-embed-v2+0+3584` -- Gemma 3 27B with QQMM
- `embedder_only+none+hf+youzexue/QQMM-embed-v2+0+3584` -- QQMM embedder only (no LLM)

---

## 9. Code Compatibility Notes

The following patches have been applied for `langchain==0.3.27` compatibility:

- `remembr/agents/remembr_agent.py` and `remembr/agents/vlm_non_agent.py`: Changed `from langchain.output_parsers` to `from langchain_core.output_parsers`
- `remembr/memory/memory_factory.py`: Milvus imports are lazy (only loaded when `backend=milvus`)
