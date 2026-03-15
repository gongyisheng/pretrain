# Pretrain

Pretraining experiments on a single GPU — different architectures, datasets, hyperparameters, and see what actually works.

Pure PyTorch, no frameworks, no magic (>=12GB VRAM).

## Roadmap

GPT-2 (MHA) is the baseline. All experiments compare against it.

### Attention

| Experiment | Status | Description |
|---|---|---|
| MHA (GPT-2 baseline) | Done | Standard multi-head attention — our reference point |
| GQA | Planned | Grouped-query attention (Llama 2 style) — fewer KV heads, same quality? |
| MQA | Planned | Multi-query attention — single KV head, how far can we push it? |
| MLA | Planned | Multi-head latent attention (DeepSeek-V2) — compress KV into latent space |
| Linear Attention | Planned | RetNet / RWKV-style — can we kill quadratic complexity and keep quality? |
| Sliding Window | Planned | Mistral-style local + global attention mix |

### Beyond Attention

| Experiment | Status | Description |
|---|---|---|
| Sparse MoE | Planned | Mixture of Experts — more parameters, same compute |
| Diffusion LM | Planned | Diffusion-based language modeling — text generation without autoregression |
| Multimodal | Planned | Vision-language from scratch — how hard is it really? |

### Frontier & Exotic

| Experiment | Status | Description |
|---|---|---|
| mHC | Planned | Multi-head chunked attention — efficient long-context |
| DeepSeek OCR | Planned | DeepSeek's optical character reasoning approach |
| DeepSeek Engram | Planned | Persistent memory across context — learned engrams |
| State Space (Mamba) | Planned | S4/Mamba — the RNN strikes back |
| Hyena | Planned | Long convolutions as attention replacement |

### Dataset

| Experiment | Status | Description |
|---|---|---|
| Coding Corpus | Planned | Train model on coding data (e.g., The Stack) — compare against OpenWebText baseline |
| Multi-language | Planned | Train on multi-language corpora — cross-lingual transfer and tokenizer efficiency |

### Infra & Kernel Speedups

| Experiment | Status | Description |
|---|---|---|
| Triton FlashAttention | Planned | Custom Triton kernel for fused attention (MHA/GQA/MQA) |
| Fused Softmax | Planned | Triton fused online softmax — avoid materializing full attention matrix |
| Fused MLP | Planned | Fused GeLU / SwiGLU MLP kernel — one read, one write |
| Fused LayerNorm / RMSNorm | Planned | Triton fused norm kernels — eliminate extra memory passes |
| Fused Cross-Entropy | Planned | Fused logit + softmax + loss — skip materializing logits |
| Custom Optimizer | Planned | Implement AdamW and Muon from scratch — custom weight decay, momentum, Newton-style updates |
| Custom Tokenizer | Planned | Byte-level BPE tokenizer from scratch — no HF dependency |
| FP8/FP4 Training | Planned | Low-precision training — measure throughput gains vs. loss quality tradeoff |

### Evaluation

| Experiment | Status | Description |
|---|---|---|
| Scaling Law | Planned | Reproduce Chinchilla/Kaplan scaling laws — sweep model size vs. tokens vs. compute |
| Tokenizer Quality | Planned | Measure compression ratio and fertility across corpora (English, code, multi-language) |
| Training Efficiency | Planned | Compare tokens/sec, peak VRAM, and FLOPs utilization across architectures and kernels |
| Weight Visualization | Planned | Layer weight distribution and gradient flow — detect vanishing/exploding gradients |
| Attention Visualization | Planned | Attention score heatmaps — inspect head specialization and pattern formation |

> Results are posted weekly. Check the `experiments/` folder for write-ups and W&B links.

## Project Structure

```
pretrain/
├── configs/                  # Base configs (YAML)
│   └── gpt2_small.yaml      # GPT-2 124M baseline
├── src/
│   ├── model/                # Model architectures
│   │   ├── components.py     # Shared blocks (attention, MLP, norms)
│   │   ├── gpt2.py           # GPT-2 (baseline)
│   │   └── registry.py       # Model registry
│   ├── data/                 # Data pipeline
│   │   ├── tokenizer.py      # Custom BPE tokenizer
│   │   └── dataset.py        # Memory-mapped .bin dataset
│   ├── training/             # Training loop
│   │   ├── trainer.py        # Core training loop
│   │   ├── optimizer.py      # AdamW + cosine warmup scheduler
│   │   └── logger.py         # W&B logging wrapper
│   └── utils/
│       └── config.py         # TrainConfig dataclass + YAML loader
├── scripts/
│   ├── run_pipeline.sh       # End-to-end: preprocess + train
│   ├── train.py              # Training entry point
│   ├── train_tokenizer.py    # Train custom BPE tokenizer
│   ├── preprocess_data.py    # Tokenize dataset → .bin files
│   └── eval.py               # Standalone evaluation
├── experiments/              # Architecture experiments (see below)
├── tests/
└── pyproject.toml
```

## Quick Start

```bash
uv sync

# Run the full pipeline (preprocess OpenWebText + train GPT-2 baseline)
nohup uv run bash scripts/run_pipeline.sh > pipeline.log 2>&1 &

# Or step by step
uv run python scripts/train_tokenizer.py --config configs/gpt2_small.yaml
uv run python scripts/preprocess_data.py --config configs/gpt2_small.yaml
uv run python scripts/train.py --config configs/gpt2_small.yaml
uv run python scripts/eval.py --config configs/gpt2_small.yaml --checkpoint checkpoints/step_50000.pt

# CLI overrides
uv run python scripts/train.py --config configs/gpt2_small.yaml --optimizer.lr 3e-4
```

## Experiments

All architecture experiments live in `experiments/`. Each experiment is self-contained with its own config, custom code, and documentation.

```
experiments/
├── gqa_attention/
│   ├── README.md         # Hypothesis, setup, results
│   ├── config.yaml       # Inherits from base + overrides
│   └── attention.py      # GQA implementation
├── moe_transformer/
│   ├── README.md
│   ├── config.yaml
│   └── moe_layer.py
└── ...
```

### Running an experiment

```bash
uv run python scripts/train.py --config experiments/gqa_attention/config.yaml
```

### Experiment workflow

1. Create `experiments/<name>/` with a `config.yaml` that inherits from a base config
2. Implement the custom component (attention, layer, model)
3. Train and compare against baseline on W&B
4. Record results in the experiment's `README.md`
5. If results are good, graduate the code into `src/`

## Adding a New Architecture

1. Create `src/model/<name>.py` with the model class
2. Add it to `MODEL_REGISTRY` in `registry.py`
3. Create a config YAML in `configs/`

The training loop, data pipeline, and scripts work unchanged — only the model is swapped.

## Data Pipeline

```
HuggingFace dataset (streaming)
  → Tokenize with custom BPE
  → Concatenate into one long sequence
  → Split into train/val (99%/1%)
  → Chunk into fixed-length blocks (1024 tokens)
  → Save as memory-mapped .bin files (uint16)
```

Data is preprocessed once and reused across all experiments.

## Training Features

| Feature | Detail |
|---|---|
| Mixed precision | `torch.amp` (fp16/bf16) |
| Gradient accumulation | Simulate larger batch sizes |
| Gradient clipping | `max_norm=1.0` |
| Activation checkpointing | Trade compute for VRAM |
| Checkpointing | Full state saved to `.pt`, resume from any checkpoint |
| W&B logging | Loss, perplexity, lr, grad norm, tokens/sec |

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/). See `pyproject.toml` for the full list.

| Package | Version | Purpose |
|---|---|---|
| `torch` | >=2.9 | Model, training, mixed precision |
| `triton` | >=3.6 | Custom GPU kernels |
| `tokenizers` | >=0.22 | BPE tokenizer (Rust-backed) |
| `datasets` | >=4.5 | HuggingFace dataset streaming |
| `wandb` | >=0.24 | Experiment tracking |
| `pyyaml` | >=6.0 | Config parsing |
| `numpy` | >=2.4 | Memory-mapped data files |
