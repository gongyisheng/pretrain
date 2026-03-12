# LLM Pretraining Codebase — Design Spec

## Overview

A pure PyTorch LLM pretraining codebase for learning, reproducing published results, and running experiments. Targets GPT-2 scale models (~124M–1.5B parameters) on a single consumer GPU (24GB VRAM).

## Goals

- **Learn**: Understand every piece of the pretraining pipeline — data, tokenizer, model, training loop, evaluation
- **Reproduce**: Validate the setup by replicating GPT-2 results
- **Experiment**: Quickly test hypotheses about architectures, optimizers, schedules, etc.

## Constraints

- Single consumer GPU (e.g. RTX 3090/4090, 24GB VRAM)
- Pure PyTorch — no heavy frameworks (no HF Trainer, no Lightning)
- Public HuggingFace datasets for training data
- W&B for monitoring

---

## Project Structure

```
pretrain/
├── configs/                      # Experiment configs (YAML)
│   ├── gpt2_small.yaml
│   └── gpt2_medium.yaml
├── src/
│   ├── model/                    # Model architectures
│   │   ├── __init__.py
│   │   ├── gpt2.py               # GPT-2 implementation
│   │   ├── components.py         # Shared building blocks (attention, MLP, norms, etc.)
│   │   ├── config.py             # Model config dataclasses
│   │   └── registry.py           # Model registry: {"gpt2": GPT2Model, ...}
│   ├── data/                     # Data pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py            # Tokenized dataset loading & chunking
│   │   └── tokenizer.py          # Custom BPE tokenizer training
│   ├── training/                 # Training loop
│   │   ├── __init__.py
│   │   ├── trainer.py            # Core training loop
│   │   ├── optimizer.py          # Optimizer & LR scheduler setup
│   │   └── logger.py             # W&B logging wrapper
│   └── utils/                    # Shared utilities
│       ├── __init__.py
│       └── config.py             # Config loading (YAML -> dataclass)
├── scripts/
│   ├── train.py                  # Entry point for training
│   ├── train_tokenizer.py        # Train custom BPE tokenizer
│   ├── preprocess_data.py        # Tokenize dataset and save to .bin files
│   └── eval.py                   # Standalone evaluation
├── experiments/                  # Code-level experiments
│   └── README.md                 # How to create and run experiments
├── requirements.txt
└── README.md
```

---

## Model Architecture

### Baseline: GPT-2 (Decoder-only Transformer)

- Causal (autoregressive) language model
- Learned positional embeddings
- Pre-LayerNorm transformer blocks
- GeLU activation
- Configurable: n_layers, n_heads, d_model, d_ff, vocab_size, max_seq_len, dropout

### Shared Components (`src/model/components.py`)

Reusable building blocks that architectures pick from. Initial implementation includes only what GPT-2 needs:

- `MultiHeadAttention` — standard scaled dot-product with causal mask
- `LayerNorm`
- `GeLU`
- `TransformerBlock` — composable from the above

Future components (add when a new architecture needs them):
- `GroupedQueryAttention`, `RMSNorm`, `SwiGLU`, `RotaryPositionalEmbedding` (RoPE)

### Model Registry (`src/model/registry.py`)

Simple dictionary mapping architecture names to model classes:

```python
MODEL_REGISTRY = {
    "gpt2": GPT2Model,
}

def build_model(config):
    return MODEL_REGISTRY[config.arch](config)
```

### Adding a New Architecture

1. Create `src/model/<name>.py` with the model class
2. Add it to `MODEL_REGISTRY`
3. Create a config YAML in `configs/`

No changes to training loop, data pipeline, or scripts required.

---

## Data Pipeline

### Tokenizer

- Train custom BPE tokenizer using HuggingFace `tokenizers` library (Rust-backed)
- Configurable vocab size (e.g. 32K, 50K)
- Script: `python scripts/train_tokenizer.py`
- Saves vocab + merges to disk

### Data Flow

```
Step 1: Train tokenizer
  python scripts/train_tokenizer.py --config configs/gpt2_small.yaml

Step 2: Preprocess data
  python scripts/preprocess_data.py --config configs/gpt2_small.yaml

  HuggingFace dataset (streaming)
    -> Tokenize with custom BPE
    -> Concatenate all tokens into one long sequence
    -> Split into train/val (e.g. 99%/1%)
    -> Chunk into fixed-length blocks (e.g. 1024 tokens)
    -> Save as memory-mapped .bin files (train.bin, val.bin)

Step 3: Train
  python scripts/train.py --config configs/gpt2_small.yaml
  DataLoader reads chunks randomly from train.bin / val.bin
```

### `.bin` File Format

- Flat array of `uint16` token IDs (sufficient for vocab sizes up to 65535)
- No header — file size / 2 = total tokens, total tokens / seq_len = number of sequences
- One file per split: `train.bin`, `val.bin`
- Created by `scripts/preprocess_data.py`, which also prints token counts and a decoded sample for verification

### Key Decisions

- **Preprocess once, train many times** — tokenize and save to `.bin` files, avoid re-tokenizing every run
- **Memory-mapped files** — handle datasets larger than RAM
- **HuggingFace streaming** — no need to download entire dataset before starting
- **Train/val split** — default 99%/1% split, configurable in YAML

---

## Training Loop

### Core Loop (`src/training/trainer.py`)

```
For each step:
  1. Fetch batch from DataLoader
  2. Forward pass (with torch.amp autocast for mixed precision)
  3. Compute loss (cross-entropy, next token prediction)
  4. Backward pass (with GradScaler for fp16 stability)
  5. Gradient clipping (max_norm=1.0)
  6. Optimizer step
  7. LR scheduler step
  8. Log metrics to W&B every N steps
  9. Evaluate on validation set every M steps
  10. Save checkpoint every K steps
```

### Training Features

| Feature | Detail |
|---------|--------|
| Mixed precision | `torch.amp` (fp16/bf16) — essential for 24GB GPU |
| Gradient accumulation | Simulate larger batch sizes without more VRAM |
| Gradient clipping | Prevent training instabilities (max_norm=1.0) |
| Activation checkpointing | `torch.utils.checkpoint` — trade compute for VRAM, needed for larger models (774M+). Configurable via `training.activation_checkpointing: true` |
| Checkpointing | Save full training state (see Checkpoint Format below). Resume from any checkpoint |
| Reproducibility | Save random seeds + RNG states, full config logged to W&B |

### Checkpoint Format

Each checkpoint is a single `.pt` file named `step_{N}.pt` containing:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "grad_scaler": scaler.state_dict(),  # if using fp16
    "step": current_step,
    "config": config_dict,               # full experiment config
    "rng_states": {                      # for exact reproducibility
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
    },
}
```

On resume, all states are restored so training continues deterministically.

### Optimizer & Scheduler (`src/training/optimizer.py`)

- Default optimizer: AdamW with weight decay
- Default schedule: linear warmup -> cosine decay
- All hyperparameters configurable via YAML

### W&B Logging (`src/training/logger.py`)

Per step:
- `train/loss`, `train/perplexity`
- `lr`, `grad_norm`, `tokens_per_sec`

Per eval:
- `val/loss`, `val/perplexity`

Also logs:
- Full config as W&B run config (for filtering/comparing runs)
- Generated text samples every eval (short completions from fixed prompts, for qualitative sanity checks)

---

## Config System

### YAML Config

A single YAML file defines an entire experiment:

```yaml
max_seq_len: 1024                   # single source of truth, shared by model and data

model:
  arch: "gpt2"
  n_layers: 12
  n_heads: 12
  d_model: 768
  d_ff: 3072                         # default: 4 * d_model
  vocab_size: 50257
  dropout: 0.1

data:
  dataset: "openwebtext"
  tokenizer_path: "tokenizers/custom_bpe_50k"
  data_dir: "data/"                # where .bin files are stored
  val_split: 0.01                  # 1% validation
  num_workers: 4

training:
  batch_size: 16
  gradient_accumulation_steps: 4
  max_steps: 100000
  mixed_precision: "bf16"
  activation_checkpointing: false  # enable for models 774M+
  grad_clip: 1.0
  checkpoint_dir: "checkpoints/"
  checkpoint_every: 5000
  eval_every: 1000
  eval_steps: 200

optimizer:
  name: "adamw"
  lr: 6e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

scheduler:
  name: "cosine"
  warmup_steps: 2000
  min_lr: 6e-5

logging:
  wandb_project: "pretrain"
  wandb_run_name: "gpt2-small-baseline"
  log_every: 10
```

### Config Loading (`src/utils/config.py`)

- YAML parsed into nested dataclasses for type safety and validation
- CLI overrides: `python scripts/train.py --config configs/gpt2_small.yaml --optimizer.lr 3e-4`
- Full config saved alongside checkpoints and logged to W&B

---

## Experiments Directory

For code-level experiments (new training tricks, custom components, etc.):

```
experiments/
├── cosine_warmup/
│   ├── README.md              # Hypothesis, setup, results
│   ├── config.yaml            # Config for this experiment
│   └── custom_scheduler.py    # Custom code override
└── muon_optimizer/
    ├── README.md
    ├── config.yaml
    └── optimizer.py
```

Rules:
- Each experiment is a self-contained folder
- Experiments import from `src/` and override only what they change
- If an experiment proves valuable, graduate it into `src/`
- README in each folder records hypothesis, setup, and results

### Override Mechanism

Experiment configs can specify custom Python modules via an `overrides` section:

```yaml
# experiments/muon_optimizer/config.yaml
overrides:
  optimizer_module: "experiments/muon_optimizer/optimizer.py"

# ... rest of config inherits from a base config
base_config: "configs/gpt2_small.yaml"
```

The training script loads the override module and uses its exported class/function in place of the default. Each overridable component (optimizer, scheduler, model) has a well-defined interface (e.g. `build_optimizer(params, config) -> Optimizer`).

### Running Experiments

```bash
# Normal run
python scripts/train.py --config configs/gpt2_small.yaml

# Experiment run (loads overrides automatically)
python scripts/train.py --config experiments/muon_optimizer/config.yaml
```

---

## Dependencies

Core:
- `torch` — model, training loop, mixed precision
- `tokenizers` — fast BPE tokenizer training
- `datasets` — HuggingFace dataset streaming/loading
- `wandb` — experiment tracking and logging
- `pyyaml` — config parsing
- `numpy` — memory-mapped data files

---

## Entry Points

| Command | Purpose |
|---------|---------|
| `python scripts/train_tokenizer.py --config configs/gpt2_small.yaml` | Train custom BPE tokenizer |
| `python scripts/train.py --config configs/gpt2_small.yaml` | Run pretraining |
| `python scripts/preprocess_data.py --config configs/gpt2_small.yaml` | Tokenize dataset and save .bin files |
| `python scripts/train.py --config configs/gpt2_small.yaml --optimizer.lr 3e-4` | Run with CLI override |
| `python scripts/eval.py --config configs/gpt2_small.yaml --checkpoint checkpoints/step_50000.pt` | Evaluate a checkpoint |
