# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-GPU LLM pretraining research codebase. Pure PyTorch, config-driven (YAML), W&B logging. Two model architectures: GPT-2 (MHA baseline) and Qwen3 (GQA + RoPE + RMSNorm + SwiGLU). Pluggable kernel backends: `torch` (torch.compile fused ops) and `triton` (hand-written Triton kernels).

## Commands

```bash
# Install
uv sync

# Tests
uv run pytest                          # all tests
uv run pytest tests/test_gpt2.py       # single file
uv run pytest tests/test_gpt2.py -k "test_forward"  # single test

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Train
uv run python scripts/train.py --config configs/gpt2_124m.yaml
uv run python scripts/train.py --config configs/qwen3_57m.yaml --no-wandb
uv run python scripts/train.py --config configs/gpt2_124m.yaml --resume checkpoints/step_1000.pt

# CLI config overrides
uv run python scripts/train.py --config configs/gpt2_124m.yaml --optimizer.lr=1e-4 --training.backend=triton

# Data preprocessing
uv run python scripts/preprocess_data.py --config configs/gpt2_124m.yaml

# Full pipeline (preprocess + train)
nohup uv run bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
```

## Architecture

### Kernel backend system

`src/model/components.py` holds global function pointers (`_rmsnorm`, `_rope`, `_swiglu`, `_flash_attn`) set by `set_backend("torch"|"triton")`. All model code calls these pointers — never imports kernel functions directly. Kernel implementations live in `src/kernel/torch/` and `src/kernel/triton/`. The backend is selected via `training.backend` in the YAML config.

Cross-entropy has its own selection path in the trainer (`src/kernel/torch/cross_entropy.py` and `src/kernel/triton/cross_entropy.py`), configured by `training.backend`.

### Model registry

Models register via `@register_model("name")` decorator in `src/model/registry.py`. `create_model(config)` dispatches by `config.model.architecture`. Adding a new architecture: create `src/model/<name>.py`, decorate the class, and add a YAML config.

### Config system

`src/utils/config.py` defines nested dataclasses (`TrainConfig` → `ModelConfig`, `DataConfig`, `TrainingConfig`, `OptimizerConfig`, `SchedulerConfig`, `LoggingConfig`, `DebugConfig`). Loaded from YAML with CLI override support. The `ModelConfig.architecture` field drives model selection. The `ModelConfig.extra` dict passes architecture-specific params (e.g., `num_kv_heads`, `rope_theta`, `qk_norm`).

### Data pipeline

Raw text → BPE tokenizer (50K vocab, `tokenizers` library) → concatenated uint16 `.bin` files (memory-mapped via numpy). `PretrainDataset` serves fixed-length chunks with next-token targets. Train/val split is 99/1.

### Training

`Trainer` in `src/training/trainer.py`: mixed precision (fp16/bf16), gradient accumulation, gradient clipping, activation checkpointing, CUDA stream prefetching, full checkpoint/resume (model + optimizer + scheduler + RNG states). Spike detection in `src/training/debug.py` watches for gradient norm anomalies.

## Development Rules

### Workflow for kernel/model changes

1. Run related tests before and after changes to confirm nothing breaks.
2. Run the relevant benchmark before and after changes to check for performance regression. Avoid merging changes that degrade performance.

### Kernel dtype handling

Kernels must support float32, float16, and bfloat16. Never hardcode a dtype or cast input tensors to a specific dtype. Preserve the caller's dtype throughout — accept it, compute in it (or an explicitly documented accumulation dtype like float32 for reductions), and return in it.

### Experiments

`experiments/` contains self-contained experiment folders (e.g., `scaling_law/`, `attn_res/`) each with their own configs, run scripts, and results. Scaling law experiments sweep model sizes (16M–145M) across both architectures.

Every experiment folder must include a `README.md` with: hypothesis, setup table (configs, key params, approx param counts), run command, results table (filled in after running), and notes.

Experiment YAML configs should explicitly set `batch_size`, `gradient_accumulation_steps`, `checkpoint_every`, `eval_every`, and `eval_steps` to the default values from `src/utils/config.py` (16, 16, 500, 100, 25) unless the experiment intentionally changes them.

Standard LR by model size (from scaling law experiments, `min_lr` = `lr / 10`):

| Model size | lr | min_lr |
|---|---|---|
| ~16M | 1e-3 | 1e-4 |
| ~30M | 8e-4 | 8e-5 |
| ~57M | 6e-4 | 6e-5 |
| ~124-145M | 6e-4 | 6e-5 |

Training duration and warmup:
- `total_tokens` = model params x 20-40 (Chinchilla range, 30x is a good default)
- `max_steps` = `total_tokens` / (`batch_size` x `gradient_accumulation_steps` x `max_seq_len`)
- `warmup_steps` = 2%-5% of `max_steps`
