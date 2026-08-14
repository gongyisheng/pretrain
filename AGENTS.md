# AGENTS.md

## Project

Single-GPU LLM pretraining research codebase: pure PyTorch, YAML-configured, with W&B logging. `TransformerLM` is the one model class and is assembled from registered attention (`mha`/`gqa`/`mla`), MLP (`dense`/`moe`), norm (`rmsnorm`/`layernorm`), positional-embedding (`rope`/`learned`), and residual components. The trainer compiles the whole model with `torch.compile(model)`; layer operations remain plain functions rather than individually compiled decorators.

## Commands

```bash
# Install
uv sync

# Tests: always pass -n explicitly. Run fast and e2e trees separately.
uv run pytest tests/fast -n 6
uv run pytest tests/fast/layers -n 6
uv run pytest tests/fast/quant/test_moe.py -n 6
uv run pytest tests/fast/kernel -n 12 --dist load
uv run pytest tests/fast/quant -n 12 --dist load
uv run pytest tests/fast/metrics -n 12 --dist load
uv run pytest tests/e2e -n 0
uv run pytest tests/fast/model/test_transformer.py -n 0 -k "test_forward"

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Train
uv run python scripts/train.py --config configs/gpt2_124m.yaml
uv run python scripts/train.py --config configs/qwen3_51m.yaml --no-wandb
uv run python scripts/train.py --config configs/gpt2_124m.yaml --resume checkpoints/step_1000.pt

# CLI config override
uv run python scripts/train.py --config configs/gpt2_124m.yaml --optimizer.lr=1e-4

# Data preprocessing
uv run python scripts/preprocess_data.py --config configs/gpt2_124m.yaml

# Full pipeline
nohup uv run bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
```

## Architecture

Reusable building blocks live in `src/layers/`; `TransformerLM` is in `src/model/transformer.py`, and `build_model(cfg)` in `src/model/__init__.py` constructs it through component registries. Losses are selected by `config.training.loss_fn` in `src/training/loss.py`. The data path is raw text → BPE tokenizer → concatenated uint16 `.bin` files; `PretrainDataset` supplies fixed-length next-token chunks. `Trainer` provides mixed precision, accumulation, clipping, activation checkpointing, prefetching, and checkpoint/resume.

## Development Rules

Run relevant tests before and after layer/model changes. For performance-sensitive work, benchmark with `benchmarks/bench_train.py` before and after.

Files under `docs/superpowers/` are local working artifacts and must never be committed. Keep the directory ignored; if any file there is already tracked, remove it from Git tracking while preserving the local file.

### GPU work

Always run `nvidia-smi` before GPU tests, training, or benchmarks, then pin a free device with `CUDA_VISIBLE_DEVICES=<idx>`. Do not assume GPU count or VRAM. Training is single-device; another GPU only isolates concurrent runs.

### Tests

Always specify `-n`. Use `-n 6` by default, including subsets and single files. Use `-n 12 --dist load` for `kernel`, `quant`, and `metrics`; use `-n 0` only for e2e or debugging.

### Configuration and components

`src/utils/config.py` owns config defaults and validation: `ModelConfig.__post_init__` fills resolved kwargs and rejects invalid combinations. Components receive explicit resolved values. Norm, positional embedding, and residual use `*_cls` plus `*_kwargs`; attention and MLP are per-layer `{*_cls, *_kwargs, layer_idx?}` lists. One unscoped entry applies to every unclaimed layer; scoped entries override named layers, and at most one unscoped fallback is allowed. `resolve_attn(i)` and `resolve_mlp(i)` are the source of per-layer resolution.

Keep canonical `model:` key order: `d_model`, `n_layers`, `vocab_size`, `attn`, `mlp`, then remaining keys. All rope-bearing layers must share a rope head dimension, and all layers must use the same `attn_implementation`. Add a component in its owning layer file and register it in the corresponding registry.

### Dtypes

Fused operations must support float32, float16, and bfloat16. Preserve the caller's dtype throughout; only use an explicitly documented accumulation dtype (such as float32 for reductions).

### Experiments

Each `experiments/` folder is self-contained and must include a `README.md` covering the hypothesis, setup table (configs, key parameters, approximate parameter count), run command, results table, and notes. Experiment YAML must explicitly set `batch_size: 16`, `gradient_accumulation_steps: 16`, `checkpoint_every: 5000`, `eval_every: 100`, and `eval_steps: 25` unless intentionally changed. Build sweep config lists in `run.sh` with nested loops over swept axes.
