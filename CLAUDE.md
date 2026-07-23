# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Single-GPU LLM pretraining research codebase. Pure PyTorch, config-driven (YAML), W&B logging. One unified model (`TransformerLM`) assembled from registered components: `mha`/`gqa`/`mla` attention, `dense`/`moe` MLP, `rmsnorm`/`layernorm` norm, `rope`/`learned` pos_emb. Layer ops are plain functions fused by the whole-model `torch.compile(model)` in the trainer, not per-op decorators.

## Commands

```bash
# Install
uv sync

# Tests
uv run pytest                          # all tests
uv run pytest tests/fast/model/test_transformer.py       # single file
uv run pytest tests/fast/model/test_transformer.py -k "test_forward"  # single test

# Lint
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/

# Train
uv run python scripts/train.py --config configs/gpt2_124m.yaml
uv run python scripts/train.py --config configs/qwen3_51m.yaml --no-wandb
uv run python scripts/train.py --config configs/gpt2_124m.yaml --resume checkpoints/step_1000.pt

# CLI config overrides
uv run python scripts/train.py --config configs/gpt2_124m.yaml --optimizer.lr=1e-4

# Data preprocessing
uv run python scripts/preprocess_data.py --config configs/gpt2_124m.yaml

# Full pipeline (preprocess + train)
nohup uv run bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
```

## Architecture

### Layers vs. models

Reusable building blocks live in `src/layers/`: `norm.py` (RMSNorm/LayerNorm, `NORM_REGISTRY`), `pos_emb.py` (RoPE + learned, `POS_EMB_REGISTRY`), `attention.py` (MHA + GQA + MLA, `ATTN_REGISTRY`; MLA = `MultiHeadLatentAttention`, DeepSeek-V2/V3 low-rank KV/Q compression with decoupled RoPE — `qk_rope_head_dim` carries position, the shared RoPE is sized to it in `transformer.py`), `activation.py` (unary `relu/gelu/silu` + gated variants, `UNGATED_ACTIVATIONS`/`GATED_ACTIVATIONS`), `mlp.py` (`DenseMLPBlock` + `SparseMoEBlock`, `MLP_REGISTRY`; SwiGLU = `DenseMLPBlock(activation="silu", gated=True)`), `residual.py` (`StandardResidual`/`AttnResidual`, `RESIDUAL_REGISTRY`), `block.py` (`TransformerBlock`). The single unified architecture `TransformerLM` lives in `src/model/transformer.py` and is the only model class. `build_model(cfg)` in `src/model/__init__.py` constructs it from the config. Layer ops (rmsnorm, rope, `gated_mlp`/`ungated_mlp`, `_sdpa`, gated activations, moe routing/scatter/ffn) are plain module-level functions in their owning file, fused by the whole-model `torch.compile(model)` in the trainer — no per-op `@torch.compile` decorators. Explicit `torch.compile` is reserved for code that has no fused eager path or runs outside the compiled model: `loss.py` (loss is computed in the trainer, outside `self.model`) and `flex_attention`/`create_block_mask` (`_flex_attn`, `_create_block_mask_compiled`). Loss functions live in `src/training/loss.py` (`LOSS_REGISTRY`, `compute_loss`), selected via `config.training.loss_fn`.

GPT-2-style configs use `attn_cls: mha`, `mlp: [{mlp_cls: dense, mlp_kwargs: {activation: gelu, gated: false, bias: true}}]`, `norm_cls: layernorm`, `pos_emb_cls: learned`. Qwen3-style configs use `attn_cls: gqa`, `mlp: [{mlp_cls: dense, mlp_kwargs: {...}}]`, `norm_cls: rmsnorm`, `pos_emb_cls: rope`. MoE configs use `mlp: [{mlp_cls: moe, mlp_kwargs: {...}}]`.

### Model registry

`build_model(config)` in `src/model/__init__.py` constructs `TransformerLM`, which dispatches to components through the registries: `ATTN_REGISTRY`, `MLP_REGISTRY`, `NORM_REGISTRY`, `POS_EMB_REGISTRY`, `RESIDUAL_REGISTRY`. To add a new component: implement the class, add it to the relevant registry in its owning layer file.

### Config system

`src/utils/config.py` defines nested dataclasses (`TrainConfig` → `ModelConfig`, `DataConfig`, `TrainingConfig`, `OptimizerConfig`, `SchedulerConfig`, `LoggingConfig`, `DebugConfig`). Loaded from YAML with CLI override support.

`ModelConfig` uses a cls+kwargs schema for attn/norm/pos_emb/residual: `attn_cls`/`attn_kwargs`, `norm_cls`/`norm_kwargs`, `pos_emb_cls`/`pos_emb_kwargs`, `residual_cls`/`residual_kwargs`. Each `*_cls` names a registry key; each `*_kwargs` dict is forwarded directly to the component constructor. Component classes own their parameter defaults. MLP is the exception: `mlp` is a list of `{mlp_cls, mlp_kwargs, layer_idx?}` items — a single item with no `layer_idx` applies uniformly to every layer (the common case); additional items can scope an `mlp_cls`/`mlp_kwargs` override to specific layers via `layer_idx: [...]`, and at most one item may omit `layer_idx` to act as the fallback/complement for whichever layers the others didn't claim. `ModelConfig.resolve_mlp(i)` returns the resolved `(mlp_cls, mlp_kwargs)` for layer `i`; `layer_mlp_classes()`, `is_moe`, `moe_layer_kwargs`, and `aux_loss_coef` are derived from the same per-layer resolution. Top-level `ModelConfig` fields: `d_model`, `n_layers`, `vocab_size`, `dropout_embd`, `tie_word_embeddings`, `lm_head_bias`.

### Data pipeline

Raw text → BPE tokenizer (50K vocab, `tokenizers` library) → concatenated uint16 `.bin` files (memory-mapped via numpy). `PretrainDataset` serves fixed-length chunks with next-token targets. Train/val split is 99/1.

### Training

`Trainer` in `src/training/trainer.py`: mixed precision (fp16/bf16), gradient accumulation, gradient clipping, activation checkpointing, CUDA stream prefetching, full checkpoint/resume (model + optimizer + scheduler + RNG states). Spike detection in `src/training/debug.py` watches for gradient norm anomalies.

## Development Rules

### Workflow for layer/model changes

1. Run related tests before and after changes to confirm nothing breaks.
2. For perf-sensitive changes, run `benchmarks/bench_train.py` before/after to guard against regressions.

### Config defaults and validation

All default config values and validation logic live in `ModelConfig.__post_init__` (and the other dataclasses) in `src/utils/config.py` — not in component constructors. Components (attention/mlp/etc.) take resolved values as explicit args; `__post_init__` fills `*_kwargs` defaults via `setdefault` and raises on invalid combinations. This keeps `config.py` the single source of truth for what a config means (e.g. MLA head dims default off `d_model // n_heads` there, and `transformer.py` reads them back from the populated `attn_kwargs`).

### Dtype handling

Fused ops must support float32, float16, and bfloat16. Never hardcode a dtype or cast input tensors to a specific dtype. Preserve the caller's dtype throughout — accept it, compute in it (or an explicitly documented accumulation dtype like float32 for reductions), and return in it.

### Experiments

`experiments/` contains self-contained experiment folders (e.g., `scaling_law/`, `attn_res/`) each with their own configs, run scripts, and results. Scaling law experiments sweep model sizes (16M–145M) across both architectures.

Every experiment folder must include a `README.md` with: hypothesis, setup table (configs, key params, approx param counts), run command, results table (filled in after running), and notes.

Experiment YAML configs should explicitly set `batch_size`, `gradient_accumulation_steps`, `checkpoint_every`, `eval_every`, and `eval_steps` to the default values from `src/utils/config.py`. Use `batch_size: 16`, `gradient_accumulation_steps: 16`, `checkpoint_every: 5000`, `eval_every: 100`, `eval_steps: 25` unless the experiment intentionally changes them (`checkpoint_every` is raised to 5000 from the config default of 500 to avoid excessive checkpoint writes across many runs).

Model configs use the cls+kwargs schema (attn/norm/pos_emb/residual); `mlp` is a list of `{mlp_cls, mlp_kwargs, layer_idx?}` items. A Qwen3-style 57M config looks like:
```yaml
model:
  d_model: 512
  n_layers: 8
  vocab_size: 50257
  attn_cls: gqa
  attn_kwargs: {n_heads: 8, n_kv_heads: 4, qk_norm: true}
  mlp:
    - {mlp_cls: dense, mlp_kwargs: {activation: silu, gated: true}}
  norm_cls: rmsnorm
  pos_emb_cls: rope
  pos_emb_kwargs: {rope_theta: 10000.0}
```
A single `mlp` item with no `layer_idx` applies uniformly to every layer (as above). To override specific layers, add more items with `layer_idx: [...]`; at most one item may omit `layer_idx`, and it acts as the fallback for whichever layers the others didn't claim.

MoE replaces the single dense item with `mlp: [{mlp_cls: moe, mlp_kwargs: {n_routed_experts: N, n_routed_experts_per_token: K, n_shared_experts: 0, ...}}]`, and can add further items scoped with `layer_idx` (e.g. to keep layer 0 dense while the rest are MoE).

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

In experiment `run.sh` scripts, build sweep config lists with nested `for` loops over the swept axes (e.g., `lrs`, `wds`) appending to a `configs` array, rather than enumerating every `lr × wd` combination by hand. Keeps the grid auditable and easy to extend.
