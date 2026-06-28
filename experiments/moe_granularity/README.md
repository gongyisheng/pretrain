# MoE Expert Granularity

## Hypothesis

Does **expert granularity** help? DeepSeekMoE argues that splitting each expert into
several finer experts (and routing to proportionally more of them) improves the loss at
the same compute, because finer experts give the router a more flexible, higher-resolution
combination of specialists.

To isolate granularity, we split each expert into `m` finer ones while scaling top-k by the
same `m`. This holds **both** the total expert pool (`E·is`) **and** the active capacity
(`k·is`) constant — only the granularity factor `m` changes. If finer-grained wins here, it
is attributable to granularity alone, not to more parameters or more compute.

## Setup

Common backbone (Qwen3-style, identical across all runs except the MLP block):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariants across the MoE sweep: pool `E·is = 8192`, active `k·is = 1024`, sparsity
`k/E = 1/8` (12.5%). `is` = per-expert `intermediate_size`. All MoE cells use
`aux_loss_coef=0.01` and dynamic capacity (no token dropping).

**Granularity** is defined (DeepSeekMoE / fine-grained scaling laws) as `G = d_ff / d_expert`,
where `d_expert = is` and `d_ff` is the intermediate width of the *equivalent dense FFN*. There
are two standard choices for `d_ff` at `d_model = 512`:
- **vanilla** `d_ff = 4·d_model = 2048` (textbook Transformer FFN) → `G = 2048/is`
- **SwiGLU-matched** `d_ff = (8/3)·d_model ≈ 1365`, rounded to a multiple of 128 → `1408`
  (param-matched gated FFN, what Qwen/Llama use) → `G = 1408/is`

The sweep's split factor `m` is granularity *relative to the `is=1024` (2·d_model) baseline*;
the absolute `G` differs by a constant. Three dense references anchor these frames: `is=1024`
(active-width baseline), `is=1408` (SwiGLU-standard, `G=1`), `is=2048` (vanilla-standard, `G=1`).

### Configs

Config filename = ckpt dir = W&B run name.

`m` = split factor (granularity vs. the `is=1024` baseline). `G@4d = 2048/is`,
`G@8/3d = 1408/is` are the absolute granularities under the two dense references.

| Config | m | is | E | k | G@4d | G@8/3d | Total | Active |
|--------|:-:|---:|---:|---:|:----:|:------:|:-----:|:------:|
| `qwen3_133m_a45m_is1024_e8_k1`  | 1 | 1024 |  8 | 1 |  2 |  1.38 | 133M | 45M |
| `qwen3_133m_a45m_is512_e16_k2`  | 2 |  512 | 16 | 2 |  4 |  2.75 | 133M | 45M |
| `qwen3_133m_a45m_is256_e32_k4`  | 4 |  256 | 32 | 4 |  8 |  5.50 | 133M | 45M |
| `qwen3_133m_a45m_is128_e64_k8`  | 8 |  128 | 64 | 8 | 16 | 11.00 | 133M | 45M |
| `qwen3_45m_is1024`              | — | 1024 |  — | — |  — |  —    |  45M | 45M |
| `qwen3_49m_is1408`              | — | 1408 |  — | — |  — |  —    |  49M | 49M |
| `qwen3_57m_is2048`              | — | 2048 |  — | — |  — |  —    |  57M | 57M |

Three dense references (no routing):
- `is=1024` (2·d_model) — same active FFN width as every MoE cell; tests whether routing buys
  anything over a plain dense FFN of equal active size.
- `is=1408` (8/3·d_model ≈ 1365, rounded to mult of 128) — the SwiGLU param-standard FFN, i.e.
  absolute granularity `G=1`.
- `is=2048` (4·d_model) — the vanilla param-standard FFN, i.e. `G=1` in the 4·d_model frame.

The dense references have *more* active params than the iso-active MoE cells (49M/57M vs 45M),
so they are upper-bound anchors for the granularity axis, not iso-compute points.

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 1e-3 → 1e-4, warmup 1000, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_granularity/run.sh > logs/moe_granularity.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_granularity/qwen3_133m_a45m_is256_e32_k4.yaml
```

W&B project: `pretrain-moe-granularity`.

## Results

| Config | m | is | E | k | Final val loss |
|--------|:-:|---:|---:|---:|:--------------:|
| `qwen3_133m_a45m_is1024_e8_k1`  | 1 | 1024 |  8 | 1 | |
| `qwen3_133m_a45m_is512_e16_k2`  | 2 |  512 | 16 | 2 | |
| `qwen3_133m_a45m_is256_e32_k4`  | 4 |  256 | 32 | 4 | |
| `qwen3_133m_a45m_is128_e64_k8`  | 8 |  128 | 64 | 8 | |
| `qwen3_45m_is1024`              | — | 1024 |  — | — | |
| `qwen3_49m_is1408`              | — | 1408 |  — | — | |
| `qwen3_57m_is2048`              | — | 2048 |  — | — | |

## Notes

- Total params (133M) and active params (45M) are constant across the MoE sweep by
  construction; the only knob is `m`. So any val-loss trend is a pure granularity effect.
- The coarse end (`is=1024, E=8, k=1`) is top-1 routing (Switch-style); the fine end
  (`is=128, E=64, k=8`) routes to 8 of 64. Sparsity (12.5%) is held fixed throughout.
