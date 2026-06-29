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

Invariants across the MoE sweep: pool `E·is = 12288`, active `k·is = 1536`, sparsity
`k/E = 1/8` (12.5%). `is` = per-expert `intermediate_size`. All MoE cells use
`aux_loss_coef=0.01` and `expert_capacity_factor=1.25` (fixed capacity, drops overflow).

The active width `k·is = 1536 = 3·d_model` is the Qwen3-0.6B FFN ratio. **Granularity** is
`G = d_ff / d_expert` (DeepSeekMoE / fine-grained scaling laws), where `d_expert = is` and
`d_ff` is the intermediate width of the equivalent dense FFN. Taking the active-width dense
twin `d_ff = 1536` gives `G = 1536/is = m` — the split factor *is* the absolute granularity
here, and the coarsest cell (`is=1536, k=1`) has `G=1`.

### Configs

Config filename = ckpt dir = W&B run name. `m` = split factor = `G = 1536/is`.

| Config | m | is | E | k | Total | Active |
|--------|:-:|---:|---:|---:|:-----:|:------:|
| `qwen3_183m_a51m_is1536_e8_k1`  | 1 | 1536 |  8 | 1 | 183M | 51M |
| `qwen3_183m_a51m_is768_e16_k2`  | 2 |  768 | 16 | 2 | 183M | 51M |
| `qwen3_183m_a51m_is384_e32_k4`  | 4 |  384 | 32 | 4 | 183M | 51M |
| `qwen3_183m_a51m_is192_e64_k8`  | 8 |  192 | 64 | 8 | 183M | 51M |
| `qwen3_51m`                     | — | 1536 |  — | — |  51M | 51M |

`qwen3_51m` is the dense reference (no routing): a plain gated SwiGLU FFN of width
`is=1536` (3·d_model), i.e. the same active FFN width as every MoE cell. It tests whether
routing buys anything over a dense FFN of equal active size. It has marginally fewer params
than the iso-active MoE cells (50.9M vs ~51M active), so it is an upper-bound anchor for the
granularity axis, not an iso-compute point.

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 5e-4 → 5e-5, warmup 1500, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_granularity/run.sh > logs/moe_granularity.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_granularity/qwen3_183m_a51m_is384_e32_k4.yaml
```

W&B project: `pretrain-moe-granularity`.

## Results

| Config | m | is | E | k | Final val loss |
|--------|:-:|---:|---:|---:|:--------------:|
| `qwen3_183m_a51m_is1536_e8_k1`  | 1 | 1536 |  8 | 1 | |
| `qwen3_183m_a51m_is768_e16_k2`  | 2 |  768 | 16 | 2 | |
| `qwen3_183m_a51m_is384_e32_k4`  | 4 |  384 | 32 | 4 | |
| `qwen3_183m_a51m_is192_e64_k8`  | 8 |  192 | 64 | 8 | |
| `qwen3_51m`                     | — | 1536 |  — | — | |

## Notes

- Total params (183M) and active params (51M) are constant across the MoE sweep by
  construction; the only knob is `m`. So any val-loss trend is a pure granularity effect.
- The coarse end (`is=1536, E=8, k=1`) is top-1 routing (Switch-style); the fine end
  (`is=192, E=64, k=8`) routes to 8 of 64. Sparsity (12.5%) is held fixed throughout.
