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

### Configs

Config filename = ckpt dir = W&B run name.

| Config | m (split) | is | E | k | Total | Active |
|--------|:---------:|---:|---:|---:|:-----:|:------:|
| `qwen3_moe_133m_a45m_is1024_e8_k1`  | 1 | 1024 |  8 | 1 | 133M | 45M |
| `qwen3_moe_133m_a45m_is512_e16_k2`  | 2 |  512 | 16 | 2 | 133M | 45M |
| `qwen3_moe_133m_a45m_is256_e32_k4`  | 4 |  256 | 32 | 4 | 133M | 45M |
| `qwen3_moe_133m_a45m_is128_e64_k8`  | 8 |  128 | 64 | 8 | 133M | 45M |
| `qwen3_dense_45m_is1024`            | — | 1024 |  — | — |  45M | 45M |

The dense `is=1024` baseline has the same active FFN width as every MoE cell (no routing) —
it tests whether routing buys anything over a plain dense FFN of equal active size.

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 6e-4 → 6e-5, warmup 1000, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_granularity/run.sh > logs/moe_granularity.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_granularity/qwen3_moe_133m_a45m_is256_e32_k4.yaml
```

W&B project: `pretrain-moe-granularity`.

## Results

| Config | m | is | E | k | Final val loss |
|--------|:-:|---:|---:|---:|:--------------:|
| `qwen3_moe_133m_a45m_is1024_e8_k1`  | 1 | 1024 |  8 | 1 | |
| `qwen3_moe_133m_a45m_is512_e16_k2`  | 2 |  512 | 16 | 2 | |
| `qwen3_moe_133m_a45m_is256_e32_k4`  | 4 |  256 | 32 | 4 | |
| `qwen3_moe_133m_a45m_is128_e64_k8`  | 8 |  128 | 64 | 8 | |
| `qwen3_dense_45m_is1024`            | — | 1024 |  — | — | |

## Notes

- Total params (133M) and active params (45M) are constant across the MoE sweep by
  construction; the only knob is `m`. So any val-loss trend is a pure granularity effect.
- The coarse end (`is=1024, E=8, k=1`) is top-1 routing (Switch-style); the fine end
  (`is=128, E=64, k=8`) routes to 8 of 64. Sparsity (12.5%) is held fixed throughout.
