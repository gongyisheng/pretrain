# MoE Pool Scaling

## Hypothesis

Does adding **more routed experts** help at fixed compute? Here we hold top-2 routing and
the per-expert size (`is=128`) fixed and grow the routed pool `E`. Active capacity
(`k·is = 256`) — and therefore FLOPs per token — stays constant; only the total expert pool
(and total params) grows. This isolates the value of *more total expert capacity* at a fixed
active budget, the complementary axis to the granularity sweep.

## Setup

Common backbone (Qwen3-style, identical across all runs except the expert count):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 128 |
| `n_routed_experts_per_token` (k) | 2 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariant: active capacity `k·is = 256` (≈35M active params) for all runs. All cells use
`aux_loss_coef=0.01` and dynamic capacity (no token dropping). `is` = per-expert
`intermediate_size`.

### Configs

Config filename = ckpt dir = W&B run name.

| Config | E (pool) | k | Pool (E·is) | Total | Active | Sparsity (k/E) |
|--------|:--------:|:-:|:-----------:|:-----:|:------:|:--------------:|
| `qwen3_45m_a35m_is128_e8_k2`    |   8 | 2 |  1024 |  45M | 35M | 25%   |
| `qwen3_57m_a35m_is128_e16_k2`   |  16 | 2 |  2048 |  57M | 35M | 12.5% |
| `qwen3_83m_a35m_is128_e32_k2`   |  32 | 2 |  4096 |  83M | 35M | 6.25% |
| `qwen3_133m_a35m_is128_e64_k2`  |  64 | 2 |  8192 | 133M | 35M | 3.1%  |
| `qwen3_234m_a36m_is128_e128_k2` | 128 | 2 | 16384 | 234M | 36M | 1.6%  |

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 1e-3 → 1e-4, warmup 1000, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_pool_scaling/run.sh > logs/moe_pool_scaling.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_pool_scaling/qwen3_83m_a35m_is128_e32_k2.yaml
```

W&B project: `pretrain-moe-pool-scaling`.

## Results

| Config | E (pool) | Total | Active | Final val loss |
|--------|:--------:|:-----:|:------:|:--------------:|
| `qwen3_45m_a35m_is128_e8_k2`    |   8 |  45M | 35M | |
| `qwen3_57m_a35m_is128_e16_k2`   |  16 |  57M | 35M | |
| `qwen3_83m_a35m_is128_e32_k2`   |  32 |  83M | 35M | |
| `qwen3_133m_a35m_is128_e64_k2`  |  64 | 133M | 35M | |
| `qwen3_234m_a36m_is128_e128_k2` | 128 | 234M | 36M | |

## Notes

- Active params (~35M) and FLOPs/token are constant; only total params (the pool) grow.
- As the pool grows, sparsity falls from 25% (E=8) to 1.6% (E=128) — fewer of the experts
  are active per token. Diminishing returns are expected once the pool is large relative to
  what top-2 routing can usefully exploit.
- This is the fixed-active-size counterpart to `moe_granularity` (which fixes the pool and
  varies how it is split).
