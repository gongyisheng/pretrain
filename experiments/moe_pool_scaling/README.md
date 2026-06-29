# MoE Pool Scaling

## Hypothesis

Does adding **more routed experts** help at fixed compute? Here we hold top-8 routing and
the per-expert size (`is=192`) fixed and grow the routed pool `E`. Active capacity
(`k·is = 1536`) — and therefore FLOPs per token — stays constant; only the total expert pool
(and total params) grows. This isolates the value of *more total expert capacity* at a fixed
active budget, the complementary axis to the granularity sweep.

## Setup

Common backbone (Qwen3-style, identical across all runs except the expert count):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 192 |
| `n_routed_experts_per_token` (k) | 8 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariant: active capacity `k·is = 1536` (≈51M active params) for all runs. All cells use
`aux_loss_coef=0.01` and dynamic capacity (no token dropping). `is` = per-expert
`intermediate_size`. `E=64` is the **`qwen3_183m_a51m` benchmark** shared across the moe
experiments; the sweep grows and shrinks the pool around it.

### Configs

Config filename = ckpt dir = W&B run name.

| Config | E (pool) | k | Pool (E·is) | Total | Active | Sparsity (k/E) |
|--------|:--------:|:-:|:-----------:|:-----:|:------:|:--------------:|
| `qwen3_70m_a51m_is192_e16_k8`   |  16 | 8 |  3072 |  70M | 51M | 50%   |
| `qwen3_108m_a51m_is192_e32_k8`  |  32 | 8 |  6144 | 108M | 51M | 25%   |
| `qwen3_183m_a51m_is192_e64_k8`  |  64 | 8 | 12288 | 183M | 51M | 12.5% |
| `qwen3_335m_a51m_is192_e128_k8` | 128 | 8 | 24576 | 335M | 51M | 6.25% |
| `qwen3_637m_a51m_is192_e256_k8` | 256 | 8 | 49152 | 637M | 51M | 3.1%  |

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 5e-4 → 5e-5, warmup 1500, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_pool_scaling/run.sh > logs/moe_pool_scaling.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_pool_scaling/qwen3_108m_a51m_is192_e32_k8.yaml
```

W&B project: `pretrain-moe-pool-scaling`.

## Results

| Config | E (pool) | Total | Active | Final val loss |
|--------|:--------:|:-----:|:------:|:--------------:|
| `qwen3_70m_a51m_is192_e16_k8`   |  16 |  70M | 51M | |
| `qwen3_108m_a51m_is192_e32_k8`  |  32 | 108M | 51M | |
| `qwen3_183m_a51m_is192_e64_k8`  |  64 | 183M | 51M | |
| `qwen3_335m_a51m_is192_e128_k8` | 128 | 335M | 51M | |
| `qwen3_637m_a51m_is192_e256_k8` | 256 | 637M | 51M | |

## Notes

- Active params (~51M) and FLOPs/token are constant; only total params (the pool) grow.
- As the pool grows, sparsity falls from 50% (E=16) to 3.1% (E=256) — fewer of the experts
  are active per token. Diminishing returns are expected once the pool is large relative to
  what top-8 routing can usefully exploit.
- This is the fixed-active-size counterpart to `moe_granularity` (which fixes the pool and
  varies how it is split).
- **Memory:** the largest cell (`E=256`, 637M total) holds ~10 GB of weights + AdamW state;
  add `--training.activation_checkpointing=true` if it OOMs.
