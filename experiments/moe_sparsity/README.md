# MoE Sparsity

## Hypothesis

Does adding **more routed experts** help at fixed compute? Here we hold the active experts
(2 shared + top-6 routed = 8) and the per-expert size (`is=192`) fixed and grow the routed
pool `E`. Active capacity (`(s+k)·is = 1536`) — and therefore FLOPs per token — stays
constant; only the total expert pool (and total params) grows. This isolates the value of
*more total expert capacity* at a fixed active budget, the complementary axis to the
granularity sweep.

## Setup

Common backbone (Qwen3-style, identical across all runs except the expert count):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 192 |
| `n_shared_experts` (s) | 2 |
| `n_routed_experts_per_token` (k) | 6 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariant: active experts per token `s + k = 8`, so active capacity `(s+k)·is = 1536`
(≈51M active params) for all runs. Load balancing is aux-loss-free (`expert_bias: true`,
`expert_bias_update_rate=0.001`, `router_score_fn: sigmoid`); no expert capacity limit (no
token drops). `is` = per-expert `intermediate_size`. `E=64` is the **`qwen3_188m_a51m`
benchmark** (the `s2_r6` cell of `moe_shared_experts`) shared across the moe experiments;
the sweep grows and shrinks the pool around it.

### Configs

Config filename = ckpt dir = W&B run name.

All cells fix `s=2` shared + `k=6` routed active experts. Sparsity is `k/E` (routed).

| Config | E (pool) | s | k | Pool (E·is) | Total | Active | Sparsity (k/E) |
|--------|:--------:|:-:|:-:|:-----------:|:-----:|:------:|:--------------:|
| `qwen3_75m_a51m_is192_e16_s2r6`   |  16 | 2 | 6 |  3072 |  75M | 51M | 37.5%  |
| `qwen3_112m_a51m_is192_e32_s2r6`  |  32 | 2 | 6 |  6144 | 112M | 51M | 18.75% |
| `qwen3_188m_a51m_is192_e64_s2r6`  |  64 | 2 | 6 | 12288 | 188M | 51M | 9.4%   |
| `qwen3_339m_a51m_is192_e128_s2r6` | 128 | 2 | 6 | 24576 | 339M | 51M | 4.7%   |
| `qwen3_642m_a51m_is192_e256_s2r6` | 256 | 2 | 6 | 49152 | 642M | 51M | 2.3%   |

Training (all runs): batch 64 × grad-accum 4 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), cosine LR 1e-3 → 1e-4, warmup 1500, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_sparsity/run.sh > logs/moe_sparsity.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_sparsity/qwen3_112m_a51m_is192_e32_s2r6.yaml
```

W&B project: `pretrain-moe-sparsity`.

## Results

| Config | E (pool) | Total | Active | Final val loss |
|--------|:--------:|:-----:|:------:|:--------------:|
| `qwen3_75m_a51m_is192_e16_s2r6`   |  16 |  75M | 51M | |
| `qwen3_112m_a51m_is192_e32_s2r6`  |  32 | 112M | 51M | |
| `qwen3_188m_a51m_is192_e64_s2r6`  |  64 | 188M | 51M | |
| `qwen3_339m_a51m_is192_e128_s2r6` | 128 | 339M | 51M | |
| `qwen3_642m_a51m_is192_e256_s2r6` | 256 | 642M | 51M | |

## Notes

- Active params (~51M) and FLOPs/token are constant; only total params (the pool) grow.
- As the pool grows, routed sparsity falls from 37.5% (E=16) to 2.3% (E=256) — fewer of the
  routed experts are active per token. Diminishing returns are expected once the pool is
  large relative to what top-6 routing can usefully exploit.
- This is the fixed-active-size counterpart to `moe_granularity` (which fixes the pool and
  varies how it is split).
- **Memory:** the largest cell (`E=256`, 642M total) holds ~10 GB of weights + optimizer state;
  reduce `batch_size` (raising `gradient_accumulation_steps` to compensate) if it OOMs.
