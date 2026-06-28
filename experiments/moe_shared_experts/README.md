# MoE Shared Experts (DeepSeekMoE)

## Hypothesis

DeepSeekMoE adds always-on **shared experts** alongside the top-k **routed experts**:
every token passes through the shared experts (which capture common knowledge), while
the router specializes the rest. The question: at a fixed per-expert width, does spending
capacity on shared experts beat (or complement) spending it on more routed experts active
per token?

We sweep both axes independently and compare final validation loss.

## Setup

Common backbone (Qwen3-style, all 12 runs identical except the MLP block):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 128 |
| `n_routed_experts` (pool) | 64 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Axes:
- **shared** = `n_shared_experts` ∈ {0, 1, 2, 4} — always-on FFN of width `shared × 128`.
- **routed** = `n_routed_experts_per_token` (top-k) ∈ {1, 2, 4} — over the fixed 64-expert pool.

All cells use `moe` with `aux_loss_coef=0.01` and dynamic capacity (no token dropping).
`shared=0` is the no-shared-expert baseline (pure routed MoE).

Active experts per token = `shared + routed`. Per-expert width is held constant, so the grid
isolates *where* the active FFN capacity is spent.

### Configs

Config filename = ckpt dir = W&B run name. `tot` = total params, `act` = active params.

| Config | shared | routed (k) | Total | Active |
|--------|:------:|:----------:|:-----:|:------:|
| `qwen3_133m_a34m_s0_r1` | 0 | 1 | 133M | 34M |
| `qwen3_133m_a35m_s0_r2` | 0 | 2 | 133M | 35M |
| `qwen3_133m_a39m_s0_r4` | 0 | 4 | 133M | 39M |
| `qwen3_135m_a35m_s1_r1` | 1 | 1 | 135M | 35M |
| `qwen3_135m_a37m_s1_r2` | 1 | 2 | 135M | 37M |
| `qwen3_135m_a40m_s1_r4` | 1 | 4 | 135M | 40M |
| `qwen3_136m_a37m_s2_r1` | 2 | 1 | 136M | 37M |
| `qwen3_136m_a39m_s2_r2` | 2 | 2 | 136M | 39M |
| `qwen3_136m_a42m_s2_r4` | 2 | 4 | 136M | 42M |
| `qwen3_139m_a40m_s4_r1` | 4 | 1 | 139M | 40M |
| `qwen3_139m_a42m_s4_r2` | 4 | 2 | 139M | 42M |
| `qwen3_139m_a45m_s4_r4` | 4 | 4 | 139M | 45M |

Training (all runs): batch 16 × grad-accum 16 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens, fixed budget for a controlled comparison), cosine LR 1e-3 → 1e-4,
warmup 1000, bf16.

## Running

```bash
# All 12 configs sequentially:
nohup bash experiments/moe_shared_experts/run.sh > logs/moe_shared_experts.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_shared_experts/qwen3_136m_a39m_s2_r2.yaml
```

W&B project: `pretrain-moe-shared-experts`.

## Results

| Config | shared | routed (k) | Active | Final val loss |
|--------|:------:|:----------:|:------:|:--------------:|
| `qwen3_133m_a34m_s0_r1` | 0 | 1 | 34M | |
| `qwen3_133m_a35m_s0_r2` | 0 | 2 | 35M | |
| `qwen3_133m_a39m_s0_r4` | 0 | 4 | 39M | |
| `qwen3_135m_a35m_s1_r1` | 1 | 1 | 35M | |
| `qwen3_135m_a37m_s1_r2` | 1 | 2 | 37M | |
| `qwen3_135m_a40m_s1_r4` | 1 | 4 | 40M | |
| `qwen3_136m_a37m_s2_r1` | 2 | 1 | 37M | |
| `qwen3_136m_a39m_s2_r2` | 2 | 2 | 39M | |
| `qwen3_136m_a42m_s2_r4` | 2 | 4 | 42M | |
| `qwen3_139m_a40m_s4_r1` | 4 | 1 | 40M | |
| `qwen3_139m_a42m_s4_r2` | 4 | 2 | 42M | |
| `qwen3_139m_a45m_s4_r4` | 4 | 4 | 45M | |

## Notes

- Compare along constant active-param contours (e.g. `s0-r2` vs `s1-r1` ≈ 35M active) to see
  whether shared experts are a better use of the same active FFN capacity than routing.
- `s0` rows are the pure-routed-MoE baselines (no shared expert).
- Total params (~133-139M) are dominated by the stored 64-expert pool and barely move across
  the grid; the controlled quantity is *active* params, not total.
