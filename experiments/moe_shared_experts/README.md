# MoE Shared Experts (DeepSeekMoE)

## Hypothesis

DeepSeekMoE adds always-on **shared experts** alongside the top-k **routed experts**:
every token passes through the shared experts (which capture common knowledge), while
the router specializes the rest. The question: at a fixed active capacity, how should the
active experts split between always-on shared and top-k routed?

We fix the number of active experts per token at **8** (`shared + routed = 8`, active
intermediate `8·192 = 1536`, the `qwen3_183m_a51m` benchmark) and slide the split.

## Setup

Common backbone (Qwen3-style, all runs identical except the MLP split):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 192 |
| `n_routed_experts` (pool) | 64 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariant: active experts per token `s + k = 8`, so active intermediate `= 1536` (≈51M
active params) and FLOPs/token are constant. All cells use `moe` with aux-loss-free
load balancing (`expert_bias: true`, `expert_bias_update_rate=0.001`)
and no expert capacity limit (no token drops). `s0_r8` is the pure-routed benchmark (no shared
expert); as `s` grows the always-on shared FFN takes over more of the fixed budget.

### Configs

Config filename = ckpt dir = W&B run name. `s` = `n_shared_experts`, `k` = top-k routed.

| Config | shared (s) | routed (k) | Total | Active |
|--------|:----------:|:----------:|:-----:|:------:|
| `qwen3_183m_a51m_s0_r8` | 0 | 8 | 183M | 51M |
| `qwen3_186m_a51m_s1_r7` | 1 | 7 | 186M | 51M |
| `qwen3_188m_a51m_s2_r6` | 2 | 6 | 188M | 51M |
| `qwen3_190m_a51m_s3_r5` | 3 | 5 | 190M | 51M |
| `qwen3_193m_a51m_s4_r4` | 4 | 4 | 193M | 51M |
| `qwen3_51m` | — | — | 51M | 51M |

`qwen3_51m` is the dense reference: a plain SwiGLU MLP with `intermediate_size: 1536`
(= the MoE's active intermediate `8·192`), so its total ≈ active ≈ 51M. It bounds what the
same active-FLOP budget buys with no routing at all.

Training (all runs): batch 32 × grad-accum 8 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens, fixed budget for a controlled comparison), cosine LR 1e-3 → 1e-4,
warmup 1500, bf16. Router uses a sigmoid score fn (`router_score_fn: sigmoid`).

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_shared_experts/run.sh > logs/moe_shared_experts.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_shared_experts/qwen3_193m_a51m_s4_r4.yaml
```

W&B project: `pretrain-moe-shared-experts`.

## Results

| Config | shared (s) | routed (k) | Active | Final val loss |
|--------|:----------:|:----------:|:------:|:--------------:|
| `qwen3_183m_a51m_s0_r8` | 0 | 8 | 51M | |
| `qwen3_186m_a51m_s1_r7` | 1 | 7 | 51M | |
| `qwen3_188m_a51m_s2_r6` | 2 | 6 | 51M | |
| `qwen3_190m_a51m_s3_r5` | 3 | 5 | 51M | |
| `qwen3_193m_a51m_s4_r4` | 4 | 4 | 51M | |
| `qwen3_51m` | — | — | 51M | |

## Notes

- Active params (~51M) and FLOPs/token are constant across the sweep; only the shared/routed
  split (and the small always-on shared-expert param count) changes. So any val-loss trend is
  a pure shared-vs-routed allocation effect.
- `s0_r8` is the pure-routed-MoE baseline (= the `183m_a51m` benchmark). More shared experts
  trade router flexibility for always-on common-knowledge capacity.
