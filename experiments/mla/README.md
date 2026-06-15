# MLA Attention: Baseline Comparison + Compression Sweeps

Evaluate Multi-head Latent Attention (MLA, DeepSeek-V2/V3) on Qwen3 57M: first
against the GQA default and MHA ceiling, then sweep MLA's three knobs
(`kv_lora_rank`, `q_lora_rank`, `qk_rope_head_dim`) one axis at a time around a
fixed center. Every variant is its own standalone YAML (no CLI overrides).

## Hypothesis

MLA trades a larger attention parameter budget (up/down projections through the
latent) for a much smaller KV cache. At fixed pretraining FLOPs and matched params
we expect MLA to roughly **match GQA/MHA quality** — the win is inference memory,
not loss. For the sweeps:

- **kv_lora_rank**: too small a KV latent bottlenecks K/V reconstruction and hurts
  loss; returns should flatten past a sweet spot (expected ~256 at this scale).
- **q_lora_rank**: query compression (V2 style) mainly saves params; at 57M it may
  cost a little quality vs `q_lora_rank=0` (V2-Lite style). Tests whether it's free.
- **qk_rope_head_dim**: the decoupled positional budget. Very small (16) may
  under-serve position; gains should saturate by 32–48.

## Setup

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8), MLA `qk_nope_head_dim=64`
/ `v_head_dim=64`, seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256,
~262K tokens/step), 6.5K steps (~1.7B tokens, ~30x params), lr=6e-4, cosine schedule
with 200 warmup steps (~3%), min_lr=6e-5, bf16, OpenWebText. Baselines (MHA/GQA) use
`qk_norm=true`; MLA uses its own latent RMSNorms instead.

The MLA **center** (`qwen3_57m_mla.yaml`: kv_lora=256, q_lora=0, qk_rope=32) is the
shared row in all three sweep tables.

### A. Baseline comparison

| Config | Attention | Approx total |
|---|---|---|
| qwen3_57m_mha | MHA, 8 heads, qk_norm | ~59.3M |
| qwen3_57m_gqa | GQA, n_kv_heads=4, qk_norm | ~57.2M |
| qwen3_57m_mla | MLA, kv_lora=256, q_lora=0, rope=32 | ~59.5M |

### B. kv_lora_rank sweep (q_lora=0, qk_rope=32 fixed)

| Config | kv_lora_rank | Approx total |
|---|---|---|
| qwen3_57m_mla_kvlora128 | 128 | ~57.9M |
| qwen3_57m_mla (center) | 256 | ~59.5M |
| qwen3_57m_mla_kvlora384 | 384 | ~61.0M |
| qwen3_57m_mla_kvlora512 | 512 | ~62.6M |

### C. q_lora_rank on/off (kv_lora=256, qk_rope=32 fixed)

| Config | q_lora_rank | Description | Approx total |
|---|---|---|---|
| qwen3_57m_mla (center) | 0 | no query compression (V2-Lite) | ~59.5M |
| qwen3_57m_mla_qlora384 | 384 | query compression (V2) | ~60.2M |

### D. qk_rope_head_dim sweep (kv_lora=256, q_lora=0 fixed)

| Config | qk_rope_head_dim | Approx total |
|---|---|---|
| qwen3_57m_mla_rope16 | 16 | ~58.9M |
| qwen3_57m_mla (center) | 32 | ~59.5M |
| qwen3_57m_mla_rope48 | 48 | ~60.0M |
| qwen3_57m_mla_rope64 | 64 | ~60.6M |

## Run

```bash
nohup bash experiments/mla/run.sh > logs/mla.log 2>&1 &
```

## Results

### A. Baseline

| Run | Final Val Loss |
|---|---|
| mha | |
| gqa | |
| mla (center) | |

### B. kv_lora_rank

| kv_lora_rank | Final Val Loss |
|---|---|
| 128 | |
| 256 (center) | |
| 384 | |
| 512 | |

### C. q_lora_rank

| q_lora_rank | Final Val Loss |
|---|---|
| 0 (center) | |
| 384 | |

### D. qk_rope_head_dim

| qk_rope_head_dim | Final Val Loss |
|---|---|
| 16 | |
| 32 (center) | |
| 48 | |
| 64 | |

## Notes

- At 57M, MLA's parameter footprint is similar to MHA and slightly above GQA — the
  KV-cache advantage doesn't show up in train-time params or loss, only in inference
  memory (not measured here).
