# MLA Attention: Baseline Comparison + Compression Sweeps

Evaluate Multi-head Latent Attention (MLA, DeepSeek-V2/V3) on Qwen3 57M in four
phases: first against the GQA default and MHA ceiling, then sweep MLA's knobs one
axis at a time around a fixed center. Every variant is its own standalone YAML
(no CLI overrides), and each phase has its own run script.

## Hypothesis

MLA trades a larger attention parameter budget (up/down projections through the
latent) for a much smaller KV cache. At matched per-head QK score dim we expect
MLA to roughly **match GQA/MHA quality** — the win is inference memory, not loss.

The MLA **center** (`qwen3_57m_mla.yaml`: nope=32, rope=32, v_head=64,
kv_lora=512, q_lora=0) is chosen so each head's QK score dim (`nope + rope` =
64) equals the MHA/GQA `head_dim` (`d_model/n_heads` = 64) — a clean
apples-to-apples comparison. kv_lora=512 makes the latent near-uncompressed
(reconstruction target `n_heads*(nope+v_head)` = 768, ratio 0.67), so the head
structure is tested before compression is dialed in. The center is the shared
reference row in the Phase 2/3/4 tables.

## Setup

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8), seq_len=1024,
batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 50K steps
(~13.1B tokens), lr=6e-4, cosine schedule with 1000 warmup steps (2%),
min_lr=6e-5, bf16, OpenWebText. Baselines (MHA/GQA) use `qk_norm=true`; MLA uses
its own latent RMSNorms instead.

KV cache per token (floats): MHA `2*8*64=1024`, GQA `2*4*64=512`, MLA
`kv_lora_rank + qk_rope_head_dim`.

### Phase 1 — attention comparison (`run_compare_attn.sh`)

| Config | Attention | KV cache | Params |
|---|---|---|---|
| qwen3_57m_mha | MHA, 8 heads, qk_norm | 1024 | 59.32M |
| qwen3_57m_gqa | GQA, n_kv_heads=4, qk_norm | 512 | 57.22M |
| qwen3_57m_mla (center) | MLA, nope32/rope32/v64/kv512 | 544 | 60.50M |

### Phase 2 — kv_lora_rank sweep (`run_kv_lora_rank.sh`)

nope=32, rope=32, v_head=64 fixed. Reconstruction target = 768 dims.

| Config | kv_lora_rank | ratio (rank/768) | KV cache | Params |
|---|---|---|---|---|
| qwen3_57m_mla_kvlora_64 | 64 | 0.083 (12x) | 96 | 55.91M |
| qwen3_57m_mla_kvlora_128 | 128 | 0.17 (6x) | 160 | 56.57M |
| qwen3_57m_mla_kvlora_256 | 256 | 0.33 (3x) | 288 | 57.88M |
| qwen3_57m_mla_kvlora_384 | 384 | 0.50 (2x) | 416 | 59.19M |
| qwen3_57m_mla (center) | 512 | 0.67 (1.5x) | 544 | 60.50M |

### Phase 3 — qk_nope:qk_rope split (`run_qk_nope_rope.sh`)

Total QK head dim fixed at 64; v_head=64, kv_lora=512 fixed. Tests how to divide
the QK budget between content matching (nope) and decoupled position (rope).

| Config | nope:rope | Params |
|---|---|---|
| qwen3_57m_mla_nope_8_rope_56 | 8:56 | 59.81M |
| qwen3_57m_mla_nope_16_rope_48 | 16:48 | 60.04M |
| qwen3_57m_mla_nope_24_rope_40 | 24:40 | 60.27M |
| qwen3_57m_mla (center) | 32:32 | 60.50M |
| qwen3_57m_mla_nope_40_rope_24 | 40:24 | 60.73M |
| qwen3_57m_mla_nope_48_rope_16 | 48:16 | 60.96M |
| qwen3_57m_mla_nope_56_rope_8 | 56:8 | 61.19M |

### Phase 4 — v_head_dim sweep (`run_v_head_dim.sh`)

nope=32, rope=32, kv_lora=512 fixed. Note: kv_lora is held fixed, so larger
v_head both widens values AND raises the latent reconstruction target
`8*(32+v_head)` — at v_head=256 the latent compresses 4.5x, so this axis is not
purely "value capacity".

| Config | v_head_dim | recon target | Params |
|---|---|---|---|
| qwen3_57m_mla_vhead_16 | 16 | 384 | 57.36M |
| qwen3_57m_mla_vhead_32 | 32 | 512 | 58.41M |
| qwen3_57m_mla (center) | 64 | 768 | 60.50M |
| qwen3_57m_mla_vhead_128 | 128 | 1280 | 64.70M |
| qwen3_57m_mla_vhead_256 | 256 | 2304 | 73.09M |

### Phase 5 — q_lora_rank sweep (`run_q_lora_rank.sh`)

nope=32, rope=32, v_head=64, kv_lora=512 fixed. Query compression: q is factored
through a q_lora_rank latent instead of a direct projection. q output is
`n_heads*qk_head` = 512, so r=512 is rank-equivalent to the direct q_proj (modulo
q_a_norm) and r<256 saves params; r=256 is param break-even with the center
(q_lora=0). A params/compute knob, not a KV-cache knob.

| Config | q_lora_rank | Params |
|---|---|---|
| qwen3_57m_mla (center) | 0 (direct, V2-Lite) | 60.50M |
| qwen3_57m_mla_qlora_64 | 64 | 58.93M |
| qwen3_57m_mla_qlora_128 | 128 | 59.45M |
| qwen3_57m_mla_qlora_256 | 256 | 60.50M |
| qwen3_57m_mla_qlora_384 | 384 | 61.55M |
| qwen3_57m_mla_qlora_512 | 512 | 62.60M |

## Run

Run Phase 1 first; its MLA center is the shared reference row for Phases 2–5.

```bash
nohup bash experiments/mla/run_compare_attn.sh  > logs/mla_compare_attn.log  2>&1 &
nohup bash experiments/mla/run_kv_lora_rank.sh  > logs/mla_kv_lora_rank.log  2>&1 &
nohup bash experiments/mla/run_qk_nope_rope.sh  > logs/mla_qk_nope_rope.log  2>&1 &
nohup bash experiments/mla/run_v_head_dim.sh    > logs/mla_v_head_dim.log    2>&1 &
nohup bash experiments/mla/run_q_lora_rank.sh   > logs/mla_q_lora_rank.log   2>&1 &
```

## Results

### Phase 1 — comparison

| Run | Final Val Loss |
|---|---|
| mha | |
| gqa | |
| mla (center) | |

### Phase 2 — kv_lora_rank

| kv_lora_rank | Final Val Loss |
|---|---|
| 64 | |
| 128 | |
| 256 | |
| 384 | |
| 512 (center) | |

### Phase 3 — qk_nope:qk_rope

| nope:rope | Final Val Loss |
|---|---|
| 8:56 | |
| 16:48 | |
| 24:40 | |
| 32:32 (center) | |
| 40:24 | |
| 48:16 | |
| 56:8 | |

### Phase 4 — v_head_dim

| v_head_dim | Final Val Loss |
|---|---|
| 16 | |
| 32 | |
| 64 (center) | |
| 128 | |
| 256 | |

### Phase 5 — q_lora_rank

| q_lora_rank | Final Val Loss |
|---|---|
| 0 (center) | |
| 64 | |
| 128 | |
| 256 | |
| 384 | |
| 512 | |

## Notes

- Phase 1 holds total QK score dim equal across MHA/GQA/MLA (64), so MLA's
  comparison is not inflated by a wider score space. MLA's parameter footprint
  (60.5M) is slightly above MHA (59.3M) and GQA (57.2M); the KV-cache advantage
  shows up in inference memory (544 vs 1024/512 floats per token), not in
  train-time params or loss.
- Phase 2 deliberately uses the near-uncompressed center (kv_lora=512) as the
  top of the sweep and scales down to find where the latent bottleneck bites.
- One-axis-at-a-time (star) design: each axis's optimum is conditional on where
  the others are pinned. Re-center later phases on earlier winners if a phase
  comes back non-flat.
