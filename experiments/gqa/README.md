# GQA KV Head Size Sweep

Sweep the number of KV heads on Qwen3 57M to measure how GQA ratio affects pretraining quality across the full spectrum from MQA to MHA.

## Hypothesis

Reducing KV heads saves attention parameters and KV-cache memory with minimal quality loss. MHA (n_kv_heads=8) should be the quality ceiling; MQA (n_kv_heads=1) may degrade noticeably. A sweet spot likely exists around GQA ratio 2-4.

## Setup

| Config | n_kv_heads | GQA ratio | Description | Approx params |
|---|---|---|---|---|
| qwen3_57m_kv1 | 1 | 8 | Multi-Query Attention | ~55M |
| qwen3_57m_kv2 | 2 | 4 | Grouped Query Attention | ~56M |
| qwen3_57m_kv4 | 4 | 2 | GQA (current default) | ~57M |
| qwen3_57m_kv8 | 8 | 1 | Multi-Head Attention | ~59M |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 6.5K steps (~1.7B tokens, ~30x params), lr=6e-4, cosine schedule with 200 warmup steps (~3%), min_lr=6e-5, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/gqa/run.sh > logs/gqa.log 2>&1 &
```

## Results

| n_kv_heads | GQA ratio | Final Val Loss |
|---|---|---|
| 1 | 8 (MQA) | |
| 2 | 4 | |
| 4 | 2 | |
| 8 | 1 (MHA) | |

## Notes
