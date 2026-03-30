# Dropout Sweep

Sweep dropout rate on Qwen3 57M to measure its effect on pretrain loss.

## Hypothesis

Higher dropout hurts pretraining by randomly zeroing activations, increasing noise and slowing convergence. Very high dropout (≥0.5) should significantly degrade loss. Low dropout (0.01–0.05) may act as mild regularization with negligible cost, while 0.0 (no dropout) should serve as the best baseline for a compute-limited pretraining run.

## Setup

| Config | Dropout |
|---|---|
| qwen3_57m_drop0.0 | 0.0 |
| qwen3_57m_drop0.01 | 0.01 |
| qwen3_57m_drop0.02 | 0.02 |
| qwen3_57m_drop0.05 | 0.05 |
| qwen3_57m_drop0.1 | 0.1 |
| qwen3_57m_drop0.2 | 0.2 |
| qwen3_57m_drop0.5 | 0.5 |
| qwen3_57m_drop0.9 | 0.9 |
| qwen3_57m_drop0.95 | 0.95 |
| qwen3_57m_drop0.99 | 0.99 |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=4, grad_accum=8 (effective batch=32), lr=1e-4, 50K steps, cosine schedule with 1K warmup steps, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/dropout/run.sh > logs/dropout.log 2>&1 &
```

## Results

| Dropout | Final Val Loss |
|---|---|
| 0.0 | |
| 0.01 | |
| 0.02 | |
| 0.05 | |
| 0.1 | |
| 0.2 | |
| 0.5 | |
| 0.9 | |
| 0.95 | |
| 0.99 | |
