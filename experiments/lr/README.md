# Learning Rate Sweep

Sweep learning rate on Qwen3 57M to find the optimal LR for this model size and token budget.

## Hypothesis

There is a compute-optimal learning rate for a given model size and training duration. Too small → slow convergence and underfitting within the step budget; too large → instability and divergence.

## Setup

| Config | LR | min_lr |
|---|---|---|
| qwen3_57m_lr1e-5 | 1e-5 | 1e-6 |
| qwen3_57m_lr2e-5 | 2e-5 | 2e-6 |
| qwen3_57m_lr5e-5 | 5e-5 | 5e-6 |
| qwen3_57m_lr1e-4 | 1e-4 | 1e-5 |
| qwen3_57m_lr2e-4 | 2e-4 | 2e-5 |
| qwen3_57m_lr5e-4 | 5e-4 | 5e-5 |
| qwen3_57m_lr1e-3 | 1e-3 | 1e-4 |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=4, grad_accum=8 (effective batch=32), 50K steps, cosine schedule with 1K warmup steps, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/lr/run.sh > logs/lr.log 2>&1 &
```

## Results

| LR | Final Val Loss |
|---|---|
| 1e-5 | |
| 2e-5 | |
| 5e-5 | |
| 1e-4 | |
| 2e-4 | |
| 5e-4 | |
| 1e-3 | |
