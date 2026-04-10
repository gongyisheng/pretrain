# Gradient Accumulation

Verify whether gradient accumulation produces identical training dynamics to a single large batch, holding effective batch size constant.

## Hypothesis

With a fixed effective batch size, varying the split between hardware batch size and accumulation steps should be mathematically equivalent. In practice, differences may arise from numerical precision in gradient summation and mixed-precision scaling. If curves diverge, gradient accumulation is not a free lunch.

## Setup

Fixed: Qwen3 57M architecture, `seq_len=1024`, `lr=6e-4`, 50K steps, `seed=42`, `use_deterministic_algo=true`.

All runs have the same effective batch of 64 samples (~64K tokens/step).

| Config | batch_size | grad_accum | Effective batch |
|---|---|---|---|
| qwen3_57m_ga_1  | 64 | 1  | 64 |
| qwen3_57m_ga_2  | 32 | 2  | 64 |
| qwen3_57m_ga_4  | 16 | 4  | 64 |
| qwen3_57m_ga_8  | 8  | 8  | 64 |
| qwen3_57m_ga_16 | 4  | 16 | 64 |
| qwen3_57m_ga_32 | 2  | 32 | 64 |

## Run

```bash
nohup bash experiments/grad_accum/run.sh > logs/grad_accum.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/grad_accum/qwen3_57m_ga_8.yaml
```

## W&B

Project: `pretrain-grad-accum`. Compare runs by `val/loss` vs `train/step`.

## Results

TODO
