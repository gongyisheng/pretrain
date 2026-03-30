# Batch Size

Measure how effective batch size (tokens per gradient step) affects final validation loss, holding all other hyperparameters fixed.

## Hypothesis

Larger batch sizes reduce gradient noise, leading to faster initial convergence but potentially worse generalization. With a fixed learning rate and step budget, smaller batches see more parameter updates over the same token budget, which may yield lower final loss.

## Setup

Fixed: Qwen3 57M architecture, `seq_len=512`, `lr=6e-4`, 50K steps.

The hardware batch size stays at 8 across all runs — only `gradient_accumulation_steps` varies to control the effective batch size.

| Config | batch_size | grad_accum | Tokens/step |
|---|---|---|---|
| qwen3_57m_bs_8k | 8 | 2 | ~8K |
| qwen3_57m_bs_32k | 8 | 8 | ~32K |
| qwen3_57m_bs_128k | 8 | 32 | ~128K |
| qwen3_57m_bs_512k | 8 | 128 | ~512K |

Note: total tokens seen = `tokens/step × max_steps`. Larger batches therefore process far more tokens total, but take the same number of optimizer steps.

## Run

```bash
nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/batch_size/qwen3_57m_bs_32k.yaml
```

## W&B

Project: `pretrain-batch-size`. Compare runs by `val/loss` vs `train/step`.

## Results

TODO
