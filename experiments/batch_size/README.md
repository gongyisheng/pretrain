# Batch Size × Learning Rate × Minimum Learning Rate

## Hypothesis

At a fixed 13.1B-token budget, the best peak and minimum learning rates may depend on effective batch size. A proportional minimum (`0.1 × lr`) may behave differently from a fixed `1e-5` floor.

## Setup

Sweep effective batch sizes of 64, 256, and 1024 sequences. For each batch size, sweep peak learning rates `1e-4`, `2e-4`, `5e-4`, `1e-3`, `2e-3`, and `5e-3`, with a minimum learning rate of either 10% of the peak or fixed at `1e-5`. This yields 33 unique runs because both minimum-LR choices are `1e-5` at peak LR `1e-4`.

## Run

Check `nvidia-smi`, then replace `0` below with a free GPU index and run the full sweep:

```bash
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs
nohup bash experiments/batch_size/run.sh > logs/batch_size.log 2>&1 &
```

Run one configuration:

```bash
uv run python scripts/train.py --config experiments/batch_size/qwen3_51m_bs_256_lr_1e-3_min_lr_1e-4.yaml
```

## Results

| Configuration | Final `val/loss` |
|---|---:|
| qwen3_51m_bs_64_lr_1e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_2e-4_min_lr_2e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_2e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_5e-4_min_lr_5e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_5e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_1e-3_min_lr_1e-4.yaml | Pending |
| qwen3_51m_bs_64_lr_1e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_2e-3_min_lr_2e-4.yaml | Pending |
| qwen3_51m_bs_64_lr_2e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_64_lr_5e-3_min_lr_5e-4.yaml | Pending |
| qwen3_51m_bs_64_lr_5e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_1e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_2e-4_min_lr_2e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_2e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_5e-4_min_lr_5e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_5e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_1e-3_min_lr_1e-4.yaml | Pending |
| qwen3_51m_bs_256_lr_1e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_2e-3_min_lr_2e-4.yaml | Pending |
| qwen3_51m_bs_256_lr_2e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_256_lr_5e-3_min_lr_5e-4.yaml | Pending |
| qwen3_51m_bs_256_lr_5e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_1e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_2e-4_min_lr_2e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_2e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_5e-4_min_lr_5e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_5e-4_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_1e-3_min_lr_1e-4.yaml | Pending |
| qwen3_51m_bs_1024_lr_1e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_2e-3_min_lr_2e-4.yaml | Pending |
| qwen3_51m_bs_1024_lr_2e-3_min_lr_1e-5.yaml | Pending |
| qwen3_51m_bs_1024_lr_5e-3_min_lr_5e-4.yaml | Pending |
| qwen3_51m_bs_1024_lr_5e-3_min_lr_1e-5.yaml | Pending |

## Notes

This experiment is inspired by [Step Law](https://arxiv.org/abs/2503.04715).

Runs log to W&B project `pretrain-batch-size`. Compare `val/loss` against `train/total_tokens`; all configurations share the same total-token budget.
