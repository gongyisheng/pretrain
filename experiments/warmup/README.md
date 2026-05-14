# Warmup Steps

Sweep learning-rate warmup duration to find the optimal warmup length for Qwen3 57M pretraining with cosine schedule.

## Hypothesis

Too little warmup risks early training instability (large Adam updates before variance estimates stabilize). Too much warmup wastes steps at sub-optimal LR, reducing effective training budget. Most LLM papers use 0.5%-2% of total steps; we sweep from 0% to 50% to map the full curve.

## Setup

Fixed: Qwen3 57M architecture, `seq_len=1024`, `lr=6e-4`, `min_lr=6e-5`, cosine schedule, `batch_size=16`, `grad_accum=16`, 5K steps.

| Config | warmup_ratio | warmup_steps | % of training |
|---|---|---|---|
| qwen3_57m_warmup0.0   | 0.0   | 0    | 0%    |
| qwen3_57m_warmup0.001 | 0.001 | 5    | 0.1%  |
| qwen3_57m_warmup0.002 | 0.002 | 10   | 0.2%  |
| qwen3_57m_warmup0.005 | 0.005 | 25   | 0.5%  |
| qwen3_57m_warmup0.01  | 0.01  | 50   | 1%    |
| qwen3_57m_warmup0.02  | 0.02  | 100  | 2%    |
| qwen3_57m_warmup0.05  | 0.05  | 250  | 5%    |
| qwen3_57m_warmup0.1   | 0.1   | 500  | 10%   |
| qwen3_57m_warmup0.2   | 0.2   | 1000 | 20%   |
| qwen3_57m_warmup0.5   | 0.5   | 2500 | 50%   |

## Run

```bash
nohup bash experiments/warmup/run.sh > logs/warmup.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/warmup/qwen3_57m_warmup0.02.yaml
```

## W&B

Project: `pretrain-warmup`. Compare runs by `val/loss` vs `train/step`.

## Results

TODO
