# MoE Warmup Steps

Sweep LR warmup duration for the Qwen3 183M-A51M MoE testbed to find the optimal warmup length with the cosine schedule and Muon optimizer.

## Hypothesis

Too little warmup risks early instability (large updates before optimizer state stabilizes), which is sharper for MoE where the router is also warming up and load balance is still forming. Too much warmup wastes steps at sub-optimal LR. This is a diagnostic run: the cosine schedule is sized to the full 50K steps but training early-stops at 10K to observe whether the model reaches balanced expert load under different warmup lengths, not a full training sweep.

## Setup

Fixed: Qwen3 183M-A51M (64 routed experts, top-8, softmax router, aux-loss 1e-3), `seq_len=1024`, `lr=1e-3`, `min_lr=1e-4`, cosine schedule over `max_steps=50000`, Muon optimizer, `batch_size=16`, `grad_accum=16`, `early_stop=10000`. Only `warmup_steps` varies.

| Config | warmup_steps | % of schedule |
|---|---|---|
| qwen3_183m_a51m_warmup500  | 500  | 1% |
| qwen3_183m_a51m_warmup1000 | 1000 | 2% |
| qwen3_183m_a51m_warmup1500 | 1500 | 3% |
| qwen3_183m_a51m_warmup2000 | 2000 | 4% |

## Run

```bash
nohup bash experiments/moe_warmup/run.sh > logs/moe_warmup.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/moe_warmup/qwen3_183m_a51m_warmup1000.yaml
```

## W&B

Project: `pretrain-moe-warmup`. Primary signal is expert load balance (`train-moe/maxvio`); also watch `val/loss` vs `train/step`.

## Results

TODO
