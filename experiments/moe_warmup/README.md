# MoE Warmup Steps

Sweep LR warmup duration for the Qwen3-A51M MoE testbed to find the optimal warmup length with the cosine schedule and Muon optimizer, across two expert layouts (no-shared vs 2-shared).

## Hypothesis

Too little warmup risks early instability (large updates before optimizer state stabilizes), which is sharper for MoE where the router is also warming up and load balance is still forming. Too much warmup wastes steps at sub-optimal LR. This is a diagnostic run: the cosine schedule is sized to the full 50K steps but training early-stops at 10K to observe whether the model reaches balanced expert load under different warmup lengths, not a full training sweep.

## Setup

Fixed: Qwen3-A51M testbed, sigmoid router, aux-loss 1e-3, `seq_len=1024`, `lr=1e-3`, `min_lr=1e-4`, cosine schedule over `max_steps=50000`, Muon optimizer, `batch_size=64`, `grad_accum=4`, `early_stop=10000`. Two expert layouts are swept in parallel (8 active experts either way), varying only `warmup_steps`:

- **qwen3_183m_a51m** — 64 routed, top-8, 0 shared.
- **qwen3_188m_a51m_s2r6** — 64 routed, top-6, 2 shared. The always-on shared experts have stable gradients from step 0, leaving the router fewer active experts to balance during warmup.

| Config | model | shared/routed | warmup_steps | % of schedule |
|---|---|---|---|---|
| qwen3_183m_a51m_warmup500       | 183M | 0 / top-8 | 500  | 1% |
| qwen3_183m_a51m_warmup1000      | 183M | 0 / top-8 | 1000 | 2% |
| qwen3_183m_a51m_warmup1500      | 183M | 0 / top-8 | 1500 | 3% |
| qwen3_183m_a51m_warmup2000      | 183M | 0 / top-8 | 2000 | 4% |
| qwen3_188m_a51m_s2r6_warmup500  | 188M | 2 / top-6 | 500  | 1% |
| qwen3_188m_a51m_s2r6_warmup1000 | 188M | 2 / top-6 | 1000 | 2% |
| qwen3_188m_a51m_s2r6_warmup1500 | 188M | 2 / top-6 | 1500 | 3% |
| qwen3_188m_a51m_s2r6_warmup2000 | 188M | 2 / top-6 | 2000 | 4% |

## Run

```bash
nohup bash experiments/moe_warmup/run_s0r8.sh > logs/moe_warmup_s0r8.log 2>&1 &   # no-shared, 183M
nohup bash experiments/moe_warmup/run_s2r6.sh > logs/moe_warmup_s2r6.log 2>&1 &   # 2-shared, 188M
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/moe_warmup/qwen3_188m_a51m_s2r6_warmup1000.yaml
```

## W&B

Project: `pretrain-moe-warmup`. Primary signal is expert load balance (`train-moe/maxvio`); also watch `val/loss` vs `train/step`.

## Results

TODO
