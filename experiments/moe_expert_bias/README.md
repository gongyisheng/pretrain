# MoE Expert-Bias Update-Rate Sweep

## Hypothesis

Auxiliary-loss-free load balancing (DeepSeek, arXiv:2408.15664) adds a
per-expert bias to the gating scores **for top-k selection only** and nudges it
each step by `b_i += u · sign(mean_load − load_i)`. No term is added to the loss,
so there is no interference gradient. The update rate `u`
(`expert_bias_update_rate`) controls balancing speed:

- Too small → bias adapts too slowly, experts stay imbalanced early in training.
- Too large → bias oscillates, routing churns and destabilizes learning.

DeepSeek-V3 uses `u = 0.001`. This sweep finds the `u` that minimizes validation
loss on our testbed.

## Setup

Fixed testbed: `qwen3_moe_133m` architecture (133M total, ~35M active), 64
experts, top-2, capacity factor 1.25. `expert_bias: true` (so `aux_loss` is off);
only `expert_bias_update_rate` varies.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 2 |
| expert_capacity_factor | 1.25 |
| batch × grad_accum × seq | 16 × 16 × 1024 (≈0.26M tok/step) |
| max_steps | 8000 (≈2.1B tokens) |
| lr / min_lr / warmup | 6e-4 / 6e-5 / 300 |

| Config | expert_bias_update_rate |
|--------|-------------------------|
| `qwen3_moe_expert_bias_rate1e-4` | 0.0001 |
| `qwen3_moe_expert_bias_rate3e-4` | 0.0003 |
| `qwen3_moe_expert_bias_rate1e-3` | 0.001 (DeepSeek-V3 default) |
| `qwen3_moe_expert_bias_rate3e-3` | 0.003 |
| `qwen3_moe_expert_bias_rate1e-2` | 0.01 |

## Run

```bash
nohup bash experiments/moe_expert_bias/run.sh > logs/moe_expert_bias.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_expert_bias/qwen3_moe_expert_bias_rate1e-3.yaml
```

## Results

| expert_bias_update_rate | val_loss | notes |
|-------------------------|----------|-------|
| 0.0001 | | |
| 0.0003 | | |
| 0.001  | | |
| 0.003  | | |
| 0.01   | | |

Best: _TBD_

## Notes

- With `expert_bias` on, the block returns no aux loss, so `train/aux_loss` is
  not logged — compare runs by `val_loss`. (Balance is enforced by the bias rule
  rather than a loss term.)
- The bias update runs once per micro-batch under `model.train()`. With
  `gradient_accumulation_steps=16`, that is 16 sign-steps per optimizer step;
  the effective rate is ~16× `u`. The sign rule is self-correcting, so this just
  shifts the useful `u` range — accounted for by sweeping two orders of magnitude.
- Compare against `../moe_aux_loss/` (same testbed) for aux-loss vs loss-free.
