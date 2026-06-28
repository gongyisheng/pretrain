# MoE Expert-Bias Update-Rate Sweep

## Hypothesis

Auxiliary-loss-free load balancing (DeepSeek, arXiv:2408.15664) adds a
per-expert bias to the gating scores **for top-k selection only** and nudges it
each step by `b_i += u · sign(mean_load − load_i)`. No term is added to the loss,
so there is no interference gradient. The update rate `u`
(`expert_bias_update_rate`) controls balancing speed:

- Too small → bias adapts too slowly, experts stay imbalanced early in training.
- Too large → bias oscillates, routing churns and destabilizes learning.

DeepSeek-V3 uses `u = 0.001`. This is a coarse decade-spaced first pass to
bracket the useful range; a finer sweep follows once the range is known.

## Setup

Fixed testbed: `qwen3_183m_a51m` architecture (183M total, ~51M active), 64
experts, top-8, capacity factor 1.25. `expert_bias: true` (so `aux_loss` is off);
only `expert_bias_update_rate` varies. Active intermediate width
`k·is = 8·192 = 1536` (3·d_model, Qwen3-0.6B FFN ratio). Same testbed as
`../moe_aux_loss/` so the two balancing schemes are directly comparable.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 8 |
| intermediate_size (per expert) | 192 |
| expert_capacity_factor | 1.25 |
| batch × grad_accum × seq | 16 × 16 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 5e-4 / 5e-5 / 1500 |

| Config | expert_bias_update_rate |
|--------|-------------------------|
| `qwen3_183m_a51m_expert_bias_rate1e-5` | 0.00001 |
| `qwen3_183m_a51m_expert_bias_rate1e-4` | 0.0001 |
| `qwen3_183m_a51m_expert_bias_rate1e-3` | 0.001 (DeepSeek-V3 default) |
| `qwen3_183m_a51m_expert_bias_rate1e-2` | 0.01 |

Plus an aux-loss baseline on the same testbed as a benchmark for the loss-free
expert-bias scheme (`expert_bias: false`, `aux_loss: true`):

| Config | aux_loss_coef |
|--------|---------------|
| `qwen3_183m_a51m_aux_coef1e-2` | 0.01 |

## Run

```bash
nohup bash experiments/moe_expert_bias/run.sh > logs/moe_expert_bias.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_expert_bias/qwen3_183m_a51m_expert_bias_rate1e-3.yaml
```

## Results

| scheme | param | val_loss | notes |
|--------|-------|----------|-------|
| expert_bias | u=0.00001 | | |
| expert_bias | u=0.0001 | | |
| expert_bias | u=0.001  | | DeepSeek-V3 default |
| expert_bias | u=0.01   | | |
| aux_loss    | coef=0.01 | | benchmark baseline |

Best: _TBD_ → pick finer range from here.

## Notes

- With `expert_bias` on, the block returns no aux loss, so `train/aux_loss` is
  not logged — compare runs by `val_loss`. (Balance is enforced by the bias rule
  rather than a loss term.)
- The bias update runs once per micro-batch under `model.train()`. With
  `gradient_accumulation_steps=32`, that is 32 sign-steps per optimizer step;
  the effective rate is ~32× `u`. The sign rule is self-correcting, so this just
  shifts the useful `u` range — accounted for by sweeping two orders of magnitude.
- Compare against `../moe_aux_loss/` (same testbed) for aux-loss vs loss-free.
