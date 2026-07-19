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
bracket the useful range; a finer sweep follows once the range is known. The
sweep is run twice — once with a **softmax** router and once with a **sigmoid**
router (`router_score_fn`) — since the useful `u` range can shift with the
gating nonlinearity.

## Setup

Fixed testbed: `qwen3_188m_a51m` architecture (188M total, ~51M active), 64
routed experts + 2 always-on shared experts, top-6 routed, no expert capacity
limit. `expert_bias: true` (so `aux_loss` is off); only `expert_bias_update_rate`
varies. Active intermediate width `(s + k)·is = 8·192 = 1536` (3·d_model,
Qwen3-0.6B FFN ratio); the expert bias balances only the 6 routed experts. Same
testbed as `../moe_aux_loss/` so the two balancing schemes are directly comparable.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_routed_experts / top-k | 64 / 6 |
| n_shared_experts | 2 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

Each `{gate}` ∈ {`softmax`, `sigmoid`}:

| Config | expert_bias_update_rate |
|--------|-------------------------|
| `qwen3_188m_a51m_{gate}_expert_bias_rate1e-4` | 0.0001 |
| `qwen3_188m_a51m_{gate}_expert_bias_rate1e-3` | 0.001 (DeepSeek-V3 default) |
| `qwen3_188m_a51m_{gate}_expert_bias_rate1e-2` | 0.01 |

Plus a per-gate aux-loss baseline on the same testbed as a benchmark for the
loss-free expert-bias scheme (`expert_bias: false`, `aux_loss: true`):

| Config | aux_loss_coef |
|--------|---------------|
| `qwen3_188m_a51m_{gate}_aux_coef1e-3` | 0.001 |

## Run

```bash
nohup bash experiments/moe_expert_bias/run_softmax.sh > logs/moe_expert_bias_softmax.log 2>&1 &
nohup bash experiments/moe_expert_bias/run_sigmoid.sh > logs/moe_expert_bias_sigmoid.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_expert_bias/qwen3_188m_a51m_softmax_expert_bias_rate1e-3.yaml
```

## Results

| gate | scheme | param | val_loss | notes |
|------|--------|-------|----------|-------|
| softmax | expert_bias | u=0.0001 | | |
| softmax | expert_bias | u=0.001  | | DeepSeek-V3 default |
| softmax | expert_bias | u=0.01   | | |
| softmax | aux_loss    | coef=0.001 | | benchmark baseline |
| sigmoid | expert_bias | u=0.0001 | | |
| sigmoid | expert_bias | u=0.001  | | DeepSeek-V3 default |
| sigmoid | expert_bias | u=0.01   | | |
| sigmoid | aux_loss    | coef=0.001 | | benchmark baseline |

Best per gate: _TBD_ → pick finer range from here.

## Notes

- With `expert_bias` on, the block returns no aux loss, so `train/aux_loss` is
  not logged — compare runs by `val_loss`. (Balance is enforced by the bias rule
  rather than a loss term.)
- The bias update runs once per micro-batch under `model.train()`. With
  `gradient_accumulation_steps=4`, that is 4 sign-steps per optimizer step;
  the effective rate is ~4× `u`. The sign rule is self-correcting, so this just
  shifts the useful `u` range — accounted for by sweeping two orders of magnitude.
- Compare against `../moe_aux_loss/` (same testbed) for aux-loss vs loss-free.
