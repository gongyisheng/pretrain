# MoE Aux-Loss Coefficient Sweep

## Hypothesis

The Switch Transformer load-balancing loss adds `α · E · Σ_i f_i·P_i` to the
objective. `α` (`aux_loss_coef`) trades off balancing strength against
interference with the language-modeling gradient:

- Too small → experts collapse, tokens dropped at capacity, val loss rises.
- Too large → the balancing gradient dominates, hurting LM quality.

There should be a sweet spot near the Switch default `0.01`. This is a coarse
first pass over decades (plus a `0` no-balancing baseline) to bracket the useful
range; a finer sweep follows once the range is known. We sweep the same coef grid
for both router score functions — **softmax** and **sigmoid** — since the two feed
different-magnitude scores into the aux-loss and may prefer different `α`.

## Setup

Fixed testbed: `qwen3_183m` (183M total, ~51M active), 64 experts, top-8, no shared
experts, no capacity limit. Active intermediate width `k·is = 8·192 = 1536`
(3·d_model, Qwen3-0.6B FFN ratio). Two sweeps — **softmax** and **sigmoid** router —
each varying only `aux_loss_coef` over `{0, 1e-3, 1e-2, 1e-1, 1e-0}`.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 8 |
| n_shared_experts | 0 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

| Config | aux_loss_coef |
|--------|---------------|
| `qwen3_183m_a51m_softmax_aux_coef0` / `qwen3_183m_a51m_sigmoid_aux_coef0` | 0 (no balancing baseline) |
| `..._aux_coef1e-3` | 0.001 |
| `..._aux_coef1e-2` | 0.01 (Switch default) |
| `..._aux_coef1e-1` | 0.1 |
| `..._aux_coef1e-0` | 1.0 |

## Run

```bash
nohup bash experiments/moe_aux_loss/run_softmax.sh > logs/moe_aux_loss_softmax.log 2>&1 &
nohup bash experiments/moe_aux_loss/run_sigmoid.sh > logs/moe_aux_loss_sigmoid.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_aux_loss/qwen3_183m_a51m_softmax_aux_coef1e-2.yaml
```

## Results

`train/aux_loss` and `val/aux_loss` (reported minus the balanced floor
`n_layers · k`) measure imbalance; lower is better balanced.

Judge balance by MaxVio / `val_aux_loss`; not comparable across score fns, so read
each sweep on its own.

**softmax**

| aux_loss_coef | val_loss | val_aux_loss (balance) | notes |
|---------------|----------|------------------------|-------|
| 0     | | | no balancing |
| 0.001 | | | |
| 0.01  | | | |
| 0.1   | | | |
| 1.0   | | | |

**sigmoid**

| aux_loss_coef | val_loss | val_aux_loss (balance) | notes |
|---------------|----------|------------------------|-------|
| 0     | | | no balancing |
| 0.001 | | | |
| 0.01  | | | |
| 0.1   | | | |
| 1.0   | | | |

Best per score fn: _TBD_ → pick finer range from here.

## Notes

- Compare against the expert-bias sweep in `../moe_expert_bias/` (same testbed)
  to see whether auxiliary-loss-free balancing matches or beats the best `α`.
