# Muon vs AdamW

Compare Muon ([Jordan et al.](https://kellerjordan.github.io/posts/muon/), [Moonshot scaling](https://arxiv.org/pdf/2502.16982)) against the AdamW benchmark on Qwen3 57M and 0.5B / OpenWebText, holding everything except the optimizer fixed.

## Hypothesis

Muon orthogonalizes each 2D weight's momentum via Newton–Schulz, equalizing the update's singular values so every direction in a weight matrix gets a comparable step — unlike AdamW's per-coordinate normalization. This should give faster convergence per step on the matrix-shaped params (attention/MLP projections), which dominate the param count. With `adjust_lr_fn="match_rms_adamw"`, Muon's update RMS is matched to AdamW so it reuses the AdamW-tuned `lr`/`wd` directly; embeddings, the output head, and 1D params (norms) still use AdamW. Expectation: Muon matches or beats AdamW at equal `lr`/`wd`/schedule, with a larger margin at 0.5B where 90% of params are Muon-routed (vs 55% at 57M).

## Setup

`torch.optim.Muon` is 2D-only, so the `muon` runs use the hybrid `MuonAdamWOptimizer`: 2D hidden weights → Muon, everything else (token/pos embeddings, `lm_head`, RMSNorm scales) → AdamW. Each `(adamw, muon)` pair is identical except `optimizer.name`.

| Config | Optimizer | Params | Muon-routed | LR | min_lr | WD | Eff. batch |
|---|---|---|---|---|---|---|---|
| qwen3_57m_adamw  | AdamW | 57.2M  | —     | 5e-4 | 5e-5 | 0.1 | 256 |
| qwen3_57m_muon   | Muon  | 57.2M  | 31.5M (55%) | 5e-4 | 5e-5 | 0.1 | 256 |
| qwen3_0.5b_adamw | AdamW | 492.0M | —     | 2e-4 | 2e-5 | 0.1 | 256 |
| qwen3_0.5b_muon  | Muon  | 492.0M | 440.4M (90%) | 2e-4 | 2e-5 | 0.1 | 256 |

- LR/WD/schedule per size match the canonical `configs/qwen3_57m.yaml` and `configs/qwen3_0.5b.yaml` (the AdamW benchmark); Muon reuses them via `match_rms_adamw`.
- Muon hyperparams at defaults: `momentum=0.95`, `nesterov=true`, `ns_steps=5`, shared `eps=1e-8`.
- All runs: seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps, cosine schedule with 1500 warmup, bf16, OpenWebText.
- `eval_every=1000`, `eval_steps=200`, `checkpoint_every=5000` for turnaround on the 50K-step runs.

## Run

```bash
nohup bash experiments/muon/run.sh > logs/muon.log 2>&1 &
```

## Results

| Model | AdamW val loss | Muon val loss | Δ |
|---|---|---|---|
| qwen3 57M  | TBD | TBD | TBD |
| qwen3 0.5B | TBD | TBD | TBD |

## Notes

- Muon stores one momentum buffer (`momentum_buffer`) per 2D param; the AdamW sub-optimizer keeps `exp_avg` + `exp_avg_sq` for the rest. `optim/momentum_norm` and `optim/variance_norm` in W&B therefore reflect only the AdamW-routed params (the `exp_avg`/`exp_avg_sq` keys); Muon's buffer is not aggregated by `metric_utils`.
- The benefit should scale with the Muon-routed fraction: 55% at 57M vs 90% at 0.5B, so watch whether the 0.5B gap exceeds the 57M gap.
- If turnaround matters, compare the 5K/10K-step intermediate eval losses before committing both pairs to 50K.
