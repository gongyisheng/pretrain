# Muon vs AdamW

Compare hybrid Muon with AdamW on Qwen3 57M / OpenWebText using the same model, data, and training schedule.

## Hypothesis

Muon orthogonalizes the momentum update of each 2D hidden weight with Newton–Schulz, giving every matrix direction a comparable step. This should converge faster than AdamW's per-coordinate normalization. `match_rms_adamw` approximates AdamW update RMS, so Muon can use the AdamW baseline LR without an LR sweep.

## Setup

`MuonAdamOptimizer` routes 2D hidden matrices (attention and MLP projections) through Muon and embeddings, output head, and RMSNorm scales through AdamW. The runs share the model, OpenWebText data, and schedule; the optimizer is the experimental variable.

| Config | Optimizer | Params | Muon-routed | LR → min_lr | WD | Effective batch |
|---|---|---:|---:|---|---:|---:|
| qwen3_57m_adamw | AdamW | 57.2M | — | 5e-4 → 5e-5 | 0.1 | 256 |
| qwen3_57m_muon | Hybrid Muon | 57.2M | 31.5M (55%) | 5e-4 → 5e-5 | 0.1 | 256 |

- Seq length 1024; batch size 16; gradient accumulation 16; 50K steps; bf16; cosine schedule with 1500 warmup steps.
- `eval_every=100`, `eval_steps=100`, and `checkpoint_every=5000`.
- W&B project: `pretrain-muon`. Checkpoints: `checkpoints/muon/qwen3_57m_adamw/` and `checkpoints/muon/qwen3_57m_muon/`.

## Run

```bash
nohup bash experiments/muon/run.sh > logs/muon.log 2>&1 &
```

## Results

| Model | AdamW val loss | Muon val loss | Δ |
|---|---:|---:|---:|
| qwen3 57M | TBD | TBD | TBD |

## Notes

- Muon keeps a momentum buffer for each routed 2D parameter. AdamW-routed parameters keep `exp_avg` and `exp_avg_sq`.
- `optim/momentum_norm` and `optim/variance_norm` read `exp_avg` and `exp_avg_sq` in `src/metrics/functional.py`, so they cover only AdamW-routed parameters in the hybrid run.
