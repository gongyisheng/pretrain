# Lion vs AdamW

Compare Lion (Chen et al. 2023, https://arxiv.org/pdf/2302.06675) against AdamW on Qwen3 57M / OpenWebText, sweeping Lion's `(lr, wd)` grid against a single AdamW anchor.

## Hypothesis

Lion's sign-based update has unit per-coordinate magnitude, so the effective step is decoupled from gradient scale. The paper recommends ~3-10x lower LR and ~3-10x higher WD than AdamW to compensate. With the right `(lr, wd)`, Lion should match or beat AdamW at half the optimizer-state memory (one momentum buffer vs two).

## Setup

| Config | Optimizer | LR | min_lr | WD | Betas |
|---|---|---|---|---|---|
| qwen3_57m_adamw_lr5e-4_wd0.1 | AdamW | 5e-4 | 5e-5 | 0.1 | (0.9, 0.95) |
| qwen3_57m_lion_lr5e-5_wd0.1 | Lion | 5e-5 | 5e-6 | 0.1 | (0.9, 0.99) |
| qwen3_57m_lion_lr5e-5_wd0.2 | Lion | 5e-5 | 5e-6 | 0.2 | (0.9, 0.99) |
| qwen3_57m_lion_lr5e-5_wd0.5 | Lion | 5e-5 | 5e-6 | 0.5 | (0.9, 0.99) |
| qwen3_57m_lion_lr5e-5_wd1.0 | Lion | 5e-5 | 5e-6 | 1.0 | (0.9, 0.99) |
| qwen3_57m_lion_lr1e-4_wd0.1 | Lion | 1e-4 | 1e-5 | 0.1 | (0.9, 0.99) |
| qwen3_57m_lion_lr1e-4_wd0.2 | Lion | 1e-4 | 1e-5 | 0.2 | (0.9, 0.99) |
| qwen3_57m_lion_lr1e-4_wd0.5 | Lion | 1e-4 | 1e-5 | 0.5 | (0.9, 0.99) |
| qwen3_57m_lion_lr1e-4_wd1.0 | Lion | 1e-4 | 1e-5 | 1.0 | (0.9, 0.99) |
| qwen3_57m_lion_lr2e-4_wd0.1 | Lion | 2e-4 | 2e-5 | 0.1 | (0.9, 0.99) |
| qwen3_57m_lion_lr2e-4_wd0.2 | Lion | 2e-4 | 2e-5 | 0.2 | (0.9, 0.99) |
| qwen3_57m_lion_lr2e-4_wd0.5 | Lion | 2e-4 | 2e-5 | 0.5 | (0.9, 0.99) |
| qwen3_57m_lion_lr2e-4_wd1.0 | Lion | 2e-4 | 2e-5 | 1.0 | (0.9, 0.99) |

- AdamW anchor matches `configs/qwen3_57m.yaml` (5e-4 is the best LR from `experiments/lr/`).
- Lion LR grid: 5e-5 (10x lower than AdamW), 1e-4 (5x lower), 2e-4 (~2.5x lower).
- Lion WD grid: 0.1 (matches AdamW to isolate optimizer effect), 0.2, 0.5, 1.0 (paper-recommended 10x scaling).
- All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4, qk_norm), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps, cosine schedule with 1K warmup, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/lion_optm/run.sh > logs/lion_optm.log 2>&1 &
```

## Results

| Config | Final Val Loss |
|---|---|
| adamw (lr=5e-4, wd=0.1) | TBD |
| lion lr=5e-5 wd=0.1 | TBD |
| lion lr=5e-5 wd=0.2 | TBD |
| lion lr=5e-5 wd=0.5 | TBD |
| lion lr=5e-5 wd=1.0 | TBD |
| lion lr=1e-4 wd=0.1 | TBD |
| lion lr=1e-4 wd=0.2 | TBD |
| lion lr=1e-4 wd=0.5 | TBD |
| lion lr=1e-4 wd=1.0 | TBD |
| lion lr=2e-4 wd=0.1 | TBD |
| lion lr=2e-4 wd=0.2 | TBD |
| lion lr=2e-4 wd=0.5 | TBD |
| lion lr=2e-4 wd=1.0 | TBD |

## Notes

- Lion's `eps` field in `OptimizerConfig` is ignored (Lion has no eps).
- Lion uses one momentum buffer per param vs AdamW's two (m + v); expect ~half the optimizer-state memory on the same param count.
- Per the LR-sweep shortcut in `experiments/lr/README.md`: if turnaround matters, check the 500-step / 5K-step intermediate eval losses and prune dominated `(lr, wd)` combos before running all to 50K.
