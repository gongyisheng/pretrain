# LR vs d_model Scaling Experiment

## Hypothesis

Optimal peak learning rate scales with model width as `lr* ∝ d_model^α` for some exponent α. Candidate predictions:
- µP: α = −1
- √-rule: α = −0.5
- Width-invariant: α = 0

The experiment measures α empirically on Qwen3, isolating width from depth and head-size effects.

## Goal

Produce a log–log plot of `optimal_lr` vs `d_model` across 9 widths with a fitted scaling exponent, plus an accompanying results table and loss curves at each staged step budget.

## Scope

- **Varied:** `d_model`, and the attention-head count (`n_heads`, `n_kv_heads`) required to keep `head_dim` fixed.
- **Fixed:** depth (`n_layers=8`), `head_dim=64`, tokenizer, data, batch size, sequence length, optimizer family, scheduler shape.
- **Out of scope:** depth sweeps, alternative optimizers (Muon, Lion, etc.), non-cosine schedules, µP-style input/output scaling multipliers, GPT-2 architecture.

## Fixed configuration

Matches the Qwen3 57M baseline from `experiments/scaling_law/qwen3/qwen3_57m.yaml` where possible:

| Setting | Value |
|---|---|
| Architecture | Qwen3 (GQA + RoPE + RMSNorm + SwiGLU + qk_norm) |
| `n_layers` | 8 |
| `head_dim` | 64 |
| `kv_heads` | `n_heads / 2` |
| `rope_theta` | 10000 |
| `vocab_size` | 50257 |
| Dataset | OpenWebText |
| Tokenizer | `tokenizers/custom_bpe_50k` |
| `max_seq_len` | 1024 |
| `batch_size` | 4 |
| `gradient_accumulation_steps` | 8 (effective batch = 32) |
| Precision | bf16 |
| Backend | torch |
| `grad_clip` | 1.0 |
| Scheduler | cosine |
| `warmup_steps` | 1500 |
| `min_lr` | `lr / 10` (per run) |
| Optimizer | AdamW, `betas=(0.9, 0.95)`, `weight_decay=0.1` |

## Swept axes

### Widths

| `d_model` | `n_heads` | `n_kv_heads` | Approx params (8L) |
|---|---|---|---|
| 256 | 4 | 2 | ~15M |
| 384 | 6 | 3 | ~25M |
| 512 | 8 | 4 | ~57M |
| 768 | 12 | 6 | ~125M |
| 1024 | 16 | 8 | ~215M |
| 1536 | 24 | 12 | ~465M |
| 2048 | 32 | 16 | ~820M |
| 3072 | 48 | 24 | ~1.8B |
| 4096 | 64 | 32 | ~3.2B |

Param counts are rough and will be re-measured from the built model before the runs launch.

### Learning rates

Stage-1 grid per width: `{1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2}`.

## Staged elimination

Follows the shortcut from `experiments/lr/README.md`.

| Stage | `max_steps` | Runs | Selection rule |
|---|---|---|---|
| 1 | 500 | 9 widths × 7 LRs = 63 | Keep top 3 LRs per width by final val loss |
| 2 | 5000 | ~27 | Keep top 1 LR per width |
| 3 | 50000 | 9 | Final curves + scaling fit |

Each stage uses the same training configuration; only `max_steps` changes. Val loss is logged every `eval_every=100` with `eval_steps=25` (consistent with `lr/` configs).

## Large-width compute handling

- d=2048 and above likely require activation checkpointing. Enable `activation_checkpointing: true` for d ≥ 2048.
- Before stage 1 launches at d=3072 / d=4096: run one forward-backward step and measure wall-clock to confirm the width is feasible. If a stage-1 sweep at a given width is projected to exceed a chosen wall-clock budget (e.g. 12h), drop that width. The scaling fit remains informative with 7 widths.
- No change to `batch_size`, `grad_accum`, or `seq_len` — kept constant so width is the only axis.

## Layout

```
experiments/lr_d_model/
├── README.md                       # hypothesis, setup, results (filled post-run)
├── configs/
│   ├── stage1/
│   │   └── qwen3_d{W}_lr{LR}.yaml  # 63 files
│   ├── stage2/
│   │   └── qwen3_d{W}_lr{LR}.yaml  # survivors, written after stage 1 picks
│   └── stage3/
│       └── qwen3_d{W}_lr{LR}.yaml  # 9 files, written after stage 2 picks
├── run_stage1.sh                   # iterates all configs/stage1/*.yaml
├── run_stage2.sh                   # iterates all configs/stage2/*.yaml
├── run_stage3.sh                   # iterates all configs/stage3/*.yaml
├── analyze.py                      # fits α, produces plots + tables
└── results/
    ├── lr_vs_d_model.png
    ├── loss_curves_stage1.png
    ├── loss_curves_stage3.png
    └── summary.csv
```

Config filenames encode width and LR, e.g. `qwen3_d512_lr5e-4.yaml`. Checkpoint dirs mirror that: `checkpoints/lr_d_model/stage{N}/qwen3_d{W}_lr{LR}/`.

## Logging

- W&B project: `pretrain-lr-d_model`
- Run name: `qwen3_d{W}_lr{LR}_s{stage}` (e.g. `qwen3_d512_lr5e-4_s1`)
- `log_every`: 10

## Analysis

`analyze.py` reads final val loss from stage-3 W&B runs (or local checkpoints) and produces:

1. **Scaling fit.** Log-log linear regression of `optimal_lr` vs `d_model` across the 9 widths; report fitted α with confidence interval.
2. **`lr_vs_d_model.png`.** Optimal LR per width on log-log axes, with fit line and reference lines for α = −1, −0.5, 0.
3. **`loss_curves_stage1.png`.** Val-loss curves for all 63 stage-1 runs, one panel per width, colored by LR.
4. **`loss_curves_stage3.png`.** Val-loss curves for the 9 stage-3 runs.
5. **`summary.csv`.** Columns: `d_model`, `n_heads`, `params`, `stage1_lrs_surviving`, `stage2_lr_surviving`, `stage3_lr`, `stage3_final_val_loss`.
6. **Results table** in `README.md` with per-width optimal LR and final val loss.

## Success criteria

- All 9 stage-1 sweeps complete (or widths are explicitly dropped with wall-clock justification recorded in `README.md`).
- Stage-3 produces a monotone or near-monotone optimal-LR curve over width; any non-monotonicity is called out in the writeup.
- Fitted α is reported with a confidence interval and compared against the three reference predictions.

## Non-goals

- Matching a specific theoretical parametrization (µP, SP). We are measuring, not prescribing.
- Repeat/seed variance studies. Single seed per (width, LR) pair.
- Architectural ablations (FFN ratio, qk_norm on/off, etc.).
