# Intermediate Size Sweep at lr=5e-4 (Qwen3 57M)

Sweep dense FFN expansion ratio `intermediate_size / d_model` at a fixed learning rate of `5e-4` (the qwen3 57M baseline LR per CLAUDE.md). Probes the loss-vs-width curve across an unusually wide range — from sub-residual (0.25x) to very wide (32x) — without re-tuning LR per width.

## Hypothesis

At fixed LR, increasing FFN width should yield monotonically decreasing validation loss with diminishing returns. Two regimes of interest:

- **Sub-residual (mult < 1).** When `intermediate_size < d_model`, the FFN becomes a bottleneck (down-projects below the residual width). Expect a sharp loss penalty; a useful lower-bound anchor.
- **Very wide (mult >= 8).** Production models cluster around 3-4x. This sweep extends to 32x to see whether loss keeps falling, plateaus, or starts hurting at fixed LR (large `intermediate_size` shifts the optimal LR for `down_proj` downward — see `down_proj_lr_sweep/`).

## Setup

9 widths x 1 LR = 9 runs. All runs share the qwen3_57m baseline (`base_lr=5e-4`, `min_lr=5e-5`, qk_norm, etc.) except for `intermediate_size` and the W&B/checkpoint paths.

| Mult | intermediate_size | ~Params |
|---|---|---|
| 0.25 | 128 | ~33M |
| 0.5 | 256 | ~35M |
| 1 | 512 | ~38M |
| 2 | 1024 | ~45M |
| 3 | 1536 | ~51M |
| 4 | 2048 | ~57M (baseline) |
| 8 | 4096 | ~82M |
| 16 | 8192 | ~133M |
| 32 | 16384 | ~233M |

Param count = ~32M (embedding + attention + norms) + 12.3K x intermediate_size.

**Fixed across all runs:** Qwen3 (8 layers, 8 heads, 4 kv_heads, d_model=512, qk_norm=true, rope_theta=10000), seq_len=1024, batch_size=8, grad_accum=32 (effective batch=256, ~262K tok/step), warmup_steps=1500, cosine schedule, bf16, OpenWebText, `debug.max_steps: 12000` (~3.14B tokens per run).

**Early stop:** `debug.max_steps: 12000` halts training while keeping the cosine schedule shaped for 50000 steps. At step 12000 the LR has decayed only ~9% from peak, so each run is compared near peak LR.

## Run

```bash
nohup bash experiments/intermediate_size/intermediate_size_sweep/run.sh > logs/intermediate_size_sweep.log 2>&1 &
```

Runs sequentially in ascending-mult order so the cheapest configs surface failures first.

## Results

Final validation loss per width (filled in after running):

| Mult | intermediate_size | val_loss |
|---|---|---|
| 0.25 | 128 | |
| 0.5 | 256 | |
| 1 | 512 | |
| 2 | 1024 | |
| 3 | 1536 | |
| 4 | 2048 | |
| 8 | 4096 | |
| 16 | 8192 | |
| 32 | 16384 | |

Headline plot: val_loss vs intermediate_size (log-x).

**Interpretation guide:**
- If **loss falls monotonically through mult=32** → FFN width is still under-saturated at this depth/d_model, and 4x is conservative.
- If **loss plateaus around mult=4-8** → standard expansion ratios are near-optimal at fixed LR; further width is wasted compute.
- If **loss rises again at mult >= 16** → fixed LR is mis-tuned for wide FFNs (consistent with μP's prediction that down_proj wants lr ∝ 1/intermediate_size). Cross-check against `down_proj_lr_sweep/` to confirm.

## W&B

Project: `pretrain-intermediate-size-sweep`. Group: `mult{M}` (one curve per width).

Suggested charts:
- val_loss vs intermediate_size (headline plot, log-x)
- val_loss curves overlaid by mult
- grad_norm overlay (watch mult=32 for instability at fixed lr=5e-4)

## Notes

### Caveats

- **Fixed LR across all widths.** No per-width LR retune. The mult=32 run in particular may be under-trained or unstable because optimal `down_proj` LR scales as `1/intermediate_size` (see `down_proj_lr_sweep/` for the μP-lite test). Read the loss-at-large-mult numbers as "lr=5e-4 result," not "best achievable at this width."
- **mult=0.25 and mult=0.5 are intentionally extreme.** `intermediate_size` below `d_model` is non-standard; results are useful as anchors for the loss-vs-width curve, not as a recommended config.
- **mult=32 (~233M params)** is much larger than the 57M nominal scale. It fits at batch=8/seq=1024 without activation checkpointing on a single 24GB+ GPU but pushes memory; if it OOMs, set `training.activation_checkpointing: true` for that one config.

### Out of scope

- Per-width LR retune (covered by the original lr_sweep, archived in git history).
- Per-layer LR scaling for `down_proj` (covered by `down_proj_lr_sweep/`).
- Depth/width tradeoffs at fixed param budget.
