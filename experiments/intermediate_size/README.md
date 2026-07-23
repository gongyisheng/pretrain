# Intermediate Size Sweep (Qwen3 d_model=512, Muon, lr=5e-4)

Sweep dense FFN expansion ratio `intermediate_size / d_model` at a fixed learning rate of `5e-4` (Muon optimizer, reusing the AdamW-tuned baseline LR via `match_rms_adamw`). Probes the loss-vs-width curve across an unusually wide range — from sub-residual (0.25x) to very wide (32x) — without re-tuning LR per width.

## Hypothesis

At fixed LR, increasing FFN width should yield monotonically decreasing validation loss with diminishing returns. Two regimes of interest:

- **Sub-residual (mult < 1).** When `intermediate_size < d_model`, the FFN becomes a bottleneck (down-projects below the residual width). Expect a sharp loss penalty; a useful lower-bound anchor.
- **Very wide (mult >= 8).** Production models cluster around 3-4x. This sweep extends to 32x to see whether loss keeps falling, plateaus, or starts hurting at fixed LR (large `intermediate_size` shifts the optimal LR for `down_proj` downward).

## Setup

15 widths x 1 LR = 15 runs. All runs share the qwen3 d_model=512 baseline (`base_lr=5e-4`, `min_lr=5e-5`, qk_norm, etc.) except for `intermediate_size` and the W&B/checkpoint paths. A fine 128-step grid (mult 2.25-3.75) densely samples the 1024-2048 region around the standard 3-4x expansion ratio.

| Mult | intermediate_size | Config | ~Params |
|---|---|---|---|
| 0.25 | 128 | `qwen3_34m_is128` | 34M |
| 0.5 | 256 | `qwen3_35m_is256` | 35M |
| 1 | 512 | `qwen3_38m_is512` | 38M |
| 2 | 1024 | `qwen3_45m_is1024` | 45M |
| 2.25 | 1152 | `qwen3_46m_is1152` | 46M |
| 2.5 | 1280 | `qwen3_48m_is1280` | 48M |
| 2.75 | 1408 | `qwen3_49m_is1408` | 49M |
| 3 | 1536 | `qwen3_51m_is1536` | 51M |
| 3.25 | 1664 | `qwen3_53m_is1664` | 53M |
| 3.5 | 1792 | `qwen3_54m_is1792` | 54M |
| 3.75 | 1920 | `qwen3_56m_is1920` | 56M |
| 4 | 2048 | `qwen3_57m_is2048` | 57M (baseline) |
| 8 | 4096 | `qwen3_82m_is4096` | 82M |
| 16 | 8192 | `qwen3_133m_is8192` | 133M |
| 32 | 16384 | `qwen3_233m_is16384` | 233M |

Config filename encodes total params (which grow with `intermediate_size`).
Param count = ~32M (embedding + attention + norms) + 12.3K x intermediate_size.

**Fixed across all runs:** Qwen3 (8 layers, 8 heads, 4 kv_heads, d_model=512, qk_norm=true, rope_theta=10000), seq_len=1024, warmup_steps=1500, `max_steps=50000` (full cosine schedule, no early stop), bf16, OpenWebText. Optimizer: Muon (hybrid — Muon for 2D hidden weights, AdamW for embeddings/head/1D params; `adjust_lr_fn=match_rms_adamw`, so the AdamW-tuned lr=5e-4 carries over).

**Batch:** effective batch is 256 (~262K tok/step) for all widths up to `is=2048`. Small widths (`<57M`) use `batch_size=16 × grad_accum=16` for throughput; `is=2048` uses `8 × 32`. The two largest raise grad-accum to put bigger models on a larger effective batch: `is=4096`/`is=8192` use `8 × 64` (~524K tok/step), `is=16384` uses `8 × 128` (~1.05M tok/step).

## Run

```bash
nohup bash experiments/intermediate_size/run.sh > logs/intermediate_size.log 2>&1 &
```

Runs sequentially in ascending-mult order so the cheapest configs surface failures first.

## Results

Final validation loss per width (filled in after running):

| Mult | intermediate_size | Config | val_loss |
|---|---|---|---|
| 0.25 | 128 | `qwen3_34m_is128` | |
| 0.5 | 256 | `qwen3_35m_is256` | |
| 1 | 512 | `qwen3_38m_is512` | |
| 2 | 1024 | `qwen3_45m_is1024` | |
| 2.25 | 1152 | `qwen3_46m_is1152` | |
| 2.5 | 1280 | `qwen3_48m_is1280` | |
| 2.75 | 1408 | `qwen3_49m_is1408` | |
| 3 | 1536 | `qwen3_51m_is1536` | |
| 3.25 | 1664 | `qwen3_53m_is1664` | |
| 3.5 | 1792 | `qwen3_54m_is1792` | |
| 3.75 | 1920 | `qwen3_56m_is1920` | |
| 4 | 2048 | `qwen3_57m_is2048` | |
| 8 | 4096 | `qwen3_82m_is4096` | |
| 16 | 8192 | `qwen3_133m_is8192` | |
| 32 | 16384 | `qwen3_233m_is16384` | |

Headline plot: val_loss vs intermediate_size (log-x).

**Interpretation guide:**
- If **loss falls monotonically through mult=32** → FFN width is still under-saturated at this depth/d_model, and 4x is conservative.
- If **loss plateaus around mult=4-8** → standard expansion ratios are near-optimal at fixed LR; further width is wasted compute.
- If **loss rises again at mult >= 16** → width hurts even with Muon's RMS-matched per-layer LR (plain AdamW would predict `down_proj` wants lr ∝ 1/intermediate_size; Muon's `match_rms_adamw` is expected to absorb most of that).

## W&B

Project: `pretrain-intermediate-size`. Group: `mult{M}` (one curve per width).

Suggested charts:
- val_loss vs intermediate_size (headline plot, log-x)
- val_loss curves overlaid by mult
- grad_norm overlay (watch mult=32 for instability at fixed lr=5e-4)

## Notes

### Caveats

- **Fixed LR across all widths.** No per-width LR retune. Muon's `match_rms_adamw` rescales each 2D weight's effective LR by its shape, which absorbs most of the `down_proj` lr ∝ 1/intermediate_size mistuning that plain AdamW suffers at large width; the AdamW-handled params (embeddings, head, 1D) still see a fixed lr. Read the loss-at-large-mult numbers as "lr=5e-4 result," not necessarily "best achievable at this width."
- **Largest widths are not iso-token.** `max_steps=50000` is fixed, but `is>=4096` raise gradient accumulation (2-4x), so they train on proportionally more tokens (~26B / ~52B vs ~13B for the rest). Their loss is not directly comparable to the smaller widths as a pure width effect.
- **mult=0.25 and mult=0.5 are intentionally extreme.** `intermediate_size` below `d_model` is non-standard; results are useful as anchors for the loss-vs-width curve, not as a recommended config.
- **mult=32 (~233M params)** is much larger than the 57M nominal scale. It fits at batch=8/seq=1024 on a single 24GB+ GPU but pushes memory; if it OOMs, lower `batch_size` and raise `gradient_accumulation_steps` to compensate.

### Out of scope

- Per-width LR retune (the AdamW LR sweep was removed; see git history).
- Depth/width tradeoffs at fixed param budget.
