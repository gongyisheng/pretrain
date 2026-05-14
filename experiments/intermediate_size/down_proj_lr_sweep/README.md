# Down-Proj LR Sweep (μP-lite for FFN width)

Sweep the per-layer LR multiplier on `down_proj` while holding base LR fixed at 5e-4. Tests whether μP's prediction (down_proj LR ∝ 1/intermediate_size) holds empirically in our SP+AdamW setup.

## Hypothesis

For the SwiGLU FFN, only `down_proj` has fan_in equal to `intermediate_size` (gate/up have fan_in = d_model). μP+Adam theory predicts the optimal LR for `down_proj` scales as `1/intermediate_size`. Anchored at mult=4 (intermediate_size=2048) with multiplier=1.0, the predicted "theory optimum" multiplier per width is `2048 / intermediate_size`:

| Mult | intermediate_size | theory_scale (μP prediction) |
|---|---|---|
| 1 | 512 | 4.00 |
| 4 | 2048 | 1.00 (reference) |
| 16 | 8192 | 0.25 |

This experiment sweeps `theory_scale × {0.25, 0.5, 1.0, 2.0, 4.0}` at each width to test whether the μP prediction (xt1) actually wins, or whether SP+AdamW's gradient normalization makes the dependence weaker.

## Setup

3 widths x 5 scales = 15 runs. All runs share the qwen3_57m baseline (`base_lr=5e-4`, `min_lr=5e-5`, qk_norm, etc.) except for `intermediate_size` and `optimizer.lr_mult.down_proj`.

**Resolved `down_proj` lr_mult per config** (= `theory_scale × xt`):

| Width | theory | xt0_25 | xt0_5 | xt1 (μP-pred) | xt2 | xt4 |
|---|---|---|---|---|---|---|
| mult=1 | 4.0 | 1.0 | 2.0 | **4.0** | 8.0 | 16.0 |
| mult=4 | 1.0 | 0.25 | 0.5 | **1.0** | 2.0 | 4.0 |
| mult=16 | 0.25 | 0.0625 | 0.125 | **0.25** | 0.5 | 1.0 |

The effective LR for `down_proj` is `base_lr × lr_mult` — e.g., for mult=1 xt1, `down_proj` trains at 5e-4 × 4.0 = 2e-3, while gate/up/attention all stay at 5e-4.

**Fixed across all runs:** Qwen3 (8 layers, 8 heads, 4 kv_heads, d_model=512, qk_norm=true, rope_theta=10000), seq_len=1024, batch_size=8, grad_accum=32, warmup_steps=1500, cosine schedule, bf16, OpenWebText, `debug.max_steps: 12000` (~3.14B tokens per run).

## Run

```bash
nohup bash experiments/intermediate_size/down_proj_lr_sweep/run.sh > logs/intermediate_size_down_proj_lr_sweep.log 2>&1 &
```

## Results

Final validation loss per (mult, xt) cell (filled in after running):

| Width | xt0_25 | xt0_5 | xt1 (μP-pred) | xt2 | xt4 | Best xt |
|---|---|---|---|---|---|---|
| mult=1 | | | | | | |
| mult=4 | | | | | | |
| mult=16 | | | | | | |

**Interpretation guide:**
- If **xt1 wins at every width** → μP prediction holds; per-layer LR scaling is real and SP+AdamW is not absorbing it.
- If **xt1 ≈ xt0_5 ≈ xt2 (flat optimum)** → AdamW's gradient normalization mostly absorbs the FFN-width dependence; μP correction is unnecessary in practice at this scale.
- If **best xt drifts systematically with width** (e.g., xt2 wins at small widths, xt0_5 at large) → there's a real dependence, but the μP exponent (1/IS) overcorrects or undercorrects.

## W&B

Project: `pretrain-intermediate-size-down-proj-lr-sweep`. Group: `mult{M}` (collapses the 5-scale sweep into one curve per width).

Compare against the SP sweep at the same widths in `pretrain-intermediate-size-lr-sweep` to see whether μP-lite improves loss vs the best-tuned global LR (no per-layer mult).

## Notes

### Caveats

- **This is μP-lite, not full μP.** Init scaling is unchanged (standard PyTorch/qwen3 init, not μP-prescribed). Forward multipliers on the residual stream are absent. So predictions like "xt1 transfers across width" are weaker than under strict μP — this experiment measures the *partial* benefit of the LR-only correction.

- **Only `down_proj` is corrected.** μP would also adjust the LR for `lm_head` (when untied) and possibly `token_emb`. In our tied setup `lm_head ≡ token_emb`, and the codebase's default `lr_mult: {lm_head: 1.0}` is a no-op, so we leave embeddings alone.

- **Boundary effects.** xt0_25 at mult=16 means down_proj trains at 5e-4 × 0.0625 = 3.125e-5, which is very small. xt4 at mult=1 means down_proj trains at 5e-4 × 16 = 8e-3, which is large enough to risk instability. Watch for grad_norm spikes in the W&B logs.

### Out of scope

- Full μP with init scaling and forward multipliers (separate experiment if results here are encouraging).
- gate/up_proj scaling (fan_in = d_model, unchanged across this sweep).
- Width transfer test: does xt1 found at one width transfer to another? Not directly tested; the design assumes μP's 1/IS prediction up front.
