# SwiGLU Up-Shift

Test whether a fixed shift on the SwiGLU up branch (`silu(gate) · (up + c)`) changes convergence at Qwen3-51M scale. Baseline `c = 0` (standard SwiGLU) vs `c = 1` (gpt-oss's value).

## Hypothesis

SwiGLU computes `silu(gate) · up`. gpt-oss ships a shifted variant, `silu(gate) · (up + 1)`. Because `up_proj` is bias-free (`up = W_up · x`), the shift is **not** absorbable into the weights:

```
silu(gate) · (up + c) = silu(gate) · up  +  c · silu(gate)
```

The `c · silu(gate)` term is a gate-dependent additive path the standard SwiGLU cannot represent, so the shift genuinely enlarges the function class (at zero extra parameters — `c` is a constant). The question is whether that extra path helps a from-scratch model, and if so whether the gpt-oss `c = 1` is a good default. A shift is **isolated** here: `alpha = 1.0` and no `act_limit`, so any difference is attributable to `up_shift` alone.

## Setup

2 runs. Both share the Qwen3-51M backbone (d_model=512, 8 layers, GQA 8/4 with qk_norm, dense SwiGLU MLP intermediate_size=1536, ~50.9M params), seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256, ~262K tokens/step), 50K steps (~13B tokens), AdamW lr=5e-4 cosine to 5e-5 with 1500 warmup steps, bf16, OpenWebText, 50257-vocab BPE, `eval_every=100`, `eval_steps=100`. The two configs differ only in `mlp_kwargs.activation_kwargs.up_shift`; parameter count is identical (the shift is a constant).

| Config | `up_shift` | Formula |
|---|---|---|
| qwen3_51m_up_shift0 | 0.0 | silu(g) · u |
| qwen3_51m_up_shift1 | 1.0 | silu(g) · (u + 1) |

`up_shift0` is exactly the standard SwiGLU baseline (identical to `experiments/activation/qwen3_51m_swiglu`), duplicated here so the folder is self-contained.

## Run

```bash
nohup bash experiments/activation_up_shift/run.sh > logs/activation_up_shift.log 2>&1 &
```

2 runs × ~5 hours per run on one A100 ≈ 10 hours wall-clock. Each run writes to its own `checkpoints/activation_up_shift/<variant>/` and W&B run name.

## Results

W&B project: `pretrain-activation-up-shift`. Fill in after the runs complete.

| Config | `up_shift` | Final Val Loss | Final Val PPL | Δ vs baseline |
|---|---|---|---|---|
| qwen3_51m_up_shift0 | 0.0 | | | 0 |
| qwen3_51m_up_shift1 | 1.0 | | | |

## Notes

- Compare runs using the mean validation loss over the final 10 evaluations.
- The shift touches only the dense MLP's up branch; attention projections, gate branch, and `lm_head` are unaffected.
- `up_shift` is a fixed scalar (baked into the activation `partial`), not learnable and not per-channel. A learnable per-channel generalization would be an `up_proj` bias — deliberately out of scope here to isolate the constant-shift effect.
- gpt-oss's full SwiGLU variant pairs `up_shift=1.0` with `alpha=1.702` and `act_limit=±7`; those are held at defaults here so the shift is isolated. See [`activation_limit`](../activation_limit/README.md) for the clamp arm.
