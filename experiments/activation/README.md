# Activation Function Ablation

Compare 19 FFN activation choices at Qwen3-57M scale with matched parameter counts.

## Hypothesis

Under fixed FFN parameter count, the choice of activation matters less than the choice of gating (Shazeer 2020). Among activations, squared variants (Primer-style ReLU²) match or exceed GeLU/SwiGLU on validation loss; PowLU matches SwiGLU while producing fewer outliers; Bilinear GLU comes close to SwiGLU, isolating the gating effect from the activation effect.

## Setup

All 19 configs share the Qwen3-57M backbone (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=16, grad_accum=16 (~262K tokens/step), 50K steps (~13B tokens), lr=6e-4 cosine to 6e-5 with 1500 warmup steps, bf16, OpenWebText, 50257-vocab BPE.

FFN parameters are matched across gated vs ungated by adjusting `intermediate_size`:

- Gated FFN params per layer = `3 · d_model · intermediate_size`
- Ungated FFN params per layer = `2 · d_model · intermediate_size`

So `intermediate_size = 2048` (gated) and `3072` (ungated) both yield `12 · d_model² = 3.15M` FFN params per layer.

### Ungated variants (8)

| Config | `mlp_activation` | Formula |
|---|---|---|
| qwen3_57m_relu | relu | relu(x) |
| qwen3_57m_gelu | gelu | gelu(x) |
| qwen3_57m_silu | silu | silu(x) |
| qwen3_57m_leaky_relu | leaky_relu | leaky_relu(x, 0.01) |
| qwen3_57m_relu2 | relu2 | relu(x)² |
| qwen3_57m_gelu2 | gelu2 | gelu(x)² |
| qwen3_57m_silu2 | silu2 | silu(x)² |
| qwen3_57m_leaky_relu2 | leaky_relu2 | leaky_relu(x, 0.01)² |

### Gated variants (11)

| Config | `mlp_activation` | Literature name | Formula |
|---|---|---|---|
| qwen3_57m_reglu | relu | ReGLU | relu(g) · u |
| qwen3_57m_geglu | gelu | GeGLU | gelu(g) · u |
| qwen3_57m_swiglu | silu | SwiGLU | silu(g) · u |
| qwen3_57m_leaky_reglu | leaky_relu | LeakyReGLU | leaky_relu(g) · u |
| qwen3_57m_reglu2 | relu2 | ReGLU² | relu(g)² · u |
| qwen3_57m_geglu2 | gelu2 | GeGLU² | gelu(g)² · u |
| qwen3_57m_swiglu2 | silu2 | SwiGLU² | silu(g)² · u |
| qwen3_57m_leaky_reglu2 | leaky_relu2 | LeakyReGLU² | leaky_relu(g)² · u |
| qwen3_57m_bilinear | bilinear | Bilinear GLU | g · u |
| qwen3_57m_bilinear2 | bilinear2 | Squared Bilinear GLU | g² · u |
| qwen3_57m_powlu | powlu | PowLU | powlu(g) · u |

PowLU formula (arXiv:2605.25704, m=3): for x > 0, `x · x^(m/(√x+1)) · σ(x)`; for x ≤ 0, `x² · σ(x)`.

## Run

```bash
nohup bash experiments/activation/run.sh > logs/activation.log 2>&1 &
```

19 runs × ~6 hours per run on one A100 ≈ 5 days wall-clock. Each run writes to its own `checkpoints/activation/<variant>/` and W&B run name.

## Results

Fill in after the runs complete.

### Ungated

| Variant | Final Val Loss | Final Val PPL | Notes |
|---|---|---|---|
| relu | | | |
| gelu | | | |
| silu | | | |
| leaky_relu | | | |
| relu2 | | | |
| gelu2 | | | |
| silu2 | | | |
| leaky_relu2 | | | |

### Gated

| Variant | Final Val Loss | Final Val PPL | Notes |
|---|---|---|---|
| reglu | | | |
| geglu | | | |
| swiglu | | | |
| leaky_reglu | | | |
| reglu2 | | | |
| geglu2 | | | |
| swiglu2 | | | |
| leaky_reglu2 | | | |
| bilinear | | | |
| bilinear2 | | | |
| powlu | | | |

## Notes

- Squaring follows **Rule A** (see `src/layers/activation.py` docstring): squared = unary `act(x)²`. Gated squared = `act(g)² · u`, NOT `(act(g) · u)²`. So SwiGLU² is `silu(g)² · u`.
- Bilinear GLU and PowLU are exposed only via the gated registry; using them with `mlp_gated: false` raises `ValueError`.
- Literature references: Shazeer 2020 "GLU Variants Improve Transformer" (SwiGLU/GeGLU/ReGLU/Bilinear), So et al. 2021 "Primer" (squared ReLU), arXiv:2605.25704 (May 2026) (PowLU).
- This sweep does not vary `intermediate_size`, learning rate, or any other hyperparameter beyond `mlp_activation` / `mlp_gated` / matched `intermediate_size`.
