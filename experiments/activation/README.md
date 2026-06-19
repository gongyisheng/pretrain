# Activation Function Ablation

Compare 19 FFN activation choices at Qwen3-57M scale with matched parameter counts.

## Hypothesis

Under fixed FFN parameter count, the choice of activation matters less than the choice of gating (Shazeer 2020). Among activations, squared variants (Primer-style ReLU²) match or exceed GeLU/SwiGLU on validation loss; PowLU matches SwiGLU while producing fewer outliers; Bilinear GLU comes close to SwiGLU, isolating the gating effect from the activation effect.

## Analysis

Each new activation fixes a concrete defect in the previous one:

- **Sigmoid / tanh** saturate at both ends, so their gradient → 0 for large |x|. Stacked deep, this is the **vanishing gradient** problem — early layers stop learning.
- **ReLU = max(0, x)** fixed vanishing gradient on the positive side (gradient is exactly 1 for x > 0, never saturates), which is what made deep nets trainable. But it has two defects: **dying ReLU** (for x < 0 both output and gradient are 0, so a neuron pushed there gets zero gradient and never recovers) and **non-differentiability at x = 0** (a non-smooth kink that hurts optimization).
- **LeakyReLU** `max(αx, x)` gives the negative side a small slope α so neurons can't fully die; still non-smooth at 0.
- **GeLU / SiLU(Swish)** are smooth everywhere with a small negative response, curing the kink and the dead-neuron problem at once. Now standard.
- **ReLU² / GeLU² (Primer)** square the activation to sharpen the nonlinearity, adding effective capacity at matched params.
- **GLU variants (SwiGLU/GeGLU/ReGLU/Bilinear)** are an orthogonal axis — a multiplicative gate `act(g)·u`, where most of the gain comes from the gate, not the activation.
- **PowLU** targets stable pre-training: SwiGLU-level quality with fewer activation outliers.

This sweep measures how much each step actually buys at fixed parameter count.

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

Final validation metrics at step 50000 (~13B tokens), pulled from W&B project `pretrain-activation`.

### Ungated

| Variant | Final Val Loss | Final Val PPL |
|---|---|---|
| relu | 3.1702 | 23.813 |
| gelu | 3.1456 | 23.233 |
| silu | 3.1522 | 23.388 |
| leaky_relu | 3.1684 | 23.770 |
| relu2 | 3.1363 | 23.019 |
| gelu2 | 3.1349 | 22.985 |
| **silu2** | **3.1324** | **22.930** |
| leaky_relu2 | 3.1353 | 22.994 |

### Gated

| Variant | Final Val Loss | Final Val PPL |
|---|---|---|
| reglu | 3.1431 | 23.176 |
| geglu | 3.1321 | 22.921 |
| swiglu | 3.1315 | 22.908 |
| leaky_reglu | 3.1413 | 23.134 |
| bilinear | 3.1442 | 23.202 |
| reglu2 | 3.1367 | 23.028 |
| **geglu2** | **3.1306** | **22.887** |
| swiglu2 | 3.1339 | 22.964 |
| leaky_reglu2 | 3.1364 | 23.021 |
| bilinear2 | 3.1375 | 23.046 |
| powlu | 3.1337 | 22.959 |

### Takeaways

- Squaring helps the ungated activations; gating closes most of the gap. The whole field spans only ~0.04 val loss (~0.9 PPL), confirming activation choice matters less than gating.
- Best overall: geglu2 (3.1306) and swiglu (3.1315). PowLU and Bilinear GLU match the hypothesis — PowLU ties SwiGLU, Bilinear GLU trails it slightly.

## Notes

- Squaring follows **Rule A** (see `src/layers/activation.py` docstring): squared = unary `act(x)²`. Gated squared = `act(g)² · u`, NOT `(act(g) · u)²`. So SwiGLU² is `silu(g)² · u`.
- Bilinear GLU and PowLU are exposed only via the gated registry; using them with `mlp_gated: false` raises `ValueError`.
- This sweep does not vary `intermediate_size`, learning rate, or any other hyperparameter beyond `mlp_activation` / `mlp_gated` / matched `intermediate_size`.

## Literature References

- Noam Shazeer. GLU variants improve transformer. *arXiv preprint* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202), 2020.
- David R. So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, and Quoc V. Le. Primer: Searching for efficient transformers for language modeling. *arXiv preprint* [arXiv:2109.08668](https://arxiv.org/abs/2109.08668), 2021.
- Peijie Jiang, Yuqi Feng, Cunyin Peng, Qian Zhao, Jia Liu, KunLong Chen, Zhiqiang Zhang, and Jun Zhou. PowLU: An activation function for stable pre-training of LLMs. *arXiv preprint* [arXiv:2605.25704](https://arxiv.org/abs/2605.25704), 2026.
