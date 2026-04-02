# Dropout Sweep

Sweep dropout rate on Qwen3 57M to measure its effect on pretrain loss.

## Hypothesis

Higher dropout hurts pretraining by randomly zeroing activations, increasing noise and slowing convergence. Very high dropout (≥0.5) should significantly degrade loss. Low dropout (0.01–0.05) may act as mild regularization with negligible cost, while 0.0 (no dropout) should serve as the best baseline for a compute-limited pretraining run.

## Setup

| Config | Dropout |
|---|---|
| qwen3_57m_drop0.0 | 0.0 |
| qwen3_57m_drop0.01 | 0.01 |
| qwen3_57m_drop0.02 | 0.02 |
| qwen3_57m_drop0.05 | 0.05 |
| qwen3_57m_drop0.1 | 0.1 |
| qwen3_57m_drop0.2 | 0.2 |
| qwen3_57m_drop0.5 | 0.5 |
| qwen3_57m_drop0.9 | 0.9 |
| qwen3_57m_drop0.95 | 0.95 |
| qwen3_57m_drop0.99 | 0.99 |

All runs share: Qwen3 57M (d_model=512, layers=8, heads=8, kv_heads=4), seq_len=1024, batch_size=4, grad_accum=8 (effective batch=32), lr=1e-4, 50K steps, cosine schedule with 1K warmup steps, bf16, OpenWebText.

## Run

```bash
nohup bash experiments/dropout/run.sh > logs/dropout.log 2>&1 &
```

## Results

| Dropout | Final Val Loss |
|---|---|
| 0.0 | |
| 0.01 | |
| 0.02 | |
| 0.05 | |
| 0.1 | |
| 0.2 | |
| 0.5 | |
| 0.9 | |
| 0.95 | |
| 0.99 | |

## Notes on Dropout Mechanics

### What dropout does

During training, dropout randomly zeros out a fraction `p` of activations and scales surviving values by `1/(1-p)`. During eval (`model.eval()`), dropout is a no-op — all activations pass through unchanged.

The scaling ensures the **expected activation magnitude is identical** between training and eval, so no adjustment is needed at inference time.

### Forward and backward

Dropout stores its binary mask during the forward pass. The backward pass multiplies incoming gradients by the same mask — so only the weights that participated in a given forward pass receive gradient updates. This is equivalent to training a different random sub-network each step.

The `1/(1-p)` scaling also applies to gradients, preserving their expected magnitude:

```
E[gradient per weight] = P(survives) × scale × base = (1-p) × 1/(1-p) × base = base
```

The **mean** gradient is preserved, but **variance** increases with `p`. At p=0.9 each weight sees gradient either 0 (90% of steps) or 10× (10% of steps) — same mean, 9× higher variance. Gradient clipping handles the spikes in practice.

### Where dropout is applied in this codebase

| Position | Config field | Notes |
|---|---|---|
| After embedding lookup | `dropout_embd` | No residual fallback — most impactful per unit of p |
| After attention out_proj | `dropout_attn` | Before residual add; residual protects against full loss |
| After FFN activation | `dropout_ffn` | Standard MLP dropout position |

The three fields allow independent control. All default to `0.0` for pretraining.

### Attention dropout design decision

The attention module uses a single `attn_dropout` applied after `out_proj`. An earlier version had two dropouts (`attn_dropout` after flash attention output + `resid_dropout` after out_proj), but this was redundant:

- True attention weight dropout (applied to softmax scores before multiplying V) is not possible here because `_flash_attn` handles the full attention computation internally.
- Two dropouts with the same `p` on the same data path compound without clear benefit.
- Modern models (LLaMA, Qwen2/3, Mistral) use at most one dropout per sublayer, often `p=0.0`.

### High dropout and training instability

With `p=0.9` the effective regularization is extreme:
- 90% of weights receive zero gradient each step
- Surviving gradients are 10× larger (high variance)
- Embedding dropout has no residual safety net — corruption propagates as the base `x` into all subsequent layers
- The model tends to collapse onto the residual connections, effectively bypassing sublayers

Pretraining at scale uses `p=0.0` because data volume is the regularizer. Dropout is most useful for small-scale training (this sweep uses a 57M model) where overfitting is a real concern.
