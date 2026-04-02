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

### 1. What dropout is: the sub-network view

Each forward pass, dropout zeros `p` fraction of activations and scales survivors by `1/(1-p)` to preserve expected magnitude. This is equivalent to sampling a random sub-network — only the active neurons participate in that step's computation and weight update. Over many steps, all sub-networks get trained, producing an implicit ensemble effect that reduces overfitting.

### 2. Why dropout must be in the forward pass

Dropout cannot be applied only in the backward pass. The backward pass is derived from the forward graph via chain rule — it computes gradients of exactly the function that was evaluated. If a unit is zeroed in the forward pass, its gradient is automatically zero; no special backward handling is needed. Conversely, zeroing gradients without zeroing activations would compute gradients of a different function than the one that produced the loss, breaking gradient descent. Forward and backward must be consistent.

### 3. Where to apply: embd, attn, ffn — not lm_head

| Position | Field | Notes |
|---|---|---|
| After embedding | `dropout_embd` | No residual fallback — corruption propagates as the base `x` into all layers |
| After attention out_proj | `dropout_attn` | Residual `x` provides a fallback if output is dropped |
| After FFN activation | `dropout_ffn` | Classic MLP dropout position; most natural location |
| lm_head output (logits) | — | Never. Logits go directly into softmax; zeroing vocab entries distorts the probability distribution and destabilizes training |

### 4. Effect: regularization vs. noise

Dropout regularizes by preventing co-adaptation between neurons, but introduces gradient noise. The `1/(1-p)` scaling preserves the **mean** gradient, but **variance** grows with `p`:

```
# p=0.9: each weight's gradient per step
step 1-9:  0.0   (dropped, 90% of the time)
step 10:   10.0  (survived, scaled 10×)
mean = 1.0 (same as no dropout), variance = 9× higher
```

High dropout means sparse, spiky weight updates. Gradient clipping bounds the spikes, but at extreme `p` (≥0.5) the model learns to rely on the residual connections and largely ignore the sublayers.

### 5. Why modern LLMs don't use dropout

At scale, **training data is the regularizer**. A model trained on hundreds of billions of tokens has little opportunity to overfit — the bottleneck is underfitting, not overfitting. Dropout slows convergence (noisier gradients, sparser updates) without providing regularization benefit when data is abundant. LLaMA, Qwen, Mistral and most post-2020 models set `p=0.0` for pretraining. Dropout remains useful for small-scale training or fine-tuning on limited data, which is why this sweep uses a 57M model.
