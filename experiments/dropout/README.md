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

## Notes on Dropout

Dropout zeros `p` fraction of activations each forward pass and scales survivors by `1/(1-p)`. It is a regularization technique that prevents overfitting by adding noise to the network — forcing it to not rely on any specific neuron. The intuition is the sub-network view: each step trains a different randomly sampled sub-network, and over many steps the model learns robust features that don't depend on any specific neuron being present.

Dropout must live in the forward pass because the backward pass is derived from it — the chain rule propagates zero gradient through zeroed activations automatically. Applying it only to gradients would compute gradients of a different function than the one that produced the loss, breaking gradient descent.

In this codebase dropout is applied at three positions: after the embedding lookup (`dropout_embd`), after the attention output projection (`dropout_attn`), and after the FFN activation (`dropout_ffn`). The embedding position is the most aggressive since there is no residual fallback — dropped dimensions corrupt the base `x` that all subsequent residual additions build on. The attention and FFN positions sit before the residual add, so the original `x` always provides a fallback. Dropout is never applied to the lm_head logits because zeroing random vocab entries distorts the softmax distribution with no regularization benefit.

The `1/(1-p)` scaling preserves the expected gradient magnitude across dropout rates, but variance grows sharply with `p`. At `p=0.9` a weight receives gradient either 0 (90% of steps) or 10× (10% of steps) — same mean, 9× higher variance. This means sparser, spikier updates and slower effective learning. At extreme rates the model collapses onto the residual connections, bypassing the sublayers entirely.

Modern LLMs (LLaMA, Qwen, Mistral) set `p=0.0` for pretraining because training data is the regularizer at scale. With hundreds of billions of tokens the bottleneck is underfitting, not overfitting, and dropout only adds noise that slows convergence. It remains useful for small models or fine-tuning on limited data — which is why this sweep is run on a 57M model.
