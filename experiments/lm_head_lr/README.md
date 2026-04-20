# LM Head LR Multiplier

Test whether lowering the learning rate of `lm_head` (relative to the rest of the model) improves training quality or stability on the Qwen3 architecture at two scales.

## Hypothesis

In untied mode the `lm_head` matrix sits directly on the softmax-CE gradient: every position contributes a dense outer product to `dL/dW_out`, and the target-row direction dominates. Meanwhile the gradient that flows back into the transformer body (`dL/dh = (p − onehot) · W_out`) is roughly one row of `W_out` and is small at initialization. This produces a large gradient-norm asymmetry (observed in the `tie_word_embd` experiment: `lm_head` grad norm ≫ `token_emb` grad norm).

AdamW normalizes per-parameter by `√v_t`, so raw magnitude differences alone don't translate directly into larger step sizes. But the `lm_head` gradient direction is highly structured — target rows get a near-deterministic negative push every step, which can drive **logit inflation** and instability. Muon-style recipes (Moonshot K2, DeepSeek-V2) put embedding + lm_head on AdamW with a fractional LR multiplier (typically ~0.3) to counter this. µP theory separately predicts that the output layer should train slower than the body as width grows.

The experiment sweeps `lr_mult.lm_head ∈ {1.0, 0.5, 0.3, 0.2, 0.1}` on untied runs (baseline + four attenuation levels) and includes one tied run per scale as a sanity-check control (in tied mode the multiplier is a no-op because `lm_head.weight is token_emb.weight`).

## Setup

**Implementation.** New `optimizer.lr_mult: Dict[str, float] = {"lm_head": 1.0}` in `OptimizerConfig`. `build_optimizer` splits any `lm_head.*` parameters into their own AdamW group with an `lr_mult` key from `lr_mult["lm_head"]`; `CosineWarmupScheduler.step()` applies `pg["lr"] = base_lr * pg.get("lr_mult", 1.0)`, so `lm_head` follows the same cosine/warmup shape as the rest of the model, just scaled. Default `1.0` = no behavior change. (`lm_head` params always go through weight decay — `lm_head.weight` is 2D and has no bias by default, so no `no_decay` split is needed.)

| Config | tie | lr_mult.lm_head | Approx params |
|---|---|---|---|
| qwen3_57m_tied | true | — (no-op) | ~57M |
| qwen3_57m_untied_mult1.0 | false | 1.0 | ~83M |
| qwen3_57m_untied_mult0.5 | false | 0.5 | ~83M |
| qwen3_57m_untied_mult0.3 | false | 0.3 | ~83M |
| qwen3_57m_untied_mult0.2 | false | 0.2 | ~83M |
| qwen3_57m_untied_mult0.1 | false | 0.1 | ~83M |
| qwen3_0.5b_tied | true | — (no-op) | ~0.5B |
| qwen3_0.5b_untied_mult1.0 | false | 1.0 | ~0.55B |
| qwen3_0.5b_untied_mult0.5 | false | 0.5 | ~0.55B |
| qwen3_0.5b_untied_mult0.3 | false | 0.3 | ~0.55B |
| qwen3_0.5b_untied_mult0.2 | false | 0.2 | ~0.55B |
| qwen3_0.5b_untied_mult0.1 | false | 0.1 | ~0.55B |

**57M runs**: Qwen3 (d_model=512, layers=8, heads=8, kv_heads=4, qk_norm=true), seq_len=1024, batch_size=16, grad_accum=2, 50K steps, lr=6e-4, warmup=200 steps, min_lr=6e-5, bf16, OpenWebText.

**0.5B runs**: Qwen3 (d_model=1024, layers=28, heads=16, kv_heads=8, qk_norm=true), seq_len=1024, batch_size=8, grad_accum=4, 50K steps, lr=2e-4, warmup=1500 steps, min_lr=2e-5, bf16, OpenWebText.

## Run

```bash
# All 12 runs
nohup bash experiments/lm_head_lr/run.sh > logs/lm_head_lr.log 2>&1 &

# 57M only
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_tied.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_untied_mult1.0.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_untied_mult0.5.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_untied_mult0.3.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_untied_mult0.2.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_57m_untied_mult0.1.yaml

# 0.5B only
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_tied.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_untied_mult1.0.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_untied_mult0.5.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_untied_mult0.3.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_untied_mult0.2.yaml
uv run python scripts/train.py --config experiments/lm_head_lr/qwen3_0.5b_untied_mult0.1.yaml
```

## Results

### Val Loss

| Model | tie | lr_mult.lm_head | Val Loss |
|---|---|---|---|
| qwen3_57m_tied | true | — | |
| qwen3_57m_untied | false | 1.0 | |
| qwen3_57m_untied | false | 0.5 | |
| qwen3_57m_untied | false | 0.3 | |
| qwen3_57m_untied | false | 0.2 | |
| qwen3_57m_untied | false | 0.1 | |
| qwen3_0.5b_tied | true | — | |
| qwen3_0.5b_untied | false | 1.0 | |
| qwen3_0.5b_untied | false | 0.5 | |
| qwen3_0.5b_untied | false | 0.3 | |
| qwen3_0.5b_untied | false | 0.2 | |
| qwen3_0.5b_untied | false | 0.1 | |

### Gradient Norm (lm_head vs body)

_To fill in after runs complete._

## Notes

_To fill in after results._
