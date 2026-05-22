# Grokking

Reproducing the canonical grokking phenomenon (Power et al. 2022, [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)) on modular arithmetic: val accuracy stays near chance long after train accuracy saturates, then suddenly jumps to ~100%.

## Hypothesis

A small Qwen3 (~1M params) trained on a fraction (30%) of all `(a, b)` pairs under a modular arithmetic operation will exhibit grokking when weight decay is strong. The strength of weight decay drives whether and when val accuracy lifts off chance.

## Setup

- **Model:** Qwen3, 4 layers, 4 heads, 2 KV heads, d_model=128 (~1.0M params, ~984k non-embedding).
- **Vocab:** 105 tokens — residues 0..99 (IDs 0..99) + operators `+ - * / =` (IDs 100..104). Fixed across all runs.
- **Data:** `a op b = c` sequences, 5 tokens per sample. p=97. Train/val = 30% / 70% partition of all valid `(a, b)` pairs (deterministic given seed).
- **Loss:** cross-entropy on the answer position only, via the HF `-100` ignore-index labels convention (`SFTDataset` masks the question positions).
- **Optimizer:** AdamW, lr=1e-3, betas=(0.9, 0.98), constant scheduler with 10-step linear warmup.
- **Batch:** 512, gradient_accumulation=1, max_steps=500k.
- **Eval:** every 100 steps; train+val accuracy logged.

| Run | op | weight_decay |
|---|---|---:|
| 1 | add | 0.0 |
| 2 | add | 0.1 |
| 3 | add | 1.0 |
| 4 | sub | 0.0 |
| 5 | sub | 0.1 |
| 6 | sub | 1.0 |
| 7 | mul | 0.0 |
| 8 | mul | 0.1 |
| 9 | mul | 1.0 |
| 10 | div | 0.0 |
| 11 | div | 0.1 |
| 12 | div | 1.0 |

## Run

```bash
# 1. Build the tokenizer (once).
uv run python experiments/grokking/generate_tokenizer.py

# 2. Generate data for all 4 ops (once).
for op in add sub mul div; do
    uv run python experiments/grokking/generate_data.py --op $op
done

# 3. Run the full sweep.
bash experiments/grokking/run_weight_decay.sh
```

Single config:
```bash
bash experiments/grokking/run_weight_decay.sh add 1.0
```

## Results

(Fill in after runs complete. Expected: with WD=1.0, val accuracy lifts off chance after train accuracy saturates — the grokking curve. With WD=0.0, the model memorizes and val accuracy stays at chance.)

| op  | WD=0.0 final val_acc | WD=0.1 final val_acc | WD=1.0 final val_acc | step of grok (WD=1.0) |
|-----|---:|---:|---:|---:|
| add | TBD | TBD | TBD | TBD |
| sub | TBD | TBD | TBD | TBD |
| mul | TBD | TBD | TBD | TBD |
| div | TBD | TBD | TBD | TBD |

## AdamW eps × loss ablation (sub, wd=0.1)

Liu et al. 2025 ([arXiv:2605.06152](https://arxiv.org/abs/2605.06152)) show that fp32 softmax/CE rounds the correct-class gradient to exactly zero in high-confidence regimes, breaking the class-grad zero-sum constraint and driving slingshot spikes / NFI drift. The `cross_entropy_fp64` loss restores the gradient direction, but the recovered magnitudes can be `~1e-12` or smaller.

AdamW's per-coordinate effective lr is `lr / (sqrt(v) + eps)`. Two opposing failure modes:

- **Large `eps` (e.g. 1e-5, LLaMA-style)**: caps max per-coord step at `lr/eps`, killing slingshot-style spikes from `v` decay — but also crushes the recovered fp64 signal (`sqrt(v) << eps` ⇒ update suppressed by `sqrt(v)/eps`).
- **Small `eps` (e.g. 1e-15)**: lets fp64-tiny grads drive learning normally — but amplifies any near-zero `v` to a huge effective step.

| variant | `loss_fn` | `eps` | hypothesis |
|---|---|---|---|
| ce_eps1e-15 | cross_entropy | 1e-15 | baseline; no fp64 fix, full adaptive scaling |
| ce_eps1e-12 | cross_entropy | 1e-12 | baseline at intermediate eps |
| ce_eps1e-8  | cross_entropy | 1e-8  | textbook default |
| ce_eps1e-7  | cross_entropy | 1e-7  | mild trust region |
| ce_eps1e-6  | cross_entropy | 1e-6  | medium trust region |
| ce_eps1e-5  | cross_entropy | 1e-5  | LLaMA-style trust region |
| ce_fp64_eps1e-15 | cross_entropy_fp64 | 1e-15 | fp64 + magnitude preserved |
| ce_fp64_eps1e-12 | cross_entropy_fp64 | 1e-12 | fp64 + mild eps damping |
| ce_fp64_eps1e-8  | cross_entropy_fp64 | 1e-8  | fp64 + default eps |
| ce_fp64_eps1e-7  | cross_entropy_fp64 | 1e-7  | fp64 + mild trust region |
| ce_fp64_eps1e-6  | cross_entropy_fp64 | 1e-6  | fp64 + medium trust region |
| ce_fp64_eps1e-5  | cross_entropy_fp64 | 1e-5  | fp64 direction-only (magnitude killed by eps) |

Run:

```bash
# all 12 (respect MAX_CONCURRENCY)
nohup bash experiments/grokking/run_adamw_eps.sh > logs/grokking/eps_sweep.log 2>&1 &

# single variant
bash experiments/grokking/run_adamw_eps.sh ce_fp64_eps1e-5
```

Reads:
- If `(ce_fp64, eps=1e-5) ≈ (ce_fp64, eps=1e-15)` → direction-fix dominates; use `eps=1e-5` for general safety.
- If they diverge → magnitude matters; recovered fp64 grads do drive late-stage updates.
- If `ce` baselines improve monotonically with eps → slingshot-style spikes were the bottleneck, not gradient absorption.

### Results

| eps | ce final val_acc | ce_fp64 final val_acc | ce grok step | ce_fp64 grok step |
|---|---:|---:|---:|---:|
| 1e-15 | TBD | TBD | TBD | TBD |
| 1e-12 | TBD | TBD | TBD | TBD |
| 1e-8  | TBD | TBD | TBD | TBD |
| 1e-7  | TBD | TBD | TBD | TBD |
| 1e-6  | TBD | TBD | TBD | TBD |
| 1e-5  | TBD | TBD | TBD | TBD |
