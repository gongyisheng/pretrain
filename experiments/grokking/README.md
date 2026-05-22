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
- **Batch:** 512, gradient_accumulation=1, max_steps=100k.
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
bash experiments/grokking/run.sh
```

Single config:
```bash
bash experiments/grokking/run.sh add 1.0
```

## Results

(Fill in after runs complete. Expected: with WD=1.0, val accuracy lifts off chance after train accuracy saturates — the grokking curve. With WD=0.0, the model memorizes and val accuracy stays at chance.)

| op  | WD=0.0 final val_acc | WD=0.1 final val_acc | WD=1.0 final val_acc | step of grok (WD=1.0) |
|-----|---:|---:|---:|---:|
| add | TBD | TBD | TBD | TBD |
| sub | TBD | TBD | TBD | TBD |
| mul | TBD | TBD | TBD | TBD |
| div | TBD | TBD | TBD | TBD |
