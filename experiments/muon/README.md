# Muon vs AdamW

Compare the Muon optimizer against AdamW on GPT-2 at fixed model size and compute budget.

## Hypothesis

Muon's Newton-Schulz orthogonalization of matrix gradients should yield faster loss reduction per step than AdamW, since it normalizes update geometry across weight matrices of different scales.

## Setup

Fixed model (d=512, 12 layers, 8 heads, ~64M params) and training budget (80k steps, batch=32k tokens). Only the optimizer differs between runs.

| Config | Optimizer | Matrix LR | Embed LR | Scalar LR |
|---|---|---|---|---|
| gpt2_adamw | AdamW | 8e-4 | 8e-4 | 8e-4 |
| gpt2_muon | Muon | 0.04 | 0.6 | 0.04 |

Both runs use the same warmup (1500 steps), cosine decay to 10% of peak LR, and identical data order.

## Run

```bash
# Individual runs
python scripts/train.py --config experiments/muon/gpt2_adamw.yaml
python scripts/train.py --config experiments/muon/gpt2_muon.yaml

# Both sequentially
bash experiments/muon/run.sh
```

## W&B

Project: `pretrain-muon`. Compare `gpt2-adamw` vs `gpt2-muon` runs.

Suggested chart: X=`train/total_tokens`, Y=`val/loss`, grouped by run.

## Results

TODO
