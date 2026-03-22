# Muon vs AdamW

Compare the Muon optimizer against AdamW on GPT-2 across multiple scales, and sweep Muon's key hyperparameters.

## Hypothesis

Muon's Newton-Schulz orthogonalization of matrix gradients should yield faster loss reduction per step than AdamW, since it normalizes update geometry across weight matrices of different scales. The advantage should be consistent across model sizes.

## Scaling comparison

Fixed training budget per size. Only the optimizer differs between paired runs.

| Config | d_model | Layers | Heads | ~Params | Steps | AdamW LR | Muon matrix LR |
|---|---|---|---|---|---|---|---|
| gpt2_16m | 256 | 4 | 4 | ~16M | 20k | 1e-3 | 0.04 |
| gpt2_30m | 384 | 6 | 6 | ~30M | 40k | 8e-4 | 0.04 |
| gpt2_55m | 512 | 8 | 8 | ~55M | 70k | 6e-4 | 0.04 |

Muon also uses `muon_embed_lr=0.6` and `muon_scalar_lr=0.04` for the Adam sub-optimizers.

```bash
bash experiments/muon/run.sh
```

W&B project: `pretrain-muon`.

## Results

TODO
