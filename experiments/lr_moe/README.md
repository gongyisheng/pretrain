# Learning Rate Sweep (MoE)

Sweep learning rate on the Qwen3 MoE 133M testbed (133M total, ~35M active) to
find the optimal LR for this model. The dense LR table was fit on dense models;
this checks whether a 133M-total / 35M-active MoE wants a different LR.

## Hypothesis

There is a compute-optimal LR for a given model size and token budget. For MoE
the right LR is ambiguous: indexing the dense table by **total** params (133M)
gives 6e-4, by **active** params (~35M) gives ~8e-4. Experts each see only
~2/64 of tokens (noisier per-expert gradients), which argues against pushing LR
much higher than the dense optimum.

## Setup

Fixed testbed: `qwen3_moe_133m` architecture, 64 experts, top-2, capacity factor
1.25, `aux_loss_coef=0.01` (Switch default). Only LR varies; `min_lr = lr/10`.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 2 |
| expert_capacity_factor | 1.25 |
| batch × grad_accum × seq | 8 × 32 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| warmup | 1000 |

| Config | LR | min_lr |
|---|---|---|
| qwen3_133m_a35m_lr1e-4 | 1e-4 | 1e-5 |
| qwen3_133m_a35m_lr2e-4 | 2e-4 | 2e-5 |
| qwen3_133m_a35m_lr3e-4 | 3e-4 | 3e-5 |
| qwen3_133m_a35m_lr5e-4 | 5e-4 | 5e-5 |
| qwen3_133m_a35m_lr1e-3 | 1e-3 | 1e-4 |
| qwen3_133m_a35m_lr2e-3 | 2e-3 | 2e-4 |
| qwen3_133m_a35m_lr3e-3 | 3e-3 | 3e-4 |
| qwen3_133m_a35m_lr5e-3 | 5e-3 | 5e-4 |
| qwen3_133m_a35m_lr5e-4_minlr_1e-4 | 5e-4 | 1e-4 |

## Run

```bash
nohup bash experiments/lr_moe/run.sh > logs/lr_moe.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/lr_moe/qwen3_133m_a35m_lr1e-3.yaml
```

## Results

| LR | Final Val Loss | notes |
|---|---|---|
| 1e-4 | | |
| 2e-4 | | |
| 3e-4 | | |
| 5e-4 | | |
| 5e-4 | | min_lr 1e-4 (vs 5e-5) |
| 1e-3 | | |
| 2e-3 | | |
| 3e-3 | | |
| 5e-3 | | |

Best: _TBD_

## Notes

- LR sweep shortcut: run all candidates for ~500 steps first; drop any higher LR
  already losing to a lower LR, then extend the survivors. Saves compute.
- Compare the winning LR against the dense `../lr/` sweep (57M dense optimum was
  5e-4–1e-3) to see how the MoE optimum shifts.
