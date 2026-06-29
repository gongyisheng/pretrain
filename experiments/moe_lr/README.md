# Learning Rate Sweep (MoE)

Sweep learning rate on the Qwen3 MoE 183M testbed to find the optimal LR. The
dense LR table was fit on dense models; this checks whether a 183M-total MoE
(~51M active) wants a different LR.

## Hypothesis

There is a compute-optimal LR for a given model size and token budget. For MoE
the right LR is ambiguous: indexing the dense table by **total** params (183M)
gives 6e-4, by **active** params (~51M) gives ~6e-4 as well. Experts each see
only a fraction of tokens (noisier per-expert gradients), which argues against
pushing LR much higher than the dense optimum.

## Setup

Fixed testbed: `qwen3_183m_a51m` architecture, 64 experts, top-8, per-expert
`intermediate_size=192` (active `k·is = 1536`), capacity factor 1.25,
`aux_loss_coef=0.001`. Only LR varies; `min_lr = lr/10`.

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_experts / top-k | 64 / 8 |
| intermediate_size (per expert) | 192 |
| expert_capacity_factor | 1.25 |
| batch × grad_accum × seq | 16 × 16 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| warmup | 1500 |

| Config | LR | min_lr |
|---|---|---|
| `qwen3_183m_a51m_lr1e-4` | 1e-4 | 1e-5 |
| `qwen3_183m_a51m_lr2e-4` | 2e-4 | 2e-5 |
| `qwen3_183m_a51m_lr3e-4` | 3e-4 | 3e-5 |
| `qwen3_183m_a51m_lr5e-4` | 5e-4 | 5e-5 |
| `qwen3_183m_a51m_lr1e-3` | 1e-3 | 1e-4 |
| `qwen3_183m_a51m_lr2e-3` | 2e-3 | 2e-4 |
| `qwen3_183m_a51m_lr3e-3` | 3e-3 | 3e-4 |
| `qwen3_183m_a51m_lr5e-3` | 5e-3 | 5e-4 |

## Run

```bash
nohup bash experiments/moe_lr/run.sh > logs/moe_lr.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_lr/qwen3_183m_a51m_lr1e-3.yaml
```

## Results

| LR | Final Val Loss | notes |
|---|---|---|
| 1e-4 | | |
| 2e-4 | | |
| 3e-4 | | |
| 5e-4 | | |
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
