# AdamW Weight-Decay Sweep

## Hypothesis

The best AdamW weight-decay coefficient for this Qwen3 51M training regime may differ from the baseline `0.1` when β₁ and β₂ remain fixed at `(0.9, 0.95)`.

## Setup

| Config | Weight decay | Approx. parameters |
|---|---:|---:|
| `qwen3_51m_adamw_wd_0` | 0 | 50.93M |
| `qwen3_51m_adamw_wd_0.01` | 0.01 | 50.93M |
| `qwen3_51m_adamw_wd_0.02` | 0.02 | 50.93M |
| `qwen3_51m_adamw_wd_0.05` | 0.05 | 50.93M |
| `qwen3_51m_adamw_wd_0.1` | 0.1 | 50.93M |
| `qwen3_51m_adamw_wd_0.2` | 0.2 | 50.93M |
| `qwen3_51m_adamw_wd_0.5` | 0.5 | 50.93M |
| `qwen3_51m_adamw_wd_1` | 1 | 50.93M |

All runs use dense Qwen3 51M on OpenWebText with the custom 50K BPE tokenizer and a 1% validation split. They use AdamW with β₁/β₂ `(0.9, 0.95)`, ε `1e-8`, sequence length 1024, effective batch size 256 (batch size 16 × accumulation 16), 50,000 steps, 13.1072B tokens, bf16, gradient clipping 1.0, and a cosine schedule from `5e-4` to `5e-5` with 1,500 warmup steps. Evaluation runs every 100 steps for 100 batches; checkpoints run every 5,000 steps.

## Run

The runner executes all eight coefficients sequentially and forwards CLI arguments to each run. Checkpoints go under `checkpoints/adamw_weight_decay/`; W&B runs use the `pretrain-adamw-weight-decay` project.

Check GPU usage, then choose a free device:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/adamw_weight_decay/run.sh
```

For a 10k-step screen that keeps the 50k-step cosine schedule, pass an early-stop override:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/adamw_weight_decay/run.sh --training.early_stop=10000
```

Run full configurations or individual finalists from scratch by invoking their YAML directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --config experiments/adamw_weight_decay/qwen3_51m_adamw_wd_0.1.yaml
```

## Results

| Weight decay | Final validation loss | Final validation BPB | Status |
|---:|---:|---:|---|
| 0 | — | — | Pending |
| 0.01 | — | — | Pending |
| 0.02 | — | — | Pending |
| 0.05 | — | — | Pending |
| 0.1 | — | — | Pending |
| 0.2 | — | — | Pending |
| 0.5 | — | — | Pending |
| 1 | — | — | Pending |

## Notes

- Compare validation loss and BPB at equal token counts.
- A screening run stops at 10k optimizer steps but retains the original 50k-step scheduler, so its learning-rate trajectory matches the prefix of a full run.
- Train selected finalists from scratch to 50k steps for the final comparison.
- The selected coefficient is conditional on the fixed β₁/β₂ values; cross-check the strongest coefficients with the best pairs from the β sweep.
