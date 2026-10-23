# AdamW β₁/β₂ Sweep

## Hypothesis

AdamW's first- and second-moment decay rates affect optimization speed and stability. The best pair for this Qwen3 51M training regime may differ from the baseline `(0.9, 0.95)`.

## Setup

| Config | β₁ | β₂ | Approx. parameters |
|---|---:|---:|---:|
| `qwen3_51m_adamw_b1_0.8_b2_0.9` | 0.8 | 0.9 | 50.93M |
| `qwen3_51m_adamw_b1_0.8_b2_0.95` | 0.8 | 0.95 | 50.93M |
| `qwen3_51m_adamw_b1_0.8_b2_0.99` | 0.8 | 0.99 | 50.93M |
| `qwen3_51m_adamw_b1_0.8_b2_0.999` | 0.8 | 0.999 | 50.93M |
| `qwen3_51m_adamw_b1_0.9_b2_0.9` | 0.9 | 0.9 | 50.93M |
| `qwen3_51m_adamw_b1_0.9_b2_0.95` | 0.9 | 0.95 | 50.93M |
| `qwen3_51m_adamw_b1_0.9_b2_0.99` | 0.9 | 0.99 | 50.93M |
| `qwen3_51m_adamw_b1_0.9_b2_0.999` | 0.9 | 0.999 | 50.93M |
| `qwen3_51m_adamw_b1_0.95_b2_0.9` | 0.95 | 0.9 | 50.93M |
| `qwen3_51m_adamw_b1_0.95_b2_0.95` | 0.95 | 0.95 | 50.93M |
| `qwen3_51m_adamw_b1_0.95_b2_0.99` | 0.95 | 0.99 | 50.93M |
| `qwen3_51m_adamw_b1_0.95_b2_0.999` | 0.95 | 0.999 | 50.93M |
| `qwen3_51m_adamw_b1_0.99_b2_0.9` | 0.99 | 0.9 | 50.93M |
| `qwen3_51m_adamw_b1_0.99_b2_0.95` | 0.99 | 0.95 | 50.93M |
| `qwen3_51m_adamw_b1_0.99_b2_0.99` | 0.99 | 0.99 | 50.93M |
| `qwen3_51m_adamw_b1_0.99_b2_0.999` | 0.99 | 0.999 | 50.93M |

All runs use dense Qwen3 51M on OpenWebText with the custom 50K BPE tokenizer and a 1% validation split. They preserve the AdamW baseline: sequence length 1024, effective batch size 256 (batch size 16 × accumulation 16), 50,000 steps, 13.1072B tokens, bf16, gradient clipping 1.0, weight decay 0.1, and a cosine schedule from `5e-4` to `5e-5` with 1,500 warmup steps. Evaluation runs every 100 steps for 100 batches; checkpoints run every 5,000 steps.

## Run

The runner executes all 16 pairs sequentially and forwards CLI arguments to each run. Checkpoints go under `checkpoints/adamw_b1_b2/`; W&B runs use the `pretrain-adamw-b1-b2` project.

Check GPU usage, then choose a free device:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/adamw_b1_b2/run.sh
```

For a 10k-step screen that keeps the 50k-step cosine schedule, pass an early-stop override:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/adamw_b1_b2/run.sh --training.early_stop=10000
```

Run full configurations or individual finalists from scratch by invoking their YAML directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --config experiments/adamw_b1_b2/qwen3_51m_adamw_b1_0.9_b2_0.95.yaml
```

## Results

| β₁ | β₂ | Final validation loss | Final validation BPB | Status |
|---:|---:|---:|---:|---|
| 0.8 | 0.9 | — | — | Pending |
| 0.8 | 0.95 | — | — | Pending |
| 0.8 | 0.99 | — | — | Pending |
| 0.8 | 0.999 | — | — | Pending |
| 0.9 | 0.9 | — | — | Pending |
| 0.9 | 0.95 | — | — | Pending |
| 0.9 | 0.99 | — | — | Pending |
| 0.9 | 0.999 | — | — | Pending |
| 0.95 | 0.9 | — | — | Pending |
| 0.95 | 0.95 | — | — | Pending |
| 0.95 | 0.99 | — | — | Pending |
| 0.95 | 0.999 | — | — | Pending |
| 0.99 | 0.9 | — | — | Pending |
| 0.99 | 0.95 | — | — | Pending |
| 0.99 | 0.99 | — | — | Pending |
| 0.99 | 0.999 | — | — | Pending |

## Notes

- Compare validation loss and BPB at equal token counts.
- A screening run stops at 10k optimizer steps but retains the original 50k-step scheduler, so its learning-rate trajectory matches the prefix of a full run.
- Train selected finalists from scratch to 50k steps for the final comparison.
