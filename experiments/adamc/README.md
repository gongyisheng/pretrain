# AdamW vs AdamC

Compare AdamW with AdamC on Qwen3 51M / OpenWebText. AdamC follows [Defazio (2025)](https://arxiv.org/abs/2506.02285) and scales decoupled weight decay with the current learning rate to correct cosine schedule decay.

## Hypothesis

AdamW applies its strongest weight decay while the learning rate is largest. AdamC reduces decay during warmup and late cosine decay for eligible matrix weights, which may improve validation loss at the same learning-rate schedule.

## Setup

| Config | Optimizer | Steps | Corrected decay groups | Approx. parameters |
|---|---|---:|---|---:|
| `qwen3_51m_adamw_steps_50k` | AdamW | 50,000 | None | 50.93M |
| `qwen3_51m_adamc_steps_50k` | AdamC | 50,000 | Matrix weights except embeddings and `lm_head` | 50.93M |
| `qwen3_51m_adamw_steps_800k` | AdamW | 800,000 | None | 50.93M |
| `qwen3_51m_adamc_steps_800k` | AdamC | 800,000 | Matrix weights except embeddings and `lm_head` | 50.93M |

All four runs use the same dense Qwen3 51M model: width 512, 8 layers, GQA with 8 query and 4 KV heads, QK normalization, and MLP intermediate size 1536. Data is OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split.

Both regimes process 13.1072B tokens. Within each regime, AdamW and AdamC have matched model, data, optimizer hyperparameters, and schedule.

| Regime | Batch size | Gradient accumulation | Tokens/step | Steps | Peak/min LR | Warmup | Eval/checkpoint/log every |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50k | 16 | 16 | 262,144 | 50,000 | `5e-4` / `5e-5` | 1,500 (3%) | 100 / 5,000 / 10 steps |
| 800k | 16 | 1 | 16,384 | 800,000 | `1.25e-4` / `1.25e-5` | 24,000 (3%) | 100 / 80,000 / 100 steps |

The 800k regime reduces the peak and minimum learning rates by four using the square-root learning-rate heuristic for its 16x smaller optimizer batch. This is a heuristic, not a proven matching rule. It scales warmup and checkpoint intervals by 16 to preserve their token cadence. Evaluation and logging run every 100 steps; evaluation uses 100 batches and yields 8,000 evaluations at 800k versus 500 at 50k.

The two regimes hold total token count fixed but change optimizer batch size, update count, and learning rate. Their results therefore cannot be attributed to update count alone. Compare AdamW and AdamC within the same regime.

AdamC keeps AdamW decay for embeddings, `lm_head`, and no-decay parameter groups. For corrected groups, it multiplies weight decay by `lr / max_lr` before the AdamW update.

## Run

Prepare the dataset and tokenizer using the repository setup instructions. Check GPU usage, then select a free GPU:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/adamc/run.sh
```

Replace `0` with a free GPU index. The runner trains AdamW then AdamC for 50k, followed by the same order for 800k. Arguments are forwarded to all runs, for example `--no-wandb --training.early_stop=1000`. Each run writes to its own `checkpoints/adamc/` directory and W&B runs use the `pretrain-adamc` project.

## Results

| Regime | Optimizer | Final validation loss | Final validation BPB | Status |
|---|---|---:|---:|---|
| 50k | AdamW | — | — | Not run |
| 50k | AdamC | — | — | Not run |
| 800k | AdamW | — | — | Not run |
| 800k | AdamC | — | — | Not run |

## Notes

- Compare `val/loss` and `val/bpb` at equal token counts within a regime.
- AdamC and AdamW are identical at peak LR for corrected groups; their decay differs as the cosine schedule changes LR.
