# AdamW vs AdamC

Compare AdamW with AdamC on Qwen3 51M / OpenWebText. AdamC follows [Defazio (2025)](https://arxiv.org/abs/2506.02285) and scales decoupled weight decay with the current learning rate to correct cosine schedule decay.

## Hypothesis

AdamW applies its strongest weight decay while the learning rate is largest. AdamC reduces decay during warmup and late cosine decay for eligible matrix weights, which may improve validation loss at the same learning-rate schedule.

## Setup

| Config | Optimizer | Corrected decay groups | Approx. parameters |
|---|---|---|---:|
| `qwen3_51m_adamw` | AdamW | None | 50.93M |
| `qwen3_51m_adamc` | AdamC | Matrix weights except embeddings and `lm_head` | 50.93M |

Both runs use the same dense Qwen3 51M model: width 512, 8 layers, GQA with 8 query and 4 KV heads, QK normalization, and MLP intermediate size 1536. Data is OpenWebText with `tokenizers/custom_bpe_50k` and a 1% validation split.

Shared training settings: sequence length 1024, batch size 16, gradient accumulation 16, 262,144 tokens per optimizer step, 50,000 steps, BF16 mixed precision, seed 42, and whole-model compilation. Both optimizers use LR 5e-4, weight decay 0.1, betas `(0.9, 0.95)`, and epsilon `1e-8`; the cosine schedule has 1,500 warmup steps and a 5e-5 minimum LR. Evaluation runs every 100 steps for 100 batches, and checkpoints are saved every 5,000 steps.

AdamC keeps AdamW decay for embeddings, `lm_head`, and no-decay parameter groups. For corrected groups, it multiplies weight decay by `lr / max_lr` before the AdamW update.

## Run

Prepare the dataset and tokenizer using the repository setup instructions. Check GPU usage, then select a free GPU:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/adamc/run.sh
```

Replace `0` with a free GPU index. The runner trains AdamW followed by AdamC; arguments are forwarded to both runs, for example `--no-wandb --training.early_stop=1000`. Each run writes to its own `checkpoints/adamc/` directory and W&B runs use the `pretrain-adamc` project.

## Results

| Optimizer | Final validation loss | Final validation BPB | Status |
|---|---:|---:|---|
| AdamW | — | — | Not run |
| AdamC | — | — | Not run |

## Notes

- Compare `val/loss` and `val/bpb` at equal optimizer steps and tokens.
- AdamC and AdamW are identical at peak LR for corrected groups; their decay differs as the cosine schedule changes LR.
