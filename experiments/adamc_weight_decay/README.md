# AdamC Weight-Decay Sweep

## Hypothesis

The best AdamC weight-decay coefficient for this Qwen3 51M training regime may differ from the baseline `0.1` when β₁ and β₂ remain fixed at `(0.9, 0.99)`.

## Setup

| Config | Weight decay | Approx. parameters |
|---|---:|---:|
| `qwen3_51m_adamc_wd_0` | 0 | 50.93M |
| `qwen3_51m_adamc_wd_0.01` | 0.01 | 50.93M |
| `qwen3_51m_adamc_wd_0.02` | 0.02 | 50.93M |
| `qwen3_51m_adamc_wd_0.05` | 0.05 | 50.93M |
| `qwen3_51m_adamc_wd_0.1` | 0.1 | 50.93M |
| `qwen3_51m_adamc_wd_0.2` | 0.2 | 50.93M |
| `qwen3_51m_adamc_wd_0.5` | 0.5 | 50.93M |
| `qwen3_51m_adamc_wd_1` | 1 | 50.93M |

All runs match the [standard 200k AdamC experiment](../adamc/qwen3_51m_adamc_steps_200k.yaml), with accumulation intentionally increased from `4` to `16` and β₂ changed from `0.95` to `0.99`, varying weight decay and output identifiers across this sweep. They use dense Qwen3 51M on OpenWebText with the custom 50K BPE tokenizer and a 1% validation split, GQA (8 query heads, 4 KV heads), QK normalization, and attention sinks disabled (the default).

AdamC uses β₁/β₂ `(0.9, 0.99)` and ε `1e-8`. Training uses sequence length 1024, effective batch size 256 (batch size 16 × accumulation 16), 262,144 tokens per update, 200,000 steps, and 52.4288B tokens total. CUDA, bf16, whole-model compilation, gradient clipping 1.0, and a cosine schedule from `2.5e-4` to `2.5e-5` with 6,000 warmup steps match the baseline. Evaluation runs every 100 steps for 100 batches, checkpoints every 20,000 steps, and logging every 50 steps.

## Run

The runner executes all eight coefficients sequentially and forwards CLI arguments to each run. Checkpoints go under `checkpoints/adamc_weight_decay/`; W&B runs use the `pretrain-adamc-weight-decay` project.

Check GPU usage, then choose a free device:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 bash experiments/adamc_weight_decay/run.sh
```

For a 10k-step screen that keeps the 200k-step cosine schedule, pass an early-stop override:

```bash
CUDA_VISIBLE_DEVICES=0 bash experiments/adamc_weight_decay/run.sh --training.early_stop=10000
```

Run full configurations or individual finalists from scratch by invoking their YAML directly:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py \
  --config experiments/adamc_weight_decay/qwen3_51m_adamc_wd_0.1.yaml
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

- AdamC applies corrected decay to eligible matrix weights. Embeddings and the LM head retain ordinary AdamW decay behavior; no-decay parameter groups remain exempt from decay.
- Compare validation loss and BPB at equal token counts.
- Accumulation 16 intentionally increases the standard 200k baseline’s effective batch size; checkpoint interval 20,000 retains the baseline cadence.
- A screening run stops at 10k optimizer steps but retains the original 200k-step scheduler, so its learning-rate trajectory matches the prefix of a full run.
- Train selected finalists from scratch to 200k steps for the final comparison.
