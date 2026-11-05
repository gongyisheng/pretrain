# BF16 versus FP32

## Hypothesis

Disabling bf16 autocast may improve Qwen3 51M pretraining loss or numerical stability, at the cost of training throughput and GPU memory. Compare both precision modes with the same baseline configuration and seed.

## Setup

| Config | Precision setting | Approx. parameters | Key settings |
|---|---|---:|---|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | `mixed_precision: bf16` | 50,931,264 | CUDA bf16 autocast |
| [qwen3_51m_fp32.yaml](qwen3_51m_fp32.yaml) | `mixed_precision: "no"` | 50,931,264 | Autocast disabled |

Both configs copy `configs/qwen3_51m.yaml`, changing only precision and output identifiers. The model uses 8 layers, d_model=512, 8 query / 4 KV heads, attention sinks, RMSNorm, RoPE, and an MLP intermediate size of 1536. The parameter count was confirmed through `build_model`.

Both runs use OpenWebText with `tokenizers/custom_bpe_50k`, sequence length 1024, batch size 16, gradient accumulation 16, 50,000 training steps, and seed 42. MuonAdam uses lr=5e-4, weight decay=0.1, AdamW betas=(0.9, 0.99), and Muon momentum=0.95. The cosine schedule has 1,500 warmup steps and min lr=5e-5. Checkpoints are saved every 5,000 steps; evaluation runs every 100 steps for 100 batches.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/fp32/run.sh
```

The runner executes bf16 first, then fp32. It forwards trainer arguments to both runs:

```bash
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/fp32/run.sh --no-wandb --training.early_stop=100
```

## Results

| Config | Final validation loss | Throughput (tokens/s) | Peak GPU memory | Notes |
|---|---:|---:|---:|---|
| `qwen3_51m_bf16` | Pending | Pending | Pending | |
| `qwen3_51m_fp32` | Pending | Pending | Pending | |

## Notes

The fp32 variant disables trainer autocast. The trainer still enables TF32 for supported CUDA matrix operations, and Muon's Newton–Schulz orthogonalization explicitly casts gradients to bf16 in both runs. This compares the existing trainer precision modes; the fp32 variant does not use strict IEEE fp32 arithmetic throughout.

W&B project: `pretrain-fp32`; run names: `qwen3-51m-bf16` and `qwen3-51m-fp32`. Checkpoints are isolated under `checkpoints/fp32/<config>/`. This is a single-seed comparison; full training results are pending.
