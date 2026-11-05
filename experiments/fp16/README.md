# BF16 versus FP16

## Hypothesis

FP16 autocast may match bf16 pretraining loss and throughput for Qwen3 51M, but its narrower exponent range may require gradient scaling to avoid numerical failures. Compare both precision modes with the same baseline configuration and seed.

## Setup

| Config | Precision setting | Approx. parameters | Key settings |
|---|---|---:|---|
| [qwen3_51m_bf16.yaml](qwen3_51m_bf16.yaml) | `mixed_precision: bf16` | 50,931,264 | CUDA bf16 autocast without GradScaler |
| [qwen3_51m_fp16.yaml](qwen3_51m_fp16.yaml) | `mixed_precision: fp16` | 50,931,264 | CUDA fp16 autocast with GradScaler |

Both configs copy `configs/qwen3_51m.yaml`, changing only precision and output identifiers. The model uses 8 layers, d_model=512, 8 query / 4 KV heads, attention sinks, RMSNorm, RoPE, and an MLP intermediate size of 1536. The parameter count was confirmed through `build_model`.

Both runs use OpenWebText with `tokenizers/custom_bpe_50k`, sequence length 1024, batch size 16, gradient accumulation 16, 50,000 training steps, and seed 42. MuonAdam uses lr=5e-4, weight decay=0.1, AdamW betas=(0.9, 0.99), and Muon momentum=0.95. The cosine schedule has 1,500 warmup steps and min lr=5e-5. Checkpoints are saved every 5,000 steps; evaluation runs every 100 steps for 100 batches.

## Run

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/fp16/run.sh
```

The runner executes bf16 first, then fp16. It forwards trainer arguments to both runs:

```bash
CUDA_VISIBLE_DEVICES=<free_idx> bash experiments/fp16/run.sh --no-wandb --training.early_stop=100
```

## Results

| Config | Final validation loss | Throughput (tokens/s) | Peak GPU memory | Notes |
|---|---:|---:|---:|---|
| `qwen3_51m_bf16` | Pending | Pending | Pending | |
| `qwen3_51m_fp16` | Pending | Pending | Pending | |

## Notes

Both variants keep model parameters in fp32 and use CUDA autocast for eligible operations. FP16 enables dynamic gradient scaling through GradScaler; bf16 does not use gradient scaling. GradScaler unscales gradients before clipping and can skip an optimizer update when gradients overflow. Muon's Newton–Schulz orthogonalization explicitly casts gradients to bf16 in both runs. The trainer also enables TF32 for supported fp32 CUDA matrix operations. This compares the existing AMP modes rather than using one dtype for every operation.

W&B project: `pretrain-fp16`; run names: `qwen3-51m-bf16` and `qwen3-51m-fp16`. Checkpoints are isolated under `checkpoints/fp16/<config>/`. This is a single-seed comparison; full training results are pending.
