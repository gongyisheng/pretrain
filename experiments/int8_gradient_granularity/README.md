# INT8 Gradient Granularity

Measure how INT8 output-gradient scale granularity affects W8A8G8 pretraining. Weight and activation quantization remain fixed, so the three W8A8G8 runs differ only in `training.quantization.scale.grad_out`.

## Hypothesis

Smaller gradient scale groups should limit outlier-driven error. We expect 1D blocks to be more stable than 2D blocks. Rowwise is a coarser comparison point, with no strict expected ordering against 2D blocks.

## Setup

| Config | Weight | Activation | Output gradient |
|---|---|---|---|
| `qwen3_51m_bf16` | BF16 | BF16 | BF16 |
| `qwen3_51m_int8_w8a8` | INT8 (32, 32) | INT8 (1, 32) | BF16 |
| `qwen3_51m_int8_w8a8g8_grad_rowwise` | INT8 (32, 32) | INT8 (1, 32) | INT8 rowwise |
| `qwen3_51m_int8_w8a8g8_grad_blockwise1d_32` | INT8 (32, 32) | INT8 (1, 32) | INT8 (1, 32) |
| `qwen3_51m_int8_w8a8g8_grad_blockwise2d_32` | INT8 (32, 32) | INT8 (1, 32) | INT8 (32, 32) |

| Shared parameter | Value |
|---|---|
| Model | Qwen3-style dense Transformer, approximately 50,931,200 parameters (51M) |
| Data | OpenWebText with `tokenizers/custom_bpe_50k`; validation split 0.01 |
| Training | sequence length 1,024; batch 128; accumulation 2; 262,144 tokens per step; 50,000 steps / 13.1072B tokens per run |
| Optimizer | Muon, momentum 0.95, Nesterov, `match_rms_adamw`, weight decay 0.1 |
| Schedule | 5e-4 cosine to 5e-5, 1,500 warmup steps |
| Precision | BF16 mixed precision; INT8 scales FP32; RNE; `lm_head` excluded |
| Evaluation / checkpoints | 100 batches every 100 steps; checkpoint every 5,000 steps |
| Seed / logging | 42; W&B project `pretrain-int8-gradient-granularity` |

## Run

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0  # Replace 0 with a free device from nvidia-smi.
mkdir -p logs
nohup bash experiments/int8_gradient_granularity/run.sh > logs/int8_gradient_granularity.log 2>&1 &
```

Prepare OpenWebText and `tokenizers/custom_bpe_50k` as described in the repository README. Select a free device from `nvidia-smi`. Each configuration writes to its own `checkpoints/int8_gradient_granularity/<config>/` directory. Batch 128 × accumulation 2 matches the source experiment; keep this experiment's separate local baselines.

## Results

Compare validation loss and BPB at equal training tokens. Compute the mean over the final ten evaluations, steps 49,100–50,000. Results are pending.

| Config | Mean final-10 val loss | Delta vs BF16 | Delta vs W8A8 | Status |
|---|---:|---:|---:|---|
| `bf16` | — | 0 | — | Pending |
| `int8_w8a8` | — | — | 0 | Pending |
| `int8_w8a8g8_grad_rowwise` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise1d_32` | — | — | — | Pending |
| `int8_w8a8g8_grad_blockwise2d_32` | — | — | — | Pending |

## Notes

- Validation runs in BF16 because `QuantizedLinear.eval()` bypasses quantization.
- Output gradients are the left operand in dgrad (`g @ W`) and wgrad (`g.T @ X`). In dgrad, 1D `(1, 32)` groups 32 output channels for one token and 2D `(32, 32)` groups 32 tokens × 32 output channels. In wgrad, 1D groups 32 tokens for one output channel and 2D groups 32 output channels × 32 tokens.
- Rowwise output gradients group all output channels for each token in dgrad and all flattened tokens for each output channel in wgrad.
- Rowwise output gradients use the BF16 backward fallback. The width-32 blockwise runs use matching-width INT8 kernels where available.
