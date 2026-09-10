# MXFP8 Scale Dtype Loss Ablation

This experiment measures how much validation loss changes when switching the scale dtype from `fp32` to `fp8_e8m0` for MXFP8 training. We compare matched FP32/E8M0 pairs for W8A16, W8A8, and W8A8G8, keeping E4M3 quantization, block shape `(1, 32)`, and all other training settings fixed within each pair.

## Mental model

Think of MXFP8 here as the accelerated variant of the same blockwise 1D `(1, 32)` FP8 setup: the control stores one FP32 scale per 32 values, while MXFP8 stores that scale as E8M0. This reduces scale metadata from 32 to 8 bits per block and enables native MXFP8 kernels on SM100+ GPUs. On older GPUs, the fallback remains useful for numerical comparison but cannot establish an MXFP8 speedup.

## Hypothesis

E8M0 power-of-two scale rounding may increase validation loss relative to FP32 scales. The paired runs measure this loss penalty for weight-only quantization (W8A16), weight and activation quantization (W8A8), and the full training recipe including output-gradient quantization (W8A8G8).

## Setup

| Config | Weights / acts | `grad_out` | Scale | Block shape | Scale bits/value | Approx. params |
|---|---|---|---|---|---:|---:|
| `qwen3_51m_bf16` | bf16 | bf16 | — | — | — | ~51M |
| `qwen3_51m_fp8_w8a16_scale_fp32` | E4M3 / bf16 | bf16 | FP32 | (1, 32) | 1.00 | ~51M |
| `qwen3_51m_fp8_w8a16_scale_e8m0` | E4M3 / bf16 | bf16 | E8M0 | (1, 32) | 0.25 | ~51M |
| `qwen3_51m_fp8_w8a8_scale_fp32` | E4M3 | bf16 | FP32 | (1, 32) | 1.00 | ~51M |
| `qwen3_51m_fp8_w8a8_scale_e8m0` | E4M3 | bf16 | E8M0 | (1, 32) | 0.25 | ~51M |
| `qwen3_51m_fp8_w8a8g8_scale_fp32` | E4M3 | E4M3 | FP32 | (1, 32) | 1.00 | ~51M |
| `qwen3_51m_fp8_w8a8g8_scale_e8m0` | E4M3 | E4M3 | E8M0 | (1, 32) | 0.25 | ~51M |

All runs use OpenWebText, sequence length 1024, batch size 16, gradient accumulation 16 (effective batch 256), 50K steps, seed 42, bf16 mixed precision, Muon with `match_rms_adamw`, lr=5e-4, weight decay=0.1, and a cosine schedule (1500 warmup, min lr=5e-5). They explicitly use `checkpoint_every: 5000`, `eval_every: 100`, and `eval_steps: 100`. The lm head remains bf16; embeddings, norms, attention, residuals, loss, and optimizer state also stay bf16/fp32.

W&B names are `qwen3-51m-bf16` and `qwen3-51m-fp8-{w8a16,w8a8,w8a8g8}-scale-{fp32,e8m0}`; checkpoints use `checkpoints/mxfp8_scale_dtype/<run_name_with_underscores>/`.

W8A16 quantizes weights only: forward and input-gradient GEMMs use the one-sided fake-quantization fallback, while the weight-gradient GEMM stays bf16. W8A8 supports two-FP8 GEMMs in forward, but its backward GEMMs still contain one bf16 operand and use the fallback. W8A8G8 quantizes both operands of all three eligible linear GEMMs. Native MXFP8 execution requires a supported GPU and backend.

## Run

The script runs BF16, then nested loops over `{W8A16, W8A8, W8A8G8}` and `{FP32, E8M0}` scale types.

```bash
nohup bash experiments/mxfp8_scale_dtype/run.sh > logs/mxfp8_scale_dtype_51m.log 2>&1 &
```

## Results

W&B project: `pretrain-mxfp8-scale-dtype`.

Report validation loss as the mean and standard deviation of the final 10 evaluations. For each recipe, the primary comparison is mean validation loss with E8M0 scales minus that with FP32 scales; a positive difference indicates a loss penalty from switching scale dtype.

| Model | Recipe | Mean Val Loss | Std. Dev. | Delta vs BF16 | Val BPB | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| 51M | BF16 | TBD | TBD | 0 | TBD | TBD |
| 51M | E4M3 W8A16, FP32 scale | TBD | TBD | TBD | TBD | n/a |
| 51M | E4M3 W8A16, E8M0 scale | TBD | TBD | TBD | TBD | n/a |
| 51M | E4M3 W8A8, FP32 scale | TBD | TBD | TBD | TBD | TBD |
| 51M | E4M3 W8A8, E8M0 scale | TBD | TBD | TBD | TBD | TBD |
| 51M | E4M3 W8A8G8, FP32 scale | TBD | TBD | TBD | TBD | TBD |
| 51M | MXFP8 W8A8G8, E8M0 scale | TBD | TBD | TBD | TBD | TBD |

## Notes

- Compare E8M0 against FP32 at fixed recipe to isolate the scale effect.
- Compare W8A16 against W8A8 at fixed scale type to measure activation quantization cost, then W8A8 against W8A8G8 to measure gradient quantization cost.
- Compare MXFP8 against BF16 for the full recipe's training-quality and throughput trade-off.
- Read per-site SQNR and underflow from `train-quant/sqnr/*` and `train-quant/underflow_rate/*`; check weight, activation, and `grad_out` series rather than validation loss alone.
- Record the GPU, software versions, and selected GEMM backend. Native MXFP8 acceleration requires SM100+; a fallback run is valid for numerical comparison but not for a native-MX speed claim.
