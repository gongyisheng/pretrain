# MXFP8 W8A8 Gradient and Scale Ablation

This experiment uses E4M3 blockwise 1D `(1, 32)` quantization with FP32 scales as the control. MXFP8 replaces those FP32 scales with E8M0 power-of-two scales; gradient quantization is a separate ablation.

## Mental model

Think of MXFP8 here as the accelerated variant of the same blockwise 1D `(1, 32)` FP8 setup: the control stores one FP32 scale per 32 values, while MXFP8 stores that scale as E8M0. This reduces scale metadata from 32 to 8 bits per block and enables native MXFP8 kernels on SM100+ GPUs. On older GPUs, the fallback remains useful for numerical comparison but cannot establish an MXFP8 speedup.

## Hypothesis

At fixed E4M3 values and blockwise 1D `(1, 32)` scaling, replacing FP32 scales with E8M0 scales may improve throughput and reduce scale metadata, but adds power-of-two scale rounding. Quantizing `grad_out` to E4M3 introduces an independent backward-GEMM error source. The full MXFP8 training recipe combines E4M3 W8A8G8 with E8M0 scales.

## Setup

| Config | Weights / acts | `grad_out` | Scale | Block shape | Scale bits/value | Approx. params |
|---|---|---|---|---|---:|---:|
| `qwen3_51m_bf16` | bf16 | bf16 | — | — | — | ~51M |
| `qwen3_51m_fp8_e4m3_w8a8g16_blockwise1d_32` | E4M3 | bf16 | FP32 | (1, 32) | 1.00 | ~51M |
| `qwen3_51m_mxfp8_w8a8g16` | E4M3 | bf16 | E8M0 | (1, 32) | 0.25 | ~51M |
| `qwen3_51m_fp8_e4m3_w8a8g8_blockwise1d_32` | E4M3 | E4M3 | FP32 | (1, 32) | 1.00 | ~51M |
| `qwen3_51m_mxfp8_w8a8g8` | E4M3 | E4M3 | E8M0 | (1, 32) | 0.25 | ~51M |

All runs use OpenWebText, sequence length 1024, batch size 16, gradient accumulation 16 (effective batch 256), 50K steps, seed 42, bf16 mixed precision, Muon with `match_rms_adamw`, lr=5e-4, weight decay=0.1, and a cosine schedule (1500 warmup, min lr=5e-5). They explicitly use `checkpoint_every: 5000`, `eval_every: 100`, and `eval_steps: 100`. The lm head remains bf16; embeddings, norms, attention, residuals, loss, and optimizer state also stay bf16/fp32.

The W8A8G16 controls quantize weights and activations while leaving `grad_out` in bf16. Each backward GEMM therefore contains one bf16 operand and uses the one-sided fake-quantization fallback. Use these cells for numerical comparison, not end-to-end FP8 throughput. W8A8G8 quantizes both operands of all three eligible linear GEMMs.

## Run

The script runs BF16, then nested loops over `{G16, G8}` and `{FP32, E8M0}` scale types.

```bash
nohup bash experiments/mxfp8/run.sh > logs/mxfp8_51m.log 2>&1 &
```

## Results

W&B project: `pretrain-mxfp8`.

Report validation loss as the mean and standard deviation of the final 10 evaluations.

| Model | Recipe | Mean Val Loss | Std. Dev. | Delta vs BF16 | Val BPB | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| 51M | BF16 | TBD | TBD | 0 | TBD | TBD |
| 51M | E4M3 W8A8G16, FP32 scale | TBD | TBD | TBD | TBD | n/a |
| 51M | E4M3 W8A8G16, E8M0 scale | TBD | TBD | TBD | TBD | n/a |
| 51M | E4M3 W8A8G8, FP32 scale | TBD | TBD | TBD | TBD | TBD |
| 51M | MXFP8 W8A8G8, E8M0 scale | TBD | TBD | TBD | TBD | TBD |

## Notes

- Compare E8M0 against FP32 at fixed gradient format: W8A8G16 isolates the scale effect with bf16 gradients; W8A8G8 isolates it with E4M3 gradients.
- Compare G8 against G16 at fixed scale type to measure gradient quantization cost.
- Compare MXFP8 against BF16 for the full recipe's training-quality and throughput trade-off.
- Read per-site SQNR and underflow from `train-quant/sqnr/*` and `train-quant/underflow_rate/*`; check weight, activation, and `grad_out` series rather than validation loss alone.
- Record the GPU, software versions, and selected GEMM backend. Native MXFP8 acceleration requires SM100+; a fallback run is valid for numerical comparison but not for a native-MX speed claim.
