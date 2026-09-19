# Int8 Weight-Scale Granularity Sweep

Sweep the scale granularity for weight-only int8 quantization on Qwen3-51M against a shared bf16 baseline. Activations and `grad_out` remain bf16, so only `scale.weight` is effective; `lm_head` is excluded from quantization. There are 11 runs: 1 bf16 baseline and 10 W8A16 granularities.

## Hypothesis

Weight-only int8 is close to lossless even with one scale per matrix. Smaller scale blocks should reduce weight quantization error, but the validation-loss curve should remain nearly flat while fp32 scale storage grows. The useful result is the coarsest granularity whose loss matches bf16 within evaluation noise.

## Setup

| Granularity | `block_shape` | Weight scale covers | Effective bits/weight |
|---|---|---|---|
| tensorwise | — | whole matrix | 8.00 |
| rowwise | — | one output row | 8.06 |
| blockwise1D | (1, 128) | 128 contiguous inputs | 8.25 |
| blockwise1D | (1, 64) | 64 contiguous inputs | 8.50 |
| blockwise1D | (1, 32) | 32 contiguous inputs | 9.00 |
| blockwise1D | (1, 16) | 16 contiguous inputs | 10.00 |
| blockwise2D | (128, 128) | 128×128 tile | 8.002 |
| blockwise2D | (64, 64) | 64×64 tile | 8.008 |
| blockwise2D | (32, 32) | 32×32 tile | 8.031 |
| blockwise2D | (16, 16) | 16×16 tile | 8.125 |

| Configs | Weight | Activation | `grad_out` | Parameters |
|---|---|---|---|---|
| `qwen3_51m_bf16` | bf16 | bf16 | bf16 | ~51M |
| `qwen3_51m_int8_w8a16_{tensorwise,rowwise}` | int8 | bf16 | bf16 | ~51M |
| `qwen3_51m_int8_w8a16_{blockwise1d,blockwise2d}_{16,32,64,128}` | int8 | bf16 | bf16 | ~51M |

All runs use seq_len=1024, batch_size=16, gradient_accumulation_steps=16, 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1,500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, and `checkpoint_every=5000`.

## Run

```bash
nohup bash experiments/int8_weight_granularity/run.sh > logs/int8_weight_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-weight-granularity`. Baseline: `qwen3_51m_bf16`.

| Granularity | `block_shape` | Effective bits/weight | Val Loss | Δ vs bf16 | Val BPB | Weight rel. err |
|---|---|---|---|---|---|---|
| bf16 | — | 16 | | 0 | | — |
| tensorwise | — | 8.00 | | | | |
| rowwise | — | 8.06 | | | | |
| blockwise1D | (1, 128) | 8.25 | | | | |
| blockwise1D | (1, 64) | 8.50 | | | | |
| blockwise1D | (1, 32) | 9.00 | | | | |
| blockwise1D | (1, 16) | 10.00 | | | | |
| blockwise2D | (128, 128) | 8.002 | | | | |
| blockwise2D | (64, 64) | 8.008 | | | | |
| blockwise2D | (32, 32) | 8.031 | | | | |
| blockwise2D | (16, 16) | 8.125 | | | | |

## Notes

- Compare each int8 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- `weight rel. err` should fall as blocks shrink; compare it with validation loss to identify the point beyond which finer scales do not improve model quality.
- The sweep measures quality, not step time. W8A16 dequantizes weights into bf16 matmuls, so it is simulated quantization rather than an int8 GEMM performance benchmark.
- `experiments/int8_tensor_dtype/` duplicates the tensorwise and rowwise W8A16 int8 runs. The matching results provide a seed-noise check across projects.
