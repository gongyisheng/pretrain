# Int4 Weight-Scale Granularity Sweep

Measure how weight-scale granularity affects W4A16 training quality on Qwen3-51M. The nine runs are a bf16 baseline and eight int4-weight layouts; activations and `grad_out` are bf16, and `lm_head` is excluded from quantization. This experiment selects a W4A16 weight-quantization recipe.

## Hypothesis

Finer weight scales should reduce quantization error, but validation quality may plateau before the finest scale granularity. The sweep identifies the stable W4A16 granularity with the best validation quality and the least scale overhead among indistinguishable candidates.

## Setup

| Granularity | `block_shape` | Weight scale covers | Logical bits/weight |
|---|---|---|---|
| blockwise1D | (1, 128) | 128 contiguous input features | 4.25 |
| blockwise1D | (1, 64) | 64 contiguous input features | 4.50 |
| blockwise1D | (1, 32) | 32 contiguous input features | 5.00 |
| blockwise1D | (1, 16) | 16 contiguous input features | 6.00 |
| blockwise2D | (128, 128) | 128 × 128 tile | 4.002 |
| blockwise2D | (64, 64) | 64 × 64 tile | 4.008 |
| blockwise2D | (32, 32) | 32 × 32 tile | 4.031 |
| blockwise2D | (16, 16) | 16 × 16 tile | 4.125 |

| Configs | Weight | Activation | `grad_out` | Parameters |
|---|---|---|---|---|
| `qwen3_51m_bf16` | bf16 | bf16 | bf16 | ~51M |
| `qwen3_51m_int4_w4a16_{blockwise1d,blockwise2d}_{16,32,64,128}` | int4 | bf16 | bf16 | ~51M |

All runs use Qwen3-51M (`d_model=512`, 8 layers, 8/4 Q/KV heads, `intermediate_size=1536`), seq_len=1024, batch_size=16, gradient_accumulation_steps=16, 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1,500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, and `checkpoint_every=5000`.

The table describes the forward weight layout. A 1D layout groups the contracted axis, so dgrad uses a different layout; square 2D tiles transpose consistently. Weights use fp32 scales and round-to-nearest-even, with global scale and rotation disabled. Weights are quantized separately for forward and dgrad, then dequantized for bf16 matmuls; wgrad remains bf16. The logical bits column is `4 + 32 / elements_per_scale`. Int4 codes currently reside in an int8 container, so this does not represent packed storage or the actual training memory footprint.

## Run

```bash
nohup bash experiments/int4_weight_granularity/run.sh > logs/int4_weight_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int4-weight-granularity`. Baseline: `qwen3_51m_bf16`.

| Granularity | `block_shape` | Logical bits/weight | Mean val. loss, final 10 evals | Δ vs bf16 | Val BPB |
|---|---|---|---|---|---|
| bf16 | — | 16 | | 0 | |
| blockwise1D | (1, 128) | 4.25 | | | |
| blockwise1D | (1, 64) | 4.50 | | | |
| blockwise1D | (1, 32) | 5.00 | | | |
| blockwise1D | (1, 16) | 6.00 | | | |
| blockwise2D | (128, 128) | 4.002 | | | |
| blockwise2D | (64, 64) | 4.008 | | | |
| blockwise2D | (32, 32) | 4.031 | | | |
| blockwise2D | (16, 16) | 4.125 | | | |

## Notes

- Select the stable run with the best observed mean validation loss over the final 10 evaluations. When candidates are indistinguishable, prefer lower scale overhead. The selected W4A16 recipe is pending results.
- Use validation BPB plus per-module weight SQNR and underflow-rate logs as diagnostics; higher SQNR indicates lower quantization error.
- Report instability, divergence, or loss spikes with the step at which they occur.
- The comparable W8A16 sweep is [int8_weight_granularity](../int8_weight_granularity/README.md).
