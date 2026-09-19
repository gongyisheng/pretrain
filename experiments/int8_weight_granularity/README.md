# Int8 Weight-Scale Granularity Sweep

Measure how weight-scale granularity affects W8A16 training quality on Qwen3-51M. The 11 runs are a bf16 baseline and 10 int8-weight layouts; activations and `grad_out` are bf16, and `lm_head` is excluded from quantization. This experiment selects a W8A16 weight-quantization recipe.

## Hypothesis

Finer weight scales should reduce quantization error, but validation quality may plateau before the finest scale granularity. The sweep identifies the stable W8A16 granularity with the best validation quality and the least scale overhead among indistinguishable candidates.

## Setup

| Granularity | `block_shape` | Weight scale covers | Logical bits/weight |
|---|---|---|---|
| tensorwise | — | whole matrix | ~8 |
| rowwise | — | one output row | `8 + 32 / input_width` (8.021–8.063) |
| blockwise1D | (1, 128) | 128 contiguous input features | 8.25 |
| blockwise1D | (1, 64) | 64 contiguous input features | 8.50 |
| blockwise1D | (1, 32) | 32 contiguous input features | 9.00 |
| blockwise1D | (1, 16) | 16 contiguous input features | 10.00 |
| blockwise2D | (128, 128) | 128 × 128 tile | 8.002 |
| blockwise2D | (64, 64) | 64 × 64 tile | 8.008 |
| blockwise2D | (32, 32) | 32 × 32 tile | 8.031 |
| blockwise2D | (16, 16) | 16 × 16 tile | 8.125 |

| Configs | Weight | Activation | `grad_out` | Parameters |
|---|---|---|---|---|
| `qwen3_51m_bf16` | bf16 | bf16 | bf16 | ~51M |
| `qwen3_51m_int8_w8a16_{tensorwise,rowwise}` | int8 | bf16 | bf16 | ~51M |
| `qwen3_51m_int8_w8a16_{blockwise1d,blockwise2d}_{16,32,64,128}` | int8 | bf16 | bf16 | ~51M |

All runs use Qwen3-51M (`d_model=512`, 8 layers, 8/4 Q/KV heads, `intermediate_size=1536`), seq_len=1024, batch_size=16, gradient_accumulation_steps=16, 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1,500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, and `checkpoint_every=5000`.

The table describes the forward weight layout. Rowwise and 1D layouts group the contracted axis, so dgrad uses a different layout; square 2D tiles transpose consistently. Weights use fp32 scales and round-to-nearest-even, with global scale and rotation disabled. Weights are quantized separately for forward and dgrad, then dequantized for bf16 matmuls; wgrad remains bf16. The logical bits column is `8 + 32 / elements_per_scale`; tensorwise is approximately 8 bits/weight. It is scale accounting, not measured training memory or throughput.

## Run

```bash
nohup bash experiments/int8_weight_granularity/run.sh > logs/int8_weight_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-weight-granularity`. Baseline: `qwen3_51m_bf16`.

| Granularity | `block_shape` | Logical bits/weight | Mean val. loss, final 10 evals | Δ vs bf16 | Val BPB |
|---|---|---|---|---|---|
| bf16 | — | 16 | | 0 | |
| tensorwise | — | ~8 | | | |
| rowwise | — | 8.021–8.063 | | | |
| blockwise1D | (1, 128) | 8.25 | | | |
| blockwise1D | (1, 64) | 8.50 | | | |
| blockwise1D | (1, 32) | 9.00 | | | |
| blockwise1D | (1, 16) | 10.00 | | | |
| blockwise2D | (128, 128) | 8.002 | | | |
| blockwise2D | (64, 64) | 8.008 | | | |
| blockwise2D | (32, 32) | 8.031 | | | |
| blockwise2D | (16, 16) | 8.125 | | | |

## Notes

- Select the stable run with the best observed mean validation loss over the final 10 evaluations. When candidates are indistinguishable, prefer lower scale overhead. The selected W8A16 recipe is pending results.
- Use validation BPB plus per-module weight SQNR and underflow-rate logs as diagnostics; higher SQNR indicates lower quantization error.
- Report instability, divergence, or loss spikes with the step at which they occur.
- The comparable W4A16 sweep is [int4_weight_granularity](../int4_weight_granularity/README.md).
