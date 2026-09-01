# Int4 Scale-Granularity Sweep

Sweep blockwise quantization scale granularity at Qwen3-51M with **int4 weights**, across two arms — **W4A16** (int4 weights, bf16 activations) and **W4A4** (int4 weights and activations) — against a shared bf16 baseline. `grad_out` stays bf16 and `lm_head` is excluded from quantization in every quantized run.

Only blockwise layouts are swept: 1D `(1, N)` and square 2D `(N, N)` at extents 16, 32, and 64. (`block_shape` must be 1D or a square tile with a contract extent that is a positive multiple of 16.) `experiments/int8_granularity/` also runs extent 128; it is dropped here because 128 elements per scale leaves int4's 16 codes far too little resolution to be informative. 13 runs: 1 baseline + 6 granularities × 2 arms. The `tensorwise` and `rowwise` anchors for int4 already exist in `experiments/int8_tensor_dtype/` (`qwen3_51m_int4_w8a{16,8}_{tensorwise,rowwise}`) and are not repeated here.

## Hypothesis

int4 has 16 codes, so a scale block that spans a wide dynamic range wastes most of them. Granularity should therefore matter far more at int4 than at int8, where `tensorwise` weight quantization is already near-lossless:

- **W4A16** applies the scale to the *weight* only. Unlike int8, weight-only int4 should be visibly lossy at coarse granularity, so the curve should be steep rather than flat: expect real gains all the way down to `(1, 16)`, where 16 contiguous inputs share one scale.
- **W4A4** additionally quantizes activations to int4. Activations carry the outliers, and 16 codes cannot absorb them — expect this arm to be much worse than its W4A16 twin at every granularity, and possibly unstable at coarse blocks.
- **1D versus 2D at equal extent** should separate here. A `(1, N)` block re-partitions the weight when it is transposed between forward and dgrad; a square `(N, N)` tile transposes consistently but covers `N×` more elements per scale. At int4 the number-of-elements-per-scale term should dominate, so `(1, N)` should beat `(N, N)` at matched `N`, with the gap widening as `N` grows.

If that holds, int4 training is a scale-granularity problem first and a rounding problem second.

## Setup

**Granularity axis** (identical in both arms). Effective bits/weight counts fp32 scale storage against the weight tensor only; the W4A4 arm additionally carries activation scales, which are transient and not counted here. int4 sets a 4-bit code range (`qmax=7`) but the tensor container is int8, so this column is a logical accounting, not measured memory.

| Granularity | `block_shape` | Weight scale covers | Activation scale covers | Effective bits/weight |
|---|---|---|---|---|
| blockwise1D | (1, 64) | 64 contiguous inputs | 64 channels of one token | 4.50 |
| blockwise1D | (1, 32) | 32 contiguous inputs | 32 channels of one token | 5.00 |
| blockwise1D | (1, 16) | 16 contiguous inputs | 16 channels of one token | 6.00 |
| blockwise2D | (64, 64) | 64×64 tile | 64 tokens × 64 channels | 4.008 |
| blockwise2D | (32, 32) | 32×32 tile | 32 tokens × 32 channels | 4.031 |
| blockwise2D | (16, 16) | 16×16 tile | 16 tokens × 16 channels | 4.125 |

**Arm axis.**

| Arm | `dtype.weight` | `dtype.act` | `dtype.grad_out` | Forward GEMM |
|---|---|---|---|---|
| W4A16 | int4 | bf16 | bf16 | dequantize → bf16 matmul (simulated) |
| W4A4 | int4 | int4 | bf16 | fused `gemm.int8_scaled_mm` (int8 container) |

Config names are `qwen3_51m_int4_<arm>_<granularity>`, e.g. `qwen3_51m_int4_w4a4_blockwise2d_64`.

All runs: ~51M params (`d_model=512`, 8 layers, 8/4 Q/KV heads, `intermediate_size=1536`), seq_len=1024, batch=16, grad_accum=16 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

## Run

```bash
nohup bash experiments/int4_granularity/run.sh > logs/int4_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int4-granularity`. Baseline: `qwen3_51m_bf16`.

### W4A16 (int4 weights, bf16 activations)

| Granularity | `block_shape` | Effective bits/weight | Val Loss | Δ vs bf16 | Val BPB | Weight rel. err |
|---|---|---|---|---|---|---|
| bf16 | — | 16 | | 0 | | — |
| blockwise1D | (1, 64) | 4.50 | | | | |
| blockwise1D | (1, 32) | 5.00 | | | | |
| blockwise1D | (1, 16) | 6.00 | | | | |
| blockwise2D | (64, 64) | 4.008 | | | | |
| blockwise2D | (32, 32) | 4.031 | | | | |
| blockwise2D | (16, 16) | 4.125 | | | | |

### W4A4 (int4 weights and activations)

| Granularity | `block_shape` | Effective bits/weight | Val Loss | Δ vs bf16 | Δ vs W4A16 twin | Val BPB | Weight rel. err |
|---|---|---|---|---|---|---|---|
| blockwise1D | (1, 64) | 4.50 | | | | | |
| blockwise1D | (1, 32) | 5.00 | | | | | |
| blockwise1D | (1, 16) | 6.00 | | | | | |
| blockwise2D | (64, 64) | 4.008 | | | | | |
| blockwise2D | (32, 32) | 4.031 | | | | | |
| blockwise2D | (16, 16) | 4.125 | | | | | |

`Δ vs W4A16 twin` isolates the cost of quantizing activations to int4 at that granularity.

### 1D vs 2D at matched extent

| Extent | W4A16 (1, N) | W4A16 (N, N) | Δ | W4A4 (1, N) | W4A4 (N, N) | Δ |
|---|---|---|---|---|---|---|
| 16 | | | | | | |
| 32 | | | | | | |
| 64 | | | | | | |

## Notes

- Compare each int4 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- `weight rel. err` should fall as blocks shrink; compare it with validation loss to isolate quantization error. It is an arm-independent weight property, so it should match between each W4A16/W4A4 pair.
- Record instability, divergence, and loss spikes, not just final-window loss. Coarse-block int4 — especially W4A4 at `(64, 64)` — is the most likely cell in the quant experiments to diverge outright; note the step if it does instead of reporting only a final number.
- **Do not compare step time across arms.** Only W4A4 reaches a fused kernel; W4A16 has no int4×bf16 GEMM and falls back to dequantizing the weight into a bf16 matmul, so it is simulated quantization and is slower than both its arm-twin and bf16. Within the W4A4 arm, step time across granularities is a fair comparison.
- int4 shares the int8 GEMM path (`gemm.int8_scaled_mm`) because int4 values are stored in an int8 container, so W4A4 buys no memory or bandwidth over int8 here. This experiment measures the *accuracy* cost of a 4-bit code range against scale granularity, not packed-int4 throughput.
- The comparable int8 curve, including its `tensorwise`, `rowwise`, and extent-128 anchors, is `experiments/int8_granularity/`; the same axes are run there at int8 with identical hyperparameters, so the two projects can be overlaid on the shared cells directly.
