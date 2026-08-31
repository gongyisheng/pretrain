# Int8 Scale-Granularity Sweep

Sweep quantization scale granularity at Qwen3-51M across two arms — **W8A16** (int8 weights, bf16 activations) and **W8A8** (int8 weights and activations) — against a shared bf16 baseline. `grad_out` stays bf16 and `lm_head` is excluded from quantization in every quantized run.

Granularity is the only axis swept within an arm; the arm is the second axis. 21 runs: 1 baseline + 10 granularities × 2 arms.

## Hypothesis

Smaller scale blocks reduce quantization error at the cost of more fp32 scale storage — but the two arms should pay off very differently:

- **W8A16** applies the scale to the *weight* only. Weight distributions per matrix are well-behaved, and weight-only int8 is close to lossless already at `tensorwise`. Expect a flat curve: granularity buys little, and the finer blocks spend bits/weight for nothing.
- **W8A8** additionally applies the scale to the *activation*. Activations are where the outliers live, and a single `tensorwise` scale set by the largest element wastes int8 codes on the bulk of the distribution. Expect a steeper curve: granularity should matter substantially more here, with the biggest gain moving from `tensorwise` to any per-token scheme.

If that holds, the practical reading is that scale granularity is an activation-quantization lever, not a weight-quantization one.

## Setup

**Granularity axis** (identical in both arms). Effective bits/weight counts the fp32 scale storage against the weight tensor only; the W8A8 arm additionally carries activation scales, which are transient and not counted here.

| Granularity | `block_shape` | Weight scale covers | Activation scale covers | Effective bits/weight |
|---|---|---|---|---|
| tensorwise | — | whole matrix | whole activation | 8.00 |
| rowwise | — | one output row | one token | 8.06 |
| blockwise1D | (1, 128) | 128 contiguous inputs | 128 channels of one token | 8.25 |
| blockwise1D | (1, 64) | 64 contiguous inputs | 64 channels of one token | 8.50 |
| blockwise1D | (1, 32) | 32 contiguous inputs | 32 channels of one token | 9.00 |
| blockwise1D | (1, 16) | 16 contiguous inputs | 16 channels of one token | 10.00 |
| blockwise2D | (128, 128) | 128×128 tile | 128 tokens × 128 channels | 8.002 |
| blockwise2D | (64, 64) | 64×64 tile | 64 tokens × 64 channels | 8.008 |
| blockwise2D | (32, 32) | 32×32 tile | 32 tokens × 32 channels | 8.031 |
| blockwise2D | (16, 16) | 16×16 tile | 16 tokens × 16 channels | 8.125 |

**Arm axis.**

| Arm | `dtype.weight` | `dtype.act` | `dtype.grad_out` | Forward GEMM |
|---|---|---|---|---|
| W8A16 | int8 | bf16 | bf16 | dequantize → bf16 matmul (simulated) |
| W8A8 | int8 | int8 | bf16 | fused `gemm.int8_scaled_mm` |

Config names are `qwen3_51m_int8_<arm>_<granularity>`, e.g. `qwen3_51m_int8_w8a8_blockwise2d_64`.

All runs: ~51M params, seq_len=1024, batch=16, grad_accum=16 (effective batch=256), 50K steps, Muon (`match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup steps, min_lr=5e-5, OpenWebText, bf16 mixed precision, seed 42, `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`.

## Run

```bash
nohup bash experiments/int8_granularity/run.sh > logs/int8_granularity.log 2>&1 &
```

## Results

W&B project: `pretrain-int8-granularity`. Baseline: `qwen3_51m_bf16`.

### W8A16 (int8 weights, bf16 activations)

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

### W8A8 (int8 weights and activations)

| Granularity | `block_shape` | Effective bits/weight | Val Loss | Δ vs bf16 | Δ vs W8A16 twin | Val BPB | Weight rel. err |
|---|---|---|---|---|---|---|---|
| tensorwise | — | 8.00 | | | | | |
| rowwise | — | 8.06 | | | | | |
| blockwise1D | (1, 128) | 8.25 | | | | | |
| blockwise1D | (1, 64) | 8.50 | | | | | |
| blockwise1D | (1, 32) | 9.00 | | | | | |
| blockwise1D | (1, 16) | 10.00 | | | | | |
| blockwise2D | (128, 128) | 8.002 | | | | | |
| blockwise2D | (64, 64) | 8.008 | | | | | |
| blockwise2D | (32, 32) | 8.031 | | | | | |
| blockwise2D | (16, 16) | 8.125 | | | | | |

`Δ vs W8A16 twin` isolates the cost of quantizing activations at that granularity — it is the quantity the hypothesis predicts will shrink as blocks get finer.

## Notes

- Compare each int8 run with this experiment's bf16 baseline using the mean validation loss over the final 10 evaluations.
- `weight rel. err` should fall as blocks shrink; compare it with validation loss to isolate quantization error. It is an arm-independent weight property, so it should match between each W8A16/W8A8 pair.
- **Do not compare step time across arms.** Only W8A8 reaches a fused kernel (`gemm.int8_scaled_mm`); W8A16 has no int8×bf16 GEMM and falls back to dequantizing the weight into a bf16 matmul, so it is simulated quantization and is slower than both its arm-twin and bf16. Within the W8A8 arm, step time across granularities is a fair comparison.
- Four cells here are duplicated by `experiments/int8_tensor_dtype/` (its `{rowwise,tensorwise}/qwen3_51m_int8_{w8a16,w8a8}_*` configs are the same runs at int8). They are rerun here so this experiment's W&B project holds the complete granularity curve without cross-project joins; the two projects should agree on those four points, which makes them a useful seed-noise check.
