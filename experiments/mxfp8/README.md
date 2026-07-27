# mxfp8 (Blockwise) — All Three GEMMs

> **Note (2026-07-26):** This run predates the unified Triton scaled-GEMM kernel.
> mxfp8 now routes through `src/kernel/gemm.py:scaled_gemm` (blockwise-32 +
> power-of-two fp32 scale) rather than the native `F.scaled_mm` BlockWise1x32 MX
> path, and is no longer Blackwell-only. See
> `docs/superpowers/specs/2026-07-26-blockwise-scaled-gemm-design.md`. The setup
> below describes the original native-MX run and is kept as a historical record.

Compare end-task loss and throughput of **mxfp8** (OCP Microscaling FP8) against a bf16 baseline at Qwen3-51M, using the in-house `src/quant/` framework (`torch.nn.functional.scaled_mm`, `ScalingType.BlockWise1x32` + `SwizzleType.SWIZZLE_32_4_4`). All three training GEMMs — forward, dgrad, and wgrad — run in mxfp8. The bf16 run anchors the loss/throughput reference.

## Hypothesis

mxfp8 (block-32 along the contraction axis, power-of-two E8M0 scale, e4m3 elements, hardware-accelerated on Blackwell SM 12.0) should:
1. Match bf16 final loss within noise — MX's per-32-element rescaling keeps each block near the e4m3 sweet spot, so the quantization noise on the Q/K/V/O and FFN GEMMs is bounded; high-dynamic-range ops (softmax, RMSNorm, residual stream, cross-entropy) stay in bf16.
2. Deliver measurable throughput gain (tokens/sec) over bf16, since **all three** GEMMs use the Blackwell MX tensor-core path (not just the forward).

### Why e4m3 on every operand (incl. gradients)

Plain fp8 uses e5m2 for gradients to buy exponent range. mxfp8 does **not**: the BlockWise1x32 MMA on this stack accepts only `e4m3 × e4m3` (mixed e5m2 is rejected), and MX's block scaling supplies the dynamic range that e5m2 would otherwise provide. So `dtype: {recipe: mxfp8}` sets weight/act/input_grad/weight_grad all to `fp8_e4m3` — this is what keeps dgrad and wgrad on the tensor-core path instead of falling back to a slow high-precision matmul.

## Setup

All hyperparameters are matched between bf16 and mxfp8. The only independent variable is the `training.quant` block. Same seed (default 42), same data order, same LR schedule.

| Config | d_model | layers | heads/kv | inter_size | quant | scaling | Approx params |
|---|---|---|---|---|---|---|---|
| qwen3_51m_bf16 | 512 | 8 | 8/4 | 1536 | off | — | ~51M |
| qwen3_51m_mxfp8 | 512 | 8 | 8/4 | 1536 | mxfp8 | blockwise-32 / E8M0 | ~51M |

`exclude: [lm_head]` (lm_head stays in bf16; numerically sensitive under tied embeddings). mxfp8 uses `dtype: {recipe: mxfp8}` → weight/act/input_grad/weight_grad all `fp8_e4m3`, scaling `{granularity: blockwise, block_size: 32, scale_dtype: e8m0}`.

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K tok/step), 50K steps (~13B tokens), Muon optimizer (`muon_adjust_lr_fn: match_rms_adamw`, momentum=0.95, nesterov), lr=5e-4, cosine schedule with 1500 warmup, min_lr=5e-5, OpenWebText. `eval_steps: 100` matches the `fp8_granularity` experiment so the two are directly comparable.

## What runs in mxfp8

For the mxfp8 run, the converter (`src/quant/convert.py:apply_quantization`) swaps eligible `nn.Linear` modules to `QuantLinear`; each of the three per-layer GEMMs (`_gemm` in `src/quant/linear.py`) dispatches to `mxfp8_gemm`, which quantizes its operands along the contraction axis (block-32) on the fly (dynamic scaling). Everything else (RoPE, RMSNorm, qk_norm, flash/flex attention, SwiGLU, residuals, embeddings, lm_head, cross-entropy, optimizer state) stays in bf16 / fp32. hp master weights are preserved.

The `quant` block:
```yaml
training:
  mixed_precision: bf16
  quant:
    enabled: true
    dtype: {recipe: mxfp8}     # e4m3 all operands + mx scaling (block-32, E8M0)
    exclude: [lm_head]
```

Hardware requirement: **SM 10.0+ (Blackwell)** — the MX MMA consumes E8M0 scales, so unlike plain fp8 (SM 8.9+) this needs Blackwell. On the dev box (RTX PRO 6000, SM 12.0) the runs use Blackwell's MX path. `apply_quantization` raises on non-Blackwell hardware.

## Run

The script runs the bf16 baseline followed by the mxfp8 run.

```bash
nohup bash experiments/mxfp8/run.sh > logs/mxfp8_51m.log 2>&1 &
```

## Results

| Model | Precision | Scaling | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|
| 51M | bf16 | — | | | | 1.00× |
| 51M | mxfp8 | blockwise-32 / E8M0 | | | | |

## Notes

- Elements are `fp8_e4m3` on every operand; the E8M0 (`torch.float8_e8m0fnu`) scale is a power of two shared by each block of 32 contiguous elements along the contraction axis.
- The wgrad GEMM contracts over the token axis, which is not always a multiple of 32; `mxfp8_gemm` zero-pads the contraction dim up to a multiple of 32 (padding contributes nothing to the accumulation) so the token count need not be aligned.
- Compare against `experiments/fp8_granularity/` (same model, same schedule): mxfp8 quantizes all three GEMMs where the fp8 rowwise recipes keep some ops higher-precision, so this is the more aggressive quantization — watch whether the per-block scaling recovers the loss the tensorwise fp8 run loses.
- Throughput gain at 51M is modest — the model is small enough that non-GEMM kernels are a meaningful fraction of step time; the all-three-GEMM coverage is where mxfp8 should edge out forward-only quantization.
