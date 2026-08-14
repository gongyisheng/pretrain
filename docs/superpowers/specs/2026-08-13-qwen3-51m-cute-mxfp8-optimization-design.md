# Qwen3-51M CuTe MXFP8 optimization design

## Goal

Narrow the SM120 CuTe MXFP8 dense-GEMM performance gap to
`torch.nn.functional.scaled_mm` for Qwen3-51M training. The target covers the
forward, activation-gradient, and weight-gradient GEMMs rather than forward only.

The primary success target is at least 85% of cuBLASLt throughput for the
model-weighted Qwen3-51M GEMM set, with no dominant MLP GEMM below 75%. Numerical
behavior must remain within the existing test tolerances. Parity remains the stretch
goal.

This work also consolidates scaled-GEMM coverage into
`benchmarks/gemm/bench_scaled_gemm.py` and deletes
`benchmarks/gemm/bench_cute_gemm.py`. The public benchmark is a dtype/scaling
matrix rather than a Qwen3-specific training mode.

## Evidence and root cause

On one RTX 5060 Ti with torch 2.11.0+cu130 and CUTLASS DSL 4.7.0, the current
Qwen3-51M MLP training shapes measure as follows:

| GEMM | M | K | N | CuTe / cuBLASLt performance |
|---|---:|---:|---:|---:|
| gate/up forward | 16384 | 512 | 3072 | 71% |
| gate/up dgrad | 16384 | 3072 | 512 | 48% |
| gate/up wgrad | 3072 | 16384 | 512 | 55% |
| down forward | 16384 | 1536 | 512 | 48% |
| down dgrad | 16384 | 512 | 1536 | 55% |
| down wgrad | 512 | 16384 | 1536 | 60% |

PyTorch profiling identifies the cuBLASLt kernel as:

```text
cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_ue8m0xe4m3_
ue8m0xe4m3_f32_bf16_bf16_128x128x128_1x1x1_0_tnn_align16_...
```

The CuTe kernel already matches its tensor-core instruction family (`s16x8x32`
MXFP8 block-scaled MMA), CTA tile (`128x128x128`), TNN operand orientation,
align-16 inputs, FP32 accumulation, and BF16 output. CuTe cannot launch the same
binary as cuBLASLt because it compiles a separate custom kernel. The enforceable
equivalence is therefore the same MMA, tile, layouts, alignment, accumulation, and
mainloop schedule family.

The remaining gap is the implementation schedule. The current CuTe mainloop uses
all-warp `cp.async` staging. CUTLASS exposes SM120 MXF8 through TMA warp-specialized
cooperative and ping-pong schedules. The existing project spec's statement that SM120
has no TMA is superseded by CUTLASS 4.7 and by the current CUTLASS dispatch policies.

The CuTe wrapper also launches two FP32-to-E8M0 scale conversions when passed the
native quantizer output. Those launches are API overhead, not the main kernel gap;
kernel-only benchmarks must pre-convert scales so they do not contaminate schedule
measurements.

## Qwen3-51M profiling workload

The default microbatch contains `16 * 1024 = 16384` tokens. Each of eight layers has:

| Projection | Count/layer | Input K | Output N |
|---|---:|---:|---:|
| query and output | 2 | 512 | 512 |
| key and value | 2 | 512 | 256 |
| fused gate/up | 1 | 512 | 3072 |
| down | 1 | 1536 | 512 |

For each `(tokens, K_in, N_out)` linear operation, the private profiling set includes:

- forward: `(tokens, K_in, N_out)`;
- dgrad: `(tokens, N_out, K_in)`;
- wgrad: `(N_out, tokens, K_in)`.

These shapes remain the private profiling and optimization workload used to choose the
CuTe schedule. They are recorded in the performance note and are not exposed as a
Qwen3-specific mode in the public benchmark script.

## Benchmark consolidation

`benchmarks/gemm/bench_scaled_gemm.py` becomes the only scaled-GEMM benchmark.

- Organize every result row by `dtype`, `granularity`, and optional `block_size`.
- Sweep both FP8 and INT8 through tensorwise, rowwise, blockwise-1D, and
  blockwise-2D scaling.
- Sweep block sizes `16`, `32`, `64`, and `128` for both blockwise granularities.
- Treat FP8 blockwise-1D with block size 32 as MXFP8. This row compares CuTe,
  Triton, and native `torch.nn.functional.scaled_mm`, and prominently reports the
  CuTe/native ratio.
- Add `cute` beside `triton` and `native`; CuTe is `n/a` outside the MXFP8 row, and
  native is `n/a` for combinations PyTorch does not support.
- Hoist quantization, E8M0 conversion, B-layout preparation, and native scale
  swizzling out of every kernel-only timed closure.
- Report latency, TFLOP/s, relative error, and supported implementation ratios in
  one table, one plot, and optional generic JSON output.
- Remove Qwen3 workload metadata, weighted aggregation, the `--qwen3-training`
  switch, and Qwen-specific threshold enforcement from the public script.
- Keep the representative linear shapes and token counts as ordinary GEMM shapes;
  they are not labeled or weighted as a Qwen training workload.
- Fold the old separate blockwise-2D appendix into the unified result matrix.
- Keep a CuTe schedule selector so the MXFP8 row can compare the baseline and
  cooperative schedules.
- Delete `bench_cute_gemm.py` and keep CPU-safe import/metadata tests for the
  consolidated script.

## Kernel equivalence and profiling

Before changing the mainloop, capture and retain evidence for both paths:

1. cuBLASLt kernel name and launch configuration from PyTorch/Nsight;
2. CuTe PTX/SASS containing the matching
   `mma.sync.aligned.m16n8k32...kind::mxf8...block_scale` instruction;
3. registers/thread, shared memory/CTA, achieved occupancy, tensor-pipe utilization,
   memory throughput, and dominant stall reasons for representative forward, dgrad,
   and wgrad shapes.

These checks prove configuration equivalence and identify whether each experimental
schedule changes the intended bottleneck. Kernel-name equality is not asserted because
the two backends produce different binaries.

## Mainloop optimization

The recommended implementation replaces the current all-warp `cp.async` mainloop with
an SM120 TMA warp-specialized design modeled on CUTLASS's MXF8 schedules.

1. Preserve the current correct scale-factor mapping, K-major operand layout, shared
   memory swizzle, edge predication, K-tail zeroing, FP32 accumulation, and vectorized
   interior epilogue.
2. Introduce dedicated producer and consumer roles. Producers issue asynchronous TMA
   transfers; consumers load fragments and issue the existing block-scaled MMA.
3. Implement the cooperative schedule first. Measure it against the frozen baseline.
4. Implement ping-pong only as a separate experiment if cooperative leaves measurable
   latency or scoreboard stalls. Change one scheduling variable at a time.
5. Retain `128x128x128` as the first configuration because it matches cuBLASLt. Add
   shape dispatch only when profiler evidence shows one schedule cannot serve both
   large-token GEMMs and wgrad's smaller M dimension.
6. Keep static K/N and dynamic M in the compile cache. Any extra schedule identifier
   becomes part of the cache key.

If CuTe DSL 4.7 cannot express the required SM120 TMA block-scaled schedule safely,
stop after documenting the limitation and benchmark the best controlled `cp.async`
variant. Dispatching to `torch._scaled_mm` is not an acceptable substitute for the
custom-kernel goal.

## Validation

Each isolated schedule change must pass:

- `tests/fast/kernel/cute/test_gemm.py`, especially the adversarial scale-placement,
  ragged-shape, partial-scale-group, dtype, and `F.scaled_mm` cross-checks;
- the consolidated dtype/scaling matrix, with special attention to every FP8
  blockwise-1D block-32 MXFP8 row;
- Ruff lint and format checks;
- the full fast kernel tree after the winning schedule is selected.

For performance-sensitive changes, record before/after kernel metrics and run
`benchmarks/bench_train.py` with the Qwen3-51M MXFP8 config after the standalone GEMM
target is met. A schedule is not accepted if it improves the aggregate by regressing a
dominant MLP shape below the 75% floor or changing numerical results.

## Out of scope

- Qwen3-188M-A51M and grouped routed-expert GEMMs;
- non-MXFP8 scaling schemes;
- replacing the custom kernel with cuBLASLt;
- unrelated quantizer or model refactors;
- CUTLASS C++ integration unless CuTe DSL is proven unable to express the required
  schedule and a separate design is approved.
