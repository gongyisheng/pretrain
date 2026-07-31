# FP8 Element-Format (dtype) Sweep

Compare end-task loss and throughput across FP8 element-format recipes against a
bf16 baseline at Qwen3-51M, holding scaling granularity fixed at **tensorwise**,
using the in-house `src/quant/` framework. The independent variable is
`training.quant.dtype`; the bf16 run anchors the loss/throughput reference.

Complements `experiments/fp8_granularity/` (granularity axis at fixed std-fp8
dtype). Design: `docs/superpowers/specs/2026-07-27-fp8-dtype-experiment-design.md`.

## Hypothesis

`QuantLinear` runs three GEMMs — forward (`act`×`weight`), dgrad
(`input_grad`×`weight`), wgrad (`weight_grad`×`act`). Classic intuition: forward
operands want e4m3 (mantissa/precision), gradients want e5m2 (exponent range).
At the coarsest scale (tensorwise), the element format matters most, since
nothing but the format supplies dynamic range.

1. **std** (e4m3 fwd / e5m2 grad) matches bf16 within noise.
2. **all-e4m3** loses loss here (unlike under mxfp8) — tensorwise scaling can't
   supply the grad range, so e4m3 clips grad outliers.
3. **all-e5m2** is the floor — the forward loses mantissa it can't afford.
4. **std_gwhp** (std with weight_grad forced to bf16) and the **actweight** pair
   (weight+act fp8, both grads bf16) recover loss progressively at throughput
   cost; the gap isolates the wgrad GEMM's cost. The two actweight runs
   (**e4m3_actweight** vs **e5m2_actweight**) isolate the forward element format
   alone — with the backward held at bf16, their gap measures pure forward
   mantissa (e4m3) vs range (e5m2) at tensorwise.
5. **weight-only** pair (**e4m3_weightonly** vs **e5m2_weightonly**): only `weight` is
   fp8, act/both grads bf16. Weights are the benign operand (stable, no outliers),
   so both should sit near bf16, and e4m3 ≥ e5m2 (weights want mantissa, not
   range). The actweight ↔ weightonly gap adds the activation's contribution to
   the forward, isolating weight vs activation sensitivity at the same format.

If (2) holds, all-e4m3's success is granularity-dependent — it needs the
fine-grained scaling `fp8_granularity`/`mxfp8` provide, not a free win at coarse scale.

## Setup

All hyperparameters matched across runs; only `training.quant.dtype` varies, and
every FP8 run uses `scaling: {granularity: tensorwise}`, `exclude: [lm_head]`.
Same seed (42), data order, LR schedule.

| Config | weight | act | input_grad | weight_grad | Approx params |
|---|---|---|---|---|---|
| qwen3_51m_bf16 | — | — | — | — | ~51M |
| qwen3_51m_fp8_std | e4m3 | e4m3 | e5m2 | e5m2 | ~51M |
| qwen3_51m_fp8_alle4m3 | e4m3 | e4m3 | e4m3 | e4m3 | ~51M |
| qwen3_51m_fp8_alle5m2 | e5m2 | e5m2 | e5m2 | e5m2 | ~51M |
| qwen3_51m_fp8_std_gwhp | e4m3 | e4m3 | e5m2 | bf16 | ~51M |
| qwen3_51m_fp8_e4m3_actweight | e4m3 | e4m3 | bf16 | bf16 | ~51M |
| qwen3_51m_fp8_e5m2_actweight | e5m2 | e5m2 | bf16 | bf16 | ~51M |
| qwen3_51m_fp8_e4m3_weightonly | e4m3 | bf16 | bf16 | bf16 | ~51M |
| qwen3_51m_fp8_e5m2_weightonly | e5m2 | bf16 | bf16 | bf16 | ~51M |

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K
tok/step), 50K steps (~13B tokens), Muon (`muon_adjust_lr_fn: match_rms_adamw`,
momentum=0.95, nesterov), lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, OpenWebText.

## What runs in FP8

The converter (`src/quant/convert.py:apply_quantization`) swaps eligible
`nn.Linear` modules to `QuantLinear`; each per-layer GEMM (`_gemm` in
`src/quant/linear.py`) uses the fused `scaled_gemm` when both operands share a
quantized family, else falls back to fake-quant + a bf16 matmul. Everything else
(RoPE, RMSNorm, qk_norm, attention, SwiGLU, residuals, embeddings, lm_head,
cross-entropy, optimizer state) stays bf16/fp32; hp master weights preserved.

- std / all-e4m3 / all-e5m2: all three GEMMs fused FP8.
- std_gwhp: std recipe with only weight_grad forced to bf16 — forward + dgrad fused; wgrad runs bf16 (not FP8-accelerated).
- e4m3_actweight / e5m2_actweight: only the forward is fused FP8 (weight+act in
  e4m3 and e5m2 respectively); both backward GEMMs run bf16.
- e4m3_weightonly / e5m2_weightonly: **no** fused FP8 GEMM. Only `weight` is quantized, so
  the forward (`act`×`weight`) and dgrad (`input_grad`×`weight`) fake-quant the
  weight and matmul in bf16; wgrad (`weight_grad`×`act`) is pure bf16 (weight
  absent). Throughput ≈ bf16 (a touch slower from fake-quant overhead) — these
  are quality probes of weight-format sensitivity, not throughput wins.

**Config note:** all-e4m3 sets `dtype` per-operand explicitly rather than
`dtype: {recipe: mxfp8}` — the mxfp8 dtype recipe auto-seeds blockwise-32/E8M0
scaling, which collides with `tensorwise` and raises at config load.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). Dev box RTX PRO 6000 (SM 12.0).

## Run

```bash
nohup bash experiments/fp8_dtype/run.sh > logs/fp8_dtype_51m.log 2>&1 &
```

## Results

| Model | dtype recipe | weight | act | input_grad | weight_grad | Final Val Loss | Val BPB | Tokens/sec | Speedup vs bf16 |
|---|---|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | — | — | — | | | | 1.00× |
| 51M | std | e4m3 | e4m3 | e5m2 | e5m2 | | | | |
| 51M | all-e4m3 | e4m3 | e4m3 | e4m3 | e4m3 | | | | |
| 51M | all-e5m2 | e5m2 | e5m2 | e5m2 | e5m2 | | | | |
| 51M | std_gwhp | e4m3 | e4m3 | e5m2 | bf16 | | | | |
| 51M | e4m3_actweight | e4m3 | e4m3 | bf16 | bf16 | | | | |
| 51M | e5m2_actweight | e5m2 | e5m2 | bf16 | bf16 | | | | |
| 51M | e4m3_weightonly | e4m3 | bf16 | bf16 | bf16 | | | | |
| 51M | e5m2_weightonly | e5m2 | bf16 | bf16 | bf16 | | | | |

## Notes

- bf16 and std duplicate the tensorwise anchor in `fp8_granularity/` — the
  numbers should match across folders (cross-experiment sanity check).
- "Best dtype" = smallest val-loss gap to bf16 at equal throughput; combine with
  `fp8_granularity`'s best granularity for the overall recipe.
- The e4m3_actweight ↔ e5m2_actweight gap isolates the forward element format with
  the backward held at bf16; the std ↔ e4m3_actweight gap adds the backward's cost.
- Throughput gain at 51M is modest — non-GEMM kernels are a meaningful fraction
  of step time at this size.
