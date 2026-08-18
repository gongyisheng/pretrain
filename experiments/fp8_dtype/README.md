# FP8 Element-Format (dtype) Sweep

Compare end-task loss and throughput across FP8 element-format recipes against a
bf16 baseline at Qwen3-51M, holding scaling granularity fixed at **tensorwise**,
using the in-house `src/quant/` framework. The independent variable is
`training.quantization.dtype`; the bf16 run anchors the loss/throughput reference.

Scope: whole-network recipes, the **backward** GEMMs, and throughput by fusion class.
The per-tensor sensitivity axis (one of weight / act / grad_input / grad_weight in fp8
at a time, × e4m3 / e5m2) lives in `experiments/fp8_tensor_dtype/`, which owns those
cells; this folder's `alle4m3` / `alle5m2` are the all-four corners of its additivity
check, and its `e4m3_w8a8` is the fourth corner of the backward 2×2 here. Complements
`experiments/fp8_granularity/` (granularity axis at fixed std-fp8 dtype). Designs:
`docs/superpowers/specs/2026-07-27-fp8-dtype-experiment-design.md`,
`docs/superpowers/specs/2026-08-17-quant-operand-and-int-bit-allocation-design.md`.

## Hypothesis

`QuantLinear` runs three GEMMs — forward (`act`×`weight`), dgrad
(`grad_input`×`weight`), wgrad (`grad_weight`×`act`). Classic intuition: forward
operands want e4m3 (mantissa/precision), gradients want e5m2 (exponent range).
At the coarsest scale (tensorwise), the element format matters most, since
nothing but the format supplies dynamic range.

1. **std** (e4m3 fwd / e5m2 grad) matches bf16 within noise.
2. **all-e4m3** loses loss here (unlike under mxfp8) — tensorwise scaling can't
   supply the grad range, so e4m3 clips grad outliers.
3. **all-e5m2** is the floor — the forward loses mantissa it can't afford.
4. **std_gwhp** (std with grad_weight forced to bf16) and **std_gihp** (std with
   grad_input forced to bf16) recover loss progressively at throughput cost. With
   std (both grads fp8) and the both-grads-bf16 corner, they form a 2×2 over the
   backward GEMMs: `std − std_gwhp` isolates the fp8 wgrad cost, `std − std_gihp`
   isolates the fp8 dgrad cost, and their sum vs `std − <both-bf16>` checks
   additivity. dgrad error chains backward through every layer, so it may be the
   more fp8-sensitive backward path — hence a dedicated cell rather than inferring
   it from the joint gap.

The 2×2's fourth corner (weight+act e4m3, both grads bf16) now lives in
`fp8_tensor_dtype/qwen3_51m_fp8_e4m3_w8a8.yaml`. Its base config is byte-identical to this
folder's, so its number drops straight into the additivity check above — run it
once, in whichever folder comes first.

If (2) holds, all-e4m3's success is granularity-dependent — it needs the
fine-grained scaling `fp8_granularity`/`mxfp8` provide, not a free win at coarse scale.

## Setup

All hyperparameters matched across runs; only `training.quantization.dtype` varies, and
every FP8 run uses `scaling: {granularity: tensorwise}`, `exclude: [lm_head]`.
Same seed (42), data order, LR schedule — except the two extra bf16 replicas, which vary
`seed` alone to measure the noise floor.

| Config | weight | act | grad_input | grad_weight | Approx params |
|---|---|---|---|---|---|
| qwen3_51m_bf16 | — | — | — | — | ~51M |
| qwen3_51m_bf16_seed43 | — | — | — | — | ~51M |
| qwen3_51m_bf16_seed44 | — | — | — | — | ~51M |
| qwen3_51m_fp8_std | e4m3 | e4m3 | e5m2 | e5m2 | ~51M |
| qwen3_51m_fp8_alle4m3 | e4m3 | e4m3 | e4m3 | e4m3 | ~51M |
| qwen3_51m_fp8_alle5m2 | e5m2 | e5m2 | e5m2 | e5m2 | ~51M |
| qwen3_51m_fp8_std_gwhp | e4m3 | e4m3 | e5m2 | bf16 | ~51M |
| qwen3_51m_fp8_std_gihp | e4m3 | e4m3 | bf16 | e5m2 | ~51M |

All runs: seq_len=1024, batch=16, grad_accum=16 (effective batch=256, ~262K
tok/step), 50K steps (~13B tokens), Muon (`muon_adjust_lr_fn: match_rms_adamw`,
momentum=0.95, nesterov), lr=5e-4, cosine with 1500 warmup, min_lr=5e-5, OpenWebText.

**Noise floor.** The gaps here are 0.1–1% of val loss, the same order as seed-to-seed
spread, and nothing in `experiments/` measured that before (`determinism/` tests
same-seed bit-identity, a different quantity). The three bf16 runs differ only in
`training.seed`. Reduce every run — these three and every cell below — to the **mean
val loss over its final 10 evals** (~steps 49.1K–50K) rather than its last single eval;
same runs, same cost, less eval noise in the statistic every conclusion rests on. The
resolution is 2σ of the three bf16 means, and a `Δ vs bf16` counts as real only when
`|Δ| > 2σ`. n=3 gives a crude σ, which is enough for an order of
magnitude. If 2σ exceeds the largest gap, the result is "all recipes within noise at
51M" — report that rather than ranking anyway. `fp8_tensor_dtype/`, `int_dtype/`, and
`int_granularity/` share this base config byte-for-byte and read their resolution off
these same three runs; changing a base hyperparameter or the eval protocol in any of them
forfeits that.

## What runs in FP8

The converter (`src/quant/convert.py:apply_quantization`) swaps eligible
`nn.Linear` modules to `QuantLinear`; each per-layer GEMM (`_gemm` in
`src/quant/linear.py`) uses the fused `scaled_mm` when both operands share a
quantized family, else falls back to fake-quant + a bf16 matmul. Everything else
(RoPE, RMSNorm, qk_norm, attention, SwiGLU, residuals, embeddings, lm_head,
cross-entropy, optimizer state) stays bf16/fp32; hp master weights preserved.

- std / all-e4m3 / all-e5m2: all three GEMMs fused FP8.
- std_gwhp: std recipe with only grad_weight forced to bf16 — forward + dgrad fused; wgrad runs bf16 (not FP8-accelerated).
- std_gihp: std recipe with only grad_input forced to bf16 — forward + wgrad fused; dgrad runs bf16 (not FP8-accelerated). Mirror of std_gwhp; same 2/3 fusion, same throughput class.
**Throughput is only comparable within a fusion class.** `scaled_mm_op`
(`src/quant/utils.py`) returns a fused kernel only when *both* operands of that GEMM are
quantized, so every cell with a bf16 operand emulates that GEMM at strictly worse than
bf16 speed. Loss comparisons are valid across all cells; tokens/sec is meaningful only
within a class — std / all-e4m3 / all-e5m2 fuse 3 GEMMs, std_gwhp / std_gihp fuse 2. The
partially-fused and unfused cells in `fp8_tensor_dtype/` fuse 1 or 0; reading their
tokens/sec as an fp8 throughput result is a category error.

**Config note:** all-e4m3 sets `dtype` per-operand explicitly rather than
`dtype: {recipe: mxfp8}` — the mxfp8 dtype recipe auto-seeds blockwise-32/E8M0
scaling, which collides with `tensorwise` and raises at config load.

Hardware requirement: SM 8.9+ (Ada/Hopper/Blackwell). Dev box RTX PRO 6000 (SM 12.0).

## Run

```bash
nohup bash experiments/fp8_dtype/run.sh > logs/fp8_dtype_51m.log 2>&1 &
```

## Results

Noise floor — three bf16 runs, `seed` the only difference:

| Config | seed | Val Loss (mean last 10 evals) | Val BPB |
|---|---|---|---|
| qwen3_51m_bf16 | 42 | | |
| qwen3_51m_bf16_seed43 | 43 | | |
| qwen3_51m_bf16_seed44 | 44 | | |
| **mean ± σ** | | | |
| **resolution (2σ)** | | | |

`Δ vs bf16` below is against the seed-42 run; `> noise?` is `|Δ| > 2σ`. Tokens/sec is
comparable only within a fusion class — see "What runs in FP8".

| Model | dtype recipe | weight | act | grad_input | grad_weight | GEMMs fused | Val Loss | Δ vs bf16 | > noise? | Val BPB | Tokens/sec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 51M | bf16 | — | — | — | — | — | | 0 | — | | |
| 51M | std | e4m3 | e4m3 | e5m2 | e5m2 | 3 | | | | | |
| 51M | all-e4m3 | e4m3 | e4m3 | e4m3 | e4m3 | 3 | | | | | |
| 51M | all-e5m2 | e5m2 | e5m2 | e5m2 | e5m2 | 3 | | | | | |
| 51M | std_gwhp | e4m3 | e4m3 | e5m2 | bf16 | 2 | | | | | |
| 51M | std_gihp | e4m3 | e4m3 | bf16 | e5m2 | 2 | | | | | |
| 51M | e4m3_w8a8 † | e4m3 | e4m3 | bf16 | bf16 | 1 | | | | | |

† The backward 2×2's fourth corner, run from `fp8_tensor_dtype/`.

## Notes

- bf16 and std duplicate the tensorwise anchor in `fp8_granularity/` — the
  numbers should match across folders (cross-experiment sanity check).
- "Best dtype" = smallest val-loss gap to bf16 at equal throughput; combine with
  `fp8_granularity`'s best granularity for the overall recipe.
- std/std_gwhp/std_gihp/e4m3_w8a8 are the four corners of the backward 2×2:
  std−std_gwhp = fp8 wgrad cost, std−std_gihp = fp8 dgrad cost. If those two
  singles sum to std−e4m3_w8a8, the backward GEMM errors are additive; a gap
  means they interact. Larger of the two singles = the more fp8-sensitive
  backward path (dgrad chains through depth, so it may dominate).
- Throughput gain at 51M is modest — non-GEMM kernels are a meaningful fraction
  of step time at this size.
- Companion experiments sharing this base config and noise floor: `fp8_tensor_dtype/`
  (per-tensor sensitivity), `int_dtype/` (weight-only int8→int4 ladder),
  `int_granularity/` (scale granularity at the cliff bit-width). Design:
  `docs/superpowers/specs/2026-08-17-quant-operand-and-int-bit-allocation-design.md`.
