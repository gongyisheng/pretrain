# MoE Performance Optimization Changelog

## Branch: feat/autoresearch-moe-perf

This document tracks all performance experiments for MoE kernels and model optimizations.
Each entry records the change, benchmark results, and whether it was kept or reverted.

---

## Baseline (2026-03-30)

**Config**: qwen3_moe_133m.yaml (64 experts, top-2, capacity_factor=1.25, d_model=512, moe_intermediate=128)
**GPU**: NVIDIA GeForce RTX 5060 Ti
**Benchmark**: 10 measured steps, 5 warmup steps

| Backend | Tokens/sec |
|---------|-----------|
| torch   | 72,197    |
| triton  | 74,916    |

Note: triton backend produces NaN loss (pre-existing flash attention dtype issue), throughput is still valid.

---

## Experiment 1: torch.compile fused MoE expert FFN

**Changes**: Extracted expert FFN compute (bmm + chunk + swiglu + bmm) into a standalone `_moe_expert_ffn` function pointer, with `@torch.compile` for torch backend and triton_swiglu-based version for triton backend.

| Backend | Before  | After   | Delta |
|---------|---------|---------|-------|
| torch   | 72,197  | 71,941  | -0.4% |
| triton  | 74,916  | 74,813  | -0.1% |

**Result**: Neutral — the compiler can't fuse across bmm boundaries.
**Decision**: KEPT (cleaner code, enables future kernel work, no regression)

---

## Experiment 2: Optimized MoE routing ops

**Changes**: Replaced `argsort` + gather with `sort` (returns both), `scatter_add_` counting with `torch.bincount`, and the for-loop aux_loss with vectorized `expert_counts / T`.

| Backend | Before  | After   | Delta |
|---------|---------|---------|-------|
| torch   | 72,197  | 71,479  | -1.0% |
| triton  | 74,916  | 74,024  | -1.2% |

**Result**: Neutral to slightly worse — routing overhead is small vs total compute.
**Decision**: KEPT (cleaner code, bincount + vectorized aux_loss are more idiomatic)

---

## Experiment 3: Triton scatter kernels for MoE dispatch

**Changes**: Added Triton kernels for scatter-in (token → padded expert input) and scatter-out (expert output → token positions with weighted accumulation). These replace PyTorch fancy indexing + scatter_add_ with explicit Triton kernels using coalesced memory access.

Micro-benchmark (isolated scatter ops, per-layer):
| Operation    | PyTorch | Triton | Speedup |
|--------------|---------|--------|---------|
| Scatter IN   | 0.204ms | 0.039ms | 5.2x   |
| Scatter OUT  | 0.329ms | 0.044ms | 7.5x   |

Full training benchmark:
| Backend | Before  | After    | Delta   |
|---------|---------|----------|---------|
| torch   | 72,197  | 71,476   | -1.0%   |
| triton  | 74,916  | 105,082  | +40.3%  |

**Result**: Massive improvement for triton backend. Torch backend neutral (uses PyTorch scatter, no Triton).
**Decision**: COMMITTED

---
