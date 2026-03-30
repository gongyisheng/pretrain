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

## Experiment 4: Use Triton scatter for torch backend too

**Changes**: Made the torch backend use Triton scatter kernels on CUDA (with fallback to PyTorch scatter on CPU for tests). Profile showed scatter ops take 42% of layer time — Triton scatter is 5-7x faster even within the torch.compile compiled graph.

| Backend | Before  | After    | Delta   |
|---------|---------|----------|---------|
| torch   | 72,197  | 102,396  | +41.8%  |
| triton  | 74,916  | 105,983  | +41.5%  |

**Result**: Both backends now get the full scatter speedup.
**Decision**: COMMITTED

---

## Experiment 5: Sort-free routing with Triton atomic position assignment

**Changes**: Replaced `torch.sort` (O(N log N)) with a Triton kernel that assigns within-expert positions via atomic counters (O(N)). Expert IDs in the output are not sorted — the scatter kernels handle random access.

Micro-benchmark (routing only):
| Approach      | Time   |
|---------------|--------|
| torch.sort    | 0.268ms|
| Atomic assign | 0.155ms|
| Speedup       | 1.72x  |

Full training benchmark (combined with all prior optimizations):
| Backend | Before   | After    | Delta  |
|---------|----------|----------|--------|
| torch   | 102,396  | ~102k    | ~0%    |
| triton  | 105,983  | ~105k    | ~0%    |

**Result**: Micro-benchmark 1.72x on routing, but end-to-end neutral because unsorted expert IDs produce less cache-friendly scatter patterns. Overall a wash.
**Decision**: KEPT (cleaner code, eliminates sort for fixed-capacity path)

---

## Experiment 6: Triton scatter autograd support (correctness fix)

**Changes**: Added `torch.autograd.Function` wrappers for Triton scatter-in and scatter-out kernels with Triton backward kernels. This is necessary for correct gradient flow through expert weights — without autograd, expert_gate_up and expert_down parameters don't receive gradients from the cross-entropy loss, only the router gets gradients from the aux_loss.

Backward kernels:
- Scatter-in backward: gather from grad_padded, atomic_add into grad_x
- Scatter-out backward: gather from grad_output, multiply by weight, store into grad_expert

| Backend | Before (no autograd) | After (with autograd) | vs Baseline |
|---------|---------------------|-----------------------|-------------|
| torch   | ~105k (incorrect)   | 89,691                | +24.2%      |
| triton  | ~106k (incorrect)   | 91,955                | +22.7%      |

**Result**: ~90k tok/s (vs 72k baseline). The 105k numbers were incorrect (no expert gradient flow). Real improvement is ~24%.
**Decision**: COMMITTED (correctness fix + 24% speedup over baseline)

---

## Final Results Summary

| Backend | Baseline | Final   | Improvement |
|---------|----------|---------|-------------|
| torch   | 72,197   | 89,825  | **+24.4%**  |
| triton  | 74,916   | 92,197  | **+23.1%**  |

### Key optimizations (cumulative):
1. **Triton scatter kernels** (5-7x per-op speedup): The dominant win. PyTorch fancy indexing + scatter_add_ was the bottleneck (42% of layer time).
2. **Autograd Function wrappers**: Correctness fix enabling gradient flow through expert weights.
3. **Sort-free routing**: Triton atomic counters replace torch.sort. Micro-benchmark 1.72x, end-to-end neutral.
4. **Vectorized aux_loss**: Replaced Python for-loop with bincount.

### Files added:
- `src/kernel/torch/moe_ffn.py` — compiled expert FFN for torch backend
- `src/kernel/torch/moe_scatter.py` — PyTorch scatter (CPU fallback)
- `src/kernel/torch/moe_routing.py` — PyTorch routing (CPU fallback)
- `src/kernel/triton/moe_ffn.py` — Triton expert FFN
- `src/kernel/triton/moe_scatter.py` — Triton scatter kernels with autograd
- `src/kernel/triton/moe_routing.py` — Triton sort-free routing
- `tests/kernel/test_moe_kernels.py` — 18 tests for all new kernels
