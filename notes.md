# perf-20260515: Qwen3 57M Training Throughput

Goal: improve `python benchmarks/bench_train.py --config configs/qwen3_57m.yaml`
on a single CUDA device (cuda:0).

Hardware: NVIDIA GeForce RTX 5060 Ti (sm_120, Blackwell, 16 GB).
Software: PyTorch 2.10.0+cu128.

## Config under test

- arch: qwen3, 8 layers, 8 heads, 4 kv heads, d_model=512, vocab=50257
- seq_len=1024, bf16, qk_norm=true, activation_checkpointing=false
- effective batch = 256 examples

## Variants tested (in order)

| # | Variant | tok/s | Δ vs main | Committed |
|---|---|---|---|---|
| 0 | baseline (main, bs=8/ga=32) | 75,550 | — | — |
| 1 | SDPA `enable_gqa=True` | 60,621 | −19.8% | no (regression) |
| 2 | qk-norm 4D + RoPE gather fused + prefetch=4 | 75,559 | −0.0% | no (neutral) |
| 3 | `mode="reduce-overhead"` (CUDA graphs) | crash | — | no |
| 4 | Force flash-only SDPA backend | 75,608 | −0.1% | no (neutral) |
| 5 | Liger fused linear+CE | 37,995 | −49.7% | no (regression) |
| 6 | bs=16/ga=16 | 77,966 | +3.2% | superseded |
| 7 | **bs=32/ga=8** | **78,914** | **+4.5%** | ✅ commit `296ec01` |
| 8 | bs=64/ga=4 | OOM | — | — |
| 9 | num_workers=8, prefetch_factor=4 | 78,900 | 0.0% | no |
| 10 | **FlexAttention + BlockMask for intra-doc** | **94,540** | **+25.1%** | ✅ this commit |

All numeric variants produced loss curves matching baseline (within bf16
tolerance) — no training-dynamics regression.

## The two wins, explained

### Win 1: rebalance bs/ga (`+4.5%`)

`bs=8/ga=32` → `bs=32/ga=8`. Effective batch unchanged (256). Cuts micro-steps
per training step from 32 → 8, amortising kernel-launch / autocast / dataloader-
handoff overhead. Math is unchanged; loss trajectory is bit-identical.

### Win 2: FlexAttention for intra-doc causal mask (`+19.8% on top`)

**Root cause** (found via `nsys profile`): the dense `(B, 1, S, S)` additive
attention mask we passed to SDPA forced PyTorch to dispatch to the
**memory-efficient attention** backend (cutlass `sm_80` kernels) — flash-attn
doesn't accept arbitrary masks. memEff is ~1.8× slower than flash at our
shape, and 23% of GPU time was in attention.

**Fix**: replace the dense additive mask with a `BlockMask` driving
`torch.nn.attention.flex_attention`. FlexAttention compiles a Triton kernel
from a user-defined `mask_mod` that exploits block sparsity — fully-masked
128×128 tiles are skipped entirely. Microbench at our exact shape:

| Backend | fwd | bwd | total |
|---|---|---|---|
| FlexAttention + BlockMask | 0.56ms | 1.73ms | **2.29ms** |
| memEff + custom mask (old path) | 3.04ms | 9.93ms | 12.97ms |
| flash + `is_causal` (no doc separation) | 0.91ms | 3.18ms | 4.09ms |

Sparse-block dispatch makes FlexAttention faster than even unmasked flash.
Intra-doc correctness is preserved.

## Code design (per follow-up review)

- **`ModelConfig.attn_implementation: str = "flex_attention"`** (or `"sdpa"`).
  Drives the kernel choice from config, not from runtime type dispatch.
- **`build_attention_mask(position_ids, device, dtype, attn_implementation)`**
  is the single mask builder; returns either a `BlockMask` (flex) or dense
  additive tensor (sdpa). Old `build_causal_mask` is folded in.
- **`MultiHeadAttention` / `GroupedQueryAttention`** take `attn_implementation`
  in `__init__` and stash `self._attn_fn = _ATTN_IMPL[impl]`. Forward just
  calls `self._attn_fn(q, k, v, attn_mask)` — no isinstance checks.
- **Trainer** passes `self.config.model.attn_implementation` to
  `build_attention_mask`; same mask format is consumed at both training and
  eval call sites.
- **Tests** parameterize `MHA`/`GQA` mask-using tests over `(impl, device)`
  pairs (`sdpa-cpu`, `flex-cuda` with skip if no CUDA). A cross-backend
  parity test pins `out_sdpa ≈ out_flex` on identical Q/K/V/mask.

## Discarded ideas (and why)

- **`enable_gqa=True` in SDPA**: 20% regression on Blackwell + PyTorch 2.10.
  Manual `expand+reshape` stays faster than SDPA's internal GQA broadcasting.
- **Liger fused linear+CE**: 50% regression. The kernel lives outside
  `torch.compile`'s graph, so we lose inductor's fusion of linear+softmax;
  at our (B*S=32k, V=50k) scale, the chunked Triton launch overhead exceeds
  the memory win. Would likely flip on a bigger model.
- **CUDA graphs (`reduce-overhead`)**: crashes on the grad-accum loop because
  of cross-iteration tensor reuse. A real integration needs
  `cudagraph_mark_step_begin` + input cloning in the trainer; deferred.
- **Forcing flash via `sdpa_kernel` context**: no effect inside the compiled
  graph — inductor picks the backend at compile time and the user-level
  context manager doesn't propagate. (FlexAttention solves the underlying
  problem differently.)
- **qk-norm reshape removal / RoPE gather fusion / `prefetch_factor` tweaks /
  forcing flash backend**: all net-zero. `torch.compile` already fuses these.

## Final results

```
baseline (main)                : 75,550 tok/s
bs=32/ga=8 (commit 296ec01)    : 78,914 tok/s   (+4.5%)
+ FlexAttention (this commit)  : 94,540 tok/s   (+25.1% vs main, +19.8% vs prior)
```

Loss curve: bit-identical to baseline through step 25 (within bf16 rounding).
Behavior preserved — intra-document causal attention still enforced.

## Where the next wins likely live

- **Inductor-visible fused linear+CE via `@triton_op`** so the kernel stays in
  the compiled graph (avoids Liger's regression). The LM-head + CE bucket is
  now ~12% of step time at this throughput, the largest remaining lever.
- **CUDA graphs** with a trainer-loop retrofit (`cudagraph_mark_step_begin`
  + input cloning) — typically 5-15% on small models like this one.
- **`max-autotune-gemm`** is locked out on this GPU ("not enough SMs"); on a
  larger card the matmul autotune alone usually buys another 5-10%.
