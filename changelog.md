# Changelog: Triton Backend Training Performance Optimization

## Summary

Systematic profiling and optimization of the Triton backend training pipeline.
Hardware: NVIDIA RTX 5060 Ti (24GB VRAM).

### Baseline Performance
| Model | Backend | Steady-state tok/s |
|-------|---------|-------------------|
| GPT-2 124M | triton | ~31,700 |
| Qwen3 145M | triton | ~31,430 |

### Final Performance (after optimization)
| Model | Backend | Steady-state tok/s | Change |
|-------|---------|-------------------|--------|
| GPT-2 124M | triton | ~31,760 | +0.2% |
| Qwen3 145M | triton | ~31,630 | +0.6% |

### Key Finding
Triton kernels (flash attention, RoPE, SwiGLU) account for only ~13% of total
training step time. The remaining ~87% is dominated by linear layer matmuls
handled by cuBLAS via torch.compile. This limits the maximum possible speedup
from Triton kernel optimization alone.

---

## Changes Kept (Committed)

### 1. Fused D computation kernel in flash attention backward
**File:** `src/kernel/triton/flashattn.py`

Added a dedicated Triton kernel `_flash_attn_D_kernel` to compute
`D = rowwise_sum(dO * O)` instead of the PyTorch expression
`D = (do.float() * o.float()).sum(dim=-1)`.

**Impact:** 9% faster flash attention backward (0.716ms → 0.651ms per call)
by eliminating two temporary fp32 tensor allocations (~25 MB each) per call.

### 2. Training loop: deferred loss.item() sync
**File:** `src/training/trainer.py`

Replaced per-micro-step `loss.item()` (which forces CUDA sync) with tensor
accumulation `accum_loss_tensor += loss.detach()`, calling `.item()` only once
per optimizer step.

**Impact:** Removes gradient_accumulation_steps - 1 unnecessary CUDA
synchronization points per training step.

### 3. Training loop: stream-based data prefetching
**File:** `src/training/trainer.py`

Added a separate CUDA stream for host-to-device data transfers, overlapping
the H2D transfer of the next micro-batch with the forward/backward compute
of the current micro-batch.

**Impact:** Overlaps ~0.3ms of H2D transfer per micro-step with compute.

### 4. GPU precision and compute settings
**File:** `src/training/trainer.py`

- `torch.set_float32_matmul_precision('high')` — enables TF32 for fp32 matmuls
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`
- `torch.backends.cudnn.benchmark = True`
- `optimizer.zero_grad(set_to_none=True)` — avoids zeroing gradient memory

**Impact:** Marginal; most compute already uses bf16 via mixed precision.

---

## Changes Attempted and Rolled Back

### 1. torch.compile mode='max-autotune-no-cudagraphs'
Autotuning matmul configs via Inductor. No improvement over default mode.

### 2. torch.compile mode='reduce-overhead' (CUDA graphs)
Crashed due to weight tying in models (lm_head.weight = token_emb.weight)
causing tensor overwrite errors in CUDA graph replay.

### 3. Inductor coordinate_descent_tuning
`torch._inductor.config.coordinate_descent_tuning = True` — actually made
training ~14% slower, likely due to suboptimal kernel selection.

### 4. torch.library custom op registration
Registered Triton ops (SwiGLU, RoPE, FlashAttn) as torch.library custom ops
to eliminate torch.compile graph breaks. Made training ~12% slower because
torch.compile generates more efficient code with autograd.Function graph breaks
than with custom op dispatch.

### 5. fp32 dtype casts in flash attention backward
Added `.to(tl.float32)` casts to backward kernel dot products for dtype
compatibility. Caused ~14% regression because fp32 dot products bypass tensor
cores. Reverted to original bf16 dot products.

### 6. Flash attention block size tuning
Swept BLOCK_Q, BLOCK_KV in {32, 64, 128} × num_warps in {2, 4, 8} ×
num_stages in {1, 2, 3} for all three kernels (fwd, dk/dv bwd, dq bwd).
The existing config (BLOCK_Q=64, BLOCK_KV=64, warps=4, stages=2) is already
near-optimal for this GPU.

### 7. SwiGLU block size tuning
Swept BLOCK_SIZE in {512, 1024, 2048, 4096, 8192} for both forward and
backward kernels. No measurable difference — kernel is bandwidth-bound.

---

## Profiling Results

### Training Step Breakdown (Qwen3 145M, triton backend)
| Phase | Time | % of Step |
|-------|------|-----------|
| Forward | ~498 ms | 39% |
| Backward | ~675 ms | 53% |
| Loss (CE) | ~63 ms | 5% |
| Optimizer | ~35 ms | 3% |
| Data loading | ~4 ms | 0.3% |

### Individual Triton Kernel Timings (B=4, H=12, S=1024, D=64)
| Kernel | Time | Notes |
|--------|------|-------|
| Flash Attn FWD | 0.176 ms | Already near-optimal |
| Flash Attn BWD | 0.651 ms | Improved from 0.716ms (9%) |
| RoPE FWD | 0.049 ms | Negligible |
| SwiGLU FWD | 0.183 ms | Bandwidth-bound |
| SwiGLU BWD | 0.324 ms | Bandwidth-bound |
