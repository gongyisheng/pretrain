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

---

## Round 2: Systematic Speed Optimization (2026-03-29)

GPU: NVIDIA GeForce RTX 5060 Ti 16GB

### New Baseline (after prior optimizations)

| Model | Backend | tok/s |
|-------|---------|-------|
| GPT2 124M | torch | 41,083 |
| GPT2 124M | triton | 37,769 |
| Qwen3 145M | torch | 33,561 |
| Qwen3 145M | triton | 33,573 |

Benchmark: `bench_train.py --all --steps 10 --warmup 10`

### Exp 1: Fused AdamW (`fused=True`) - KEPT

**File:** `src/training/optimizer.py`

Added `fused=True` to `torch.optim.AdamW`. This uses a single CUDA kernel per
parameter group for the optimizer step instead of separate kernels per parameter.

| Model | Backend | Before | After | Change |
|-------|---------|--------|-------|--------|
| GPT2 | torch | 41,083 | 41,542 | +1.1% |
| GPT2 | triton | 37,769 | 38,130 | +1.0% |
| Qwen3 | torch | 33,561 | 34,054 | +1.5% |
| Qwen3 | triton | 33,573 | 34,071 | +1.5% |

### Exp 2: DataLoader persistent_workers - KEPT (code quality)

**File:** `src/training/trainer.py`

Added `persistent_workers=True` when `num_workers > 0` to avoid worker process
respawn overhead between epochs. No measurable throughput change since both
configs use `num_workers=0` (data loading is ~0.3% of step time due to
memory-mapped datasets + prefetch stream). Code change kept for correctness.

### Exp 3: CUDA expandable segments allocator - NO CHANGE

Tested `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. No measurable
improvement — memory is not fragmented in single-model training.

### Exp 4: Qwen3 batch size increase (8→12) - KEPT

**File:** `configs/qwen3_145m.yaml`

Increased Qwen3 batch_size from 8 to 12, reduced gradient_accumulation_steps
from 4 to 3. Larger micro-batches improve GPU utilization by amortizing kernel
launch overhead over more tokens. Peak memory goes from ~10GB to ~12.3GB (fits
16GB GPU). GPT2 bs=16 was also tested but showed no gain (already saturated at bs=12).

| Model | Backend | Before | After | Change |
|-------|---------|--------|-------|--------|
| Qwen3 | torch | 34,054 | 34,763 | +2.1% |
| Qwen3 | triton | 34,071 | 34,744 | +2.0% |

### Exp 5: torch.compile mode=max-autotune - REVERTED

`max-autotune` crashes with CUDA graph tensor overwrite (weight tying).
`max-autotune-no-cudagraphs` showed no improvement (within noise).

### Exp 6: Pad vocab to multiple of 128 for matmul alignment - KEPT

**Files:** `src/model/gpt2.py`, `src/model/qwen3.py`

Padded vocab_size from 50257 to 50304 (next multiple of 128) by increasing
the embedding dimension. The lm_head matmul `(B*S, D) @ (D, V)` benefits
enormously from V being aligned — cuBLAS and Triton can pick optimal tiling
instead of falling back to slow remainder-handling paths.

Weight tying preserved (lm_head.weight = token_emb.weight). The 47 extra
vocab entries participate in softmax but receive near-zero probability.
Measured with 20 steps + 15 warmup for stability.

| Model | Backend | Before | After | Change |
|-------|---------|--------|-------|--------|
| GPT2 | torch | 41,984 | 44,608 | +6.3% |
| GPT2 | triton | 38,598 | 44,329 | +14.8% |
| Qwen3 | torch | 35,248 | 36,109 | +2.4% |
| Qwen3 | triton | 35,137 | 39,295 | +11.8% |

### Exp 7: Fix triton CE backward hardcoded dtype - KEPT (correctness)

**File:** `src/kernel/triton/cross_entropy.py`

Removed hardcoded `.to(tl.bfloat16)` in backward kernel store. The output
tensor inherits dtype from input logits; Triton handles implicit conversion.
No throughput impact but fixes correctness for float16/float32 dtypes.
