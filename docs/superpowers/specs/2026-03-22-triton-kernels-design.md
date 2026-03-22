# Triton Kernels Design Spec

## Goal

Replace the three `@torch.compile` fused ops in `src/model/components.py` with hand-written Triton kernels. Primary motivation: learn Triton kernel programming while achieving real performance gains on a single 24GB GPU.

## Scope

Three kernels, implemented in order:

1. **RMSNorm** (fwd + bwd) — row-wise reduction, fp32 accumulation, weight scaling
2. **SwiGLU** (fwd + bwd) — element-wise silu(gate) * up fusion
3. **RoPE** (fwd + bwd) — element-wise rotate-half with 2D grid indexing

## File Structure

```
src/kernel/
    __init__.py
    triton/
        __init__.py      # re-exports: triton_rmsnorm, triton_rope, triton_swiglu
        rmsnorm.py
        rope.py
        swiglu.py
```

## Per-File Pattern

Each file contains four layers:

1. **Forward kernel** — `@triton.jit` function on raw pointers + strides
2. **Backward kernel** — `@triton.jit` for gradient computation
3. **Autograd Function** — `torch.autograd.Function` subclass; `forward()` launches fwd kernel and saves tensors, `backward()` launches bwd kernel
4. **Functional API** — e.g. `triton_rmsnorm(x, weight, eps)` calling `TritonRMSNorm.apply`; this is what `components.py` will call

## Integration

The functional APIs are drop-in replacements for the existing `@torch.compile` functions in `components.py`:

| Current function | Triton replacement |
|------------------|--------------------|
| `_rms_norm(x, weight, eps)` | `triton_rmsnorm(x, weight, eps)` |
| `_swiglu(gate, up)` | `triton_swiglu(gate, up)` |
| `_apply_rope(x, cos, sin)` | `triton_rope(x, cos, sin)` |

The `nn.Module` wrappers (`RMSNorm`, `RoPE`, `SwiGluFFN`) remain unchanged — only the internal function call is swapped.

No auto-fallback. If Triton is unavailable, it fails explicitly.

## Kernel Details

### RMSNorm

- **Forward:** Each Triton program handles one row (token). Load row into SRAM, cast to fp32, compute mean of squares, rsqrt, scale by weight, cast back to input dtype, store.
- **Backward:** Two gradients: `dx` (chain rule through the norm) and `dweight` (reduction across all rows). `dweight` requires a cross-row reduction — accumulate per-program partial sums, then reduce.
- **Key concepts taught:** `tl.load`/`tl.store`, pointer arithmetic, masking for non-power-of-2 dimensions, `tl.sum` reduction, fp32 accumulation.

### SwiGLU

- **Forward:** Element-wise `silu(gate) * up`. One program per block of elements. Simple load-compute-store.
- **Backward:** Product rule. `dgate = dsilu(gate) * up * dout`, `dup = silu(gate) * dout`. Where `dsilu(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))`.
- **Key concepts taught:** Element-wise fusion pattern, multiple output pointers in backward.

### RoPE

- **Forward:** For each (batch, head, seq_pos, d_head) element, apply rotation: `x1*cos - x2*sin` and `x2*cos + x1*sin` where x1/x2 are the first/second half of d_head.
- **Backward:** Same rotation structure with transposed rotation matrix (negate sin).
- **Key concepts taught:** 2D grid launch, multi-dimensional pointer indexing, no reduction needed.

## Conventions

- All reductions accumulate in fp32
- Inputs assumed contiguous (asserted in the Python wrapper)
- `BLOCK_SIZE` passed as `tl.constexpr` parameter
- Each kernel has a `# Shape contract` comment documenting expected tensor shapes

## Teaching Flow (per kernel)

1. Concept brief — math, memory-boundedness, why fusion helps
2. Grid & block design — how to partition work across Triton programs
3. Forward kernel walkthrough — pointer arithmetic, masking, accumulation; user writes the code
4. Test forward — compare Triton output vs PyTorch reference with `torch.allclose`
5. Backward derivation — gradient math, then user writes the bwd kernel
6. Test backward — `torch.autograd.gradcheck`
7. Wrap up — autograd Function + functional API, plug into `components.py`

## Out of Scope

- Benchmark harness (deferred to after all three kernels are done)
- AttnRes aggregate kernels (more complex, future work)
- Auto-fallback / feature flags
