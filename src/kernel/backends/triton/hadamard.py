import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda
from src.kernel.utils import to_hadamard_scales


@triton.jit
def _butterfly(
    v,
    BLOCK_OTHER: tl.constexpr,
    BLOCK_ROT: tl.constexpr,
    HAD: tl.constexpr,
    LOG_HAD: tl.constexpr,
):
    """Hadamard-transform each HAD-wide group of `v`'s last axis, in registers.

    H_HAD factors into LOG_HAD two-point stages, one per index bit, so no matrix is
    ever loaded. Stage `s` pairs entries whose indices differ in bit `s`: reshaping
    the axis to (..., 2, low) puts that bit on its own axis, and because every group
    transforms identically the leading groups collapse into one axis.
    """
    for stage in tl.static_range(LOG_HAD):
        # `1 << stage` is the stride of the paired bit; a named constexpr cannot be
        # rebound across loop iterations, so it stays inline.
        pairs = tl.reshape(v, (BLOCK_OTHER, BLOCK_ROT // (2 << stage), 2, 1 << stage))
        # `tl.split` consumes a trailing axis of size 2, so move the paired bit last.
        even, odd = tl.split(tl.permute(pairs, (0, 1, 3, 2)))
        joined = tl.permute(tl.join(even + odd, even - odd), (0, 1, 3, 2))
        v = tl.reshape(joined, (BLOCK_OTHER, BLOCK_ROT))
    return v


@triton.jit
def _rotate_kernel(
    x_ptr,
    out_ptr,
    signs_ptr,
    n_rot,
    n_other,
    stride_batch,
    stride_rot,
    stride_other,
    HAD: tl.constexpr,
    LOG_HAD: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    POST_SCALE: tl.constexpr,
    HAS_SIGNS: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_OTHER: tl.constexpr,
    BLOCK_ROT: tl.constexpr,
):
    batch = tl.program_id(2)
    off_other = tl.program_id(0) * BLOCK_OTHER + tl.arange(0, BLOCK_OTHER)
    off_rot = tl.program_id(1) * BLOCK_ROT + tl.arange(0, BLOCK_ROT)
    base = batch * stride_batch
    in_other, in_rot = off_other < n_other, off_rot < n_rot

    offsets = off_other[:, None] * stride_other + off_rot[None, :] * stride_rot
    mask = in_other[:, None] & in_rot[None, :]
    v = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # HAD divides both n_rot and BLOCK_ROT, so a masked tail zeroes whole rotation
    # blocks; they transform to zero and are dropped by the store mask.
    v = v.to(tl.float32) * PRE_SCALE
    if HAS_SIGNS:
        signs = tl.load(signs_ptr + (off_rot % HAD)).to(tl.float32)
        if not INVERSE:
            v = v * signs[None, :]
    v = _butterfly(v, BLOCK_OTHER, BLOCK_ROT, HAD, LOG_HAD)
    if POST_SCALE != 1.0:
        v = v * POST_SCALE
    if HAS_SIGNS and INVERSE:
        v = v * signs[None, :]

    v = v.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + base + offsets, v, mask=mask)


@register_kernel(
    op="hadamard.rotate",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda()}),
)
@triton_op("jit_kernel::hadamard_rotate", mutates_args={})
def rotate(
    x: torch.Tensor,
    hadamard_block: int,
    sign_vector: torch.Tensor | None,
    inverse: bool,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Block-Hadamard rotation along the final axis, fused into one pass."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if hadamard_block == 1:
        return x.to(out_dtype, copy=True)
    shape = x.shape
    rows, cols = x.shape[-2:]
    # Flatten only the leading batch axes. This remains a view for regular layouts,
    # including the standard -2 transpose, and materializes only non-flattenable
    # leading permutations before the kernel addresses batches by one stride.
    x = x.reshape(-1, rows, cols)
    # The kernel stores through `x`'s strides, so `out` must mirror them. `empty_like`
    # does that for any non-overlapping layout; materialize the views it cannot mirror.
    out = torch.empty_like(x, dtype=out_dtype)
    if out.stride() != x.stride():
        x = x.contiguous()
        out = torch.empty_like(x, dtype=out_dtype)
    pre_scale, post_scale = to_hadamard_scales(hadamard_block)
    batch = x.shape[0]
    stride_batch = x.stride(0)

    n_rot, n_other = cols, rows
    stride_rot, stride_other = x.stride(-1), x.stride(-2)

    # BLOCK_ROT must cover whole rotation blocks, so never fall below `hadamard_block`.
    block_rot = max(hadamard_block, 64)
    block_other = max(2048 // block_rot, 8)
    grid = (
        triton.cdiv(n_other, block_other),
        triton.cdiv(n_rot, block_rot),
        batch,
    )
    wrap_triton(_rotate_kernel)[grid](
        x,
        out,
        sign_vector,
        n_rot,
        n_other,
        stride_batch,
        stride_rot,
        stride_other,
        HAD=hadamard_block,
        LOG_HAD=hadamard_block.bit_length() - 1,
        PRE_SCALE=pre_scale,
        POST_SCALE=post_scale,
        HAS_SIGNS=sign_vector is not None,
        INVERSE=inverse,
        BLOCK_OTHER=block_other,
        BLOCK_ROT=block_rot,
        num_warps=4,
    )
    return out.reshape(shape)
