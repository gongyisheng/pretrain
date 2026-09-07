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
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
    LOG2_HADAMARD_BLOCK: tl.constexpr,
):
    """Hadamard-transform each Hadamard-block-wide group of `v` in registers.

    A Hadamard block factors into LOG2_HADAMARD_BLOCK two-point stages, one per
    index bit, so no matrix is ever loaded. Stage `s` pairs entries whose indices
    differ in bit `s`: reshaping the axis to (..., 2, low) puts that bit on its own
    axis, and because every group transforms identically the leading groups collapse
    into one axis.
    """
    for stage in tl.static_range(LOG2_HADAMARD_BLOCK):
        # `1 << stage` is the stride of the paired bit; a named constexpr cannot be
        # rebound across loop iterations, so it stays inline.
        pairs = tl.reshape(v, (BLOCK_ROW, BLOCK_COL // (2 << stage), 2, 1 << stage))
        # `tl.split` consumes a trailing axis of size 2, so move the paired bit last.
        even, odd = tl.split(tl.permute(pairs, (0, 1, 3, 2)))
        joined = tl.permute(tl.join(even + odd, even - odd), (0, 1, 3, 2))
        v = tl.reshape(joined, (BLOCK_ROW, BLOCK_COL))
    return v


@triton.jit
def _rotate_kernel(
    x_ptr,
    out_ptr,
    signs_ptr,
    n_row,
    n_col,
    stride_batch,
    stride_row,
    stride_col,
    HADAMARD_BLOCK: tl.constexpr,
    LOG2_HADAMARD_BLOCK: tl.constexpr,
    PRE_SCALE: tl.constexpr,
    POST_SCALE: tl.constexpr,
    HAS_SIGNS: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    off_row = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    off_col = tl.program_id(1) * BLOCK_COL + tl.arange(0, BLOCK_COL)
    base = batch_idx * stride_batch
    in_row, in_col = off_row < n_row, off_col < n_col

    offsets = off_row[:, None] * stride_row + off_col[None, :] * stride_col
    mask = in_row[:, None] & in_col[None, :]
    v = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)

    # HADAMARD_BLOCK divides both n_col and BLOCK_COL, so a masked tail zeroes
    # whole rotation blocks; they transform to zero and are dropped by the store mask.
    v = v.to(tl.float32) * PRE_SCALE
    if HAS_SIGNS:
        signs = tl.load(signs_ptr + (off_col % HADAMARD_BLOCK)).to(tl.float32)
        if not INVERSE:
            v = v * signs[None, :]
    v = _butterfly(v, BLOCK_ROW, BLOCK_COL, LOG2_HADAMARD_BLOCK)
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
    """Block-Hadamard rotation along the final axis of logical [batch, row, col]."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    if hadamard_block == 1:
        return x.to(out_dtype, copy=True)
    shape = x.shape
    n_row, n_col = x.shape[-2:]
    x = x.reshape(-1, n_row, n_col)
    out = torch.empty_like(x, dtype=out_dtype)
    if out.stride() != x.stride():
        x = x.contiguous()
        out = torch.empty_like(x, dtype=out_dtype)
    pre_scale, post_scale = to_hadamard_scales(hadamard_block)
    n_batch = x.shape[0]
    stride_batch = x.stride(0)

    stride_row, stride_col = x.stride(-2), x.stride(-1)

    # BLOCK_COL must cover whole rotation blocks, so never fall below `hadamard_block`.
    block_col = max(hadamard_block, 64)
    block_row = max(2048 // block_col, 8)
    grid = (
        triton.cdiv(n_row, block_row),
        triton.cdiv(n_col, block_col),
        n_batch,
    )
    wrap_triton(_rotate_kernel)[grid](
        x,
        out,
        sign_vector,
        n_row,
        n_col,
        stride_batch,
        stride_row,
        stride_col,
        HADAMARD_BLOCK=hadamard_block,
        LOG2_HADAMARD_BLOCK=hadamard_block.bit_length() - 1,
        PRE_SCALE=pre_scale,
        POST_SCALE=post_scale,
        HAS_SIGNS=sign_vector is not None,
        INVERSE=inverse,
        BLOCK_ROW=block_row,
        BLOCK_COL=block_col,
        num_warps=4,
    )
    return out.reshape(shape)
