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
    for stage in tl.static_range(LOG2_HADAMARD_BLOCK):
        pairs = tl.reshape(v, (BLOCK_ROW, BLOCK_COL // (2 << stage), 2, 1 << stage))
        even, odd = tl.split(tl.permute(pairs, (0, 1, 3, 2)))
        joined = tl.permute(tl.join(even + odd, even - odd), (0, 1, 3, 2))
        v = tl.reshape(joined, (BLOCK_ROW, BLOCK_COL))
    return v


_ROTATE_CFG = [
    (64, 16, 8),
    (32, 64, 4),
    (32, 64, 8),
    (64, 64, 8),
    (16, 128, 4),
    (16, 128, 8),
    (32, 128, 8),
    (8, 256, 4),
    (4, 512, 4),
    (4, 512, 8),
]
_ROTATE_CONFIGS = [
    triton.Config({"BLOCK_ROW": block_row, "BLOCK_COL": block_col}, num_warps=warps)
    for block_row, block_col, warps in _ROTATE_CFG
]


@triton.autotune(
    configs=_ROTATE_CONFIGS,
    key=["n_row", "n_col", "stride_col", "HADAMARD_BLOCK"],
)
@triton.heuristics(
    values={"BLOCK_COL": lambda meta: max(meta["BLOCK_COL"], meta["HADAMARD_BLOCK"])}
)
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
    off_row = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    off_col = tl.program_id(1) * BLOCK_COL + tl.arange(0, BLOCK_COL)
    batch_idx = tl.program_id(2)
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
    shape = x.shape
    n_row, n_col = (1, x.shape[-1]) if x.ndim == 1 else x.shape[-2:]
    x = x.reshape(-1, n_row, n_col)
    out = torch.empty_like(x, dtype=out_dtype)
    if out.stride() != x.stride():
        x = x.contiguous()
        out = torch.empty_like(x, dtype=out_dtype)
    pre_scale, post_scale = to_hadamard_scales(hadamard_block)
    n_batch = x.shape[0]
    stride_batch = x.stride(0)

    stride_row, stride_col = x.stride(-2), x.stride(-1)

    def grid(meta):
        return (
            triton.cdiv(n_row, meta["BLOCK_ROW"]),
            triton.cdiv(n_col, meta["BLOCK_COL"]),
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
    )
    return out.reshape(shape)
