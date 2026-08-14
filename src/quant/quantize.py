from __future__ import annotations

import torch
import torch.nn.functional as F

from src.quant.constants import EPS
from src.quant.utils import (
    is_int8s,
    str_to_qmax,
    str_to_dtype,
)


def _scale_block_map(offs, n_rows, block_size) -> tuple[torch.Tensor, int]:
    """Map each row to its scale block for an axis that is ragged over groups.

    Rows are sorted by group and cut by the cumulative end-offsets `offs`, so a
    scale block must never span two groups: pooling amax across groups is what
    lets one hot expert set the scale for a cold one. `block_size` 0
    (row/tensorwise) gives one block per group. A positive one (blockwise)
    re-tiles `block_size` rows inside each group, so group sizes that are not
    multiples of `block_size` cost a short trailing block rather than a
    straddling one.

    Returns `(row_blocks, n_blocks)`: `row_blocks` is an int64 `(n_rows,)` tensor
    indexing torch reductions; `n_blocks` is a Python int so the scale buffer's
    shape stays static under torch.compile.
    """
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    # right=True: row r belongs to the group whose end-offset first exceeds r.
    # Rows past offs[-1] cannot occur in the MoE dispatch (offs[-1] == n_rows);
    # clamping only keeps the gathers below in bounds if one ever did.
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if not block_size:
        return group.long(), n_groups
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)])
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    row_blocks = first_block[group].long() + local.long()
    # sum_g ceil(m_g/bs) <= floor(M/bs) + n_groups for every offs, so this bound
    # is static even though the per-group counts are not.
    return row_blocks, n_rows // block_size + n_groups


def _compute_scale(
    amax: torch.Tensor, fmt: str, scale_dtype: str | None
) -> torch.Tensor:
    """fp32 dequant scale from per-block amax."""
    if scale_dtype == "fp8_e8m0":
        # e8m0 is a bare 8-bit exponent: a power of two, biased to cover +-127.
        exp = torch.ceil(torch.log2(amax.clamp_min(EPS) / str_to_qmax(fmt)))
        return torch.exp2(exp.clamp(-127, 127))
    return (amax / str_to_qmax(fmt)).clamp_min(EPS)


def _compute_codes(xf: torch.Tensor, scale: torch.Tensor, fmt: str) -> torch.Tensor:
    """Scale into the format's range and cast — the tail every quantizer shares.

    `scale` must broadcast against `xf`; both are float32.
    """
    xq = xf / scale
    if is_int8s(fmt):
        xq = torch.round(xq)
    qmax = str_to_qmax(fmt)
    return xq.clamp(-qmax, qmax).to(str_to_dtype(fmt))


def _check_tile_dim(dim):
    """The tiling helpers split one of the last two dims; anything else is a bug."""
    if dim not in (-2, -1):
        raise ValueError(f"dim must be -2 or -1, got {dim}")


def _tile(a, dim, block_size):
    """View `dim` as (n_blocks, block_size), zero-padding it to a multiple.

    The tile axis lands at -1 for `dim == -1` and -2 for `dim == -2`. Both are free
    views on contiguous input -- splitting a dim never transposes.
    """
    # Both entry points validate their dim up front; this stays because the -2 branch
    # would otherwise silently claim an out-of-domain dim and mis-tile at the right
    # shape.
    _check_tile_dim(dim)
    length = a.shape[dim]
    n_blocks = (length + block_size - 1) // block_size
    pad = n_blocks * block_size - length
    if dim == -1:
        if pad:
            a = F.pad(a, (0, pad))
        return a.reshape(*a.shape[:-1], n_blocks, block_size)
    if pad:
        a = F.pad(a, (0, 0, 0, pad))
    return a.reshape(*a.shape[:-2], n_blocks, block_size, a.shape[-1])


def _untile(a, dim, length):
    """Inverse of `_tile`: fold the tile axis back in and drop the padding."""
    _check_tile_dim(dim)
    if dim == -1:
        return a.flatten(-2).narrow(-1, 0, length).contiguous()
    return a.flatten(-3, -2).narrow(-2, 0, length).contiguous()


def _tile2d(a, tile):
    """View the last two dims as (n_rows, tile, n_cols, tile), zero-padding both.

    A square tile needs no `contract_dim` branch -- the reshape is identical whichever
    of the last two dims is the contraction. Only the scale's expansion back to the 1D
    layout, in `_quantize_blockwise_2d`, is orientation-dependent.
    """
    rows, cols = a.shape[-2:]
    pad_rows, pad_cols = (-rows) % tile, (-cols) % tile
    if pad_rows or pad_cols:
        a = F.pad(a, (0, pad_cols, 0, pad_rows))
    return a.reshape(
        *a.shape[:-2], a.shape[-2] // tile, tile, a.shape[-1] // tile, tile
    )


def _untile2d(v, rows, cols):
    """Inverse of `_tile2d`: fold both tile axes back in and drop the padding."""
    a = v.reshape(*v.shape[:-4], v.shape[-4] * v.shape[-3], v.shape[-2] * v.shape[-1])
    return a[..., :rows, :cols].contiguous()


def _tile_amax(a, dim, block_size):
    """Dense per-block amax along `dim`, giving `n_blocks` there."""
    return _tile(a, dim, block_size).amax(dim=-1 if dim == -1 else -2)


def _segment_amax(a, dim, row_blocks, n_blocks):
    """Per-block amax along a segmented `dim`, giving `n_blocks` there.

    include_self with a zeros buffer is safe because `a` is abs(): an empty block
    keeps 0 and clamps to EPS in _compute_scale.
    """
    shape = list(a.shape)
    shape[dim] = n_blocks
    return a.new_zeros(shape).index_reduce_(
        dim % a.ndim, row_blocks, a, "amax", include_self=True
    )


def _check_dims(x, contract_dim, ragged_dim, offs):
    if contract_dim not in (-2, -1):
        raise ValueError(f"contract_dim must be -2 or -1, got {contract_dim}")
    if (offs is None) != (ragged_dim is None):
        raise ValueError("offs and ragged_dim must be given together")
    if ragged_dim is None:
        return
    if ragged_dim not in (-2, -1):
        raise ValueError(f"ragged_dim must be -2 or -1, got {ragged_dim}")
    if x.ndim != 2:
        raise ValueError(f"a ragged axis needs a 2D operand, got {x.ndim}D")


def _quantize_segmented_contraction(
    xf, contract_dim, block_size, fmt, scale_dtype, offs
):
    """The contraction is ragged: one scale block per group (or per `block_size` rows
    inside a group), so no scale pools amax across two experts.

    Shared verbatim by rowwise and blockwise -- `block_size` 0 vs N is the only
    difference between them here. The ragged map already gives one scale entry per row,
    so the operand is never tiled: reshaping it would only undo itself.
    """
    row_blocks, n_blocks = _scale_block_map(offs, xf.shape[contract_dim], block_size)
    amax = _segment_amax(xf.abs(), contract_dim, row_blocks, n_blocks)
    scale = _compute_scale(amax, fmt, scale_dtype)
    return _compute_codes(xf, scale.index_select(contract_dim, row_blocks), fmt), scale


def _quantize_tensorwise(xf, contract_dim, fmt, scale_dtype, offs, ragged_dim):
    """One scale per tensor -- the only granularity that pools the outer axis.

    That is also why the scale-layout restore lives only here: every other granularity
    leaves the outer axis at full length already.
    """
    outer_dim = -1 if contract_dim == -2 else -2
    if offs is None:
        # (-2, -1), never (-1, -2): inductor compiles the two orders to markedly
        # different kernels, and this is the hot path.
        amax = xf.abs().amax((-2, -1), keepdim=True)
        row_blocks = None
    else:
        # Collapse the axis that is not ragged first, then segment the ragged one, so
        # the index_reduce_ runs over a single row/column instead of the full tensor.
        dense_dim = -1 if ragged_dim == -2 else -2
        row_blocks, n_blocks = _scale_block_map(offs, xf.shape[ragged_dim], 0)
        amax = _segment_amax(
            xf.abs().amax(dense_dim, keepdim=True), ragged_dim, row_blocks, n_blocks
        )
    scale = _compute_scale(amax, fmt, scale_dtype)

    # one selection, on whichever axis is ragged
    div = scale if row_blocks is None else scale.index_select(ragged_dim, row_blocks)
    codes = _compute_codes(xf, div, fmt)

    # the invariant: outer restored to full length, contract left at n_blocks
    if ragged_dim == outer_dim:
        return codes, div  # the gather already restored the outer axis
    shape = list(scale.shape)
    shape[outer_dim] = xf.shape[outer_dim]
    return codes, scale.expand(shape)


def _quantize_rowwise(xf, contract_dim, fmt, scale_dtype, offs, ragged_dim):
    """One scale per row: a single block spans the whole contraction."""
    if ragged_dim == contract_dim:
        return _quantize_segmented_contraction(
            xf, contract_dim, 0, fmt, scale_dtype, offs
        )
    # a ragged outer axis cannot change a per-row scale, so it is ignored
    scale = _compute_scale(xf.abs().amax(contract_dim, keepdim=True), fmt, scale_dtype)
    return _compute_codes(xf, scale, fmt), scale


def _quantize_blockwise_1d(
    xf, contract_dim, block_size, fmt, scale_dtype, offs, ragged_dim
):
    """One scale per `block_size` elements along the contraction (a 1 x B block)."""
    if ragged_dim == contract_dim:
        return _quantize_segmented_contraction(
            xf, contract_dim, block_size, fmt, scale_dtype, offs
        )
    # a ragged outer axis cannot change a within-row scale, so it is ignored
    scale = _compute_scale(
        _tile_amax(xf.abs(), contract_dim, block_size), fmt, scale_dtype
    )
    tile_axis = -1 if contract_dim == -1 else -2
    codes = _untile(
        _compute_codes(
            _tile(xf, contract_dim, block_size), scale.unsqueeze(tile_axis), fmt
        ),
        contract_dim,
        xf.shape[contract_dim],
    )
    return codes, scale


def _quantize_blockwise_2d(xf, contract_dim, tile, fmt, scale_dtype):
    """One scale per `tile` x `tile` square, pooling amax over both axes.

    The scale is expanded along the outer axis before returning, so it leaves in the
    layout a 1 x `tile` blockwise scale would -- outer at full length, contract at
    n_blocks. That is what lets every kernel, `dequantize_operand` and the metrics fold
    stay identical to the 1D path: a square tile shows up only as neighbouring rows
    sharing a value, which nothing downstream can observe.
    """
    rows, cols = xf.shape[-2:]
    tiled = _tile2d(xf, tile)
    scale = _compute_scale(tiled.abs().amax((-3, -1)), fmt, scale_dtype)
    codes = _untile2d(
        _compute_codes(tiled, scale[..., :, None, :, None], fmt), rows, cols
    )
    # the outer axis is whichever of the last two is not the contraction
    outer_dim = -2 if contract_dim == -1 else -1
    length = rows if outer_dim == -2 else cols
    scale = scale.repeat_interleave(tile, outer_dim).narrow(outer_dim, 0, length)
    return codes, scale


def quantize_operand(x, contract_dim, fmt, scaling, offs=None, ragged_dim=None):
    """Quantize `x` into scale groups tiled along `contract_dim`.

    `contract_dim` and `ragged_dim` each name one of the last two dims (-2 or -1), so a
    2D weight and a 3D expert stack are the same call; leading dims are batch and are
    never pooled into a scale group. `offs` marks `ragged_dim` as ragged over groups and
    keeps every scale inside one group -- which only changes the result where a scale
    tile spans more than one element along that axis, so rowwise and blockwise ignore a
    ragged axis that is not the contraction.

    Returns codes in `fmt`'s dtype and an fp32 dequant scale whose outer axis is at full
    length and whose contract axis is the number of scale blocks -- the layout
    `scaled_gemm` reads.

    A 2D (square) `block_shape` pools amax over a `T x T` tile and then expands the
    scale along the outer axis, so the returned layout is the same either way -- the
    square tile is a quantization policy, not a second scale layout.
    """
    _check_dims(x, contract_dim, ragged_dim, offs)
    granularity = scaling["granularity"]
    block_outer, block_size = scaling["block_shape"]
    scale_dtype = scaling.get("scale_dtype")
    xf = x.float()
    if granularity == "tensorwise":
        codes, scale = _quantize_tensorwise(
            xf, contract_dim, fmt, scale_dtype, offs, ragged_dim
        )
    elif granularity == "rowwise":
        codes, scale = _quantize_rowwise(
            xf, contract_dim, fmt, scale_dtype, offs, ragged_dim
        )
    elif granularity == "blockwise":
        if block_outer > 1:
            if offs is not None:
                raise NotImplementedError(
                    f"2D block_shape {(block_outer, block_size)} with a ragged axis "
                    "is not supported; a square tile can straddle two groups on the "
                    "ragged axis, which is what per-group scales exist to prevent"
                )
            codes, scale = _quantize_blockwise_2d(
                xf, contract_dim, block_outer, fmt, scale_dtype
            )
        else:
            codes, scale = _quantize_blockwise_1d(
                xf, contract_dim, block_size, fmt, scale_dtype, offs, ragged_dim
            )
    else:
        raise ValueError(f"unknown granularity: {granularity!r}")
    return codes.contiguous(), scale.contiguous()


def dequantize_operand(xq, scale, contract_dim, scaling, offs=None, ragged_dim=None):
    """Invert `quantize_operand` in fp32, given the layout it was called with."""
    _check_dims(xq, contract_dim, ragged_dim, offs)
    block_size = scaling["block_shape"][1]
    qf, sf = xq.float(), scale.float()
    if ragged_dim == contract_dim and offs is not None:
        row_blocks, _ = _scale_block_map(offs, xq.shape[contract_dim], block_size)
        return (qf * sf.index_select(contract_dim, row_blocks)).contiguous()
    length = xq.shape[contract_dim]
    return _untile(
        _tile(qf, contract_dim, block_size or length)
        * sf.unsqueeze(-1 if contract_dim == -1 else -2),
        contract_dim,
        length,
    )
