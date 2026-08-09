from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from src.quant.constants import EPS
from src.quant.utils import (
    is_int8s,
    str_to_emax,
    str_to_qmax,
    str_to_dtype,
)


class RaggedScaleBlocks(NamedTuple):
    """Row -> scale-block mapping for an axis that is ragged over groups.

    Rows are sorted by group and cut by the cumulative end-offsets `offs`, so a
    scale block must never span two groups: pooling amax across groups is what
    lets one hot expert set the scale for a cold one. `row_blocks` is int64
    because it indexes torch reductions; `n_blocks` is a Python int so the scale
    buffer's shape stays static under torch.compile.
    """

    row_blocks: torch.Tensor  # (n_rows,) int64, scale row for each row
    n_blocks: int  # static upper bound on the scale buffer's height


def ragged_scale_blocks(offs, n_rows, block_size) -> RaggedScaleBlocks:
    """Scale blocks for a ragged axis, restarting the tiling at each group.

    `block_size` 0 (row/tensorwise) gives one block per group. A positive one
    (blockwise) re-tiles `block_size` rows inside each group, so group sizes that are
    not multiples of `block_size` cost a short trailing block rather than a
    straddling one.
    """
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    # right=True: row r belongs to the group whose end-offset first exceeds r.
    # Rows past offs[-1] cannot occur in the MoE dispatch (offs[-1] == n_rows);
    # clamping only keeps the gathers below in bounds if one ever did.
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if not block_size:
        return RaggedScaleBlocks(group.long(), n_groups)
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)])
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    row_blocks = first_block[group].long() + local.long()
    # sum_g ceil(m_g/bs) <= floor(M/bs) + n_groups for every offs, so this bound
    # is static even though the per-group counts are not.
    return RaggedScaleBlocks(row_blocks, n_rows // block_size + n_groups)


def _compute_scale(
    amax: torch.Tensor, fmt: str, scale_dtype: str | None
) -> torch.Tensor:
    """fp32 dequant scale from per-block amax."""
    if scale_dtype == "fp8_e8m0":
        e8m0_emax = str_to_emax("fp8_e8m0")
        exp = (torch.floor(torch.log2(amax.clamp_min(EPS))) - str_to_emax(fmt)).clamp(
            -e8m0_emax, e8m0_emax
        )
        return torch.exp2(exp)
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
    # `dim` is rank-relative, so a caller normalizing an absolute contraction axis can
    # land outside the last two dims. Without this the -2 branch would silently claim
    # it and mis-tile at the right shape.
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


def _tile_amax(a, dim, block_size):
    """Dense per-block amax along `dim`, giving `n_blocks` there."""
    return _tile(a, dim, block_size).amax(dim=-1 if dim == -1 else -2)


def _segment_amax(a, dim, blocks):
    """Per-block amax along a segmented `dim`, giving `blocks.n_blocks` there.

    include_self with a zeros buffer is safe because `a` is abs(): an empty block
    keeps 0 and clamps to EPS in _compute_scale.
    """
    shape = list(a.shape)
    shape[dim] = blocks.n_blocks
    return a.new_zeros(shape).index_reduce_(
        dim % a.ndim, blocks.row_blocks, a, "amax", include_self=True
    )


def _quantize(
    x, contract_dim, fmt, granularity, block_size, scale_dtype, blocks, ragged_dim
):
    """Tile x into scale groups, amax each, divide. One path for every case.

    `contract_dim` and `ragged_dim` each name one of the last two dims; leading dims
    are batch and are never pooled into a scale group. Segmenting an axis only matters
    where a tile spans more than one element along it -- which is why rowwise and
    blockwise ignore a ragged axis that is not the contraction.
    """
    xf = x.float()
    outer_dim = -1 if contract_dim == -2 else -2
    width = block_size or x.shape[contract_dim]
    # only tensorwise pools the outer axis into a tile, so only there can a ragged
    # outer axis fold it down to one row per group -- rowwise/blockwise leave it at
    # full length and must not be re-indexed.
    outer_segmented = granularity == "tensorwise" and ragged_dim == outer_dim

    amax = xf.abs()
    if granularity == "tensorwise":
        amax = (
            _segment_amax(amax, outer_dim, blocks)
            if outer_segmented
            else amax.amax(outer_dim, keepdim=True)
        )
    if ragged_dim == contract_dim:
        amax = _segment_amax(amax, contract_dim, blocks)
    else:
        amax = _tile_amax(amax, contract_dim, width)
    scale = _compute_scale(amax, fmt, scale_dtype)

    if ragged_dim == contract_dim:
        codes = _compute_codes(
            xf, scale.index_select(contract_dim, blocks.row_blocks), fmt
        )
    else:
        div = scale
        if outer_segmented:
            div = div.index_select(outer_dim, blocks.row_blocks)
        tile_axis = -1 if contract_dim == -1 else -2
        codes = _untile(
            _compute_codes(
                _tile(xf, contract_dim, width), div.unsqueeze(tile_axis), fmt
            ),
            contract_dim,
            x.shape[contract_dim],
        )

    # the invariant: outer restored to full length, contract left at n_blocks
    if granularity == "tensorwise":
        if outer_segmented:
            scale = scale.index_select(outer_dim, blocks.row_blocks)
        else:
            shape = list(scale.shape)
            shape[outer_dim] = x.shape[outer_dim]
            scale = scale.expand(shape)
    return codes.contiguous(), scale.contiguous()


def quantize_operand(
    x,
    contract_dim,
    granularity,
    block_size,
    fmt,
    *,
    scale_dtype=None,
    ragged=None,
):
    """Quantize 2D operand x for scaled_gemm along contract_dim (-1: A, 0: B).

    `ragged` (a RaggedScaleBlocks) marks the row axis as ragged over groups and keeps
    every scale inside one group.
    """
    # this signature's ragged axis is always the row axis, i.e. -2: the contraction
    # when contract_dim is 0, the outer axis otherwise
    return _quantize(
        x,
        -2 if contract_dim == 0 else -1,
        fmt,
        granularity,
        block_size,
        scale_dtype,
        ragged,
        None if ragged is None else -2,
    )


def dequantize_operand(xq, scale, contract_dim, block_size, ragged=None):
    # contract_dim is positive for the stacked-expert layouts (dim 1 of an (E,K,N)
    # slab), so normalize onto the {-2, -1} the tiling helpers take.
    dim = contract_dim if contract_dim < 0 else contract_dim - xq.ndim
    qf, sf = xq.float(), scale.float()
    if ragged is not None and contract_dim == 0:
        return (qf * sf.index_select(dim, ragged.row_blocks)).contiguous()
    length = xq.shape[dim]
    width = block_size or length  # 0: the one block spans the whole contraction
    return _untile(
        _tile(qf, dim, width) * sf.unsqueeze(-1 if dim == -1 else -2), dim, length
    )


class QuantizationSnapshot(NamedTuple):
    """What one GEMM operand was quantized to, recorded for metrics.

    Holds references to live tensors, not copies — read it during the GEMM that
    produced it, not later. Carrying the layout alongside the data is what lets
    `dequantize_operand` invert it without re-supplying (and possibly transposing)
    the contraction axis.
    """

    source_tensor: torch.Tensor  # pre-quantization, in the compute dtype
    quantized_tensor: torch.Tensor  # low-precision codes
    scale: torch.Tensor  # fp32 dequant scale
    contract_dim: int
    block_size: int  # 0: one scale block spans the whole contraction
    offs: torch.Tensor | None = None  # set by grouped GEMMs, marks an expert axis
    fmt: str = ""  # the element format the codes are in, for qmax
