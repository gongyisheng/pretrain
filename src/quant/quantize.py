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


def _compute_ragged_scale(src, ragged, fmt, scale_dtype):
    """Scale from a per-block amax of the (M,C') rows of `src`, giving (n_blocks,C').

    include_self with a zeros buffer is safe because `src` is abs(): an empty block
    keeps 0 and clamps to EPS in _compute_scale.
    """
    amax = src.new_zeros(ragged.n_blocks, src.shape[1]).index_reduce_(
        0, ragged.row_blocks, src, "amax", include_self=True
    )
    return _compute_scale(amax, fmt, scale_dtype)


def _to_row_scale(scale, ragged):
    """Per-block scale -> per-row, so it broadcasts against the (M,C) operand."""
    return scale.index_select(0, ragged.row_blocks)


def _quantize_ragged_tensorwise(x, ragged, contract_dim, *, fmt, scale_dtype):
    """Quantize (M,C) `x` with one amax per group of `ragged` instead of one for the
    whole tensor, so no scale mixes two groups. `contract_dim` picks the layout the
    kernel reads: (n_blocks,C) along the contracted axis, else (M,1) per row.
    """
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True)  # (M,1)
    scale = _compute_ragged_scale(amax, ragged, fmt, scale_dtype)  # (n_blocks,1)
    row_scale = _to_row_scale(scale, ragged)  # (M,1)
    xq = _compute_codes(xf, row_scale, fmt)
    out_scale = scale.expand(-1, xf.shape[1]) if contract_dim == 0 else row_scale
    return xq.contiguous(), out_scale.contiguous()


def _quantize_ragged_blockwise(x, ragged, *, fmt, scale_dtype):
    """Quantize (M,C) `x` whose contracted axis M is ragged: a per-column amax within
    each scale block of `ragged`, so no scale mixes two groups. Returns codes (M,C)
    and an (n_blocks,C) fp32 scale — already the layout the wgrad kernel reads.
    """
    xf = x.float()
    scale = _compute_ragged_scale(xf.abs(), ragged, fmt, scale_dtype)  # (n_blocks,C)
    xq = _compute_codes(xf, _to_row_scale(scale, ragged), fmt)
    return xq.contiguous(), scale.contiguous()


def _quantize_tensorwise(xf, fmt, scale_dtype):
    scale = _compute_scale(xf.abs().amax(), fmt, scale_dtype)  # 0-dim
    return _compute_codes(xf, scale, fmt), scale


def _quantize_blockwise(xf, fmt, scale_dtype, block_size):
    K = xf.shape[-1]
    n_blocks = (K + block_size - 1) // block_size
    pad = n_blocks * block_size - K
    xp = F.pad(xf, (0, pad)) if pad else xf
    xb = xp.reshape(*xp.shape[:-1], n_blocks, block_size)  # (..., n_blocks, block_size)
    scale = _compute_scale(xb.abs().amax(dim=-1), fmt, scale_dtype)  # (..., n_blocks)
    xq = _compute_codes(xb, scale.unsqueeze(-1), fmt)  # (..., n_blocks, block_size)
    return xq.flatten(-2)[..., :K].contiguous(), scale  # (..., K)


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

    `ragged` (a RaggedScaleBlocks) marks the row axis as ragged over groups and
    keeps every scale inside one group. Required when the contracted axis is ragged
    (contract_dim == 0); for contract_dim == -1 it changes tensorwise only, while
    rowwise/blockwise are already per-row and ignore it.
    """
    if ragged is not None and granularity == "tensorwise":
        # Checked before contract_dim: for contract_dim == -1 only tensorwise has an
        # amax wide enough to span groups (rowwise/blockwise are already per row).
        return _quantize_ragged_tensorwise(
            x, ragged, contract_dim, fmt=fmt, scale_dtype=scale_dtype
        )

    if ragged is not None and contract_dim == 0:
        return _quantize_ragged_blockwise(x, ragged, fmt=fmt, scale_dtype=scale_dtype)

    if granularity == "tensorwise":
        # tensorwise branch
        xq, scale = _quantize_tensorwise(x.float(), fmt, scale_dtype)
        rows, cols = xq.shape
        s = scale.reshape(1, 1)
        scale2d = s.expand(rows, 1) if contract_dim == -1 else s.expand(1, cols)

        # scale shape:
        # if contract_dim == -1, (rows, 1)
        # if contract_dim == 0,  (1, cols)
        return xq.contiguous(), scale2d.contiguous()

    else:
        # rowwise/blockwise branch
        xf = x.movedim(contract_dim, -1).float().contiguous()  # (..., K)
        xq, scale = _quantize_blockwise(
            xf, fmt, scale_dtype, block_size or xf.shape[-1]
        )
        xq = xq.movedim(-1, contract_dim)
        scale = scale.movedim(-1, contract_dim)

        # scale shape:
        # if contract_dim == -1, (M, n_blocks)
        # if contract_dim == 0,  (n_blocks, N)
        return xq.contiguous(), scale.contiguous()


def dequantize_operand(xq, scale, contract_dim, block_size, ragged=None):
    if ragged is not None and contract_dim == 0:
        return xq.float() * _to_row_scale(scale.float(), ragged)
    qf = xq.movedim(contract_dim, -1).float()  # (..., K)
    K = qf.shape[-1]
    sf = scale.movedim(contract_dim, -1).float()  # (..., n_block)
    n_block = sf.shape[-1]
    _block_size = block_size or K  # 0: the one block spans the whole contraction
    pad = n_block * _block_size - K
    qp = F.pad(qf, (0, pad)) if pad else qf
    deq = (qp.reshape(*qp.shape[:-1], n_block, _block_size) * sf.unsqueeze(-1)).flatten(
        -2
    )[..., :K]
    return deq.movedim(-1, contract_dim).contiguous()


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
