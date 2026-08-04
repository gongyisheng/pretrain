from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from src.quant.constants import EPS
from src.quant.utils import (
    is_int8s,
    str_to_emax,
    str_to_qmax,
    str_to_store_dtype,
)


def effective_block_size(granularity: str, block_size: int, K: int) -> int:
    """Elements per scale block along the contraction axis (K for row/tensorwise)."""
    return block_size if granularity == "blockwise" else K


class RaggedScaleBlocks(NamedTuple):
    """Row -> scale-block mapping for an axis that is ragged over groups.

    Rows are sorted by group and cut by the cumulative end-offsets `offs`, so a
    scale block must never span two groups: pooling amax across groups is what
    lets one hot expert set the scale for a cold one. `row_blocks` is int64
    because it indexes torch reductions; `first_block` is int32 to match `offs`
    where the Triton kernel reads it; `n_blocks` is a Python int so the scale
    buffer's shape stays static under torch.compile.
    """

    row_blocks: torch.Tensor  # (n_rows,) int64, scale row for each row
    first_block: torch.Tensor  # (n_groups+1,) int32, group g owns [g], [g+1])
    n_blocks: int  # static upper bound on the scale buffer's height


def ragged_scale_blocks(offs, n_rows, granularity, block_size) -> RaggedScaleBlocks:
    """Scale blocks for a ragged axis, restarting the tiling at each group.

    `tensorwise`/`rowwise` give one block per group. `blockwise` re-tiles
    `block_size` rows inside each group, so group sizes that are not multiples of
    `block_size` cost a short trailing block rather than a straddling one.
    """
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    # right=True: row r belongs to the group whose end-offset first exceeds r.
    # Rows past offs[-1] cannot occur in the MoE dispatch (offs[-1] == n_rows);
    # clamping only keeps the gathers below in bounds if one ever did.
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if granularity != "blockwise":
        return RaggedScaleBlocks(
            group.long(),
            torch.arange(n_groups + 1, device=offs.device, dtype=torch.int32),
            n_groups,
        )
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)]).to(torch.int32)
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    row_blocks = first_block[group].long() + local.long()
    # sum_g ceil(m_g/bs) <= floor(M/bs) + n_groups for every offs, so this bound
    # is static even though the per-group counts are not.
    return RaggedScaleBlocks(row_blocks, first_block, n_rows // block_size + n_groups)


def _amax_to_scale(
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


def _quantize_ragged_axis(x, ragged, granularity, *, fmt, scale_dtype):
    """Quantize (M,C) `x` whose contracted axis M is ragged: one scale row per
    block of `ragged`, so no scale mixes two groups. Returns codes (M,C) and an
    (n_blocks,C) fp32 scale — already the layout the wgrad kernel reads.
    """
    xf = x.float()
    # amax over the rows of each block. include_self with a zeros buffer is safe
    # because the source is abs(): an empty block keeps 0 and clamps to EPS below.
    amax = xf.new_zeros(ragged.n_blocks, xf.shape[1]).index_reduce_(
        0, ragged.row_blocks, xf.abs(), "amax", include_self=True
    )
    if granularity == "tensorwise":
        amax = amax.amax(dim=-1, keepdim=True).expand_as(amax)
    scale = _amax_to_scale(amax, fmt, scale_dtype)
    q = xf / scale.index_select(0, ragged.row_blocks)
    if is_int8s(fmt):
        q = torch.round(q)
    qmax = str_to_qmax(fmt)
    q = q.clamp(-qmax, qmax).to(str_to_store_dtype(fmt))
    return q.contiguous(), scale.contiguous()


def _quantize_last_axis(xf, K, bs, *, fmt, scale_dtype):
    """Blockwise quantize a (..., K) fp32 tensor along the last axis."""
    nkb = (K + bs - 1) // bs
    pad = nkb * bs - K
    xp = F.pad(xf, (0, pad)) if pad else xf
    xb = xp.reshape(*xp.shape[:-1], nkb, bs)  # (..., nkb, bs)
    amax = xb.abs().amax(dim=-1)  # (..., nkb)
    scale = _amax_to_scale(amax, fmt, scale_dtype)
    q = xb / scale.unsqueeze(-1)
    if is_int8s(fmt):
        q = torch.round(q)
    qmax = str_to_qmax(fmt)
    q = q.clamp(-qmax, qmax).to(str_to_store_dtype(fmt))
    xq = q.flatten(-2)[..., :K].contiguous()
    return xq, scale


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
    keeps every scale inside one group. Required when contract_dim == 0; currently
    ignored for contract_dim == -1.
    """
    if ragged is not None and contract_dim == 0:
        return _quantize_ragged_axis(
            x, ragged, granularity, fmt=fmt, scale_dtype=scale_dtype
        )
    xf = x.movedim(contract_dim, -1).float().contiguous()  # (..., K)
    K = xf.shape[-1]

    # tensorwise branch
    if granularity == "tensorwise":
        amax = xf.abs().amax()
        scale = _amax_to_scale(amax, fmt, scale_dtype)  # 0-dim
        q = xf / scale
        if is_int8s(fmt):
            q = torch.round(q)
        qmax = str_to_qmax(fmt)
        q = q.clamp(-qmax, qmax).to(str_to_store_dtype(fmt))
        xq = q.movedim(-1, contract_dim).contiguous()
        # canonical broadcast: A -> (M,1), B -> (1,N)
        rows = xq.shape[0]
        cols = xq.shape[1]
        s = scale.reshape(1, 1)
        scale2d = s.expand(rows, 1) if contract_dim == -1 else s.expand(1, cols)
        return xq, scale2d.contiguous()

    # rowwise/blockwise branch
    bs = effective_block_size(granularity, block_size, K)
    xq_last, scale = _quantize_last_axis(xf, K, bs, fmt=fmt, scale_dtype=scale_dtype)
    xq = xq_last.movedim(-1, contract_dim).contiguous()
    scale = scale.movedim(-1, contract_dim).contiguous()  # A:(M,nkb) B:(nkb,N)
    return xq, scale


def dequantize_operand(
    xq, scale, contract_dim, granularity, block_size, *, ragged=None
):
    """float32 `xq * scale` with contract_dim restored — inverse of `quantize_operand`.

    `granularity`/`block_size` must be the values `quantize_operand` used. The block
    size cannot be recovered from nkb/K, which is ambiguous whenever the last block
    is partial (e.g. K=100, block_size=32 -> nkb=4 but ceil(100/4)=25 != 32).
    `ragged` must likewise be the mapping `quantize_operand` was given; it is only
    consulted when the contracted axis is the ragged one, since a contract_dim==-1
    scale is per row and already inverts without it.
    """
    if ragged is not None and contract_dim == 0:
        return xq.float() * scale.float().index_select(0, ragged.row_blocks)
    qf = xq.movedim(contract_dim, -1).float()  # (..., K)
    K = qf.shape[-1]
    sf = scale.movedim(contract_dim, -1).float()  # (..., nkb)
    nkb = sf.shape[-1]
    bs = effective_block_size(granularity, block_size, K)
    pad = nkb * bs - K
    qp = F.pad(qf, (0, pad)) if pad else qf
    deq = (qp.reshape(*qp.shape[:-1], nkb, bs) * sf.unsqueeze(-1)).flatten(-2)[..., :K]
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
    granularity: str
    block_size: int
    offs: torch.Tensor | None = None  # set by grouped GEMMs, marks an expert axis
