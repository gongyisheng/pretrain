from __future__ import annotations

import torch
import torch.nn.functional as F

from src.quant.constants import EPS
from src.quant.utils import (
    is_fp8,
    is_int8s,
    str_to_fp8_ulp,
    str_to_qmax,
    str_to_dtype,
)


def _scale_block_map(
    offs: torch.Tensor, n_rows: int, block_size: int
) -> tuple[torch.Tensor, int]:
    """Map rows to blocks that never cross `offs` groups; `n_blocks` is static under torch.compile."""
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    # `right=True` maps each row to its containing end-offset group; clamp keeps gathers in bounds.
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if not block_size:
        return group.long(), n_groups
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)])
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    row_blocks = first_block[group].long() + local.long()
    # `floor(n_rows / block_size) + n_groups` is a static upper bound.
    return row_blocks, n_rows // block_size + n_groups


def _compute_scale(
    amax: torch.Tensor, fmt: str, scale_dtype: torch.dtype
) -> torch.Tensor:
    """Return fp32 or fp8_e8m0 dequantization scales from block maxima."""
    if scale_dtype is torch.float8_e8m0fnu:
        # Clamp E8M0 exponents, preserving its full lower range; `log2(0)` selects 2**-127.
        exp = torch.ceil(torch.log2(amax / str_to_qmax(fmt)))
        # E8M0 cast corrects CUDA `exp2(-127)` being one ULP low.
        return torch.exp2(exp.clamp(-127, 127)).to(scale_dtype).float()
    return (amax / str_to_qmax(fmt)).clamp_min(EPS)


def _compute_codes(
    xf: torch.Tensor, scale: torch.Tensor, fmt: str, stochastic_rounding: bool
) -> torch.Tensor:
    """Scale float32 `xf` by broadcastable float32 `scale` and cast to `fmt`."""
    qmax = str_to_qmax(fmt)
    xq = (xf / scale).clamp(-qmax, qmax)
    if is_int8s(fmt):
        if stochastic_rounding:
            lower = torch.floor(xq)
            xq = lower + (torch.rand_like(xq) < xq - lower).to(xq.dtype)
        else:
            xq = torch.round(xq)
    elif is_fp8(fmt):
        if stochastic_rounding:
            # Construct binade ULPs from fp32 exponents; clamping gives subnormals their shared spacing.
            mantissa_bits, min_ulp_exp = str_to_fp8_ulp(fmt)
            exponent = ((xq.view(torch.int32) >> 23) & 0xFF) - mantissa_bits
            ulp = (exponent.clamp_min(127 + min_ulp_exp) << 23).view(torch.float32)
            # `floor` makes the stochastic interval sign-safe.
            lower = torch.floor(xq / ulp) * ulp
            probability = (xq - lower) / ulp
            xq = torch.where(torch.rand_like(xq) < probability, lower + ulp, lower)
        else:
            # Hardware cast supplies RNE.
            pass
    else:
        raise ValueError(f"Unknown fmt:{fmt}")
    return xq.to(str_to_dtype(fmt))


def _check_tile_dim(dim: int) -> None:
    """Reject dimensions outside the final two axes."""
    if dim not in (-2, -1):
        raise ValueError(f"dim must be -2 or -1, got {dim}")


def _tile(a: torch.Tensor, dim: int, block_size: int) -> torch.Tensor:
    """Pad and split `dim` into fixed-size blocks."""
    # Validate because any non--1 dimension otherwise takes the -2 branch.
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


def _untile(a: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    """Merge tiled blocks and remove padding."""
    _check_tile_dim(dim)
    if dim == -1:
        return a.flatten(-2).narrow(-1, 0, length).contiguous()
    return a.flatten(-3, -2).narrow(-2, 0, length).contiguous()


def _tile2d(a: torch.Tensor, tile: int) -> torch.Tensor:
    """Pad and split both final dimensions into square tiles."""
    rows, cols = a.shape[-2:]
    pad_rows, pad_cols = (-rows) % tile, (-cols) % tile
    if pad_rows or pad_cols:
        a = F.pad(a, (0, pad_cols, 0, pad_rows))
    return a.reshape(
        *a.shape[:-2], a.shape[-2] // tile, tile, a.shape[-1] // tile, tile
    )


def _untile2d(v: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Merge square tiles and remove padding."""
    a = v.reshape(*v.shape[:-4], v.shape[-4] * v.shape[-3], v.shape[-2] * v.shape[-1])
    return a[..., :rows, :cols].contiguous()


def _tile_amax(a: torch.Tensor, dim: int, block_size: int) -> torch.Tensor:
    """Return dense block maxima along `dim`."""
    return _tile(a, dim, block_size).amax(dim=-1 if dim == -1 else -2)


def _segment_amax(
    a: torch.Tensor, dim: int, row_blocks: torch.Tensor, n_blocks: int
) -> torch.Tensor:
    """Return segmented block maxima along `dim`."""
    # Empty absolute-value blocks remain zero before `_compute_scale` clamps to EPS.
    shape = list(a.shape)
    shape[dim] = n_blocks
    return a.new_zeros(shape).index_reduce_(
        dim % a.ndim, row_blocks, a, "amax", include_self=True
    )


def _check_dims(
    x: torch.Tensor,
    contract_dim: int,
    ragged_dim: int | None,
    offs: torch.Tensor | None,
) -> None:
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
    xf: torch.Tensor,
    contract_dim: int,
    block_size: int,
    fmt: str,
    scale_dtype: torch.dtype,
    offs: torch.Tensor,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ragged contraction without pooling maxima across groups."""
    row_blocks, n_blocks = _scale_block_map(offs, xf.shape[contract_dim], block_size)
    amax = _segment_amax(xf.abs(), contract_dim, row_blocks, n_blocks)
    scale = _compute_scale(amax, fmt, scale_dtype)
    return (
        _compute_codes(
            xf,
            scale.index_select(contract_dim, row_blocks),
            fmt,
            stochastic_rounding,
        ),
        scale,
    )


def _quantize_tensorwise(
    xf: torch.Tensor,
    contract_dim: int,
    fmt: str,
    scale_dtype: torch.dtype,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one scale per tensor or ragged group."""
    outer_dim = -1 if contract_dim == -2 else -2
    if offs is None:
        # This reduction order is Inductor-sensitive.
        amax = xf.abs().amax((-2, -1), keepdim=True)
        row_blocks = None
    else:
        # Reduce the dense axis before segmented `index_reduce_`.
        dense_dim = -1 if ragged_dim == -2 else -2
        row_blocks, n_blocks = _scale_block_map(offs, xf.shape[ragged_dim], 0)
        amax = _segment_amax(
            xf.abs().amax(dense_dim, keepdim=True), ragged_dim, row_blocks, n_blocks
        )
    scale = _compute_scale(amax, fmt, scale_dtype)

    div = scale if row_blocks is None else scale.index_select(ragged_dim, row_blocks)
    codes = _compute_codes(xf, div, fmt, stochastic_rounding)

    # Returned scales expand the outer axis and remain blockwise on the contraction axis.
    if ragged_dim == outer_dim:
        return codes, div
    shape = list(scale.shape)
    shape[outer_dim] = xf.shape[outer_dim]
    return codes, scale.expand(shape)


def _quantize_rowwise(
    xf: torch.Tensor,
    contract_dim: int,
    fmt: str,
    scale_dtype: torch.dtype,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one scale across each contraction row."""
    if ragged_dim == contract_dim:
        return _quantize_segmented_contraction(
            xf, contract_dim, 0, fmt, scale_dtype, offs, stochastic_rounding
        )
    scale = _compute_scale(xf.abs().amax(contract_dim, keepdim=True), fmt, scale_dtype)
    return _compute_codes(xf, scale, fmt, stochastic_rounding), scale


def _quantize_blockwise_1d(
    xf: torch.Tensor,
    contract_dim: int,
    block_size: int,
    fmt: str,
    scale_dtype: torch.dtype,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with one scale per contraction-axis block."""
    if ragged_dim == contract_dim:
        return _quantize_segmented_contraction(
            xf, contract_dim, block_size, fmt, scale_dtype, offs, stochastic_rounding
        )
    scale = _compute_scale(
        _tile_amax(xf.abs(), contract_dim, block_size), fmt, scale_dtype
    )
    tile_axis = -1 if contract_dim == -1 else -2
    codes = _untile(
        _compute_codes(
            _tile(xf, contract_dim, block_size),
            scale.unsqueeze(tile_axis),
            fmt,
            stochastic_rounding,
        ),
        contract_dim,
        xf.shape[contract_dim],
    )
    return codes, scale


def _axis_amax(
    a: torch.Tensor,
    dim: int,
    tile: int,
    block_map: tuple[torch.Tensor, int] | None,
) -> torch.Tensor:
    """Return segmented or dense block maxima along `dim`."""
    if block_map is None:
        return _tile_amax(a, dim, tile)
    row_blocks, n_blocks = block_map
    return _segment_amax(a, dim, row_blocks, n_blocks)


def _axis_expand(
    scale: torch.Tensor,
    dim: int,
    tile: int,
    length: int,
    block_map: tuple[torch.Tensor, int] | None,
) -> torch.Tensor:
    """Expand block scales by gather on ragged axes or repetition on dense axes."""
    if block_map is None:
        return scale.repeat_interleave(tile, dim).narrow(dim, 0, length)
    return scale.index_select(dim, block_map[0])


def _quantize_blockwise_2d(
    xf: torch.Tensor,
    contract_dim: int,
    block_size: int,
    fmt: str,
    scale_dtype: torch.dtype,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize with square blocks that stay within ragged groups."""
    outer_dim = -1 if contract_dim == -2 else -2
    if offs is None:
        # Dense 2D avoids the ragged path's slower materialized divisor.
        rows, cols = xf.shape[-2:]
        tiled = _tile2d(xf, block_size)
        blocks = _compute_scale(tiled.abs().amax((-3, -1)), fmt, scale_dtype)
        codes = _untile2d(
            _compute_codes(
                tiled, blocks[..., :, None, :, None], fmt, stochastic_rounding
            ),
            rows,
            cols,
        )
        return codes, _axis_expand(
            blocks, outer_dim, block_size, rows if outer_dim == -2 else cols, None
        )

    # Ragged 2D uses a block map and gathered divisor instead of reshape/broadcast.
    ragged_map = _scale_block_map(offs, xf.shape[ragged_dim], block_size)
    dense_dim = contract_dim if ragged_dim == outer_dim else outer_dim
    amax = _axis_amax(
        _axis_amax(xf.abs(), dense_dim, block_size, None),
        ragged_dim,
        block_size,
        ragged_map,
    )
    blocks = _compute_scale(amax, fmt, scale_dtype)
    maps = {ragged_dim: ragged_map, dense_dim: None}
    scale = _axis_expand(
        blocks, outer_dim, block_size, xf.shape[outer_dim], maps[outer_dim]
    )
    div = _axis_expand(
        scale, contract_dim, block_size, xf.shape[contract_dim], maps[contract_dim]
    )
    return _compute_codes(xf, div, fmt, stochastic_rounding), scale


def quantize_operand(
    x: torch.Tensor,
    contract_dim: int,
    fmt: str,
    scale_cfg: dict,
    offs: torch.Tensor | None = None,
    ragged_dim: int | None = None,
    stochastic_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize `x` with scales along `contract_dim`.

    `contract_dim` and `ragged_dim` must be -2 or -1. When given, `offs` keeps
    ragged-axis scale blocks within groups; 2D blockwise quantization uses either
    ragged axis. Returns codes in `fmt` and fp32 or E8M0 scales with the outer axis
    expanded and the contraction axis blockwise.
    """
    _check_dims(x, contract_dim, ragged_dim, offs)
    granularity = scale_cfg["granularity"]
    block_outer, block_size = scale_cfg["block_shape"]
    scale_dtype = scale_cfg["scale_dtype"]
    xf = x.float()
    if granularity == "tensorwise":
        codes, scale = _quantize_tensorwise(
            xf, contract_dim, fmt, scale_dtype, offs, ragged_dim, stochastic_rounding
        )
    elif granularity == "rowwise":
        codes, scale = _quantize_rowwise(
            xf, contract_dim, fmt, scale_dtype, offs, ragged_dim, stochastic_rounding
        )
    elif granularity == "blockwise":
        if block_outer > 1:
            codes, scale = _quantize_blockwise_2d(
                xf,
                contract_dim,
                block_size,
                fmt,
                scale_dtype,
                offs,
                ragged_dim,
                stochastic_rounding,
            )
        else:
            codes, scale = _quantize_blockwise_1d(
                xf,
                contract_dim,
                block_size,
                fmt,
                scale_dtype,
                offs,
                ragged_dim,
                stochastic_rounding,
            )
    else:
        raise ValueError(f"unknown granularity: {granularity!r}")
    return codes.contiguous(), scale.contiguous()


def dequantize_operand(
    xq: torch.Tensor,
    scale: torch.Tensor,
    contract_dim: int,
    scale_cfg: dict,
    offs: torch.Tensor | None = None,
    ragged_dim: int | None = None,
) -> torch.Tensor:
    """Dequantize `xq` in fp32 using `quantize_operand`'s scale layout."""
    _check_dims(xq, contract_dim, ragged_dim, offs)
    block_size = scale_cfg["block_shape"][1]
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
