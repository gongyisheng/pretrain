import torch
import torch.nn.functional as F

from src.kernel.ops.quantize import pack_e2m1_rne
from src.quant.constants import EPS, _FP4_E2M1_VALUES
from src.quant.rotation import Rotation
from src.quant.utils import (
    is_fp4,
    is_fp8,
    is_int8s,
    str_to_fp8_ulp,
    str_to_qmax,
    str_to_qmin,
    str_to_dtype,
)


def _e2m1_stochastic_codes(xq: torch.Tensor) -> torch.Tensor:
    """Encode scaled values into unpacked signed E2M1 nibbles."""
    magnitude = torch.nan_to_num(xq.abs(), nan=6.0, posinf=6.0).clamp(max=6.0)
    random_bits = torch.randint(
        0, 256, magnitude.shape, dtype=torch.int32, device=magnitude.device
    )
    unit_random = random_bits.to(magnitude.dtype) * (1.0 / 256.0)
    step = torch.where(
        magnitude >= 4.0,
        2.0,
        torch.where(magnitude >= 2.0, 1.0, 0.5),
    )
    dithered = torch.addcmul(magnitude, unit_random, step)
    rounded_step = torch.where(
        dithered >= 4.0,
        2.0,
        torch.where(dithered >= 2.0, 1.0, 0.5),
    )
    rounded = (torch.floor(dithered / rounded_step) * rounded_step).clamp(max=6.0)
    values = xq.new_tensor(_FP4_E2M1_VALUES)
    code = torch.searchsorted(values, rounded).clamp(max=7)
    negative = torch.signbit(xq) & ~torch.isnan(xq)
    return code.to(torch.uint8) | negative.to(torch.uint8) << 3


def _pack_e2m1(codes: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack low-nibble-first E2M1 codes along `dim`."""
    _check_tile_dim(dim)
    axis = dim % codes.ndim
    packed_dim = codes.shape[axis]
    if packed_dim % 2:
        raise ValueError(f"fp4_e2m1 contraction extent must be even, got {packed_dim}")
    moved = codes.movedim(axis, -1).contiguous()
    packed = moved[..., 0::2] | moved[..., 1::2] << 4
    return packed.movedim(-1, axis).contiguous()


def _unpack_e2m1(codes: torch.Tensor, dim: int) -> torch.Tensor:
    """Unpack low-nibble-first signed E2M1 codes into float32 values."""
    _check_tile_dim(dim)
    axis = dim % codes.ndim
    moved = codes.movedim(axis, -1)
    nibbles = torch.stack((moved & 0xF, moved >> 4), dim=-1).flatten(-2)
    magnitude = torch.tensor(_FP4_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    values = magnitude[nibbles.long() & 0x7]
    return torch.where(nibbles & 0x8 != 0, -values, values).movedim(-1, axis)


unpack_e2m1 = _unpack_e2m1


def _scale_block_map(
    offs: torch.Tensor, n_rows: int, block_size: int
) -> tuple[torch.Tensor, int]:
    """Map rows to blocks without crossing `offs` groups."""
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    # `right=True` selects each row's end-offset group.
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if not block_size:
        return group.long(), n_groups
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)])
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    row_blocks = first_block[group].long() + local.long()
    # Static upper bound for torch.compile.
    return row_blocks, n_rows // block_size + n_groups


def _compute_scale(
    amax: torch.Tensor, fmt: str, scale_dtype: torch.dtype
) -> torch.Tensor:
    """Return dequantization scales in `scale_dtype` from block maxima."""
    if scale_dtype is torch.float8_e8m0fnu:
        # Clamp E8M0 exponents; `log2(0)` selects 2**-127.
        exp = torch.ceil(torch.log2(amax / str_to_qmax(fmt)))
        # E8M0 cast corrects CUDA `exp2(-127)` being one ULP low.
        return torch.exp2(exp.clamp(-127, 127)).to(scale_dtype)
    if scale_dtype is torch.float8_e4m3fn:
        # Clamp to E4M3's finite, nonzero range.
        low, high = str_to_qmin("fp8_e4m3"), str_to_qmax("fp8_e4m3")
        exact = (amax / str_to_qmax(fmt)).clamp(low, high)
        coded = exact.to(scale_dtype)
        # Round up after a downward cast to avoid clipping block maxima.
        # Positive E4M3 encodings are ordered, so the next byte is the next scale.
        bits = coded.contiguous().view(torch.uint8)
        return torch.where(coded.float() < exact, bits + 1, bits).view(scale_dtype)
    return (amax / str_to_qmax(fmt)).clamp_min(EPS)


def _compute_codes(
    xf: torch.Tensor, scale: torch.Tensor, fmt: str, stochastic_rounding: bool
) -> torch.Tensor:
    """Scale `xf` by broadcastable `scale` and cast to `fmt`; FP4 returns scaled values."""
    qmax = str_to_qmax(fmt)
    xq = (xf / scale.float()).clamp(-qmax, qmax)
    if is_int8s(fmt):
        if stochastic_rounding:
            lower = torch.floor(xq)
            xq = lower + (torch.rand_like(xq) < xq - lower).to(xq.dtype)
        else:
            xq = torch.round(xq)
    elif is_fp8(fmt):
        if stochastic_rounding:
            # Clamp binade ULPs to the shared subnormal spacing.
            mantissa_bits, min_ulp_exp = str_to_fp8_ulp(fmt)
            exponent = ((xq.view(torch.int32) >> 23) & 0xFF) - mantissa_bits
            ulp = (exponent.clamp_min(127 + min_ulp_exp) << 23).view(torch.float32)
            # `floor` keeps the stochastic interval sign-safe.
            lower = torch.floor(xq / ulp) * ulp
            probability = (xq - lower) / ulp
            xq = torch.where(torch.rand_like(xq) < probability, lower + ulp, lower)
        else:
            # Hardware cast supplies RNE.
            pass
    elif is_fp4(fmt):
        return xq
    else:
        raise ValueError(f"Unknown fmt:{fmt}")
    return xq.to(str_to_dtype(fmt))


def _check_tile_dim(dim: int) -> None:
    """Reject dimensions outside the final two axes."""
    if dim not in (-2, -1):
        raise ValueError(f"dim must be -2 or -1, got {dim}")


def _tile(a: torch.Tensor, dim: int, block_size: int) -> torch.Tensor:
    """Pad and split `dim` into fixed-size blocks."""
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
    # Empty blocks have zero maxima.
    shape = list(a.shape)
    shape[dim] = n_blocks
    return a.new_zeros(shape).index_reduce_(
        dim % a.ndim, row_blocks, a, "amax", include_self=True
    )


def _global_amax(
    xf: torch.Tensor,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
) -> torch.Tensor:
    """Return absolute maxima for operands, experts, or ragged groups."""
    if offs is None:
        return xf.abs().amax((-2, -1)).reshape(-1)
    dense_dim = -1 if ragged_dim == -2 else -2
    row_blocks, n_blocks = _scale_block_map(offs, xf.shape[ragged_dim], 0)
    return _segment_amax(
        xf.abs().amax(dense_dim, keepdim=True), ragged_dim, row_blocks, n_blocks
    ).reshape(-1)


def _global_divisor(
    global_scale: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor | None,
    ragged_dim: int | None,
) -> torch.Tensor:
    """Broadcast global scales over `x`."""
    if offs is None:
        if x.ndim == 3:
            return global_scale.reshape(-1, 1, 1)
        return global_scale.reshape(())
    row_blocks, _ = _scale_block_map(offs, x.shape[ragged_dim], 0)
    expanded = global_scale.index_select(0, row_blocks)
    return expanded.reshape(-1, 1) if ragged_dim == -2 else expanded.reshape(1, -1)


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


def _check_e2m1_dims(
    x: torch.Tensor,
    contract_dim: int,
    ragged_dim: int | None,
    offs: torch.Tensor | None,
) -> None:
    """Validate the logical E2M1 layout expected by NVFP4 kernels."""
    logical_k = x.shape[contract_dim]
    if logical_k % 16:
        raise ValueError(
            f"fp4_e2m1 contraction extent must be a multiple of 16, got {logical_k}"
        )
    if offs is None or ragged_dim != contract_dim:
        return
    valid = torch.all(offs.remainder(2) == 0) & (offs[-1] == logical_k)
    message = "fp4_e2m1 ragged contraction offsets must be even and end at logical K"
    if torch.compiler.is_compiling():
        torch._assert_async(valid, message)
    else:
        torch._assert(valid, message)


def _check_rotation_dims(
    contract_dim: int,
    ragged_dim: int | None,
    offs: torch.Tensor | None,
    rotation: Rotation,
) -> None:
    """Require ragged contraction groups to begin on rotation-block boundaries."""
    if ragged_dim != contract_dim:
        return
    aligned = torch.all(offs.remainder(rotation.alignment) == 0)
    if torch.compiler.is_compiling():
        # Fullgraph requires an asynchronous assertion.
        torch._assert_async(
            aligned, "ragged contraction boundaries must align with rotation blocks"
        )
    else:
        torch._assert(
            aligned, "ragged contraction boundaries must align with rotation blocks"
        )


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

    # Expand scales across the outer axis.
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
        # Dense 2D avoids a materialized ragged divisor.
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

    # Ragged blocks use a gathered divisor.
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
    rotation: Rotation | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize `x` with scales along `contract_dim`.

    `contract_dim` and `ragged_dim` are -2 or -1; `offs` and `ragged_dim` are
    supplied together, and `offs` keeps blocks within groups. In 2D blockwise
    quantization, either axis may be ragged. Returns codes in `fmt` (packed uint8
    for fp4_e2m1), scales expanded on the outer axis and blockwise on the
    contraction axis, and `global_scale`. Output strides are unspecified.

    `rotation` preconditions `x` and is inverted by `dequantize_operand`; ragged
    contraction boundaries must align with its blocks. With `enable_global_scale`,
    `global_scale` is an fp32 `(G,)` factor that block scales are relative to;
    otherwise it is None. Pass it to dequantization or scaled GEMM.
    """
    _check_dims(x, contract_dim, ragged_dim, offs)
    if is_fp4(fmt):
        _check_e2m1_dims(x, contract_dim, ragged_dim, offs)
    if rotation is not None:
        _check_rotation_dims(contract_dim, ragged_dim, offs, rotation)
    # Rotation retains fp32 values.
    xf = x.float() if rotation is None else rotation(x, contract_dim, torch.float32)
    granularity = scale_cfg["granularity"]
    block_outer, block_size = scale_cfg["block_shape"]
    scale_dtype = scale_cfg["scale_dtype"]

    global_scale = None
    if scale_cfg["enable_global_scale"]:
        # Fit block scales in the finite range of `scale_dtype`.
        global_scale = (
            _global_amax(xf, offs, ragged_dim)
            / (str_to_qmax(fmt) * float(torch.finfo(scale_dtype).max))
        ).clamp_min(EPS)
        # Block scales are relative to the global factor.
        xf = xf / _global_divisor(global_scale, xf, offs, ragged_dim)

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
    if is_fp4(fmt):
        codes = (
            _pack_e2m1(_e2m1_stochastic_codes(codes), contract_dim)
            if stochastic_rounding
            else pack_e2m1_rne(codes, contract_dim)
        )
    return codes, scale, global_scale


def dequantize_operand(
    xq: torch.Tensor,
    scale: torch.Tensor,
    contract_dim: int,
    scale_cfg: dict,
    offs: torch.Tensor | None = None,
    ragged_dim: int | None = None,
    rotation: Rotation | None = None,
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize `xq` in fp32 using `quantize_operand`'s scale layout."""
    if xq.dtype is torch.uint8:
        xq = _unpack_e2m1(xq, contract_dim)
    _check_dims(xq, contract_dim, ragged_dim, offs)
    if rotation is not None:
        _check_rotation_dims(contract_dim, ragged_dim, offs, rotation)
    block_size = scale_cfg["block_shape"][1]
    qf, sf = xq.float(), scale.float()
    if ragged_dim == contract_dim and offs is not None:
        row_blocks, _ = _scale_block_map(offs, xq.shape[contract_dim], block_size)
        deq = (qf * sf.index_select(contract_dim, row_blocks)).contiguous()
    else:
        length = xq.shape[contract_dim]
        deq = _untile(
            _tile(qf, contract_dim, block_size or length)
            * sf.unsqueeze(-1 if contract_dim == -1 else -2),
            contract_dim,
            length,
        )
    if global_scale is not None:
        # Restore the global factor before inverse rotation.
        deq = deq * _global_divisor(global_scale, deq, offs, ragged_dim)
    if rotation is not None:
        deq = rotation.inverse(deq, contract_dim)
    return deq
