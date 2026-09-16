"""Shared reference quantization helpers."""

import torch
import torch.nn.functional as F

from src.kernel.registry import register_kernel

EPS = 1e-30


def scale_block_map(
    offs: torch.Tensor, n_rows: int, block_size: int
) -> tuple[torch.Tensor, int]:
    """Map ragged rows to their group-local scale blocks."""
    n_groups = offs.shape[0]
    rows = torch.arange(n_rows, device=offs.device, dtype=offs.dtype)
    group = torch.searchsorted(offs, rows, right=True).clamp_(max=n_groups - 1)
    if not block_size:
        return group.long(), n_groups
    starts = torch.cat([offs.new_zeros(1), offs[:-1]])
    counts = offs - starts
    per_group = torch.div(counts + block_size - 1, block_size, rounding_mode="floor")
    first_block = torch.cat([offs.new_zeros(1), per_group.cumsum(0)])
    local = torch.div(rows - starts[group], block_size, rounding_mode="floor")
    blocks = first_block[group].long() + local.long()
    # Keep output shapes independent of offset values for torch.compile.
    return blocks, n_rows // block_size + n_groups


def scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode float scales to float32 without E8M0 conversion kernels."""
    if scale.dtype is torch.float8_e8m0fnu:
        code = scale.view(torch.uint8).to(torch.int32)
        # Code 0 is 2**-127 (an FP32 subnormal); code 255 is NaN.
        bits = torch.where(
            code == 0,
            0x00400000,
            torch.where(code == 255, 0x7F800001, code << 23),
        )
        return bits.view(torch.float32)
    return scale.float()


def tile(a: torch.Tensor, dim: int, block_size: int) -> torch.Tensor:
    """Pad and split either of the final two dimensions into blocks."""
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


def untile(a: torch.Tensor, dim: int, length: int) -> torch.Tensor:
    """Merge blocks along either final dimension and remove padding."""
    if dim == -1:
        return a.flatten(-2).narrow(-1, 0, length).contiguous()
    return a.flatten(-3, -2).narrow(-2, 0, length).contiguous()


def pack_e2m1(codes: torch.Tensor, dim: int) -> torch.Tensor:
    """Pack low-nibble-first E2M1 codes along ``dim``."""
    axis = dim % codes.ndim
    moved = codes.movedim(axis, -1).contiguous()
    return (moved[..., 0::2] | moved[..., 1::2] << 4).movedim(-1, axis).contiguous()


def layout_codes(codes: torch.Tensor, output_layout: str) -> torch.Tensor:
    """Store dense matrix codes in the requested physical layout."""
    if output_layout == "row_major":
        return codes.contiguous()
    if output_layout == "column_major":
        return codes.mT.contiguous().mT
    raise ValueError("output_layout must be row_major or column_major")


def global_amax(xf: torch.Tensor) -> torch.Tensor:
    """Return absolute maxima for dense operands."""
    return xf.abs().amax((-2, -1)).reshape(-1)


def global_amax_grouped(
    xf: torch.Tensor, offs: torch.Tensor, ragged_dim: int
) -> torch.Tensor:
    """Return absolute maxima for ragged groups."""
    dense_dim = -1 if ragged_dim == -2 else -2
    row_blocks, n_blocks = scale_block_map(offs, xf.shape[ragged_dim], 0)
    return _segment_amax(
        xf.abs().amax(dense_dim, keepdim=True), ragged_dim, row_blocks, n_blocks
    ).reshape(-1)


def global_divisor(global_scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast dense global scales over ``x``."""
    if x.ndim == 3:
        return global_scale.reshape(-1, 1, 1)
    return global_scale.reshape(())


def global_divisor_grouped(
    global_scale: torch.Tensor,
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
) -> torch.Tensor:
    """Broadcast ragged-group global scales over ``x``."""
    row_blocks, _ = scale_block_map(offs, x.shape[ragged_dim], 0)
    expanded = global_scale.index_select(0, row_blocks)
    return expanded.reshape(-1, 1) if ragged_dim == -2 else expanded.reshape(1, -1)


def _quantization_stats(
    source: torch.Tensor,
    codes: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor | None,
    reduction_dims: tuple[int, ...],
    numel: int,
) -> torch.Tensor:
    """Reduce original-domain energies and zero counts, before code packing."""
    source, codes, scale = source.detach().float(), codes.detach(), scale.detach()
    if codes.dtype is torch.uint8:
        magnitudes = source.new_tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
        decoded = magnitudes[(codes & 7).long()]
        decoded = torch.where((codes & 8) != 0, -decoded, decoded)
    else:
        decoded = codes.float()
    reconstructed = decoded * scale_to_float(scale)
    if global_scale is not None:
        reconstructed = reconstructed * global_scale.detach().reshape(())
    nonzero = source != 0
    partials = torch.stack(
        (
            source.square().sum(reduction_dims),
            (source - reconstructed).square().sum(reduction_dims),
            (nonzero & (decoded == 0)).sum(reduction_dims, dtype=torch.float32),
            nonzero.sum(reduction_dims, dtype=torch.float32),
        )
    ).reshape(4, -1)
    totals = partials.sum(-1)
    return torch.stack(
        (totals[0], totals[1], totals[2], totals.new_full((), numel), totals[3])
    )


def _quantize_dense(
    xf: torch.Tensor,
    contract_dim: int,
    dtype: torch.dtype,
    qmax: float,
    block_shape: tuple[int, int],
    scale_dtype: torch.dtype,
    stochastic_rounding: bool = False,
    source: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize dense FP32 values; uint8 returns E2M1 nibbles."""
    block_outer, block_size = block_shape
    quantization_stats = None
    if block_shape == (0, 0):
        outer_dim = -1 if contract_dim == -2 else -2
        scale = _compute_scale(xf.abs().amax((-2, -1), keepdim=True), qmax, scale_dtype)
        codes = _compute_codes(xf, scale, dtype, qmax, stochastic_rounding)
        if source is not None:
            quantization_stats = _quantization_stats(
                source, codes, scale, global_scale, (-2, -1), source.numel()
            )
        shape = list(scale.shape)
        shape[outer_dim] = xf.shape[outer_dim]
        return codes, scale.expand(shape), quantization_stats
    if block_shape == (1, 0):
        scale = _compute_scale(
            xf.abs().amax(contract_dim, keepdim=True), qmax, scale_dtype
        )
        codes = _compute_codes(xf, scale, dtype, qmax, stochastic_rounding)
        if source is not None:
            quantization_stats = _quantization_stats(
                source, codes, scale, global_scale, (contract_dim,), source.numel()
            )
        return codes, scale, quantization_stats
    if block_outer == 1:
        scale = _compute_scale(
            tile(xf.abs(), contract_dim, block_size).amax(
                dim=-1 if contract_dim == -1 else -2
            ),
            qmax,
            scale_dtype,
        )
        axis = -1 if contract_dim == -1 else -2
        codes = _compute_codes(
            tile(xf, contract_dim, block_size),
            scale.unsqueeze(axis),
            dtype,
            qmax,
            stochastic_rounding,
        )
        if source is not None:
            reduction_dims = (
                (-3, -2, -1)
                if dtype is torch.uint8 and stochastic_rounding
                else (axis,)
            )
            quantization_stats = _quantization_stats(
                tile(source, contract_dim, block_size),
                codes,
                scale.unsqueeze(axis),
                global_scale,
                reduction_dims,
                source.numel(),
            )
        codes = untile(codes, contract_dim, xf.shape[contract_dim])
        return codes, scale, quantization_stats

    outer_dim = -1 if contract_dim == -2 else -2
    rows, cols = xf.shape[-2:]
    pad_rows, pad_cols = (-rows) % block_size, (-cols) % block_size
    padded = F.pad(xf, (0, pad_cols, 0, pad_rows)) if pad_rows or pad_cols else xf
    tiled = padded.reshape(
        *padded.shape[:-2],
        padded.shape[-2] // block_size,
        block_size,
        padded.shape[-1] // block_size,
        block_size,
    )
    blocks = _compute_scale(tiled.abs().amax((-3, -1)), qmax, scale_dtype)
    codes = _compute_codes(
        tiled,
        blocks[..., :, None, :, None],
        dtype,
        qmax,
        stochastic_rounding,
    )
    if source is not None:
        original = (
            F.pad(source, (0, pad_cols, 0, pad_rows))
            if pad_rows or pad_cols
            else source
        )
        quantization_stats = _quantization_stats(
            original.reshape(tiled.shape),
            codes,
            blocks[..., :, None, :, None],
            global_scale,
            (-3, -1),
            source.numel(),
        )
    codes = codes.reshape(
        *codes.shape[:-4],
        codes.shape[-4] * codes.shape[-3],
        codes.shape[-2] * codes.shape[-1],
    )
    codes = codes[..., :rows, :cols].contiguous()
    length = rows if outer_dim == -2 else cols
    scale = blocks.repeat_interleave(block_size, outer_dim).narrow(outer_dim, 0, length)
    return codes, scale, quantization_stats


def _quantize_grouped(
    xf: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    dtype: torch.dtype,
    qmax: float,
    block_shape: tuple[int, int],
    scale_dtype: torch.dtype,
    stochastic_rounding: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize grouped FP32 values; uint8 returns E2M1 nibbles."""
    block_outer, block_size = block_shape
    if block_shape == (0, 0):
        outer_dim = -1 if contract_dim == -2 else -2
        dense_dim = -1 if ragged_dim == -2 else -2
        row_blocks, n_blocks = scale_block_map(offs, xf.shape[ragged_dim], 0)
        scale = _compute_scale(
            _segment_amax(
                xf.abs().amax(dense_dim, keepdim=True),
                ragged_dim,
                row_blocks,
                n_blocks,
            ),
            qmax,
            scale_dtype,
        )
        divisor = scale.index_select(ragged_dim, row_blocks)
        codes = _compute_codes(xf, divisor, dtype, qmax, stochastic_rounding)
        if ragged_dim == outer_dim:
            return codes, divisor
        shape = list(scale.shape)
        shape[outer_dim] = xf.shape[outer_dim]
        return codes, scale.expand(shape)
    if block_shape == (1, 0):
        if ragged_dim == contract_dim:
            return _quantize_segmented_contraction(
                xf,
                contract_dim,
                0,
                dtype,
                qmax,
                scale_dtype,
                offs,
                stochastic_rounding,
            )
        scale = _compute_scale(
            xf.abs().amax(contract_dim, keepdim=True), qmax, scale_dtype
        )
        return _compute_codes(xf, scale, dtype, qmax, stochastic_rounding), scale
    if block_outer == 1:
        if ragged_dim == contract_dim:
            return _quantize_segmented_contraction(
                xf,
                contract_dim,
                block_size,
                dtype,
                qmax,
                scale_dtype,
                offs,
                stochastic_rounding,
            )
        scale = _compute_scale(
            tile(xf.abs(), contract_dim, block_size).amax(
                dim=-1 if contract_dim == -1 else -2
            ),
            qmax,
            scale_dtype,
        )
        axis = -1 if contract_dim == -1 else -2
        codes = untile(
            _compute_codes(
                tile(xf, contract_dim, block_size),
                scale.unsqueeze(axis),
                dtype,
                qmax,
                stochastic_rounding,
            ),
            contract_dim,
            xf.shape[contract_dim],
        )
        return codes, scale

    outer_dim = -1 if contract_dim == -2 else -2
    ragged_map = scale_block_map(offs, xf.shape[ragged_dim], block_size)
    dense_dim = contract_dim if ragged_dim == outer_dim else outer_dim
    dense_amax = tile(xf.abs(), dense_dim, block_size).amax(
        dim=-1 if dense_dim == -1 else -2
    )
    blocks = _compute_scale(
        _segment_amax(dense_amax, ragged_dim, ragged_map[0], ragged_map[1]),
        qmax,
        scale_dtype,
    )
    if ragged_dim == outer_dim:
        scale = blocks.index_select(outer_dim, ragged_map[0])
    else:
        scale = blocks.repeat_interleave(block_size, outer_dim).narrow(
            outer_dim, 0, xf.shape[outer_dim]
        )
    if ragged_dim == contract_dim:
        divisor = scale.index_select(contract_dim, ragged_map[0])
    else:
        divisor = scale.repeat_interleave(block_size, contract_dim).narrow(
            contract_dim, 0, xf.shape[contract_dim]
        )
    return _compute_codes(xf, divisor, dtype, qmax, stochastic_rounding), scale


def _compute_scale(
    amax: torch.Tensor, qmax: float, scale_dtype: torch.dtype
) -> torch.Tensor:
    if scale_dtype is torch.float8_e8m0fnu:
        exponent = torch.ceil(torch.log2(amax / qmax))
        code = torch.where(exponent.isnan(), 255, exponent.clamp(-127, 127) + 127)
        return code.to(torch.uint8).view(scale_dtype)
    if scale_dtype is torch.float8_e4m3fn:
        exact = (amax / qmax).clamp(2.0**-9, 448.0)
        coded = exact.to(scale_dtype)
        bits = coded.contiguous().view(torch.uint8)
        # Round scales upward so the block maximum does not clip.
        return torch.where(coded.float() < exact, bits + 1, bits).view(scale_dtype)
    return (amax / qmax).clamp_min(EPS)


def _compute_codes(
    xf: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype,
    qmax: float,
    stochastic_rounding: bool,
) -> torch.Tensor:
    normalized = (xf / scale_to_float(scale)).clamp(-qmax, qmax)
    if dtype is torch.int8:
        if stochastic_rounding:
            lower = torch.floor(normalized)
            normalized = lower + (torch.rand_like(normalized) < normalized - lower)
        else:
            normalized = torch.round(normalized)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if stochastic_rounding:
            mantissa_bits, min_ulp_exp = (
                (3, -9) if dtype is torch.float8_e4m3fn else (2, -16)
            )
            exponent = ((normalized.view(torch.int32) >> 23) & 0xFF) - mantissa_bits
            ulp = (exponent.clamp_min(127 + min_ulp_exp) << 23).view(torch.float32)
            lower = torch.floor(normalized / ulp) * ulp
            probability = (normalized - lower) / ulp
            normalized = torch.where(
                torch.rand_like(normalized) < probability, lower + ulp, lower
            )
    elif dtype is torch.uint8:
        if stochastic_rounding:
            magnitude = torch.nan_to_num(normalized.abs(), nan=qmax, posinf=qmax)
            magnitude = magnitude.clamp(max=qmax)
            step = torch.where(magnitude >= 2.0, 1.0, 0.5)
            step = torch.where(magnitude >= 4.0, 2.0, step)
            lower = torch.floor(magnitude / step) * step
            probability = (magnitude - lower) / step
            rounded = lower + (torch.rand_like(magnitude) < probability) * step
            values = normalized.new_tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0))
            code = torch.searchsorted(values, rounded).clamp(max=7)
            negative = torch.signbit(normalized) & ~torch.isnan(normalized)
            return code.to(torch.uint8) | negative.to(torch.uint8) << 3
        magnitude = torch.nan_to_num(normalized.abs(), nan=6.0, posinf=6.0).clamp(
            max=6.0
        )
        codes = (
            (magnitude > 0.25).to(torch.uint8)
            + (magnitude >= 0.75).to(torch.uint8)
            + (magnitude > 1.25).to(torch.uint8)
            + (magnitude >= 1.75).to(torch.uint8)
            + (magnitude > 2.5).to(torch.uint8)
            + (magnitude >= 3.5).to(torch.uint8)
            + (magnitude > 5.0).to(torch.uint8)
        )
        negative = torch.signbit(normalized) & ~torch.isnan(normalized)
        return codes | negative.to(torch.uint8) << 3
    return normalized.to(dtype)


def _segment_amax(
    a: torch.Tensor, dim: int, row_blocks: torch.Tensor, n_blocks: int
) -> torch.Tensor:
    shape = list(a.shape)
    shape[dim] = n_blocks
    return a.new_zeros(shape).index_reduce_(
        dim % a.ndim, row_blocks, a, "amax", include_self=True
    )


def _quantize_segmented_contraction(
    xf: torch.Tensor,
    contract_dim: int,
    block_size: int,
    dtype: torch.dtype,
    qmax: float,
    scale_dtype: torch.dtype,
    offs: torch.Tensor,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_blocks, n_blocks = scale_block_map(offs, xf.shape[contract_dim], block_size)
    scale = _compute_scale(
        _segment_amax(xf.abs(), contract_dim, row_blocks, n_blocks), qmax, scale_dtype
    )
    return _compute_codes(
        xf,
        scale.index_select(contract_dim, row_blocks),
        dtype,
        qmax,
        stochastic_rounding,
    ), scale


@register_kernel(
    op="quantize.dequantize_dense",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def dequantize_dense(
    xq: torch.Tensor,
    scale: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a dense operand into FP32."""
    if xq.dtype is torch.uint8:
        xq = unpack_e2m1(xq, contract_dim)
    length = xq.shape[contract_dim]
    block_size = block_shape[1] or length
    values = untile(
        tile(xq.float(), contract_dim, block_size)
        * scale_to_float(scale).unsqueeze(-1 if contract_dim == -1 else -2),
        contract_dim,
        length,
    )
    if global_scale is not None:
        values = values * global_divisor(global_scale, values)
    return values


@register_kernel(
    op="quantize.dequantize_grouped",
    backend="eager",
    build="eager",
    autograd=False,
    reference=True,
)
def dequantize_grouped(
    xq: torch.Tensor,
    scale: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    global_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize a grouped operand into FP32."""
    if xq.dtype is torch.uint8:
        xq = unpack_e2m1(xq, contract_dim)
    if ragged_dim == contract_dim:
        row_blocks, _ = scale_block_map(offs, xq.shape[contract_dim], block_shape[1])
        values = (
            xq.float() * scale_to_float(scale).index_select(contract_dim, row_blocks)
        ).contiguous()
    else:
        length = xq.shape[contract_dim]
        block_size = block_shape[1] or length
        values = untile(
            tile(xq.float(), contract_dim, block_size)
            * scale_to_float(scale).unsqueeze(-1 if contract_dim == -1 else -2),
            contract_dim,
            length,
        )
    if global_scale is not None:
        values = values * global_divisor_grouped(global_scale, values, offs, ragged_dim)
    return values


@register_kernel(
    op="quantize.unpack_e2m1",
    backend="eager",
    build="eager",
    autograd=False,
    capabilities=frozenset(),
    reference=True,
)
def unpack_e2m1(codes: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unpack low-nibble-first E2M1 codes to FP32."""
    axis = dim % codes.ndim
    moved = codes.movedim(axis, -1)
    nibbles = torch.stack((moved & 0xF, moved >> 4), dim=-1).flatten(-2)
    magnitude = torch.tensor(
        (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
        dtype=torch.float32,
        device=codes.device,
    )
    values = magnitude[nibbles.long() & 0x7]
    return torch.where(nibbles & 0x8 != 0, -values, values).movedim(-1, axis)
