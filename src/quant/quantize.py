import torch

from src.kernel.ops.quantize import (
    dequantize_dense,
    dequantize_grouped,
    quantize_fp8,
    quantize_fp8_grouped,
    quantize_int8,
    quantize_int8_grouped,
    quantize_mxfp8,
    quantize_mxfp8_grouped,
    quantize_nvfp4,
    quantize_nvfp4_grouped,
)
from src.quant.rotation import Rotation
from src.quant.utils import (
    is_fp4,
    is_fp8,
    is_int8s,
    is_quantized,
    str_to_qmax,
    str_to_dtype,
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


def _check_e2m1_dims(
    x: torch.Tensor,
    contract_dim: int,
    ragged_dim: int | None,
    offs: torch.Tensor | None,
) -> None:
    """Validate the E2M1 contraction layout."""
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
    """Check ragged boundaries align with rotation blocks."""
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


def quantize_operand(
    x: torch.Tensor,
    contract_dim: int,
    fmt: str,
    scale_cfg: dict,
    offs: torch.Tensor | None = None,
    ragged_dim: int | None = None,
    stochastic_rounding: bool = False,
    rotation: Rotation | None = None,
    return_quantization_stats: bool = False,
    output_layout: str | None = None,
    backend: str | None = None,
    scale_layout: str = "row_major",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return codes, scales, and an optional global scale.

    For unrotated dense 2D inputs, `return_quantization_stats` appends a detached
    FP32 tensor in (src_sq, err_sq, under, numel, nonzero) order. Backends without
    fused statistics append None so callers can fall back to post-quantization
    monitoring without changing the selected quantizer. An omitted layout retains
    legacy rank-3 code layouts; explicit layouts use the dense storage contract.
    """
    if return_quantization_stats and (
        x.ndim != 2 or offs is not None or rotation is not None
    ):
        raise ValueError(
            "quantization statistics require an unrotated dense 2D operand"
        )
    layout_was_omitted = output_layout is None
    if layout_was_omitted:
        output_layout = "row_major"
    elif output_layout not in ("row_major", "column_major"):
        raise ValueError("output_layout must be row_major or column_major")
    if offs is not None and output_layout != "row_major":
        raise ValueError("output_layout is supported only for dense quantization")
    _check_dims(x, contract_dim, ragged_dim, offs)
    if is_fp4(fmt):
        _check_e2m1_dims(x, contract_dim, ragged_dim, offs)
    if rotation is not None:
        _check_rotation_dims(contract_dim, ragged_dim, offs, rotation)
    granularity = scale_cfg["granularity"]
    block_shape = tuple(scale_cfg["block_shape"])
    scale_dtype = scale_cfg["scale_dtype"]
    if granularity == "tensorwise":
        block_shape = (0, 0)
    elif granularity == "rowwise":
        block_shape = (1, 0)
    elif granularity != "blockwise":
        raise ValueError(f"unknown granularity: {granularity!r}")
    enable_global_scale = scale_cfg["enable_global_scale"]
    if scale_layout != "row_major" and (
        offs is not None or scale_dtype is not torch.float8_e8m0fnu
    ):
        raise ValueError("packed scales require dense MXFP8 quantization")
    if scale_dtype is torch.float8_e8m0fnu and is_quantized(fmt) and fmt != "fp8_e4m3":
        raise ValueError("MXFP8 quantization requires fp8_e4m3")
    # Rotation retains FP32 values before quantization.
    source = x if rotation is None else rotation(x, contract_dim, torch.float32)
    if scale_dtype is torch.float8_e8m0fnu:
        if offs is None:
            return quantize_mxfp8(
                source,
                contract_dim,
                block_shape,
                fmt,
                stochastic_rounding,
                output_layout=output_layout,
                return_quantization_stats=return_quantization_stats,
                backend=backend,
                scale_layout=scale_layout,
            )
        return quantize_mxfp8_grouped(
            source, offs, ragged_dim, contract_dim, block_shape, stochastic_rounding
        )
    if is_fp8(fmt):
        if offs is None:
            if layout_was_omitted and source.ndim == 3:
                return quantize_fp8(
                    source,
                    contract_dim,
                    str_to_dtype(fmt),
                    block_shape,
                    stochastic_rounding,
                    backend="eager",
                    scale_dtype=scale_dtype,
                    enable_global_scale=enable_global_scale,
                    preserve_strides=True,
                    return_quantization_stats=return_quantization_stats,
                    output_layout=output_layout,
                )
            return quantize_fp8(
                source,
                contract_dim,
                str_to_dtype(fmt),
                block_shape,
                stochastic_rounding,
                scale_dtype=scale_dtype,
                enable_global_scale=enable_global_scale,
                return_quantization_stats=return_quantization_stats,
                output_layout=output_layout,
                backend=backend,
            )
        return quantize_fp8_grouped(
            source,
            offs,
            ragged_dim,
            contract_dim,
            str_to_dtype(fmt),
            block_shape,
            stochastic_rounding,
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    if is_int8s(fmt):
        bits = int(fmt[3:])
        if offs is None:
            int8_output_layout = output_layout
            if (
                layout_was_omitted
                and source.ndim == 3
                and block_shape[1] == 0
                and source.stride(-2) == 1
            ):
                int8_output_layout = "column_major"
            return quantize_int8(
                source,
                contract_dim,
                block_shape,
                bits,
                stochastic_rounding,
                scale_dtype=scale_dtype,
                enable_global_scale=enable_global_scale,
                return_quantization_stats=return_quantization_stats,
                output_layout=int8_output_layout,
                backend=backend,
            )
        return quantize_int8_grouped(
            source,
            offs,
            ragged_dim,
            contract_dim,
            block_shape,
            bits,
            stochastic_rounding,
            scale_dtype=scale_dtype,
            enable_global_scale=enable_global_scale,
        )
    if is_fp4(fmt):
        if offs is None:
            return quantize_nvfp4(
                source,
                contract_dim,
                block_shape,
                enable_global_scale,
                stochastic_rounding,
                scale_dtype=scale_dtype,
                qmax=str_to_qmax(fmt),
                return_quantization_stats=return_quantization_stats,
                output_layout=output_layout,
                backend=backend,
            )
        return quantize_nvfp4_grouped(
            source,
            offs,
            ragged_dim,
            contract_dim,
            block_shape,
            enable_global_scale,
            stochastic_rounding,
            scale_dtype=scale_dtype,
            qmax=str_to_qmax(fmt),
        )
    raise ValueError(f"unsupported quantization format: {fmt!r}")


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
    _check_dims(xq, contract_dim, ragged_dim, offs)
    if rotation is not None:
        _check_rotation_dims(contract_dim, ragged_dim, offs, rotation)
    block_shape = tuple(scale_cfg["block_shape"])
    if offs is None:
        deq = dequantize_dense(xq, scale, contract_dim, block_shape, global_scale)
    else:
        deq = dequantize_grouped(
            xq,
            scale,
            offs,
            ragged_dim,
            contract_dim,
            block_shape,
            global_scale,
        )
    if rotation is not None:
        deq = rotation.inverse(deq, contract_dim)
    return deq
