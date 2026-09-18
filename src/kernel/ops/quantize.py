"""Backend-neutral quantization operations."""

import torch

import src.kernel.backends.cuda  # noqa: F401
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.selector import _platform_for, dispatch, select_kernel


_SUPPORTED_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})
_CODE_DTYPES = frozenset(
    {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}
)
_SCALE_DTYPES = frozenset({torch.float32, torch.float8_e4m3fn, torch.float8_e8m0fnu})


def _check_input(x: torch.Tensor, contract_dim: int, allow_empty: bool = False) -> None:
    if x.ndim not in (2, 3) or (not allow_empty and any(size == 0 for size in x.shape)):
        raise ValueError("quantization requires a nonempty 2D or 3D tensor")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("x must have dtype float32, float16, or bfloat16")
    if contract_dim not in (-2, -1):
        raise ValueError("contract_dim must be -2 or -1")


def _check_block_shape(block_shape: tuple[int, int]) -> None:
    if (
        not isinstance(block_shape, (tuple, list))
        or len(block_shape) != 2
        or any(type(extent) is not int for extent in block_shape)
    ):
        raise ValueError("block_shape must contain two integer extents")
    outer, contract = block_shape
    if (outer, contract) not in ((0, 0), (1, 0)) and (
        contract <= 0 or outer not in (1, contract)
    ):
        raise ValueError("block_shape must be (0, 0), (1, 0), (1, B), or (B, B)")


def _check_grouped(x: torch.Tensor, offs: torch.Tensor, ragged_dim: int) -> None:
    if x.ndim != 2:
        raise ValueError("grouped quantization requires a 2D tensor")
    if ragged_dim not in (-2, -1):
        raise ValueError("ragged_dim must be -2 or -1")
    if offs.ndim != 1 or offs.numel() == 0:
        raise ValueError("offs must be a nonempty 1D tensor")
    if offs.dtype not in (torch.int32, torch.int64):
        raise ValueError("offs must have dtype int32 or int64")
    if offs.device != x.device or not offs.is_contiguous():
        raise ValueError("offs must be contiguous and on the input device")


def _check_rounding(stochastic_rounding: bool) -> None:
    if type(stochastic_rounding) is not bool:
        raise ValueError("stochastic_rounding must be a bool")


def _check_output_layout(output_layout: str) -> None:
    if output_layout not in ("row_major", "column_major"):
        raise ValueError("output_layout must be row_major or column_major")


def _check_mxfp8_block_shape(block_shape: tuple[int, int]) -> None:
    _check_block_shape(block_shape)
    contract = block_shape[1]
    if contract <= 0 or (contract != 16 and contract % 32):
        raise ValueError("MXFP8 block extent must be 16 or a multiple of 32")


def _check_nvfp4(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    enable_global_scale: bool,
    scale_dtype: torch.dtype,
    qmax: float,
) -> None:
    _check_input(x, contract_dim)
    _check_block_shape(block_shape)
    _check_scale(scale_dtype, enable_global_scale)
    if x.shape[contract_dim] % 16:
        raise ValueError("NVFP4 contraction extent must be a multiple of 16")
    if type(qmax) not in (int, float) or qmax not in (4.0, 6.0):
        raise ValueError("NVFP4 qmax must be 4 or 6")


def _check_bits(bits: int) -> None:
    if type(bits) is not int or bits not in (4, 5, 6, 7, 8):
        raise ValueError("bits must be an integer from 4 to 8")


def _check_scale(scale_dtype: torch.dtype, enable_global_scale: bool) -> None:
    if scale_dtype not in (torch.float32, torch.float8_e4m3fn):
        raise ValueError("scale_dtype must be float32 or float8_e4m3fn")
    if type(enable_global_scale) is not bool:
        raise ValueError("enable_global_scale must be a bool")


def _check_dequantize(
    xq: torch.Tensor,
    scale: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    global_scale: torch.Tensor | None,
) -> None:
    if xq.ndim not in (2, 3) or contract_dim not in (-2, -1):
        raise ValueError("dequantization requires a 2D or 3D tensor and dim -2 or -1")
    if xq.dtype not in _CODE_DTYPES and xq.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("unsupported code dtype")
    _check_block_shape(block_shape)
    if scale.dtype not in _SCALE_DTYPES:
        raise ValueError("unsupported scale dtype")
    if scale.ndim != xq.ndim or scale.device != xq.device:
        raise ValueError("scale must have the same rank and device as codes")
    if global_scale is not None and (
        global_scale.dtype is not torch.float32
        or global_scale.device != xq.device
        or global_scale.ndim != 1
    ):
        raise ValueError("global_scale must be a 1D float32 tensor on the code device")


def _dispatch_quantize(op, args, backend, device, return_quantization_stats):
    if type(return_quantization_stats) is not bool:
        raise ValueError("return_quantization_stats must be a bool")
    if not return_quantization_stats:
        return dispatch(op, args, {}, backend, device=device)
    if args[0].ndim != 2:
        raise ValueError("quantization statistics require a dense 2D operand")
    kernel = select_kernel(op, backend, _platform_for(device.type, device.index))
    return kernel.fn(*args, return_quantization_stats=True)


def quantize_fp8(
    x: torch.Tensor,
    contract_dim: int,
    dtype: torch.dtype,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
    preserve_strides: bool = False,
    return_quantization_stats: bool = False,
    output_layout: str = "row_major",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return FP8 codes, dequantization scales, and an optional FP32 global scale.

    `block_shape` orders outer/contraction extents; zero spans the entire axis.
    Global scaling applies only to E4M3 scales; FP32 scales return None.
    Codes use row-major storage by default; `output_layout="column_major"`
    stores column-major codes. `preserve_strides` retains the eager input layout.
    When requested, append FP32 statistics in (src_sq, err_sq, under, numel,
    nonzero) order. Backends without fused statistics append None without
    changing quantization.
    """
    _check_input(x, contract_dim)
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("dtype must be float8_e4m3fn or float8_e5m2")
    _check_block_shape(block_shape)
    _check_rounding(stochastic_rounding)
    _check_scale(scale_dtype, enable_global_scale)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    if type(preserve_strides) is not bool:
        raise ValueError("preserve_strides must be a bool")
    _check_output_layout(output_layout)
    if preserve_strides and output_layout != "row_major":
        raise ValueError("preserve_strides cannot be combined with column_major output")
    if backend is None and (scale_dtype is not torch.float32 or preserve_strides):
        backend = "eager"
    return _dispatch_quantize(
        "quantize.quantize_fp8",
        (
            x,
            contract_dim,
            dtype,
            tuple(block_shape),
            stochastic_rounding,
            scale_dtype,
            enable_global_scale,
            preserve_strides,
            output_layout,
        ),
        backend,
        x.device,
        return_quantization_stats,
    )


def quantize_fp8_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    dtype: torch.dtype,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return FP8 codes, group-local scales, and optional per-group global scales.

    `offs` contains cumulative group ends along `ragged_dim`, including empty
    groups. `block_shape` is (outer, contraction); zero spans the group axis.
    Ragged contraction scales reserve length // block_size + num_groups slots,
    or num_groups slots for tensorwise/rowwise scaling.
    """
    _check_input(x, contract_dim)
    _check_grouped(x, offs, ragged_dim)
    if dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError("dtype must be float8_e4m3fn or float8_e5m2")
    _check_block_shape(block_shape)
    _check_rounding(stochastic_rounding)
    _check_scale(scale_dtype, enable_global_scale)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    return dispatch(
        "quantize.quantize_fp8_grouped",
        (
            x,
            offs,
            ragged_dim,
            contract_dim,
            dtype,
            tuple(block_shape),
            stochastic_rounding,
            scale_dtype,
            enable_global_scale,
        ),
        {},
        backend,
        device=x.device,
    )


def quantize_int8(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    bits: int = 8,
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
    return_quantization_stats: bool = False,
    output_layout: str = "row_major",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return signed integer codes in int8 storage, scales, and optional global scales.

    `bits` selects 4 through 8 effective bits. Block shapes follow `quantize_fp8`.
    Global scaling applies only to E4M3 scales; FP32 scales return None.
    When requested, append FP32 statistics in (src_sq, err_sq, under, numel,
    nonzero) order. Backends without fused statistics append None without
    changing quantization.
    """
    _check_input(x, contract_dim)
    _check_block_shape(block_shape)
    _check_bits(bits)
    _check_rounding(stochastic_rounding)
    _check_scale(scale_dtype, enable_global_scale)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    _check_output_layout(output_layout)
    if backend is None and scale_dtype is not torch.float32:
        backend = "eager"
    return _dispatch_quantize(
        "quantize.quantize_int8",
        (
            x,
            contract_dim,
            tuple(block_shape),
            bits,
            stochastic_rounding,
            scale_dtype,
            enable_global_scale,
            output_layout,
        ),
        backend,
        x.device,
        return_quantization_stats,
    )


def quantize_int8_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    bits: int = 8,
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    enable_global_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return int8 codes, scales, and optional globals in `quantize_fp8_grouped`'s layout."""
    _check_input(x, contract_dim)
    _check_grouped(x, offs, ragged_dim)
    _check_block_shape(block_shape)
    _check_bits(bits)
    _check_rounding(stochastic_rounding)
    _check_scale(scale_dtype, enable_global_scale)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    return dispatch(
        "quantize.quantize_int8_grouped",
        (
            x,
            offs,
            ragged_dim,
            contract_dim,
            tuple(block_shape),
            bits,
            stochastic_rounding,
            scale_dtype,
            enable_global_scale,
        ),
        {},
        backend,
        device=x.device,
    )


def quantize_mxfp8(
    x: torch.Tensor,
    contract_dim: int = -1,
    block_shape: tuple[int, int] = (1, 32),
    fmt: str = "fp8_e4m3",
    stochastic_rounding: bool = False,
    backend: str | None = None,
    output_layout: str = "row_major",
    return_quantization_stats: bool = False,
    scale_layout: str = "row_major",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return E4M3 codes, E8M0 dequantization scales, and None for the global scale.

    `block_shape` is (outer, contraction). Scales replace the contraction axis
    with its block count; square-tile scales repeat along the outer axis.
    CUDA supports (1, B) and (B, B) for B in {16, 32, 64, 128}.
    `output_layout` selects row-major or column-major code matrices; logical
    shapes are unchanged, including for batched inputs. Scales are row-major by
    default; `swizzled_32_4_4` packs rank-2 block-32 scales for cuBLASLt, orienting
    the non-contracting dimension first and padding it to 128 and blocks to 4.
    The default is PyTorch/eager so whole-model compilation can fuse this operation.
    When requested, append FP32 statistics in (src_sq, err_sq, under, numel,
    nonzero) order. Backends without fused statistics append None without
    changing quantization.
    """
    _check_input(x, contract_dim, allow_empty=True)
    if fmt != "fp8_e4m3":
        raise ValueError("MXFP8 quantization requires fp8_e4m3")
    _check_mxfp8_block_shape(block_shape)
    _check_rounding(stochastic_rounding)
    _check_output_layout(output_layout)
    if scale_layout not in ("row_major", "swizzled_32_4_4"):
        raise ValueError("unsupported MXFP8 scale layout")
    if scale_layout == "swizzled_32_4_4" and (x.ndim != 2 or block_shape[1] != 32):
        raise ValueError("swizzled MXFP8 scales require rank-2 block-32 quantization")
    if backend is None:
        backend = "eager"
    return _dispatch_quantize(
        "quantize.quantize_mxfp8",
        (
            x,
            contract_dim,
            tuple(block_shape),
            fmt,
            stochastic_rounding,
            output_layout,
            scale_layout,
        ),
        backend,
        x.device,
        return_quantization_stats,
    )


def quantize_mxfp8_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    stochastic_rounding: bool = False,
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return grouped E4M3 codes, E8M0 scales, and None for the global scale."""
    _check_input(x, contract_dim, allow_empty=True)
    _check_grouped(x, offs, ragged_dim)
    _check_mxfp8_block_shape(block_shape)
    _check_rounding(stochastic_rounding)
    return dispatch(
        "quantize.quantize_mxfp8_grouped",
        (x, offs, ragged_dim, contract_dim, tuple(block_shape), stochastic_rounding),
        {},
        backend,
        device=x.device,
    )


def quantize_nvfp4(
    x: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    enable_global_scale: bool = True,
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    qmax: float = 6.0,
    return_quantization_stats: bool = False,
    output_layout: str = "row_major",
) -> (
    tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]
):
    """Return packed E2M1 codes, dequantization scales, and optional FP32 global scales.

    `qmax` selects standard E2M1 (6) or the 4-over-6 recipe (4).
    Global scaling applies only to E4M3 scales, with shape (1,) or (batch,).
    FP32 scales return None. Block shapes follow `quantize_fp8`.
    When requested, append FP32 statistics in (src_sq, err_sq, under, numel,
    nonzero) order. Backends without fused statistics append None without
    changing quantization.
    """
    _check_nvfp4(x, contract_dim, block_shape, enable_global_scale, scale_dtype, qmax)
    _check_rounding(stochastic_rounding)
    _check_output_layout(output_layout)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    if backend is None and (
        scale_dtype is not torch.float8_e4m3fn
        or tuple(block_shape) not in ((1, 16), (16, 16))
    ):
        backend = "eager"
    return _dispatch_quantize(
        "quantize.quantize_nvfp4",
        (
            x,
            contract_dim,
            tuple(block_shape),
            enable_global_scale,
            stochastic_rounding,
            scale_dtype,
            qmax,
            output_layout,
        ),
        backend,
        x.device,
        return_quantization_stats,
    )


def quantize_nvfp4_grouped(
    x: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    enable_global_scale: bool = True,
    stochastic_rounding: bool = False,
    backend: str | None = None,
    scale_dtype: torch.dtype = torch.float8_e4m3fn,
    qmax: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Return packed E2M1 codes, grouped scales, and optional per-group global scales.

    FP4 packs along `contract_dim`; ragged contraction ends must be even and
    cover the logical contraction extent. Global scales are FP32 with shape (E,).
    """
    _check_nvfp4(x, contract_dim, block_shape, enable_global_scale, scale_dtype, qmax)
    _check_grouped(x, offs, ragged_dim)
    _check_rounding(stochastic_rounding)
    if ragged_dim == contract_dim:
        valid = torch.all(offs.remainder(2) == 0) & (offs[-1] == x.shape[contract_dim])
        message = "NVFP4 ragged contraction ends must be even and cover logical K"
        if torch.compiler.is_compiling():
            torch._assert_async(valid, message)
        else:
            torch._assert(valid, message)
    enable_global_scale = enable_global_scale and scale_dtype is torch.float8_e4m3fn
    return dispatch(
        "quantize.quantize_nvfp4_grouped",
        (
            x,
            offs,
            ragged_dim,
            contract_dim,
            tuple(block_shape),
            enable_global_scale,
            stochastic_rounding,
            scale_dtype,
            qmax,
        ),
        {},
        backend,
        device=x.device,
    )


def dequantize_dense(
    xq: torch.Tensor,
    scale: torch.Tensor,
    contract_dim: int,
    block_shape: tuple[int, int],
    global_scale: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Decode dense codes and scales to contiguous FP32, including global scaling."""
    _check_dequantize(xq, scale, contract_dim, block_shape, global_scale)
    if global_scale is not None and global_scale.numel() != (
        xq.shape[0] if xq.ndim == 3 else 1
    ):
        raise ValueError("global_scale must contain one value per dense batch")
    return dispatch(
        "quantize.dequantize_dense",
        (xq, scale, contract_dim, tuple(block_shape), global_scale),
        {},
        backend,
        device=xq.device,
    )


def dequantize_grouped(
    xq: torch.Tensor,
    scale: torch.Tensor,
    offs: torch.Tensor,
    ragged_dim: int,
    contract_dim: int,
    block_shape: tuple[int, int],
    global_scale: torch.Tensor | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """Decode grouped codes to contiguous FP32; offsets refer to unpacked values."""
    _check_dequantize(xq, scale, contract_dim, block_shape, global_scale)
    _check_grouped(xq, offs, ragged_dim)
    if global_scale is not None and global_scale.numel() != offs.numel():
        raise ValueError("global_scale must contain one value per group")
    return dispatch(
        "quantize.dequantize_grouped",
        (xq, scale, offs, ragged_dim, contract_dim, tuple(block_shape), global_scale),
        {},
        backend,
        device=xq.device,
    )


def unpack_e2m1(
    codes: torch.Tensor, dim: int = -1, backend: str | None = None
) -> torch.Tensor:
    """Decode low-nibble-first packed E2M1 codes to FP32."""
    if dim not in (-2, -1) or codes.ndim < -dim:
        raise ValueError("dim must be -2 or -1 and within the tensor rank")
    if codes.dtype is not torch.uint8:
        raise ValueError("packed E2M1 codes must have dtype uint8")
    return dispatch(
        "quantize.unpack_e2m1", (codes, dim), {}, backend, device=codes.device
    )
