from collections.abc import Mapping
from typing import Any

import torch

import src.kernel.backends.cublaslt as cublaslt_backend
import src.kernel.backends.eager  # noqa: F401
import src.kernel.backends.triton  # noqa: F401
from src.kernel.registry import register_operation
from src.kernel.selector import dispatch
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    BF16,
    FP16,
    FP32,
    FP8_E4M3,
    FP8_E5M2,
    FP8_E8M0,
    INT8,
    INT32,
    INT64,
    check_condition,
    check_compute_capability_at_least,
    check_contiguous,
    check_cuda_tensors,
    check_dimension_size_multiple_of,
    check_dimension_size_match,
    check_dtypes,
    check_matrix_layout,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    check_stride,
    check_torch_tensors,
    check_values,
    first_failure,
)

__all__ = [
    "grouped_gemm",
    "scaled_gemm",
    "scaled_grouped_gemm",
    "can_implement_grouped_gemm_eager",
    "can_implement_scaled_gemm_eager",
    "can_implement_scaled_grouped_gemm_eager",
    "can_implement_grouped_gemm_triton",
    "can_implement_scaled_gemm_triton",
    "can_implement_scaled_grouped_gemm_triton",
    "can_implement_scaled_gemm_cublaslt",
    "can_implement_scaled_grouped_gemm_cublaslt",
]


_FLOAT_DTYPES = FP32 | BF16 | FP16
_QUANTIZED_DTYPES = INT8 | FP8_E4M3 | FP8_E5M2
_OFFSET_DTYPES = INT32 | INT64
_CUBLASLT_EXTENSION_AVAILABLE = cublaslt_backend._C is not None
_CUBLASLT_EXTENSION_ERROR = cublaslt_backend._C_IMPORT_ERROR


def _validate_grouped_gemm_shared_contract(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    checks = (check_condition(len(args) == 4, "expected 4 arguments"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    a, b, offs, bias = args
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    result = check_torch_tensors(tensors, "grouped GEMM")
    if not result.ok:
        return result
    checks = (
        check_same_device(tensors, "grouped GEMM"),
        check_dtypes((a, b), _FLOAT_DTYPES, "grouped GEMM operands", require_same=True),
        check_ndim(a, frozenset({2, 3}), "grouped GEMM A"),
        check_ndim(b, frozenset({2, 3}), "grouped GEMM B"),
        check_ndim(offs, frozenset({1}), "grouped GEMM offsets"),
        check_dtypes((offs,), _OFFSET_DTYPES, "grouped GEMM offsets"),
        check_nonempty(offs, "grouped GEMM offsets"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = (
        check_condition(a.ndim != 3 or b.ndim != 3, "3D x 3D not supported"),
        check_dimension_size_match(
            ((a, -1), (b, -2)), "grouped GEMM contraction dimensions"
        ),
    )
    if a.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((a, 0), (offs, 0)), "grouped GEMM A group count"
            ),
        )
    if b.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((b, 0), (offs, 0)), "grouped GEMM B group count"
            ),
        )
    if bias is not None:
        checks += (
            check_condition(a.ndim != 3, "bias not supported for ragged-N"),
            check_dtypes((bias,), frozenset({a.dtype}), "grouped GEMM bias"),
            check_shape(bias, (offs.numel(), b.shape[-1]), "grouped GEMM bias"),
        )
    result = first_failure(checks)
    return result


def _validate_scaled_gemm_shared_contract(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    checks = (check_condition(len(args) == 8, "expected 8 arguments"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias = args
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    result = check_torch_tensors(tensors, "scaled GEMM")
    if not result.ok:
        return result
    checks = (
        check_condition(
            isinstance(out_dtype, torch.dtype),
            "scaled GEMM output dtype must be torch.dtype",
        ),
        check_condition(
            isinstance(scale_dtype, torch.dtype),
            "scaled GEMM scale_dtype must be torch.dtype",
        ),
        check_condition(
            type(block_size) is int and block_size >= 0,
            "block_size must be a non-negative integer",
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = (
        check_same_device(tensors, "scaled GEMM"),
        check_dtypes((aq, bq), _QUANTIZED_DTYPES, "scaled GEMM operands"),
        check_dtypes((sa, sb), FP32, "scaled GEMM scales"),
        check_dtypes((out_dtype,), _FLOAT_DTYPES, "scaled GEMM output dtype"),
        check_dtypes((scale_dtype,), FP32 | FP8_E8M0, "scaled GEMM scale_dtype"),
        check_ndim(aq, frozenset({2}), "scaled GEMM A"),
        check_ndim(bq, frozenset({2}), "scaled GEMM B"),
        check_ndim(sa, frozenset({2}), "scaled GEMM A scale"),
        check_ndim(sb, frozenset({2}), "scaled GEMM B scale"),
    )
    if bias is not None:
        checks += (check_dtypes((bias,), _FLOAT_DTYPES, "scaled GEMM bias"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    checks = (
        check_dimension_size_match(
            ((aq, -1), (bq, -2)), "scaled GEMM contraction dimensions"
        ),
        check_shape(sa, (aq.shape[0], blocks), "scaled GEMM A scale"),
        check_shape(sb, (blocks, bq.shape[1]), "scaled GEMM B scale"),
    )
    if bias is not None:
        checks += (check_shape(bias, (bq.shape[1],), "scaled GEMM bias"),)
    result = first_failure(checks)
    return result


def _validate_scaled_grouped_gemm_shared_contract(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    del kwargs
    checks = (check_condition(len(args) == 9, "expected 9 arguments"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias = args
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    result = check_torch_tensors(tensors, "scaled grouped GEMM")
    if not result.ok:
        return result
    checks = (
        check_condition(
            isinstance(out_dtype, torch.dtype),
            "scaled grouped GEMM output dtype must be torch.dtype",
        ),
        check_condition(
            isinstance(scale_dtype, torch.dtype),
            "scaled grouped GEMM scale_dtype must be torch.dtype",
        ),
        check_condition(
            type(block_size) is int and block_size >= 0,
            "block_size must be a non-negative integer",
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = (
        check_same_device(tensors, "scaled grouped GEMM"),
        check_dtypes((aq, bq), _QUANTIZED_DTYPES, "scaled grouped GEMM operands"),
        check_dtypes((sa, sb), FP32, "scaled grouped GEMM scales"),
        check_dtypes((offs,), _OFFSET_DTYPES, "scaled grouped GEMM offsets"),
        check_dtypes((out_dtype,), _FLOAT_DTYPES, "scaled grouped GEMM output dtype"),
        check_dtypes(
            (scale_dtype,), FP32 | FP8_E8M0, "scaled grouped GEMM scale_dtype"
        ),
        check_ndim(aq, frozenset({2, 3}), "scaled grouped GEMM A"),
        check_ndim(bq, frozenset({2, 3}), "scaled grouped GEMM B"),
        check_ndim(sa, frozenset({2, 3}), "scaled grouped GEMM A scale"),
        check_ndim(sb, frozenset({2, 3}), "scaled grouped GEMM B scale"),
        check_ndim(offs, frozenset({1}), "scaled grouped GEMM offsets"),
        check_nonempty(offs, "scaled grouped GEMM offsets"),
    )
    if bias is not None:
        checks += (check_dtypes((bias,), _FLOAT_DTYPES, "scaled grouped GEMM bias"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = (
        check_condition(aq.ndim != 3 or bq.ndim != 3, "3D x 3D not supported"),
        check_dimension_size_match(
            ((aq, -1), (bq, -2)), "scaled grouped GEMM contraction dimensions"
        ),
    )
    if aq.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((aq, 0), (offs, 0)), "scaled grouped GEMM A group count"
            ),
        )
    if bq.ndim == 3:
        checks += (
            check_dimension_size_match(
                ((bq, 0), (offs, 0)), "scaled grouped GEMM B group count"
            ),
        )
    if bias is not None:
        checks += (
            check_condition(aq.ndim != 3, "bias not supported for ragged-N"),
            check_shape(bias, (offs.numel(), bq.shape[-1]), "scaled grouped GEMM bias"),
        )
    result = first_failure(checks)
    return result


def can_implement_grouped_gemm_eager(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_grouped_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del args, kwargs
    return CheckResult(True)


def can_implement_scaled_gemm_eager(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del args, kwargs
    return CheckResult(True)


def can_implement_scaled_grouped_gemm_eager(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_grouped_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias = args
    checks = ()
    if aq.ndim == 2 and bq.ndim == 3:
        scale_blocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
        valid_a_scale = sa.shape == (aq.shape[0], scale_blocks)
        valid_b_scale = sb.shape == (
            bq.shape[0],
            scale_blocks,
            bq.shape[2],
        )
    elif aq.ndim == 2 and bq.ndim == 2:
        valid_a_scale = (
            sa.ndim == 2
            and sa.shape[0] == aq.shape[0]
            and (aq.shape[1] == 0 or sa.shape[1] > 0)
        )
        valid_b_scale = (
            sb.ndim == 2 and sb.shape[0] == sa.shape[1] and sb.shape[1] == bq.shape[1]
        )
        if block_size == 0:
            valid_a_scale = valid_a_scale and sa.shape[1] == offs.numel()
    else:
        scale_blocks = (aq.shape[2] + block_size - 1) // block_size if block_size else 1
        valid_a_scale = sa.shape == (
            aq.shape[0],
            aq.shape[1],
            scale_blocks,
        )
        valid_b_scale = sb.shape == (scale_blocks, bq.shape[1])
    checks += (
        check_condition(
            valid_a_scale,
            "scaled grouped GEMM has invalid A scale layout",
        ),
        check_condition(
            valid_b_scale,
            "scaled grouped GEMM has invalid B scale layout",
        ),
    )
    result = first_failure(checks)
    return result


def can_implement_grouped_gemm_triton(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_grouped_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    a, b, offs, bias = args
    tensors = (a, b, offs) if bias is None else (a, b, offs, bias)
    checks = (
        check_cuda_tensors(tensors, "grouped GEMM"),
        check_dtypes((a, b), BF16, "grouped GEMM operands"),
        check_contiguous(offs, "grouped GEMM offsets"),
        check_dtypes((offs,), INT32, "grouped GEMM offsets"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    return check_compute_capability_at_least(a.device, (8, 0), "grouped GEMM")


def _check_triton_quantized_operands(
    aq: torch.Tensor, bq: torch.Tensor, feature: str
) -> CheckResult:
    return check_condition(
        aq.dtype.is_floating_point == bq.dtype.is_floating_point,
        f"{feature} requires operands from the same dtype family",
    )


def can_implement_scaled_gemm_triton(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias = args
    tensors = (aq, bq, sa, sb) + (() if bias is None else (bias,))
    checks = (
        check_cuda_tensors(tensors, "scaled GEMM"),
        _check_triton_quantized_operands(aq, bq, "scaled GEMM"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    nblocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
    checks = (
        check_condition(
            nblocks <= 1 or (block_size >= 16 and (block_size & (block_size - 1)) == 0),
            "block_size must be a power of two >= 16",
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    minimum_capability = (8, 9) if aq.dtype.is_floating_point else (8, 0)
    return check_compute_capability_at_least(
        aq.device, minimum_capability, "scaled GEMM"
    )


def can_implement_scaled_grouped_gemm_triton(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_grouped_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias = args
    tensors = (aq, bq, sa, sb, offs) + (() if bias is None else (bias,))
    checks = (
        check_cuda_tensors(tensors, "scaled grouped GEMM"),
        check_contiguous(offs, "scaled grouped GEMM offsets"),
        check_dtypes((offs,), INT32, "scaled grouped GEMM offsets"),
        _check_triton_quantized_operands(aq, bq, "scaled grouped GEMM"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    checks = ()
    if aq.ndim == 2 and bq.ndim == 3:
        nblocks = (aq.shape[1] + block_size - 1) // block_size if block_size else 1
        valid_scales = sa.shape == (aq.shape[0], nblocks) and sb.shape == (
            bq.shape[0],
            nblocks,
            bq.shape[2],
        )
    elif aq.ndim == 2 and bq.ndim == 2:
        valid_scales = (
            sa.ndim == 2
            and sb.ndim == 2
            and sa.shape[0] == aq.shape[0]
            and sa.shape[1] == sb.shape[0]
            and sb.shape[1] == bq.shape[1]
        )
    else:
        nblocks = (aq.shape[2] + block_size - 1) // block_size if block_size else 1
        valid_scales = sa.shape == (aq.shape[0], aq.shape[1], nblocks) and sb.shape == (
            nblocks,
            bq.shape[1],
        )
    checks += (check_condition(valid_scales, "invalid scale layout"),)
    result = first_failure(checks)
    if not result.ok:
        return result
    minimum_capability = (8, 9) if aq.dtype.is_floating_point else (8, 0)
    return check_compute_capability_at_least(
        aq.device, minimum_capability, "scaled grouped GEMM"
    )


def can_implement_scaled_gemm_cublaslt(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias = args
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    if not _CUBLASLT_EXTENSION_AVAILABLE:
        return CheckResult(
            False, f"cuBLASLt extension unavailable: {_CUBLASLT_EXTENSION_ERROR}"
        )
    checks = (
        check_cuda_tensors(tensors, "MXFP8 GEMM"),
        check_dtypes((aq, bq), FP8_E4M3, "MXFP8 GEMM operands"),
        check_values((block_size,), frozenset({32}), "MXFP8 GEMM block_size"),
        check_dtypes((scale_dtype,), FP8_E8M0, "MXFP8 GEMM scale_dtype"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    a_allowed_strides = ((aq.shape[1], 1),)
    b_allowed_strides = ((bq.shape[1], 1), (1, bq.shape[0]))
    checks = (
        check_contiguous(sa, "MXFP8 GEMM A scale"),
        check_contiguous(sb, "MXFP8 GEMM B scale"),
        check_matrix_layout(aq, 16, "MXFP8 GEMM A"),
        check_matrix_layout(bq, 16, "MXFP8 GEMM B"),
        check_stride(aq, a_allowed_strides, "MXFP8 GEMM A"),
        check_stride(bq, b_allowed_strides, "MXFP8 GEMM B"),
        check_dimension_size_multiple_of(
            ((aq, -1), (bq, -1)), 16, "MXFP8 GEMM dimensions"
        ),
        check_dtypes((out_dtype,), BF16, "MXFP8 GEMM output dtype"),
    )
    if bias is not None:
        checks += (
            check_dtypes((bias,), BF16, "MXFP8 GEMM bias"),
            check_contiguous(bias, "MXFP8 GEMM bias"),
        )
    result = first_failure(checks)
    if not result.ok:
        return result
    return check_compute_capability_at_least(aq.device, (10, 0), "MXFP8 GEMM")


def can_implement_scaled_grouped_gemm_cublaslt(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> CheckResult:
    result = _validate_scaled_grouped_gemm_shared_contract(args, kwargs)
    if not result.ok:
        return result
    del kwargs
    aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias = args
    if not _CUBLASLT_EXTENSION_AVAILABLE:
        return CheckResult(
            False, f"cuBLASLt extension unavailable: {_CUBLASLT_EXTENSION_ERROR}"
        )
    tensors = (aq, bq, sa, sb, offs)
    checks = (
        check_condition(bias is None, "MXFP8 grouped GEMM does not support bias"),
        check_cuda_tensors(tensors, "MXFP8 grouped GEMM"),
        check_dtypes((aq, bq), FP8_E4M3, "MXFP8 grouped GEMM operands"),
        check_values((block_size,), frozenset({32}), "MXFP8 grouped GEMM block_size"),
        check_dtypes((scale_dtype,), FP8_E8M0, "MXFP8 grouped GEMM scale_dtype"),
        check_dtypes((out_dtype,), BF16, "MXFP8 grouped GEMM output dtype"),
        check_dtypes((offs,), INT32, "MXFP8 grouped GEMM offsets"),
        check_ndim(aq, frozenset({2}), "MXFP8 grouped GEMM A"),
        check_ndim(bq, frozenset({3}), "MXFP8 grouped GEMM B"),
        check_ndim(sa, frozenset({2}), "MXFP8 grouped GEMM A scale"),
        check_ndim(sb, frozenset({3}), "MXFP8 grouped GEMM B scale"),
        check_contiguous(sa, "MXFP8 grouped GEMM A scale"),
        check_contiguous(sb, "MXFP8 grouped GEMM B scale"),
        check_contiguous(offs, "MXFP8 grouped GEMM offsets"),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    blocks = (aq.shape[1] + 31) // 32
    checks = (
        check_shape(sa, (aq.shape[0], blocks), "MXFP8 grouped GEMM A scale"),
        check_shape(
            sb,
            (bq.shape[0], blocks, bq.shape[2]),
            "MXFP8 grouped GEMM B scale",
        ),
        check_matrix_layout(aq, 16, "MXFP8 grouped GEMM A"),
        check_matrix_layout(bq, 16, "MXFP8 grouped GEMM B"),
        check_stride(aq, ((aq.shape[1], 1),), "MXFP8 grouped GEMM A"),
        check_stride(
            bq,
            (
                (bq.shape[1] * bq.shape[2], bq.shape[2], 1),
                (bq.shape[1] * bq.shape[2], 1, bq.shape[1]),
            ),
            "MXFP8 grouped GEMM B",
        ),
        check_dimension_size_multiple_of(
            ((aq, -1), (bq, -1)), 16, "MXFP8 grouped GEMM dimensions"
        ),
    )
    result = first_failure(checks)
    if not result.ok:
        return result
    capability = torch.cuda.get_device_capability(aq.device)
    return check_condition(
        capability[0] == 10 or capability == (11, 0),
        f"MXFP8 grouped GEMM runs on SM{capability[0]}{capability[1]} at "
        f"{aq.device}; requires SM10x or SM110",
    )


register_operation(
    "gemm.grouped",
    {
        "eager": can_implement_grouped_gemm_eager,
        "triton": can_implement_grouped_gemm_triton,
    },
)
register_operation(
    "gemm.scaled",
    {
        "eager": can_implement_scaled_gemm_eager,
        "triton": can_implement_scaled_gemm_triton,
        "cublaslt": can_implement_scaled_gemm_cublaslt,
    },
)
register_operation(
    "gemm.scaled_grouped",
    {
        "eager": can_implement_scaled_grouped_gemm_eager,
        "triton": can_implement_scaled_grouped_gemm_triton,
        "cublaslt": can_implement_scaled_grouped_gemm_cublaslt,
    },
)


def grouped_gemm(a, b, offs, bias=None, backend="auto"):
    return dispatch("gemm.grouped", (a, b, offs, bias), {}, backend)


def scaled_gemm(
    aq,
    bq,
    sa,
    sb,
    out_dtype,
    block_size,
    scale_dtype,
    bias=None,
    backend="auto",
):
    return dispatch(
        "gemm.scaled",
        (aq, bq, sa, sb, out_dtype, block_size, scale_dtype, bias),
        {},
        backend,
    )


def scaled_grouped_gemm(
    aq,
    bq,
    sa,
    sb,
    offs,
    out_dtype,
    block_size,
    scale_dtype,
    bias=None,
    backend="auto",
):
    return dispatch(
        "gemm.scaled_grouped",
        (aq, bq, sa, sb, offs, out_dtype, block_size, scale_dtype, bias),
        {},
        backend,
    )
