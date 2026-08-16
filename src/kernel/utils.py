import torch
import torch.nn.functional as F

from src.kernel.spec import CheckResult


FP32 = frozenset({torch.float32})
BF16 = frozenset({torch.bfloat16})
FP16 = frozenset({torch.float16})
FP8_E4M3 = frozenset({torch.float8_e4m3fn})
FP8_E5M2 = frozenset({torch.float8_e5m2})
FP8_E8M0 = frozenset({torch.float8_e8m0fnu})
INT8 = frozenset({torch.int8})
INT32 = frozenset({torch.int32})
INT64 = frozenset({torch.int64})


def first_failure(results: tuple[CheckResult, ...]) -> CheckResult:
    for result in results:
        if not result.ok:
            return result
    return CheckResult(True)


def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(-2) == 1:
        return tensor
    return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)


def to_swizzle_32_4_4(scale: torch.Tensor) -> torch.Tensor:
    rows, blocks = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_blocks = (blocks + 3) // 4 * 4
    padded = F.pad(scale, (0, padded_blocks - blocks, 0, padded_rows - rows))
    return (
        padded.to(torch.float8_e8m0fnu)
        .reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def check_condition(condition: bool, reason: str) -> CheckResult:
    return CheckResult(condition, None if condition else reason)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _format_capability(capability: tuple[int, int]) -> str:
    return f"SM{capability[0]}{capability[1]}"


def _format_major_requirements(majors: frozenset[int]) -> str:
    requirements = [f"SM{major}0" for major in sorted(majors)]
    if len(requirements) < 2:
        return "".join(requirements)
    return ", ".join(requirements[:-1]) + f" or {requirements[-1]}"


def check_callable(owner: object, attribute: str) -> CheckResult:
    owner_name = getattr(owner, "__name__", type(owner).__name__)
    missing = object()
    value = getattr(owner, attribute, missing)
    if callable(value):
        return CheckResult(True)
    if value is missing:
        return CheckResult(False, f"{owner_name}.{attribute} is unavailable")
    return CheckResult(
        False,
        f"{owner_name}.{attribute} is {type(value).__name__}, requires a callable",
    )


def check_attribute(owner: object, attribute: str) -> CheckResult:
    owner_name = getattr(owner, "__name__", type(owner).__name__)
    if getattr(owner, "__module__", None) == "torch.nn.functional":
        owner_name = f"torch.nn.functional.{owner_name.removeprefix('_')}"
    if getattr(owner, attribute, None) is not None:
        return CheckResult(True)
    return CheckResult(False, f"{owner_name}.{attribute} is unavailable")


def check_torch_tensors(tensors: tuple[object, ...], feature: str) -> CheckResult:
    invalid = tuple(
        type(tensor).__name__
        for tensor in tensors
        if not isinstance(tensor, torch.Tensor)
    )
    if invalid:
        actual = ", ".join(invalid)
        return CheckResult(
            False, f"{feature} received {actual}; requires torch.Tensor inputs"
        )
    return CheckResult(True)


def check_cuda_tensors(tensors: tuple[torch.Tensor, ...], feature: str) -> CheckResult:
    devices = tuple(tensor.device for tensor in tensors)
    non_cuda = tuple(device for device in devices if device.type != "cuda")
    if not non_cuda:
        return CheckResult(True)
    actual = ", ".join(str(device) for device in non_cuda)
    return CheckResult(
        False,
        f"{feature} received device(s) {actual}; requires CUDA tensors",
    )


def check_same_device(tensors: tuple[torch.Tensor, ...], feature: str) -> CheckResult:
    devices = tuple(dict.fromkeys(tensor.device for tensor in tensors))
    if len(devices) <= 1:
        return CheckResult(True)
    actual = ", ".join(str(device) for device in devices)
    return CheckResult(
        False,
        f"{feature} received devices {actual}; requires tensors on the same device",
    )


def check_dtypes(
    values: tuple[torch.Tensor | torch.dtype, ...],
    allowed_dtypes: frozenset[torch.dtype],
    feature: str,
    require_same: bool = False,
) -> CheckResult:
    actual_dtypes = frozenset(
        value if isinstance(value, torch.dtype) else value.dtype for value in values
    )
    rejected = actual_dtypes.difference(allowed_dtypes)
    if rejected:
        actual = ", ".join(_dtype_name(dtype) for dtype in sorted(rejected, key=str))
        allowed = ", ".join(
            _dtype_name(dtype) for dtype in sorted(allowed_dtypes, key=str)
        )
        return CheckResult(
            False,
            f"{feature} received dtype(s) {actual}; requires one of {allowed}",
        )
    if require_same and len(actual_dtypes) > 1:
        actual = ", ".join(
            _dtype_name(dtype) for dtype in sorted(actual_dtypes, key=str)
        )
        return CheckResult(
            False,
            f"{feature} received dtype(s) {actual}; requires the same dtype",
        )
    return CheckResult(True)


def check_values(
    values: tuple[object, ...],
    allowed_values: frozenset[object],
    feature: str,
) -> CheckResult:
    rejected = tuple(value for value in values if value not in allowed_values)
    if not rejected:
        return CheckResult(True)
    actual = ", ".join(repr(value) for value in rejected)
    allowed = ", ".join(repr(value) for value in sorted(allowed_values, key=repr))
    return CheckResult(
        False,
        f"{feature} received value(s) {actual}; requires one of {allowed}",
    )


def check_nonempty(tensor: torch.Tensor, feature: str) -> CheckResult:
    if tensor.numel() > 0:
        return CheckResult(True)
    return CheckResult(False, f"{feature} is empty; requires a nonempty tensor")


def check_shape(
    tensor: torch.Tensor,
    expected_shape: tuple[int, ...],
    feature: str,
) -> CheckResult:
    actual_shape = tuple(tensor.shape)
    if actual_shape == expected_shape:
        return CheckResult(True)
    return CheckResult(
        False,
        f"{feature} has shape {actual_shape}; requires shape {expected_shape}",
    )


def check_dimension_size_match(
    tensor_dimensions: tuple[tuple[torch.Tensor, int], ...],
    feature: str,
) -> CheckResult:
    sizes = []
    for tensor, dimension in tensor_dimensions:
        if not -tensor.ndim <= dimension < tensor.ndim:
            return CheckResult(
                False,
                f"{feature} received dimension {dimension} for a rank {tensor.ndim} "
                "tensor",
            )
        sizes.append(tensor.shape[dimension])
    if len(set(sizes)) <= 1:
        return CheckResult(True)
    actual = ", ".join(str(size) for size in sizes)
    return CheckResult(
        False,
        f"{feature} received sizes {actual}; requires matching sizes",
    )


def check_dimension_size_multiple_of(
    tensor_dimensions: tuple[tuple[torch.Tensor, int], ...],
    multiple: int,
    feature: str,
) -> CheckResult:
    sizes = []
    for tensor, dimension in tensor_dimensions:
        if not -tensor.ndim <= dimension < tensor.ndim:
            return CheckResult(
                False,
                f"{feature} received dimension {dimension} for a rank {tensor.ndim} "
                "tensor",
            )
        sizes.append(tensor.shape[dimension])
    rejected = tuple(size for size in sizes if size % multiple != 0)
    if not rejected:
        return CheckResult(True)
    actual = ", ".join(str(size) for size in rejected)
    return CheckResult(
        False,
        f"{feature} received size(s) {actual}; requires multiples of {multiple}",
    )


def check_compute_capability_at_least(
    device: torch.device,
    minimum: tuple[int, int],
    feature: str,
) -> CheckResult:
    if device.type != "cuda":
        return CheckResult(
            False,
            f"{feature} runs on {device}; requires a CUDA device for capability checks",
        )
    actual = torch.cuda.get_device_capability(device)
    if actual >= minimum:
        return CheckResult(True)
    return CheckResult(
        False,
        f"{feature} runs on {_format_capability(actual)} at {device}; requires "
        f"{_format_capability(minimum)} or newer",
    )


def check_compute_capability_in(
    device: torch.device,
    supported_majors: frozenset[int],
    feature: str,
) -> CheckResult:
    if device.type != "cuda":
        return CheckResult(
            False,
            f"{feature} runs on {device}; requires a CUDA device for capability checks",
        )
    actual = torch.cuda.get_device_capability(device)
    if actual[0] in supported_majors:
        return CheckResult(True)
    requirement = _format_major_requirements(supported_majors)
    return CheckResult(
        False,
        f"{feature} runs on {_format_capability(actual)} at {device}; requires "
        f"{requirement}",
    )


def check_ndim(
    tensor: torch.Tensor, allowed_ndims: frozenset[int], feature: str
) -> CheckResult:
    if tensor.ndim in allowed_ndims:
        return CheckResult(True)
    allowed = ", ".join(str(ndim) for ndim in sorted(allowed_ndims))
    return CheckResult(
        False,
        f"{feature} has rank {tensor.ndim}; requires rank in {{{allowed}}}",
    )


def check_contiguous(tensor: torch.Tensor, feature: str) -> CheckResult:
    if tensor.is_contiguous():
        return CheckResult(True)
    return CheckResult(
        False,
        f"{feature} is non-contiguous; requires a contiguous tensor",
    )


def check_stride(
    tensor: torch.Tensor,
    allowed_strides: tuple[tuple[int, ...], ...],
    feature: str,
) -> CheckResult:
    actual_stride = tensor.stride()
    if tensor.numel() == 0 or actual_stride in allowed_strides:
        return CheckResult(True)
    allowed = ", ".join(str(stride) for stride in allowed_strides)
    return CheckResult(
        False,
        f"{feature} has stride {actual_stride}; requires one of {allowed}",
    )


def check_alignment(
    tensor: torch.Tensor, alignment_bytes: int, feature: str
) -> CheckResult:
    if alignment_bytes <= 0:
        return CheckResult(
            False,
            f"{feature} received {alignment_bytes}-byte alignment; requires a positive "
            "alignment",
        )
    stride_bytes = tuple(
        stride * tensor.element_size() for stride in tensor.stride() if stride != 1
    )
    misaligned = tuple(
        stride for stride in stride_bytes if stride % alignment_bytes != 0
    )
    if not misaligned:
        return CheckResult(True)
    actual = ", ".join(str(stride) for stride in misaligned)
    return CheckResult(
        False,
        f"{feature} has non-unit stride byte(s) {actual}; requires {alignment_bytes}-byte "
        "alignment",
    )


def check_matrix_layout(
    tensor: torch.Tensor, alignment_bytes: int, feature: str
) -> CheckResult:
    if tensor.ndim < 2:
        return CheckResult(
            False,
            f"{feature} has rank {tensor.ndim}; requires a matrix with rank 2 or greater",
        )
    row_extent, column_extent = tensor.shape[-2:]
    row_stride, column_stride = tensor.stride()[-2:]
    row_major = column_stride == 1 and (row_extent <= 1 or row_stride >= column_extent)
    column_major = row_stride == 1 and (
        column_extent <= 1 or column_stride >= row_extent
    )
    if row_major or column_major:
        return check_alignment(tensor, alignment_bytes, feature)
    if column_stride != 1 and row_stride != 1:
        return CheckResult(
            False,
            f"{feature} has matrix strides {(row_stride, column_stride)}; requires a "
            "row-major or column-major layout",
        )
    return CheckResult(
        False,
        f"{feature} has matrix shape {(row_extent, column_extent)} and strides "
        f"{(row_stride, column_stride)} with overlapping logical matrix dimensions",
    )
