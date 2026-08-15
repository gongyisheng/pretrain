import torch

from src.kernel.spec import CheckResult


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _format_capability(capability: tuple[int, int]) -> str:
    return f"SM{capability[0]}{capability[1]}"


def _format_major_requirements(majors: frozenset[int]) -> str:
    requirements = [f"SM{major}0" for major in sorted(majors)]
    if len(requirements) < 2:
        return "".join(requirements)
    return ", ".join(requirements[:-1]) + f" or {requirements[-1]}"


def check_callable(owner: object, owner_name: str, attribute: str) -> CheckResult:
    value = getattr(owner, attribute, None)
    if callable(value):
        return CheckResult(True)
    return CheckResult(
        False,
        f"{owner_name}.{attribute} is {type(value).__name__}, requires a callable",
    )


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
    tensors: tuple[torch.Tensor, ...],
    allowed_dtypes: frozenset[torch.dtype],
    feature: str,
) -> CheckResult:
    actual_dtypes = frozenset(tensor.dtype for tensor in tensors)
    rejected = actual_dtypes.difference(allowed_dtypes)
    if not rejected:
        return CheckResult(True)
    actual = ", ".join(_dtype_name(dtype) for dtype in sorted(rejected, key=str))
    allowed = ", ".join(_dtype_name(dtype) for dtype in sorted(allowed_dtypes, key=str))
    return CheckResult(
        False,
        f"{feature} received dtype(s) {actual}; requires one of {allowed}",
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


def check_rank(
    tensor: torch.Tensor, allowed_ranks: frozenset[int], feature: str
) -> CheckResult:
    if tensor.ndim in allowed_ranks:
        return CheckResult(True)
    allowed = ", ".join(str(rank) for rank in sorted(allowed_ranks))
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
