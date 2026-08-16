from types import SimpleNamespace

import pytest
import torch

from src.kernel.backends.cublaslt import gemm as cublaslt_gemm
from src.kernel.backends.eager import gemm as eager_gemm
from src.kernel.backends.torch import gemm as torch_gemm
from src.kernel.backends.triton import gemm as triton_gemm
from src.kernel.spec import CheckResult
from src.kernel.utils import (
    check_alignment,
    check_callable,
    check_compute_capability_at_least,
    check_compute_capability_in,
    check_contiguous,
    check_cuda_tensors,
    check_dimension_sizes_match,
    check_dimension_sizes_multiple_of,
    check_dtypes,
    check_matrix_layout,
    check_ndim,
    check_nonempty,
    check_same_device,
    check_shape,
    check_stride,
    check_values,
    first_failure,
)


class _Owner:
    available = staticmethod(lambda: None)
    not_callable = None


def test_first_failure_returns_first_failed_result():
    first_failed_result = CheckResult(False, "first")
    second_failed_result = CheckResult(False, "second")

    assert (
        first_failure((CheckResult(True), first_failed_result, second_failed_result))
        is first_failed_result
    )


def test_first_failure_accepts_all_successful_results():
    assert first_failure((CheckResult(True), CheckResult(True))) == CheckResult(True)


def test_check_callable_accepts_callable_attribute():
    assert check_callable(_Owner, "test owner", "available").ok


@pytest.mark.parametrize("attribute", ["missing", "not_callable"])
def test_check_callable_rejects_missing_or_non_callable_attribute(attribute: str):
    result = check_callable(_Owner, "test owner", attribute)

    assert not result.ok
    assert result.reason is not None
    assert "test owner" in result.reason
    assert attribute in result.reason
    assert "callable" in result.reason


def test_check_cuda_tensors_reports_actual_device_and_requirement():
    result = check_cuda_tensors((torch.ones(1, device="cpu"),), "scaled GEMM")

    assert not result.ok
    assert result.reason is not None
    assert "scaled GEMM" in result.reason
    assert "cpu" in result.reason
    assert "CUDA" in result.reason


def test_check_cuda_tensors_accepts_cuda_devices():
    tensor = SimpleNamespace(device=torch.device("cuda:2"))

    assert check_cuda_tensors((tensor,), "scaled GEMM").ok


def test_check_same_device_accepts_matching_devices():
    tensors = (torch.ones(1), torch.zeros(1))

    assert check_same_device(tensors, "grouped GEMM").ok


def test_check_same_device_reports_actual_devices():
    tensors = (
        SimpleNamespace(device=torch.device("cuda:0")),
        SimpleNamespace(device=torch.device("cuda:1")),
    )

    result = check_same_device(tensors, "grouped GEMM")

    assert not result.ok
    assert result.reason is not None
    assert "grouped GEMM" in result.reason
    assert "cuda:0" in result.reason
    assert "cuda:1" in result.reason
    assert "same device" in result.reason


def test_check_dtypes_reports_actual_and_allowed_dtypes():
    tensors = (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float16))

    result = check_dtypes(tensors, frozenset({torch.float32}), "eager GEMM")

    assert not result.ok
    assert result.reason is not None
    assert "eager GEMM" in result.reason
    assert "float16" in result.reason
    assert "float32" in result.reason


def test_check_dtypes_requires_matching_dtypes_when_requested():
    tensors = (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float16))

    result = check_dtypes(
        tensors,
        frozenset({torch.float32, torch.float16}),
        "grouped GEMM operands",
        require_same=True,
    )

    assert not result.ok
    assert result.reason is not None
    assert "grouped GEMM operands" in result.reason
    assert "float16" in result.reason
    assert "float32" in result.reason
    assert "same dtype" in result.reason


def test_check_dtypes_accepts_matching_dtypes_when_required():
    tensors = (torch.ones(1, dtype=torch.float16), torch.zeros(1, dtype=torch.float16))

    assert check_dtypes(
        tensors,
        frozenset({torch.float16, torch.float32}),
        "grouped GEMM operands",
        require_same=True,
    ).ok


def test_check_dtypes_reports_disallowed_dtype_before_mismatch():
    tensors = (torch.ones(1, dtype=torch.float32), torch.ones(1, dtype=torch.float16))

    result = check_dtypes(
        tensors,
        frozenset({torch.float32}),
        "grouped GEMM operands",
        require_same=True,
    )

    assert not result.ok
    assert result.reason is not None
    assert "requires one of float32" in result.reason


def test_check_dtypes_accepts_dtype_values():
    assert check_dtypes(
        (torch.bfloat16,),
        frozenset({torch.float16, torch.bfloat16}),
        "output dtype",
    ).ok


def test_check_dtypes_requires_tensor_and_dtype_value_to_match():
    result = check_dtypes(
        (torch.empty(1, dtype=torch.float16), torch.bfloat16),
        frozenset({torch.float16, torch.bfloat16}),
        "input and output dtypes",
        require_same=True,
    )

    assert not result.ok
    assert result.reason is not None
    assert "same dtype" in result.reason


def test_check_values_accepts_allowed_values():
    assert check_values(
        (32, "fp8_e8m0"),
        frozenset({32, "fp8_e8m0"}),
        "MXFP8 metadata",
    ).ok


def test_check_values_reports_rejected_and_allowed_values():
    result = check_values(
        (64, "fp32"),
        frozenset({32, "fp8_e8m0"}),
        "MXFP8 metadata",
    )

    assert not result.ok
    assert result.reason is not None
    assert "MXFP8 metadata" in result.reason
    assert "64" in result.reason
    assert "'fp32'" in result.reason
    assert "32" in result.reason
    assert "'fp8_e8m0'" in result.reason


def test_check_nonempty_rejects_empty_tensor():
    result = check_nonempty(torch.empty(0), "grouped GEMM offsets")

    assert not result.ok
    assert result.reason is not None
    assert "grouped GEMM offsets" in result.reason
    assert "nonempty" in result.reason


def test_check_nonempty_accepts_populated_tensor():
    assert check_nonempty(torch.ones(1), "grouped GEMM offsets").ok


def test_check_shape_reports_actual_and_expected_shapes():
    result = check_shape(torch.empty(2, 3), (2, 4), "scaled GEMM A scale")

    assert not result.ok
    assert result.reason is not None
    assert "scaled GEMM A scale" in result.reason
    assert "(2, 3)" in result.reason
    assert "(2, 4)" in result.reason


def test_check_shape_accepts_expected_shape():
    assert check_shape(torch.empty(2, 3), (2, 3), "scaled GEMM A scale").ok


def test_check_dimension_sizes_match_accepts_selected_dimensions():
    tensors = ((torch.empty(2, 3), -1), (torch.empty(3, 4), 0))

    assert check_dimension_sizes_match(tensors, "GEMM contraction dimensions").ok


def test_check_dimension_sizes_match_accepts_more_than_two_dimensions():
    tensors = (
        (torch.empty(2, 3), 0),
        (torch.empty(2, 4), 0),
        (torch.empty(5, 2), 1),
    )

    assert check_dimension_sizes_match(tensors, "group counts").ok


def test_check_dimension_sizes_match_reports_selected_sizes():
    tensors = ((torch.empty(2, 3), -1), (torch.empty(4, 5), 0))

    result = check_dimension_sizes_match(tensors, "GEMM contraction dimensions")

    assert not result.ok
    assert result.reason is not None
    assert "GEMM contraction dimensions" in result.reason
    assert "3" in result.reason
    assert "4" in result.reason
    assert "matching sizes" in result.reason


def test_check_dimension_sizes_match_rejects_invalid_dimension():
    result = check_dimension_sizes_match(
        ((torch.empty(3), -2), (torch.empty(3, 4), -2)),
        "scaled GEMM contraction dimensions",
    )

    assert not result.ok
    assert result.reason is not None
    assert "dimension -2" in result.reason


def test_check_dimension_sizes_multiple_of_accepts_selected_dimensions():
    tensor_dimensions = (
        (torch.empty(2, 16), -1),
        (torch.empty(32, 3), 0),
    )

    assert check_dimension_sizes_multiple_of(
        tensor_dimensions, 16, "GEMM dimensions"
    ).ok


def test_check_dimension_sizes_multiple_of_reports_rejected_sizes():
    tensor_dimensions = (
        (torch.empty(2, 15), -1),
        (torch.empty(18, 3), 0),
    )

    result = check_dimension_sizes_multiple_of(tensor_dimensions, 16, "GEMM dimensions")

    assert not result.ok
    assert result.reason is not None
    assert "GEMM dimensions" in result.reason
    assert "15" in result.reason
    assert "18" in result.reason
    assert "multiples of 16" in result.reason


def test_check_dimension_sizes_multiple_of_rejects_invalid_dimension():
    result = check_dimension_sizes_multiple_of(
        ((torch.empty(3), -2),),
        16,
        "scaled GEMM dimensions",
    )

    assert not result.ok
    assert result.reason is not None
    assert "dimension -2" in result.reason


@pytest.mark.parametrize("ndim", [2, 3])
def test_check_ndim_accepts_allowed_ndims(ndim: int):
    tensor = torch.empty((1,) * ndim)

    assert check_ndim(tensor, frozenset({2, 3}), "grouped operand").ok


def test_check_ndim_reports_actual_and_allowed_ndims():
    result = check_ndim(torch.ones(1), frozenset({2, 3}), "grouped operand")

    assert not result.ok
    assert result.reason is not None
    assert "grouped operand" in result.reason
    assert "1" in result.reason
    assert "2" in result.reason
    assert "3" in result.reason


def test_check_contiguous_reports_layout_requirement():
    tensor = torch.empty(2, 3).t()

    result = check_contiguous(tensor, "scale tensor")

    assert not result.ok
    assert result.reason is not None
    assert "scale tensor" in result.reason
    assert "non-contiguous" in result.reason
    assert "contiguous" in result.reason


def test_check_stride_accepts_allowed_stride():
    tensor = torch.empty_strided((2, 3), (3, 1))

    assert check_stride(tensor, ((3, 1),), "matrix").ok


def test_check_stride_accepts_empty_tensor_with_disallowed_stride():
    tensor = torch.empty_strided((0, 3), (4, 1))

    assert check_stride(tensor, ((3, 1),), "matrix").ok


def test_check_stride_reports_actual_and_allowed_strides():
    tensor = torch.empty_strided((2, 3), (4, 1))

    result = check_stride(tensor, ((3, 1), (1, 2)), "matrix")

    assert not result.ok
    assert result.reason is not None
    assert "matrix" in result.reason
    assert "(4, 1)" in result.reason
    assert "(3, 1)" in result.reason
    assert "(1, 2)" in result.reason


def test_check_alignment_accepts_aligned_non_unit_strides():
    assert check_alignment(torch.empty(4, 4), 16, "grouped operand").ok


def test_check_alignment_reports_stride_bytes_and_requirement():
    result = check_alignment(torch.empty(4, 3), 16, "grouped operand")

    assert not result.ok
    assert result.reason is not None
    assert "grouped operand" in result.reason
    assert "12" in result.reason
    assert "16-byte" in result.reason


@pytest.mark.parametrize("alignment_bytes", [0, -16])
def test_check_alignment_rejects_non_positive_alignment(alignment_bytes: int):
    result = check_alignment(torch.empty(4, 4), alignment_bytes, "grouped operand")

    assert not result.ok
    assert result.reason is not None
    assert str(alignment_bytes) in result.reason
    assert "positive" in result.reason


@pytest.mark.parametrize(
    ("shape", "strides"),
    [
        ((8, 8), (8, 1)),
        ((8, 8), (1, 8)),
        ((64, 64), (80, 1)),
        ((64, 64), (1, 80)),
        ((2, 8, 8), (64, 8, 1)),
        ((2, 8, 8), (64, 1, 8)),
        ((2, 64, 48), (4096, 64, 1)),
        ((2, 64, 48), (3840, 1, 80)),
        ((2, 64, 48), (0, 64, 1)),
        ((1, 64), (0, 1)),
        ((64, 1), (1, 0)),
    ],
)
def test_check_matrix_layout_accepts_aligned_row_or_column_major_layouts(
    shape: tuple[int, ...], strides: tuple[int, ...]
):
    tensor = torch.empty_strided(shape, strides, dtype=torch.bfloat16)

    assert check_matrix_layout(tensor, 16, "grouped operand").ok


@pytest.mark.parametrize(
    ("shape", "strides", "reason"),
    [
        ((8, 8), (512, 8), "row-major or column-major"),
        ((64, 64), (8, 1), "overlapping"),
        ((2, 64, 48), (4096, 8, 1), "overlapping"),
        ((8, 7), (7, 1), "16-byte"),
        ((2, 8, 8), (63, 8, 1), "16-byte"),
    ],
)
def test_check_matrix_layout_rejects_exotic_or_misaligned_layouts(
    shape: tuple[int, ...], strides: tuple[int, ...], reason: str
):
    tensor = torch.empty_strided(shape, strides, dtype=torch.bfloat16)

    result = check_matrix_layout(tensor, 16, "grouped operand")

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


@pytest.mark.parametrize(
    ("actual", "ok"),
    [((8, 8), False), ((8, 9), True), ((9, 0), True), ((12, 0), True)],
)
def test_check_compute_capability_at_least_uses_actual_device(
    monkeypatch: pytest.MonkeyPatch, actual: tuple[int, int], ok: bool
):
    seen: list[torch.device] = []

    def get_device_capability(device: torch.device) -> tuple[int, int]:
        seen.append(device)
        return actual

    monkeypatch.setattr(torch.cuda, "get_device_capability", get_device_capability)
    device = torch.device("cuda:3")

    result = check_compute_capability_at_least(device, (8, 9), "torch scaled GEMM")

    assert result.ok is ok
    assert seen == [device]
    if not ok:
        assert result.reason is not None
        assert "torch scaled GEMM" in result.reason
        assert "SM88" in result.reason
        assert "SM89 or newer" in result.reason


@pytest.mark.parametrize(
    ("actual", "ok"),
    [((8, 9), False), ((9, 0), True), ((10, 0), True), ((12, 0), False)],
)
def test_check_compute_capability_in_uses_closed_major_set(
    monkeypatch: pytest.MonkeyPatch, actual: tuple[int, int], ok: bool
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: actual)

    result = check_compute_capability_in(
        torch.device("cuda:1"),
        frozenset({9, 10}),
        "torch scaled grouped GEMM",
    )

    assert result.ok is ok
    if not ok:
        assert result.reason is not None
        assert "torch scaled grouped GEMM" in result.reason
        assert f"SM{actual[0]}{actual[1]}" in result.reason
        assert "SM90 or SM100" in result.reason


@pytest.mark.parametrize(
    "check",
    [
        lambda device: check_compute_capability_at_least(
            device, (8, 9), "torch scaled GEMM"
        ),
        lambda device: check_compute_capability_in(
            device, frozenset({9, 10}), "torch scaled grouped GEMM"
        ),
    ],
)
def test_compute_capability_checks_reject_non_cuda_without_cuda_query(
    monkeypatch: pytest.MonkeyPatch, check
):
    def unexpected_query(device: torch.device) -> tuple[int, int]:
        raise AssertionError(f"unexpected CUDA query for {device}")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_query)

    result = check(torch.device("cpu"))

    assert not result.ok
    assert result.reason is not None
    assert "cpu" in result.reason
    assert "CUDA" in result.reason


class _TensorMeta:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: str = "cuda:7",
        contiguous: bool = True,
        strides: tuple[int, ...] | None = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self.is_cuda = self.device.type == "cuda"
        self._contiguous = contiguous
        if strides is None:
            running = 1
            calculated = []
            for size in reversed(shape):
                calculated.append(running)
                running *= size
            self._strides = tuple(reversed(calculated))
        else:
            self._strides = strides

    def is_contiguous(self) -> bool:
        return self._contiguous

    def stride(self, dim: int | None = None):
        return self._strides if dim is None else self._strides[dim]

    def element_size(self) -> int:
        return self.dtype.itemsize

    def numel(self) -> int:
        total = 1
        for size in self.shape:
            total *= size
        return total


def _grouped_metadata(layout: str, with_bias: bool):
    offs = _TensorMeta((2,), torch.int32)
    bias = _TensorMeta((2, 48), torch.bfloat16) if with_bias else None
    if layout == "ragged_m":
        return (
            _TensorMeta((64, 64), torch.bfloat16),
            _TensorMeta((2, 64, 48), torch.bfloat16),
            offs,
            bias,
        )
    if layout == "ragged_k":
        return (
            _TensorMeta((32, 64), torch.bfloat16),
            _TensorMeta((64, 48), torch.bfloat16),
            offs,
            bias,
        )
    return (
        _TensorMeta((2, 32, 64), torch.bfloat16),
        _TensorMeta((64, 48), torch.bfloat16),
        offs,
        bias,
    )


def _default_grouped_metadata():
    return _grouped_metadata("ragged_m", False)


def test_eager_grouped_requires_matching_allowed_operand_dtypes():
    args = (
        torch.empty(64, 64, dtype=torch.float16),
        torch.empty(2, 64, 48, dtype=torch.float32),
        torch.tensor([32, 64], dtype=torch.int32),
        None,
    )

    result = eager_gemm.can_implement_grouped_gemm(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "same dtype" in result.reason


@pytest.mark.parametrize(
    "predicate",
    [
        torch_gemm.can_implement_grouped_gemm,
        triton_gemm.can_implement_grouped_gemm,
    ],
)
def test_grouped_backends_reject_empty_offsets(
    monkeypatch: pytest.MonkeyPatch, predicate
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_default_grouped_metadata())
    args[2] = _TensorMeta((0,), torch.int32)

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def test_eager_grouped_rejects_empty_offsets():
    args = (
        torch.empty(64, 64, dtype=torch.bfloat16),
        torch.empty(2, 64, 48, dtype=torch.bfloat16),
        torch.empty(0, dtype=torch.int32),
        None,
    )

    result = eager_gemm.can_implement_grouped_gemm(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


@pytest.mark.parametrize("layout", ["ragged_m", "ragged_k"])
def test_torch_grouped_accepts_supported_bias_layouts(
    monkeypatch: pytest.MonkeyPatch, layout: str
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = _grouped_metadata(layout=layout, with_bias=True)

    assert torch_gemm.can_implement_grouped_gemm(args, {}).ok


def test_torch_grouped_rejects_ragged_n_bias(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    result = torch_gemm.can_implement_grouped_gemm(
        _grouped_metadata(layout="ragged_n", with_bias=True), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert "ragged-N" in result.reason


@pytest.mark.parametrize(
    ("bias", "reason"),
    [
        (_TensorMeta((48,), torch.bfloat16), "rank"),
        (
            _TensorMeta((2, 48), torch.bfloat16, contiguous=False),
            "contiguous",
        ),
        (_TensorMeta((2, 48), torch.float16), "dtype"),
        (_TensorMeta((2, 47), torch.bfloat16), "shape"),
    ],
)
def test_torch_grouped_requires_contiguous_bf16_grouped_bias(
    monkeypatch: pytest.MonkeyPatch, bias: _TensorMeta, reason: str
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_grouped_metadata(layout="ragged_m", with_bias=False))
    args[3] = bias

    result = torch_gemm.can_implement_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


def _scaled_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((1, 48), torch.float32, device),
        torch.bfloat16,
        0,
        None,
        None,
    )


def _cublaslt_scaled_metadata():
    return (
        _TensorMeta((129, 144), torch.float8_e4m3fn),
        _TensorMeta((144, 160), torch.float8_e4m3fn),
        _TensorMeta((129, 5), torch.float32),
        _TensorMeta((5, 160), torch.float32),
        torch.bfloat16,
        32,
        None,
        "fp8_e8m0",
    )


def _torch_scaled_metadata(
    a_dtype: torch.dtype = torch.float8_e4m3fn,
    b_dtype: torch.dtype = torch.float8_e4m3fn,
    block_size: int = 0,
    scale_dtype: str = "fp32",
    m: int = 64,
    n: int = 48,
    k: int = 64,
):
    scale_blocks = (k + block_size - 1) // block_size if block_size else 1
    return (
        _TensorMeta((m, k), a_dtype),
        _TensorMeta((k, n), b_dtype),
        _TensorMeta((m, scale_blocks), torch.float32),
        _TensorMeta((scale_blocks, n), torch.float32),
        torch.bfloat16,
        block_size,
        None,
        scale_dtype,
    )


def _scaled_grouped_metadata(device: str = "cuda:7"):
    return (
        _TensorMeta((64, 64), torch.float8_e4m3fn, device),
        _TensorMeta((2, 64, 48), torch.float8_e4m3fn, device),
        _TensorMeta((64, 1), torch.float32, device),
        _TensorMeta((2, 1, 48), torch.float32, device),
        _TensorMeta((2,), torch.int32, device),
        torch.bfloat16,
        0,
        None,
        None,
    )


@pytest.mark.parametrize(
    "predicate",
    [
        torch_gemm.can_implement_scaled_grouped_gemm,
        triton_gemm.can_implement_scaled_grouped_gemm,
    ],
)
def test_scaled_grouped_backends_reject_empty_offsets(
    monkeypatch: pytest.MonkeyPatch, predicate
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[4] = _TensorMeta((0,), torch.int32)

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def test_eager_scaled_grouped_rejects_empty_offsets():
    args = (
        torch.empty(64, 64, dtype=torch.float8_e4m3fn),
        torch.empty(2, 64, 48, dtype=torch.float8_e4m3fn),
        torch.empty(64, 1, dtype=torch.float32),
        torch.empty(2, 1, 48, dtype=torch.float32),
        torch.empty(0, dtype=torch.int32),
        torch.bfloat16,
        0,
        None,
        None,
    )

    result = eager_gemm.can_implement_scaled_grouped_gemm(args, {})

    assert not result.ok
    assert result.reason is not None
    assert "nonempty" in result.reason


def _triton_scaled_metadata(block_size: int = 0, k: int = 64):
    nblocks = (k + block_size - 1) // block_size if block_size else 1
    return (
        _TensorMeta((64, k), torch.float8_e4m3fn),
        _TensorMeta((k, 48), torch.float8_e4m3fn),
        _TensorMeta((64, nblocks), torch.float32),
        _TensorMeta((nblocks, 48), torch.float32),
        torch.float32,
        block_size,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("name", "mutate", "expected_reason"),
    [
        (
            "e5m2_operands",
            lambda args: args.__setitem__(
                0, _TensorMeta((129, 144), torch.float8_e5m2)
            ),
            "operands",
        ),
        ("wrong_scale_dtype", lambda args: args.__setitem__(7, "fp32"), "scale_dtype"),
        ("wrong_block_size", lambda args: args.__setitem__(5, 64), "block_size"),
        (
            "invalid_logical_scale_shapes",
            lambda args: args.__setitem__(2, _TensorMeta((129, 4), torch.float32)),
            "A scale",
        ),
        ("fp16_output", lambda args: args.__setitem__(4, torch.float16), "bfloat16"),
        (
            "invalid_bias",
            lambda args: args.__setitem__(6, _TensorMeta((160,), torch.float16)),
            "dtype",
        ),
    ],
)
def test_cublaslt_scaled_gemm_rejects_invalid_contracts(
    monkeypatch: pytest.MonkeyPatch, name: str, mutate, expected_reason: str
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_cublaslt_scaled_metadata())
    mutate(args)

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(tuple(args), {})

    assert not result.ok, name
    assert result.reason is not None
    assert expected_reason in result.reason


def test_cublaslt_scaled_gemm_accepts_sm120_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(
        _cublaslt_scaled_metadata(), {}
    )

    assert result.ok


def test_cublaslt_scaled_gemm_rejects_invalid_ndim_before_stride_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_cublaslt_scaled_metadata())
    args[0] = _TensorMeta((144,), torch.float8_e4m3fn)

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "rank" in result.reason


def test_cublaslt_scaled_gemm_rejects_missing_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", "extension unavailable")

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(
        _cublaslt_scaled_metadata(), {}
    )

    assert not result.ok
    assert result.reason == "extension unavailable"


def test_cublaslt_scaled_gemm_checks_contract_before_sm100(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(cublaslt_gemm, "_EXTENSION_ERROR", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    args = list(_cublaslt_scaled_metadata())
    args[0] = _TensorMeta((129, 144), torch.int8)

    result = cublaslt_gemm.can_implement_scaled_gemm_mxfp8(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "operands" in result.reason


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype"),
    [
        (torch.int8, torch.int8),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ],
)
def test_torch_scaled_gemm_accepts_supported_operand_pairs(
    monkeypatch: pytest.MonkeyPatch,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(a_dtype=a_dtype, b_dtype=b_dtype), {}
    )

    assert result.ok, result.reason


@pytest.mark.parametrize(
    ("a_dtype", "b_dtype"),
    [
        (torch.float8_e4m3fn, torch.float8_e5m2),
        (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float8_e4m3fn, torch.int8),
    ],
)
def test_torch_scaled_gemm_rejects_unsupported_operand_pairs(
    monkeypatch: pytest.MonkeyPatch,
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(a_dtype=a_dtype, b_dtype=b_dtype), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert "operand pair" in result.reason


@pytest.mark.parametrize("block_size", [0, 16, 32, 64, 128])
def test_torch_scaled_gemm_accepts_supported_block_sizes(
    monkeypatch: pytest.MonkeyPatch, block_size: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(block_size=block_size, k=256), {}
    )

    assert result.ok, result.reason


@pytest.mark.parametrize("scale_dtype", ["fp32", "fp8_e8m0"])
def test_torch_scaled_gemm_accepts_supported_scale_dtypes(
    monkeypatch: pytest.MonkeyPatch, scale_dtype: str
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(scale_dtype=scale_dtype), {}
    )

    assert result.ok, result.reason


@pytest.mark.parametrize(
    ("m", "n", "k"),
    [(0, 48, 64), (64, 0, 64), (64, 48, 0)],
)
def test_torch_scaled_gemm_rejects_zero_dimensions(
    monkeypatch: pytest.MonkeyPatch, m: int, n: int, k: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(
            a_dtype=torch.int8,
            b_dtype=torch.int8,
            m=m,
            n=n,
            k=k,
        ),
        {},
    )

    assert not result.ok
    assert result.reason == "torch scaled GEMM dimensions must be positive"


def test_torch_scaled_gemm_int8_rejects_unaligned_output_rows(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(
            a_dtype=torch.int8,
            b_dtype=torch.int8,
            block_size=16,
            m=17,
        ),
        {},
    )

    assert not result.ok
    assert result.reason is not None
    assert "torch scaled GEMM INT8 output rows" in result.reason
    assert "17" in result.reason
    assert "multiples of 32" in result.reason


@pytest.mark.parametrize(
    ("n", "k"),
    [(512, 512), (3072, 512), (512, 1536), (384, 512), (512, 192)],
)
def test_torch_scaled_gemm_int8_accepts_shared_model_shapes(
    monkeypatch: pytest.MonkeyPatch, n: int, k: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(
            a_dtype=torch.int8,
            b_dtype=torch.int8,
            block_size=16,
            m=8192,
            n=n,
            k=k,
        ),
        {},
    )

    assert result.ok, result.reason


@pytest.mark.parametrize(("block_size", "k"), [(8, 64), (96, 256)])
def test_torch_scaled_gemm_rejects_unusable_tiling_block_size(
    monkeypatch: pytest.MonkeyPatch, block_size: int, k: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(block_size=block_size, k=k), {}
    )

    assert not result.ok
    assert result.reason == "block_size must be a power of two >= 16"


def test_torch_scaled_gemm_int8_requires_int_mm(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch, "_int_mm", None)

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(a_dtype=torch.int8, b_dtype=torch.int8), {}
    )

    assert not result.ok
    assert result.reason is not None
    assert "torch._int_mm" in result.reason
    assert "callable" in result.reason


def test_torch_scaled_gemm_int8_does_not_require_scaled_mm(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.nn.functional, "scaled_mm", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(a_dtype=torch.int8, b_dtype=torch.int8), {}
    )

    assert result.ok, result.reason


def test_torch_scaled_gemm_fp8_requires_scaled_mm(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.nn.functional, "scaled_mm", None)

    result = torch_gemm.can_implement_scaled_gemm(_torch_scaled_metadata(), {})

    assert not result.ok
    assert result.reason is not None
    assert "torch.nn.functional.scaled_mm" in result.reason
    assert "callable" in result.reason


def test_torch_scaled_gemm_fp8_does_not_require_int_mm(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch, "_int_mm", None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))

    result = torch_gemm.can_implement_scaled_gemm(_torch_scaled_metadata(), {})

    assert result.ok, result.reason


@pytest.mark.parametrize(
    ("actual", "expected"),
    [((7, 5), False), ((8, 0), True), ((8, 9), True), ((12, 0), True)],
)
def test_torch_scaled_gemm_int8_requires_sm80(
    monkeypatch: pytest.MonkeyPatch,
    actual: tuple[int, int],
    expected: bool,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: actual)

    result = torch_gemm.can_implement_scaled_gemm(
        _torch_scaled_metadata(a_dtype=torch.int8, b_dtype=torch.int8), {}
    )

    assert result.ok is expected
    if not expected:
        assert result.reason is not None
        assert "SM80 or newer" in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory", "actual", "expected"),
    [
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            (8, 9),
            True,
        ),
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            (9, 0),
            True,
        ),
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            (10, 0),
            True,
        ),
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            (12, 0),
            True,
        ),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (8, 9), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (9, 0), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (10, 0), True),
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata, (12, 0), True),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (8, 9),
            False,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (9, 0),
            True,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (10, 0),
            True,
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            (12, 0),
            False,
        ),
    ],
)
def test_torch_gemm_validation_architecture_matrix(
    monkeypatch: pytest.MonkeyPatch,
    predicate,
    args_factory,
    actual: tuple[int, int],
    expected: bool,
):
    seen = []
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: seen.append(device) or actual,
    )

    result = predicate(args_factory(), {})

    assert result.ok is expected
    assert seen == [torch.device("cuda:7")]
    if actual == (12, 0) and predicate is torch_gemm.can_implement_scaled_grouped_gemm:
        assert result.reason is not None
        assert "SM120" in result.reason
        assert "SM90 or SM100" in result.reason


@pytest.mark.parametrize(
    ("attribute", "predicate", "args_factory"),
    [
        (
            "grouped_mm",
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
        ),
        ("scaled_mm", torch_gemm.can_implement_scaled_gemm, _scaled_metadata),
        (
            "scaled_grouped_mm",
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
        ),
    ],
)
def test_torch_gemm_validation_requires_callable_functional(
    monkeypatch: pytest.MonkeyPatch, attribute: str, predicate, args_factory
):
    monkeypatch.setattr(torch.nn.functional, attribute, None)

    result = predicate(args_factory(), {})

    assert not result.ok
    assert result.reason is not None
    assert attribute in result.reason
    assert "callable" in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory"),
    [
        (torch_gemm.can_implement_scaled_gemm, _scaled_metadata),
        (torch_gemm.can_implement_scaled_grouped_gemm, _scaled_grouped_metadata),
    ],
)
def test_torch_scaled_gemm_validation_requires_rowwise_scaling_enum(
    monkeypatch: pytest.MonkeyPatch, predicate, args_factory
):
    monkeypatch.setattr(torch.nn.functional, "ScalingType", type("ScalingType", (), {}))

    result = predicate(args_factory(), {})

    assert not result.ok
    assert result.reason is not None
    assert "ScalingType.RowWise" in result.reason


@pytest.mark.parametrize(
    ("a_strides", "b_strides", "expected"),
    [
        ((64, 1), (3072, 48, 1), True),
        ((1, 64), (3072, 1, 64), True),
        ((512, 8), (3072, 48, 1), False),
        ((64, 1), (3072, 512, 8), False),
    ],
)
def test_torch_grouped_gemm_validation_matrix_layouts(
    monkeypatch: pytest.MonkeyPatch,
    a_strides: tuple[int, ...],
    b_strides: tuple[int, ...],
    expected: bool,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_default_grouped_metadata())
    args[0] = _TensorMeta((64, 64), torch.bfloat16, contiguous=False, strides=a_strides)
    args[1] = _TensorMeta(
        (2, 64, 48), torch.bfloat16, contiguous=False, strides=b_strides
    )

    result = torch_gemm.can_implement_grouped_gemm(tuple(args), {})

    assert result.ok is expected
    if not expected:
        assert result.reason is not None
        assert "row-major or column-major" in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory", "index", "replacement", "reason"),
    [
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            2,
            _TensorMeta((2,), torch.int64),
            "int32",
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            4,
            _TensorMeta((2,), torch.int32, contiguous=False, strides=(2,)),
            "contiguous",
        ),
        (
            torch_gemm.can_implement_scaled_gemm,
            _scaled_metadata,
            4,
            torch.float32,
            "output dtype",
        ),
        (
            torch_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            5,
            torch.float16,
            "output dtype",
        ),
        (
            triton_gemm.can_implement_scaled_grouped_gemm,
            _scaled_grouped_metadata,
            8,
            "unknown",
            "scale_dtype",
        ),
    ],
)
def test_gemm_validation_rejects_offset_dtype_or_layout_and_output_or_scale_dtype(
    monkeypatch: pytest.MonkeyPatch,
    predicate,
    args_factory,
    index: int,
    replacement,
    reason: str,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(args_factory())
    args[index] = replacement

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


@pytest.mark.parametrize(
    ("predicate", "args_factory", "index", "replacement", "reason"),
    [
        (
            torch_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            1,
            _TensorMeta((2, 32, 48), torch.bfloat16),
            "contraction",
        ),
        (
            torch_gemm.can_implement_scaled_gemm,
            _scaled_metadata,
            1,
            _TensorMeta((32, 48), torch.float8_e4m3fn),
            "contraction",
        ),
        (
            torch_gemm.can_implement_scaled_gemm,
            _scaled_metadata,
            2,
            _TensorMeta((64, 2), torch.float32),
            "A scale",
        ),
        (
            torch_gemm.can_implement_scaled_gemm,
            _scaled_metadata,
            5,
            32,
            "A scale",
        ),
        (
            torch_gemm.can_implement_scaled_gemm,
            _scaled_metadata,
            6,
            _TensorMeta((64,), torch.float32),
            "dtype",
        ),
        (
            triton_gemm.can_implement_grouped_gemm,
            _default_grouped_metadata,
            3,
            _TensorMeta((2, 47), torch.bfloat16),
            "shape",
        ),
    ],
)
def test_gemm_validation_rejects_shape_block_and_bias_contracts(
    monkeypatch: pytest.MonkeyPatch,
    predicate,
    args_factory,
    index: int,
    replacement,
    reason: str,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(args_factory())
    args[index] = replacement

    result = predicate(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


def test_triton_scaled_gemm_validation_requires_same_device():
    args = list(_scaled_metadata())
    args[2] = _TensorMeta((64, 1), torch.float32, "cuda:6")

    result = triton_gemm.can_implement_scaled_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert "same device" in result.reason


def test_triton_scaled_grouped_gemm_validation_rejects_malformed_scale_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[0] = _TensorMeta((4, 8), torch.int8)
    args[1] = _TensorMeta((2, 8, 6), torch.int8)
    args[2] = _TensorMeta((4, 2), torch.float32)
    args[3] = _TensorMeta((2, 2, 5), torch.float32)

    result = triton_gemm.can_implement_scaled_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason == "invalid scale layout"


def test_triton_grouped_gemm_validation_rejects_bias_for_ragged_n(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_default_grouped_metadata())
    args[0] = _TensorMeta((2, 64, 64), torch.bfloat16)
    args[1] = _TensorMeta((64, 48), torch.bfloat16)
    args[3] = _TensorMeta((2, 48), torch.bfloat16)

    result = triton_gemm.can_implement_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason == "bias is not supported for the ragged-N layout"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("contraction", "contraction dimensions"),
        ("sa_rows", "A scale"),
        ("sa_1d", "rank"),
        ("sb_cols", "B scale"),
        ("sb_blocks", "B scale"),
        ("sb_1d", "rank"),
        ("scale_blocks_vs_block_size", "A scale"),
        ("bias_len", "bias"),
        ("bias_2d", "bias"),
        ("cross_family_dtype", "same dtype family"),
    ],
)
def test_triton_scaled_gemm_validation_rejects_malformed_metadata(
    monkeypatch: pytest.MonkeyPatch, case: str, reason: str
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_triton_scaled_metadata(block_size=32, k=256))
    if case == "contraction":
        args[1] = _TensorMeta((128, 48), torch.float8_e4m3fn)
    elif case == "sa_rows":
        args[2] = _TensorMeta((32, 8), torch.float32)
    elif case == "sa_1d":
        args[2] = _TensorMeta((64,), torch.float32)
    elif case == "sb_cols":
        args[3] = _TensorMeta((8, 24), torch.float32)
    elif case == "sb_blocks":
        args[3] = _TensorMeta((1, 48), torch.float32)
    elif case == "sb_1d":
        args[3] = _TensorMeta((48,), torch.float32)
    elif case == "scale_blocks_vs_block_size":
        args[2] = _TensorMeta((64, 1), torch.float32)
        args[3] = _TensorMeta((1, 48), torch.float32)
    elif case == "bias_len":
        args[6] = _TensorMeta((24,), torch.bfloat16)
    elif case == "bias_2d":
        args[6] = _TensorMeta((1, 48), torch.bfloat16)
    else:
        args[1] = _TensorMeta((256, 48), torch.int8)

    result = triton_gemm.can_implement_scaled_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason is not None
    assert reason in result.reason


@pytest.mark.parametrize(("block_size", "nblocks"), [(8, 32), (96, 3)])
def test_triton_scaled_gemm_validation_rejects_unusable_block_size(
    monkeypatch: pytest.MonkeyPatch, block_size: int, nblocks: int
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_triton_scaled_metadata(block_size=block_size, k=256))

    assert args[2].shape == (64, nblocks)
    result = triton_gemm.can_implement_scaled_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason == "block_size must be a power of two >= 16"


def test_triton_scaled_grouped_gemm_validation_rejects_bias_for_ragged_n(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    args = list(_scaled_grouped_metadata())
    args[0] = _TensorMeta((2, 64, 64), torch.float8_e4m3fn)
    args[1] = _TensorMeta((64, 48), torch.float8_e4m3fn)
    args[2] = _TensorMeta((2, 64, 1), torch.float32)
    args[3] = _TensorMeta((1, 48), torch.float32)
    args[7] = _TensorMeta((2, 48), torch.bfloat16)

    result = triton_gemm.can_implement_scaled_grouped_gemm(tuple(args), {})

    assert not result.ok
    assert result.reason == "bias is not supported for the ragged-N layout"


def _prune_scaled_configs(configs, named_args):
    return triton_gemm._early_prune_scaled_configs(
        configs, named_args, named_args["SCALE_BLOCK_SIZE"], False, None, False
    )


def _prune_scaled_grouped_configs(configs, named_args):
    return triton_gemm._early_prune_scaled_grouped_configs(
        configs,
        named_args,
        36,
        True,
        False,
        named_args["SCALE_BLOCK_SIZE"],
        False,
        None,
        False,
    )


@pytest.mark.parametrize(
    "prune,configs",
    [
        (_prune_scaled_configs, triton_gemm._SCALED_CONFIGS),
        (_prune_scaled_grouped_configs, triton_gemm._SCALED_GROUPED_CONFIGS),
    ],
    ids=["scaled", "scaled_grouped"],
)
def test_triton_autotune_pruning_caps_block_k_at_scale_block(prune, configs):
    for scale_block in (16, 32, 128, 4096):
        kept = prune(list(configs), {"SCALE_BLOCK_SIZE": scale_block})
        assert kept
        assert all(config.kwargs["BLOCK_K"] <= max(32, scale_block) for config in kept)
        assert {id(config) for config in kept} == {
            id(config)
            for config in configs
            if config.kwargs["BLOCK_K"] <= max(32, scale_block)
        }

    assert prune(list(configs), {"SCALE_BLOCK_SIZE": 0}) == list(configs)
