from types import SimpleNamespace

import pytest
import torch

from src.kernel.utils import (
    check_alignment,
    check_callable,
    check_compute_capability_at_least,
    check_compute_capability_in,
    check_contiguous,
    check_cuda_tensors,
    check_dtypes,
    check_matrix_layout,
    check_rank,
    check_same_device,
)


class _Owner:
    available = staticmethod(lambda: None)
    not_callable = None


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


@pytest.mark.parametrize("ndim", [2, 3])
def test_check_rank_accepts_allowed_ranks(ndim: int):
    tensor = torch.empty((1,) * ndim)

    assert check_rank(tensor, frozenset({2, 3}), "grouped operand").ok


def test_check_rank_reports_actual_and_allowed_ranks():
    result = check_rank(torch.ones(1), frozenset({2, 3}), "grouped operand")

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
