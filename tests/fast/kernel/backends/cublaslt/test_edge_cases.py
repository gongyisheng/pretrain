"""Edge contracts for the cuBLASLt MXFP8 backend."""

import runpy
from pathlib import Path

import pytest
import setuptools
import torch

from benchmarks.gemm.bench_scaled_mm import _require_mxfp8_precision
from src.kernel.ops.gemm import mxfp8_scaled_mm
from src.kernel.selector import select_kernel


cuda_mxfp8_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuBLASLt MXFP8 GEMM requires CUDA"
)


def _require_mxfp8_device() -> None:
    try:
        from src.kernel.backends.cublaslt import _C  # noqa: F401
    except ImportError as error:
        pytest.skip(f"cuBLASLt extension unavailable: {error}")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("cuBLASLt MXFP8 GEMM requires SM100 or newer")


def _fp8_values(count: int, dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    values = (torch.arange(count, device="cuda", dtype=torch.float32) % 17 - 8) / 16
    return values.to(dtype)


def _view_operands(case: str) -> tuple[object, ...]:
    m, k, n = 16, 32, 16
    a = _fp8_values(m * k).view(m, k)
    b = _fp8_values(k * n).view(k, n)
    if case == "padded_a":
        a = _fp8_values(m * (k + 16)).view(m, k + 16)[:, :k]
        assert a.stride() == (k + 16, 1)
    elif case == "padded_b":
        b = _fp8_values(n * (k + 16)).view(n, k + 16)[:, :k].t()
        assert b.stride() == (1, k + 16)
    elif case == "offset_a":
        a = _fp8_values(m * k + 1)[1:].view(m, k)
        assert a.data_ptr() % 16 != 0
    else:
        b = _fp8_values(n * k + 1)[1:].view(n, k).t()
        assert b.data_ptr() % 16 != 0
    scale_a = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((1, n), device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b, torch.bfloat16, 32, None


@cuda_mxfp8_only
@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("padded_a", "MXFP8 GEMM A has stride"),
        ("padded_b", "MXFP8 GEMM B has stride"),
    ],
)
def test_forced_backend_rejects_unsafe_operand_view(case: str, reason: str):
    _require_mxfp8_device()
    args = _view_operands(case)

    with pytest.raises(ValueError, match=reason):
        mxfp8_scaled_mm(*args, backend="cublaslt")


def _valid_dense_operands(block_size: int) -> tuple[object, ...]:
    m, k, n = 16, 32, 16
    a = _fp8_values(m * k).view(m, k)
    b = _fp8_values(k * n).view(k, n)
    scale_a = torch.ones((m, 1), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((1, n), device="cuda", dtype=torch.float32)
    return a, b, scale_a, scale_b, torch.bfloat16, block_size, None


@cuda_mxfp8_only
def test_forced_backend_rejects_bad_scale_config():
    """cuBLASLt reads scales as bare 32-wide e8m0 blocks -- unlike Triton it does
    not replicate a wider scale across blocks, so block_size must be checked
    explicitly rather than left to silently miscompute."""
    _require_mxfp8_device()
    args = _valid_dense_operands(64)

    with pytest.raises(ValueError, match="requires block_size=32"):
        mxfp8_scaled_mm(*args, backend="cublaslt")


@cuda_mxfp8_only
@pytest.mark.parametrize("backend", ["cublaslt"])
@pytest.mark.parametrize("case", ["offset_a", "offset_b"])
def test_misaligned_operand_executes_with_cublaslt(case: str, backend: str):
    _require_mxfp8_device()
    args = _view_operands(case)

    selected = select_kernel("gemm.mxfp8_scaled_mm", backend=backend)
    expected = mxfp8_scaled_mm(*args, backend="eager")
    actual = mxfp8_scaled_mm(*args, backend=backend)

    assert selected.backend == "cublaslt"
    _require_mxfp8_precision(f"misaligned {case} {backend}", actual, expected)


def _zero_operands(shape: tuple[int, int, int], with_bias: bool) -> tuple[object, ...]:
    m, k, n = shape
    blocks = (k + 31) // 32
    a = torch.empty((m, k), device="cuda", dtype=torch.float8_e4m3fn)
    b = torch.empty((k, n), device="cuda", dtype=torch.float8_e4m3fn)
    scale_a = torch.ones((m, blocks), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((blocks, n), device="cuda", dtype=torch.float32)
    bias = (
        torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)
        if with_bias
        else None
    )
    return (
        a,
        b,
        scale_a,
        scale_b,
        torch.bfloat16,
        32,
        bias,
    )


@cuda_mxfp8_only
@pytest.mark.parametrize("backend", [None, "cublaslt"])
@pytest.mark.parametrize(
    ("shape", "with_bias"),
    [
        ((0, 16, 16), False),
        ((16, 16, 0), False),
        ((16, 0, 16), False),
        ((16, 0, 16), True),
    ],
)
def test_zero_size_gemm_matches_eager(
    backend: str | None, shape: tuple[int, int, int], with_bias: bool
):
    _require_mxfp8_device()
    args = _zero_operands(shape, with_bias)

    expected = mxfp8_scaled_mm(*args, backend="eager")
    actual = mxfp8_scaled_mm(*args, backend=backend)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("out_values", "reference_values"),
    [
        ([0.0], [0.0]),
        ([0.0], [float("inf")]),
    ],
)
def test_precision_gate_rejects_nonfinite_diagnostics(
    out_values: list[float], reference_values: list[float]
):
    out = torch.tensor(out_values)
    reference = torch.tensor(reference_values)

    with pytest.raises(AssertionError, match="MXFP8 eager-relative error gate"):
        _require_mxfp8_precision("test", out, reference)


def test_cublaslt_build_is_classified_as_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))

    runpy.run_path(str(Path(__file__).parents[5] / "setup.py"), run_name="__main__")

    extensions = captured["ext_modules"]
    cublaslt = next(
        extension
        for extension in extensions
        if extension.name == "src.kernel.backends.cublaslt._C"
    )
    assert any(Path(source).suffix == ".cu" for source in cublaslt.sources)
