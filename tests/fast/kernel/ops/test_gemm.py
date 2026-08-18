import inspect
from types import SimpleNamespace

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
from src.kernel.registry import KERNEL_REGISTRY
from src.kernel.selector import select_kernel
from src.kernel.spec import PlatformInfo


def test_public_gemm_import_surface():
    from src.kernel.ops import (
        fp8_scaled_grouped_mm,
        fp8_scaled_mm,
        grouped_mm,
        int8_scaled_grouped_mm,
        int8_scaled_mm,
        mxfp8_scaled_grouped_mm,
        mxfp8_scaled_mm,
    )

    assert callable(grouped_mm)
    assert callable(int8_scaled_mm)
    assert callable(fp8_scaled_mm)
    assert callable(mxfp8_scaled_mm)
    assert callable(int8_scaled_grouped_mm)
    assert callable(fp8_scaled_grouped_mm)
    assert callable(mxfp8_scaled_grouped_mm)


_DENSE_SCALED_PARAMETERS = [
    ("aq", inspect.Parameter.empty),
    ("bq", inspect.Parameter.empty),
    ("sa", inspect.Parameter.empty),
    ("sb", inspect.Parameter.empty),
    ("out_dtype", inspect.Parameter.empty),
    ("block_size", inspect.Parameter.empty),
    ("bias", None),
    ("backend", None),
]

_GROUPED_SCALED_PARAMETERS = [
    ("aq", inspect.Parameter.empty),
    ("bq", inspect.Parameter.empty),
    ("sa", inspect.Parameter.empty),
    ("sb", inspect.Parameter.empty),
    ("offs", inspect.Parameter.empty),
    ("out_dtype", inspect.Parameter.empty),
    ("block_size", inspect.Parameter.empty),
    ("bias", None),
    ("backend", None),
]


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (
            gemm_module.grouped_mm,
            [
                ("a", inspect.Parameter.empty),
                ("b", inspect.Parameter.empty),
                ("offs", inspect.Parameter.empty),
                ("bias", None),
                ("backend", None),
            ],
        ),
        (gemm_module.int8_scaled_mm, _DENSE_SCALED_PARAMETERS),
        (gemm_module.fp8_scaled_mm, _DENSE_SCALED_PARAMETERS),
        (gemm_module.mxfp8_scaled_mm, _DENSE_SCALED_PARAMETERS),
        (gemm_module.int8_scaled_grouped_mm, _GROUPED_SCALED_PARAMETERS),
        (gemm_module.fp8_scaled_grouped_mm, _GROUPED_SCALED_PARAMETERS),
        (gemm_module.mxfp8_scaled_grouped_mm, _GROUPED_SCALED_PARAMETERS),
    ],
)
def test_public_gemm_signatures(function, parameters):
    signature = inspect.signature(function)

    assert [
        (parameter.name, parameter.default)
        for parameter in signature.parameters.values()
    ] == parameters


_ALL_OPS = (
    "gemm.grouped_mm",
    "gemm.int8_scaled_mm",
    "gemm.fp8_scaled_mm",
    "gemm.mxfp8_scaled_mm",
    "gemm.int8_scaled_grouped_mm",
    "gemm.fp8_scaled_grouped_mm",
    "gemm.mxfp8_scaled_grouped_mm",
)

_CUBLASLT_POLICY_OPS = ("gemm.mxfp8_scaled_mm",)


@pytest.mark.parametrize("arch", [(8, 0), (8, 9), (10, 0), (11, 0), (12, 0)])
@pytest.mark.parametrize(
    "op", [op for op in _ALL_OPS if op not in _CUBLASLT_POLICY_OPS]
)
def test_ops_without_cublaslt_policy_resolve_without_ambiguity_on_cuda(op, arch):
    """These ops have one optimized backend, so `backend=None` must resolve through
    the real eligibility path on every arch -- either to Triton or, below its SM
    floor, to the eager reference tier. A raise means the windows overlap.
    """
    spec = select_kernel(op, platform=PlatformInfo("cuda", arch))

    assert spec.op == op
    assert spec.backend in ("triton", "eager")


# The mxfp8 policy has two separable axes, tested separately below:
#   _is_kernel_available -- metadata (SM window + build availability)
#   supports_mxfp8_scaled_mm -- per-call (scale width, out dtype, alignment)
_ELIGIBILITY = [
    ("gemm.mxfp8_scaled_mm", (8, 9), False),
    ("gemm.mxfp8_scaled_mm", (10, 0), True),
    ("gemm.mxfp8_scaled_mm", (11, 0), True),
    ("gemm.mxfp8_scaled_mm", (12, 0), True),
]


@pytest.mark.parametrize("op,arch,expected", _ELIGIBILITY)
def test_kernel_availability_follows_the_cublaslt_sm_window(
    monkeypatch: pytest.MonkeyPatch, op, arch, expected
):
    """The metadata axis: hardware window plus build availability, no tensors."""
    monkeypatch.setattr(
        gemm_module, "_platform_for", lambda *_: PlatformInfo("cuda", arch)
    )
    gemm_module._is_kernel_available.cache_clear()
    try:
        assert (
            gemm_module._is_kernel_available(op, "cublaslt", torch.device("cuda", 0))
            is expected
        )
    finally:
        gemm_module._is_kernel_available.cache_clear()


def test_kernel_availability_is_false_without_the_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unbuilt `_C` registers no cuBLASLt spec, so the policy must name Triton
    rather than a backend `select_kernel` would then reject.
    """
    monkeypatch.setattr(
        gemm_module, "_platform_for", lambda *_: PlatformInfo("cuda", (10, 0))
    )
    original = KERNEL_REGISTRY.implementations
    monkeypatch.setattr(
        gemm_module.KERNEL_REGISTRY,
        "implementations",
        lambda op: tuple(s for s in original(op) if s.backend != "cublaslt"),
    )
    gemm_module._is_kernel_available.cache_clear()
    try:
        assert not gemm_module._is_kernel_available(
            "gemm.mxfp8_scaled_mm", "cublaslt", torch.device("cuda", 0)
        )
    finally:
        gemm_module._is_kernel_available.cache_clear()


@pytest.mark.parametrize(
    ("supports", "cublaslt_available", "triton_available", "expected"),
    [
        (True, True, True, "cublaslt"),
        (True, False, True, "triton"),
        (False, True, True, "triton"),
        (False, False, False, None),
    ],
)
def test_select_mxfp8_scaled_mm_backend_needs_both_axes(
    monkeypatch: pytest.MonkeyPatch,
    supports,
    cublaslt_available,
    triton_available,
    expected,
):
    """cuBLASLt only when the hardware allows it and it supports this call."""
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)

    def supports_mxfp8_scaled_mm(aq, bq, out_dtype, block_size):
        return supports

    def kernel_available(op, backend, device):
        return {
            "cublaslt": cublaslt_available,
            "triton": triton_available,
        }[backend]

    monkeypatch.setattr(
        gemm_module,
        "_cublaslt",
        SimpleNamespace(supports_mxfp8_scaled_mm=supports_mxfp8_scaled_mm),
    )
    monkeypatch.setattr(
        gemm_module,
        "_is_kernel_available",
        kernel_available,
    )

    chosen = gemm_module._select_mxfp8_scaled_mm_backend(aq, bq, torch.bfloat16, 32)

    assert chosen == expected


def test_select_mxfp8_scaled_mm_backend_uses_triton_without_the_extension(
    monkeypatch: pytest.MonkeyPatch,
):
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(gemm_module, "_cublaslt", None)

    checked = []

    def kernel_available(op, backend, device):
        checked.append(backend)
        return backend == "triton"

    monkeypatch.setattr(
        gemm_module,
        "_is_kernel_available",
        kernel_available,
    )

    assert (
        gemm_module._select_mxfp8_scaled_mm_backend(aq, bq, torch.bfloat16, 32)
        == "triton"
    )
    assert checked == ["triton"]


def test_mxfp8_scaled_mm_uses_eager_fallback_on_cpu():
    aq = torch.ones(2, 32, dtype=torch.float8_e4m3fn)
    bq = torch.ones(32, 3, dtype=torch.float8_e4m3fn)
    sa = torch.ones(2, 1)
    sb = torch.ones(1, 3)

    result = gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.float32, block_size=32)

    torch.testing.assert_close(result, torch.full((2, 3), 32.0))


def test_grouped_mm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend, device=device)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    a = torch.empty(2, 3)
    b = torch.empty(3, 4)
    offs = torch.tensor([3], dtype=torch.int32)
    bias = torch.empty(1, 4)

    result = gemm_module.grouped_mm(a, b, offs, bias=bias, backend="triton")

    assert result is sentinel
    assert seen == {
        "op": "gemm.grouped_mm",
        "args": (a, b, offs, bias),
        "kwargs": {},
        "backend": "triton",
        "device": a.device,
    }


@pytest.mark.parametrize(
    ("dtype", "expected_backend"),
    [
        (torch.float32, "eager"),
        (torch.float16, None),
        (torch.bfloat16, None),
    ],
)
def test_grouped_mm_leaves_supported_dtypes_for_capability_dispatch(
    monkeypatch: pytest.MonkeyPatch, dtype, expected_backend
):
    """BF16 and FP16 use capability dispatch; unsupported FP32 stays on eager."""
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    a = torch.empty(2, 3, dtype=dtype)
    b = torch.empty(3, 4, dtype=dtype)
    offs = torch.tensor([2], dtype=torch.int32)

    gemm_module.grouped_mm(a, b, offs)

    assert seen["backend"] == expected_backend


@pytest.mark.parametrize(
    ("function", "op"),
    [
        (gemm_module.int8_scaled_mm, "gemm.int8_scaled_mm"),
        (gemm_module.fp8_scaled_mm, "gemm.fp8_scaled_mm"),
    ],
)
def test_dense_scaled_mm_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch, function, op
):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend, device=device)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.int8)
    bq = torch.empty(4, 3, dtype=torch.int8)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)
    bias = torch.empty(3)

    result = function(aq, bq, sa, sb, torch.bfloat16, 0, bias=bias, backend="triton")

    assert result is sentinel
    assert seen == {
        "op": op,
        "args": (aq, bq, sa, sb, torch.bfloat16, 0, bias),
        "kwargs": {},
        "backend": "triton",
        "device": aq.device,
    }


def test_mxfp8_scaled_mm_forwards_the_selected_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend, kwargs=kwargs)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)

    def fake_selector(
        selected_aq, selected_bq, selected_out_dtype, selected_block_size
    ):
        seen.update(
            selector_args=(
                selected_aq,
                selected_bq,
                selected_out_dtype,
                selected_block_size,
            )
        )
        return "cublaslt"

    monkeypatch.setattr(gemm_module, "_select_mxfp8_scaled_mm_backend", fake_selector)

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32)

    assert seen == {
        "backend": "cublaslt",
        "kwargs": {},
        "selector_args": (aq, bq, torch.bfloat16, 32),
    }


def test_mxfp8_scaled_mm_fallback_has_no_private_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend, kwargs=kwargs)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)

    monkeypatch.setattr(
        gemm_module,
        "_select_mxfp8_scaled_mm_backend",
        lambda aq, bq, out_dtype, block_size: "triton",
    )

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32)

    assert seen == {"backend": "triton", "kwargs": {}}


def test_mxfp8_scaled_mm_explicit_cublaslt_has_no_private_kwargs(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend, kwargs=kwargs)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32, backend="cublaslt")

    assert seen == {"backend": "cublaslt", "kwargs": {}}


def test_mxfp8_scaled_grouped_mm_forwards_the_default_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(1, 4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 1, 3)
    offs = torch.tensor([2], dtype=torch.int32)

    gemm_module.mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, torch.bfloat16, 32)

    assert seen["backend"] is None


def test_mxfp8_scaled_grouped_mm_forwards_an_explicit_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(1, 4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 1, 3)
    offs = torch.tensor([2], dtype=torch.int32)

    gemm_module.mxfp8_scaled_grouped_mm(
        aq, bq, sa, sb, offs, torch.bfloat16, 32, backend="triton"
    )

    assert seen["backend"] == "triton"


@pytest.mark.parametrize(
    ("function", "op"),
    [
        (gemm_module.int8_scaled_grouped_mm, "gemm.int8_scaled_grouped_mm"),
        (gemm_module.fp8_scaled_grouped_mm, "gemm.fp8_scaled_grouped_mm"),
        (gemm_module.mxfp8_scaled_grouped_mm, "gemm.mxfp8_scaled_grouped_mm"),
    ],
)
def test_grouped_scaled_mm_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch, function, op
):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend, device=device)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.int8)
    bq = torch.empty(1, 4, 3, dtype=torch.int8)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 1, 3)
    offs = torch.tensor([1], dtype=torch.int32)
    bias = torch.empty(1, 3)

    result = function(
        aq, bq, sa, sb, offs, torch.bfloat16, 0, bias=bias, backend="triton"
    )

    assert result is sentinel
    assert seen == {
        "op": op,
        "args": (aq, bq, sa, sb, offs, torch.bfloat16, 0, bias),
        "kwargs": {},
        "backend": "triton",
        "device": aq.device,
    }
