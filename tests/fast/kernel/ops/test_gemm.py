import inspect

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
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

_WRAPPER_DEFAULT_BACKEND = {
    "gemm.mxfp8_scaled_mm": gemm_module._MXFP8_DENSE_BACKEND,
    "gemm.mxfp8_scaled_grouped_mm": gemm_module._MXFP8_GROUPED_BACKEND,
}


@pytest.mark.parametrize("arch", [(8, 0), (8, 9), (10, 0), (11, 0), (12, 0)])
@pytest.mark.parametrize("op", _ALL_OPS)
def test_every_op_resolves_without_ambiguity_on_cuda(op, arch):
    """Each wrapper must resolve to exactly one backend on every arch, using the
    registry directly (no GPU needed): its own pinned default for mxfp8, or the
    sole eligible optimized backend otherwise. A raise here means a wrapper forgot
    to apply its default -- the gap that let mxfp8_scaled_grouped_mm raise
    KernelSelectionError on SM100/SM110 (both triton and cublaslt eligible)
    undetected, since existing tests always passed backend="triton" explicitly.
    """
    platform = PlatformInfo("cuda", arch)
    spec = select_kernel(
        op, backend=_WRAPPER_DEFAULT_BACKEND.get(op), platform=platform
    )
    assert spec.op == op


def test_grouped_gemm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
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
    [(torch.float32, "eager"), (torch.bfloat16, None)],
)
def test_grouped_gemm_routes_non_bf16_to_eager_by_default(
    monkeypatch: pytest.MonkeyPatch, dtype, expected_backend
):
    """Triton's grouped kernel is bf16-only; the capability gate has no dtype axis,
    so the wrapper itself must route away from triton for every other dtype."""
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


def test_mxfp8_scaled_mm_defaults_to_the_pinned_dense_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32)

    assert seen["backend"] == gemm_module._MXFP8_DENSE_BACKEND


def test_mxfp8_scaled_mm_explicit_backend_overrides_the_pinned_default(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend=None, device=None):
        seen.update(backend=backend)
        return None

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(2, 4, dtype=torch.float8_e4m3fn)
    bq = torch.empty(4, 3, dtype=torch.float8_e4m3fn)
    sa = torch.empty(2, 1)
    sb = torch.empty(1, 3)

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32, backend="triton")

    assert seen["backend"] == "triton"


def test_mxfp8_scaled_grouped_mm_defaults_to_the_pinned_grouped_backend(
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

    assert seen["backend"] == gemm_module._MXFP8_GROUPED_BACKEND


def test_mxfp8_scaled_grouped_mm_explicit_backend_overrides_the_pinned_default(
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
