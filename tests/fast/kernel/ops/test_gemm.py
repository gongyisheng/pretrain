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
#   _cublaslt_eligible -- metadata (SM window + whether the extension was built)
#   serves_dense -- per-call (scale width, out dtype, alignment)
_ELIGIBILITY = [
    ("gemm.mxfp8_scaled_mm", (8, 9), False),
    ("gemm.mxfp8_scaled_mm", (10, 0), True),
    ("gemm.mxfp8_scaled_mm", (11, 0), True),
    ("gemm.mxfp8_scaled_mm", (12, 0), True),
]


@pytest.mark.parametrize("op,arch,expected", _ELIGIBILITY)
def test_cublaslt_eligibility_follows_the_sm_window(
    monkeypatch: pytest.MonkeyPatch, op, arch, expected
):
    """The metadata axis: hardware window plus build availability, no tensors."""
    monkeypatch.setattr(
        gemm_module, "_platform_for", lambda *_: PlatformInfo("cuda", arch)
    )
    gemm_module._cublaslt_eligible.cache_clear()
    try:
        assert gemm_module._cublaslt_eligible(op, "cuda", 0) is expected
    finally:
        gemm_module._cublaslt_eligible.cache_clear()


def test_cublaslt_eligibility_is_false_without_the_extension(
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
    gemm_module._cublaslt_eligible.cache_clear()
    try:
        assert not gemm_module._cublaslt_eligible("gemm.mxfp8_scaled_mm", "cuda", 0)
    finally:
        gemm_module._cublaslt_eligible.cache_clear()


@pytest.mark.parametrize(
    ("serves", "eligible", "expected"),
    [(True, True, "cublaslt"), (True, False, "triton"), (False, True, "triton")],
)
def test_mxfp8_backend_needs_both_axes(
    monkeypatch: pytest.MonkeyPatch, serves, eligible, expected
):
    """cuBLASLt only when the hardware allows it *and* it can serve this call."""
    monkeypatch.setattr(gemm_module, "_cublaslt_eligible", lambda *_: eligible)

    chosen = gemm_module._mxfp8_backend(
        "gemm.mxfp8_scaled_mm", serves, torch.device("cuda", 0)
    )

    assert chosen == expected


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
    [(torch.float32, "eager"), (torch.bfloat16, None)],
)
def test_grouped_mm_routes_non_bf16_to_eager_by_default(
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


def test_mxfp8_scaled_mm_forwards_the_resolved_backend(
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
        gemm_module, "_cublaslt", SimpleNamespace(serves_dense=lambda *_: True)
    )
    monkeypatch.setattr(gemm_module, "_mxfp8_backend", lambda *_: "cublaslt")

    gemm_module.mxfp8_scaled_mm(aq, bq, sa, sb, torch.bfloat16, 32)

    assert seen == {"backend": "cublaslt", "kwargs": {"_dense_contract_checked": True}}


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
        gemm_module, "_cublaslt", SimpleNamespace(serves_dense=lambda *_: False)
    )
    monkeypatch.setattr(gemm_module, "_mxfp8_backend", lambda *_: "triton")

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
