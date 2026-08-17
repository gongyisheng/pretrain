import inspect

import pytest
import torch

from src.kernel.ops import gemm as gemm_module


def test_public_gemm_import_surface():
    from src.kernel.ops import grouped_gemm, scaled_gemm, scaled_grouped_gemm

    assert callable(grouped_gemm)
    assert callable(scaled_gemm)
    assert callable(scaled_grouped_gemm)


@pytest.mark.parametrize(
    ("function", "parameters"),
    [
        (
            gemm_module.grouped_gemm,
            [
                ("a", inspect.Parameter.empty),
                ("b", inspect.Parameter.empty),
                ("offs", inspect.Parameter.empty),
                ("bias", None),
                ("backend", None),
            ],
        ),
        (
            gemm_module.scaled_gemm,
            [
                ("aq", inspect.Parameter.empty),
                ("bq", inspect.Parameter.empty),
                ("sa", inspect.Parameter.empty),
                ("sb", inspect.Parameter.empty),
                ("out_dtype", inspect.Parameter.empty),
                ("block_size", inspect.Parameter.empty),
                ("scale_dtype", inspect.Parameter.empty),
                ("bias", None),
                ("backend", None),
            ],
        ),
        (
            gemm_module.scaled_grouped_gemm,
            [
                ("aq", inspect.Parameter.empty),
                ("bq", inspect.Parameter.empty),
                ("sa", inspect.Parameter.empty),
                ("sb", inspect.Parameter.empty),
                ("offs", inspect.Parameter.empty),
                ("out_dtype", inspect.Parameter.empty),
                ("block_size", inspect.Parameter.empty),
                ("scale_dtype", inspect.Parameter.empty),
                ("bias", None),
                ("backend", None),
            ],
        ),
    ],
)
def test_public_gemm_signatures(function, parameters):
    signature = inspect.signature(function)

    assert [
        (parameter.name, parameter.default)
        for parameter in signature.parameters.values()
    ] == parameters


def test_scaled_gemm_requires_scale_dtype():
    aq = bq = sa = sb = object()

    with pytest.raises(TypeError, match="scale_dtype"):
        inspect.signature(gemm_module.scaled_gemm).bind(
            aq, bq, sa, sb, torch.bfloat16, 0
        )


def test_scaled_grouped_gemm_requires_scale_dtype():
    aq = bq = sa = sb = offs = object()

    with pytest.raises(TypeError, match="scale_dtype"):
        inspect.signature(gemm_module.scaled_grouped_gemm).bind(
            aq, bq, sa, sb, offs, torch.bfloat16, 0
        )


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

    result = gemm_module.grouped_gemm(a, b, offs, bias=bias, backend="triton")

    assert result is sentinel
    assert seen == {
        "op": "gemm.grouped",
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

    gemm_module.grouped_gemm(a, b, offs)

    assert seen["backend"] == expected_backend


def test_scaled_gemm_forwards_arguments(monkeypatch: pytest.MonkeyPatch):
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

    result = gemm_module.scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        0,
        torch.float8_e8m0fnu,
        bias=bias,
        backend="triton",
    )

    assert result is sentinel
    assert seen == {
        "op": "gemm.scaled",
        "args": (
            aq,
            bq,
            sa,
            sb,
            torch.bfloat16,
            0,
            torch.float8_e8m0fnu,
            bias,
        ),
        "kwargs": {},
        "backend": "triton",
        "device": aq.device,
    }


def test_scaled_grouped_gemm_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
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

    result = gemm_module.scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.bfloat16,
        0,
        torch.float8_e8m0fnu,
        bias=bias,
        backend="triton",
    )

    assert result is sentinel
    assert seen == {
        "op": "gemm.scaled_grouped",
        "args": (
            aq,
            bq,
            sa,
            sb,
            offs,
            torch.bfloat16,
            0,
            torch.float8_e8m0fnu,
            bias,
        ),
        "kwargs": {},
        "backend": "triton",
        "device": aq.device,
    }
