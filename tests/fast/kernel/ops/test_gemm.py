import inspect

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
from src.quant.quantize import quantize_operand


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GEMM compilation requires CUDA"
)


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
                ("backend", "auto"),
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
                ("bias", None),
                ("scale_dtype", None),
                ("backend", "auto"),
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
                ("bias", None),
                ("scale_dtype", None),
                ("backend", "auto"),
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


def test_grouped_gemm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    a, b, offs, bias = object(), object(), object(), object()

    result = gemm_module.grouped_gemm(a, b, offs, bias=bias, backend="torch")

    assert result is sentinel
    assert seen == {
        "op": "gemm.grouped",
        "args": (a, b, offs, bias),
        "kwargs": {},
        "backend": "torch",
    }


def test_scaled_gemm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq, bq, sa, sb, bias = object(), object(), object(), object(), object()

    result = gemm_module.scaled_gemm(
        aq,
        bq,
        sa,
        sb,
        torch.bfloat16,
        32,
        bias=bias,
        scale_dtype="fp8_e8m0",
        backend="triton",
    )

    assert result is sentinel
    assert seen == {
        "op": "gemm.scaled",
        "args": (aq, bq, sa, sb, torch.bfloat16, 32, bias, "fp8_e8m0"),
        "kwargs": {},
        "backend": "triton",
    }


def test_scaled_grouped_gemm_forwards_normalized_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq, bq, sa, sb, offs, bias = (
        object(),
        object(),
        object(),
        object(),
        object(),
        object(),
    )

    result = gemm_module.scaled_grouped_gemm(
        aq,
        bq,
        sa,
        sb,
        offs,
        torch.bfloat16,
        32,
        bias=bias,
        scale_dtype="fp8_e8m0",
        backend="triton",
    )

    assert result is sentinel
    assert seen == {
        "op": "gemm.scaled_grouped",
        "args": (aq, bq, sa, sb, offs, torch.bfloat16, 32, bias, "fp8_e8m0"),
        "kwargs": {},
        "backend": "triton",
    }


def _scaling(block_size: int) -> dict[str, object]:
    return {
        "granularity": "blockwise",
        "block_shape": (1, block_size),
        "scale_dtype": "fp32",
    }


@cuda_only
def test_grouped_gemm_auto_compiles_fullgraph():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(2, 64, 48, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    compiled = torch.compile(
        lambda: gemm_module.grouped_gemm(a, b, offs), fullgraph=True
    )

    assert torch.isfinite(compiled()).all()


@cuda_only
def test_scaled_gemm_auto_compiles_fullgraph():
    a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 48, device="cuda", dtype=torch.bfloat16)
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", _scaling(32))
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", _scaling(32))

    compiled = torch.compile(
        lambda: gemm_module.scaled_gemm(aq, bq, sa, sb, torch.bfloat16, 32),
        fullgraph=True,
    )

    assert torch.isfinite(compiled()).all()


@cuda_only
def test_scaled_grouped_gemm_auto_compiles_fullgraph():
    aq = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
    bq = torch.randn(2, 64, 48, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(64, 1, device="cuda")
    sb = torch.ones(2, 1, 48, device="cuda")
    offs = torch.tensor([32, 64], device="cuda", dtype=torch.int32)

    compiled = torch.compile(
        lambda: gemm_module.scaled_grouped_gemm(
            aq, bq, sa, sb, offs, torch.bfloat16, 0
        ),
        fullgraph=True,
    )

    assert torch.isfinite(compiled()).all()
