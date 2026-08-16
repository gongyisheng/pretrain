import inspect

import pytest
import torch

from src.kernel.ops import gemm as gemm_module
from src.kernel.selector import KernelSelectionError


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


def test_grouped_gemm_rejects_input_without_tensor_metadata():
    with pytest.raises(
        KernelSelectionError,
        match=r"grouped GEMM received object; requires torch\.Tensor inputs",
    ):
        gemm_module.grouped_gemm(
            object(),
            torch.empty(1, 1),
            torch.tensor([1], dtype=torch.int32),
            backend="eager",
        )


def test_scaled_gemm_rejects_input_without_tensor_metadata():
    tensor = torch.empty(1, 1)

    with pytest.raises(
        KernelSelectionError,
        match=r"scaled GEMM received object; requires torch\.Tensor inputs",
    ):
        gemm_module.scaled_gemm(
            tensor,
            tensor,
            tensor,
            tensor,
            torch.float32,
            0,
            bias=object(),
            backend="eager",
        )


def test_scaled_grouped_gemm_rejects_input_without_tensor_metadata():
    tensor = torch.empty(1, 1)

    with pytest.raises(
        KernelSelectionError,
        match=r"scaled grouped GEMM received object; requires torch\.Tensor inputs",
    ):
        gemm_module.scaled_grouped_gemm(
            tensor,
            tensor,
            tensor,
            tensor,
            object(),
            torch.float32,
            0,
            backend="eager",
        )


def test_grouped_gemm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    a = torch.empty(0)
    b = torch.empty(0)
    offs = torch.empty(0, dtype=torch.int32)
    bias = torch.empty(0)

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
    aq = torch.empty(0)
    bq = torch.empty(0)
    sa = torch.empty(0)
    sb = torch.empty(0)
    bias = torch.empty(0)

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
        "args": (
            aq,
            bq,
            sa,
            sb,
            torch.bfloat16,
            32,
            bias,
            torch.float8_e8m0fnu,
        ),
        "kwargs": {},
        "backend": "triton",
    }


def test_scaled_gemm_defaults_scale_dtype_to_fp32(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen["args"] = args

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    tensor = torch.empty(0)

    gemm_module.scaled_gemm(
        tensor,
        tensor,
        tensor,
        tensor,
        torch.bfloat16,
        32,
    )

    assert seen["args"][-1] == torch.float32


def test_scaled_grouped_gemm_forwards_normalized_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
        return sentinel

    monkeypatch.setattr(gemm_module, "dispatch", fake_dispatch)
    aq = torch.empty(0)
    bq = torch.empty(0)
    sa = torch.empty(0)
    sb = torch.empty(0)
    offs = torch.empty(0, dtype=torch.int32)
    bias = torch.empty(0)

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
        "args": (
            aq,
            bq,
            sa,
            sb,
            offs,
            torch.bfloat16,
            32,
            bias,
            torch.float8_e8m0fnu,
        ),
        "kwargs": {},
        "backend": "triton",
    }
