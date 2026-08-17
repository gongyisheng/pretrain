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
                ("scale_dtype", inspect.Parameter.empty),
                ("bias", None),
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
                ("scale_dtype", inspect.Parameter.empty),
                ("bias", None),
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
            torch.float32,
            bias=object(),
            backend="eager",
        )


def test_scaled_gemm_rejects_non_tensor_before_invalid_scale_dtype():
    tensor = torch.empty(1, 1)

    with pytest.raises(
        KernelSelectionError,
        match=r"scaled GEMM received object; requires torch\.Tensor inputs",
    ):
        gemm_module.scaled_gemm(
            object(),
            tensor,
            tensor,
            tensor,
            torch.float32,
            0,
            scale_dtype="invalid",
        )


def test_scaled_gemm_rejects_invalid_scale_dtype_after_tensor_validation():
    with pytest.raises(KernelSelectionError, match="scale_dtype must be torch.dtype"):
        gemm_module.scaled_gemm(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(4, 3, dtype=torch.int8),
            torch.empty(2, 1),
            torch.empty(1, 3),
            torch.bfloat16,
            0,
            scale_dtype="invalid",
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
            torch.float32,
            backend="eager",
        )


def test_scaled_grouped_gemm_rejects_non_tensor_before_invalid_scale_dtype():
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
            scale_dtype="invalid",
        )


def test_scaled_grouped_gemm_rejects_invalid_scale_dtype_after_tensor_validation():
    with pytest.raises(KernelSelectionError, match="scale_dtype must be torch.dtype"):
        gemm_module.scaled_grouped_gemm(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(1, 4, 3, dtype=torch.int8),
            torch.empty(2, 1),
            torch.empty(1, 1, 3),
            torch.tensor([1], dtype=torch.int32),
            torch.bfloat16,
            0,
            scale_dtype="invalid",
        )


def test_grouped_gemm_rejects_contract_failure_through_selector():
    with pytest.raises(KernelSelectionError, match="contraction dimensions"):
        gemm_module.grouped_gemm(
            torch.empty(2, 3),
            torch.empty(4, 5),
            torch.tensor([2], dtype=torch.int64),
        )


def test_scaled_gemm_rejects_contract_failure_through_selector():
    with pytest.raises(KernelSelectionError, match="A scale"):
        gemm_module.scaled_gemm(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(4, 3, dtype=torch.int8),
            torch.empty(2, 2),
            torch.empty(1, 3),
            torch.float32,
            0,
            torch.float32,
        )


def test_scaled_grouped_gemm_rejects_contract_failure_through_selector():
    with pytest.raises(KernelSelectionError, match="3D x 3D"):
        gemm_module.scaled_grouped_gemm(
            torch.empty(2, 3, 4, dtype=torch.int8),
            torch.empty(2, 4, 5, dtype=torch.int8),
            torch.empty(2, 3, 1),
            torch.empty(2, 1, 5),
            torch.tensor([3, 3], dtype=torch.int64),
            torch.float32,
            0,
            torch.float32,
        )


def test_scaled_grouped_gemm_rejects_rank_one_a_scale_through_selector():
    with pytest.raises(KernelSelectionError, match="scaled grouped GEMM A scale"):
        gemm_module.scaled_grouped_gemm(
            torch.empty(2, 4, dtype=torch.int8),
            torch.empty(4, 3, dtype=torch.int8),
            torch.empty(2),
            torch.empty(1, 3),
            torch.tensor([4], dtype=torch.int32),
            torch.float32,
            0,
            torch.float32,
            backend="eager",
        )


def test_grouped_gemm_forwards_normalized_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
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
    }


def test_scaled_gemm_forwards_arguments(monkeypatch: pytest.MonkeyPatch):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
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
    }


def test_scaled_grouped_gemm_forwards_arguments(
    monkeypatch: pytest.MonkeyPatch,
):
    seen = {}
    sentinel = object()

    def fake_dispatch(op, args, kwargs, backend="auto"):
        seen.update(op=op, args=args, kwargs=kwargs, backend=backend)
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
    }
