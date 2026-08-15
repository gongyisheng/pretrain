"""Torch grouped GEMM parity against the eager correctness backend."""

import pytest
import torch
import torch.nn.functional as F

from src.kernel.backends.eager import gemm as eager_gemm
from src.kernel.backends.torch import gemm as torch_gemm
from tests.fast.kernel.backends.helper import (
    GROUPED_LAYOUTS,
    GROUPED_PROJECTIONS,
    make_grouped_inputs,
)


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="Torch grouped GEMM is CUDA only"
    ),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and (
            not callable(getattr(F, "grouped_mm", None))
            or torch.cuda.get_device_capability() < (8, 0)
        ),
        reason="Torch grouped GEMM requires torch.nn.functional.grouped_mm and SM80+",
    ),
]


@pytest.mark.parametrize("projection", GROUPED_PROJECTIONS, ids=lambda case: case.name)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("with_bias", [False, True], ids=["no-bias", "bias"])
def test_grouped_gemm_precision(projection, layout, with_bias):
    if layout == "ragged_n" and with_bias:
        pytest.skip("ragged-N has no defined per-expert output-channel bias")
    a, b, offs, bias = make_grouped_inputs(projection, layout, with_bias)

    actual = torch_gemm.grouped_gemm(a, b, offs, bias)
    expected = eager_gemm.grouped_gemm(a, b, offs, bias)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
