"""Validation for optional dense quantization statistics."""

import pytest
import torch

from src.kernel import ops

QUANTIZERS = ["quantize_fp8", "quantize_int8", "quantize_mxfp8", "quantize_nvfp4"]
INVALID_REQUESTS = [((2, 32, 32), True), ((32, 32), 1), ((32, 32), "yes")]


@pytest.mark.parametrize("quantizer", QUANTIZERS)
@pytest.mark.parametrize("invalid_request", INVALID_REQUESTS)
def test_quantize_quantization_stats_raise_error(quantizer, invalid_request):
    shape, collect = invalid_request
    with pytest.raises(ValueError):
        getattr(ops, quantizer)(
            torch.ones(shape),
            contract_dim=-1,
            block_shape=(1, 16),
            backend="eager",
            return_quantization_stats=collect,
        )
