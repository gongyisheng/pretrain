"""CuTe DSL mxfp8 kernels (sm120). Numerics are checked against the same
oracle the Triton backend uses, so the two cannot silently diverge."""

import pytest
import torch

from src.quant.quantize import quantize_operand
from tests.fast.kernel._refs import scaled_gemm_ref

E4M3 = torch.float8_e4m3fn
MX = {"granularity": "blockwise", "block_size": 32, "scale_dtype": "fp8_e8m0"}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0),
    reason="CuTe mxfp8 kernels are sm120-only",
)


def _operands(M, K, N, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    aq, sa = quantize_operand(a, -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b, -2, "fp8_e4m3", MX)
    return aq, bq, sa, sb


@pytest.mark.parametrize("shape", [(128, 128, 128), (256, 128, 256)])
def test_scaled_gemm_mxfp8_matches_oracle(shape):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    want = scaled_gemm_ref(aq, bq, sa, sb, 32)

    assert got.shape == (M, N)
    assert got.dtype == torch.float32
    # relative Frobenius norm, matching the bar the Triton mxfp8 kernel is held to
    # in tests/fast/kernel/triton/test_gemm.py::test_scaled_gemm_mx_matches_oracle.
    # Elementwise tolerances give spurious failures on block-quantized outputs.
    assert (got - want).norm() / want.norm() < 0.02
