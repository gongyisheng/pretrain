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


def _operands_spread_scales(M, K, N, seed=0):
    """Operands whose per-scale-block magnitudes span 2^-8..2^7.

    A's magnitude varies per (row, K-block) and B's per (K-block, column), so both
    axes of *both* scale-factor fragment mappings carry signal. Uniform magnitudes
    would leave SFB's N axis untested, which is the half of the mapping that does not
    mirror SFA -- it reads a different thread coordinate and regroups its modes.
    """
    torch.manual_seed(seed)
    n_blocks = K // 32
    exp = torch.randint(-8, 8, (M, n_blocks), device="cuda").float()
    a = torch.randn(M, n_blocks, 32, device="cuda") * torch.exp2(exp)[:, :, None]
    exp = torch.randint(-8, 8, (n_blocks, N), device="cuda").float()
    b = torch.randn(n_blocks, 32, N, device="cuda") * torch.exp2(exp)[:, None, :]
    aq, sa = quantize_operand(a.reshape(M, K).bfloat16(), -1, "fp8_e4m3", MX)
    bq, sb = quantize_operand(b.reshape(K, N).bfloat16(), -2, "fp8_e4m3", MX)
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
    # in tests/fast/kernel/triton/test_triton_gemm.py::test_scaled_gemm_mx_matches_oracle.
    # Elementwise tolerances give spurious failures on block-quantized outputs.
    assert (got - want).norm() / want.norm() < 0.02


@pytest.mark.parametrize("shape", [(256, 512, 128), (128, 256, 256), (384, 128, 128)])
def test_scaled_gemm_mxfp8_adds_no_error_over_dequantizing(shape):
    """Applying the e8m0 scales inside the MMA must be as exact as dequantizing first.

    The bound is fp32-rounding tight (measured 1.1e-7..2.9e-7) on purpose. It sits an
    order of magnitude above the Triton test's 2e-8 because these operands span 2^15
    of per-block magnitude, and summing terms of such different size costs fp32
    precision -- that spread is the point, since it denies a misplaced scale anywhere
    to hide.

    What the tight bound buys over the 2e-2 oracle comparison in
    `test_scaled_gemm_mxfp8_matches_oracle` is precision, not placement. Verified by
    mutation: rounding the result path through bf16 -- standing in for scales or
    accumulation applied at less than fp32 -- passes every 2e-2 assertion in this file
    and fails all three shapes here. Placement is caught either way, because an e8m0
    scale is a power of two, so landing one on the wrong row, column or k-tile is at
    least a factor-2 error and never sits inside the quantization noise.

    This is the assertion that pins the instruction's arithmetic, and the one the
    pipelining work -- which rewrites which k-tile's scales are resident when the MMA
    fires -- must keep green. It is the CuTe counterpart of the Triton backend's
    `test_scaled_gemm_mxfp8_adds_no_error_over_dequantizing`.
    """
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands_spread_scales(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    oracle = scaled_gemm_ref(aq, bq, sa, sb, 32)
    rel = (got - oracle).norm() / oracle.norm()
    assert rel < 1e-6, rel


def test_compile_cache_reused():
    """M is dynamic in the compiled kernel, so a batch-size change must not recompile."""
    from src.kernel.cute import gemm as cute_gemm

    aq, bq, sa, sb = _operands(128, 128, 128)
    cute_gemm.scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32)
    n_after_first = len(cute_gemm._COMPILED)

    aq2, bq2, sa2, sb2 = _operands(256, 128, 128, seed=1)  # different M, same K/N
    got = cute_gemm.scaled_gemm_mxfp8(aq2, bq2, sa2, sb2, torch.float32)

    # without this, dropping the store into _COMPILED would leave both checks below
    # passing while every call silently re-JITs -- the exact cost the cache prevents
    assert n_after_first >= 1, "the first call must have cached a compiled kernel"
    assert len(cute_gemm._COMPILED) == n_after_first, "M must not trigger a recompile"
    # a cached kernel with M baked static would satisfy the count check above while
    # returning garbage here, so assert the reused kernel is actually correct
    want = scaled_gemm_ref(aq2, bq2, sa2, sb2, 32)
    assert (got - want).norm() / want.norm() < 0.02
