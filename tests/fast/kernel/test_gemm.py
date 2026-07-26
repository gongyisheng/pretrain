"""Parity of the Triton grouped GEMM prototype vs torch._grouped_mm (bf16)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton grouped GEMM is CUDA+bf16 only"
)


def _ref_and_triton(counts, K, N, seed=0):
    """Build (a, b=w.mT, offs) and return (torch._grouped_mm, triton_grouped_mm)."""
    from src.kernel.gemm import triton_grouped_mm

    torch.manual_seed(seed)
    E = len(counts)
    R = sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = (
        torch.randn(E, N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    )  # (E, out=N, in=K)
    b = w.mT  # (E, K, N), transposed view — non-contiguous
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    ref = torch._grouped_mm(a, b, offs=offs)
    got = triton_grouped_mm(a, b, offs)
    return ref, got


def test_parity_balanced_groups():
    ref, got = _ref_and_triton(counts=[128, 128, 128, 128], K=64, N=48)
    assert got.shape == ref.shape
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
