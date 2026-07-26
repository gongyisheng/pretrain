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


def test_parity_empty_and_uneven_groups():
    # includes an empty group (0 rows) and uneven sizes
    ref, got = _ref_and_triton(counts=[5, 0, 130, 41, 1], K=64, N=48)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_parity_single_group():
    ref, got = _ref_and_triton(counts=[200], K=192, N=64)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize(
    "K,N",
    [
        (64, 384),  # e512_k48 gate_up
        (192, 64),  # e512_k48 down
        (512, 384),  # e64_k6 gate_up
        (192, 512),  # e64_k6 down
    ],
)
def test_parity_config_shapes(K, N):
    # skewed load across 16 experts, summing to a realistic row count
    counts = [300, 5, 120, 0, 44, 210, 7, 90, 33, 150, 1, 260, 12, 80, 40, 400]
    ref, got = _ref_and_triton(counts=counts, K=K, N=N, seed=1)
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)
