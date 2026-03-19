"""Tests for Triton RMSNorm kernels: single-pass and tiled."""

import itertools

import pytest
import torch

from src.kernel.rmsnorm import triton_rmsnorm, triton_rmsnorm_tiled


def pytorch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Reference PyTorch implementation."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return weight * x.to(dtype)


# -- Test ranges ---------------------------------------------------------------

SHAPES = [
    (1, 1, 64),
    (2, 128, 768),
    (4, 512, 1024),
    (1, 1, 1),
    (1, 1, 127),       # non-power-of-2
    (2, 256, 1600),    # GPT-2 large
]
DTYPES = [
    (torch.float32, 1e-5),
    (torch.float16, 1e-2),
    (torch.bfloat16, 1e-2),
]


# -- Single-pass kernel --------------------------------------------------------

class TestRMSNormSinglePass:

    @pytest.mark.parametrize("B, T, D", SHAPES)
    def test_correctness(self, B, T, D):
        x = torch.randn(B, T, D, device="cuda")
        w = torch.randn(D, device="cuda")
        torch.testing.assert_close(
            triton_rmsnorm(x, w), pytorch_rmsnorm(x, w), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("dtype, atol", DTYPES)
    def test_dtypes(self, dtype, atol):
        x = torch.randn(2, 128, 768, device="cuda", dtype=dtype)
        w = torch.randn(768, device="cuda", dtype=dtype)
        torch.testing.assert_close(
            triton_rmsnorm(x, w), pytorch_rmsnorm(x, w), atol=atol, rtol=atol
        )

    def test_preserves_shape(self):
        x = torch.randn(2, 64, 256, device="cuda")
        w = torch.randn(256, device="cuda")
        assert triton_rmsnorm(x, w).shape == x.shape

    def test_preserves_dtype(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 64, 256, device="cuda", dtype=dtype)
            w = torch.randn(256, device="cuda", dtype=dtype)
            assert triton_rmsnorm(x, w).dtype == dtype

    def test_2d_input(self):
        x = torch.randn(128, 768, device="cuda")
        w = torch.randn(768, device="cuda")
        torch.testing.assert_close(
            triton_rmsnorm(x, w), pytorch_rmsnorm(x, w), atol=1e-5, rtol=1e-5
        )


# -- Tiled kernel --------------------------------------------------------------

class TestRMSNormTiled:

    @pytest.mark.parametrize("B, T, D", SHAPES)
    def test_correctness(self, B, T, D):
        x = torch.randn(B, T, D, device="cuda")
        w = torch.randn(D, device="cuda")
        torch.testing.assert_close(
            triton_rmsnorm_tiled(x, w), pytorch_rmsnorm(x, w), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("dtype, atol", DTYPES)
    def test_dtypes(self, dtype, atol):
        x = torch.randn(2, 128, 768, device="cuda", dtype=dtype)
        w = torch.randn(768, device="cuda", dtype=dtype)
        torch.testing.assert_close(
            triton_rmsnorm_tiled(x, w), pytorch_rmsnorm(x, w), atol=atol, rtol=atol
        )

    @pytest.mark.parametrize("D", [4096, 8192, 16384])
    def test_large_d(self, D):
        """Tiled kernel should handle large D without register spilling."""
        x = torch.randn(256, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            triton_rmsnorm_tiled(x, w), pytorch_rmsnorm(x, w), atol=1e-2, rtol=1e-2
        )

    def test_matches_single_pass(self):
        """Tiled and single-pass should produce identical results."""
        x = torch.randn(4, 128, 768, device="cuda")
        w = torch.randn(768, device="cuda")
        torch.testing.assert_close(
            triton_rmsnorm(x, w), triton_rmsnorm_tiled(x, w), atol=1e-5, rtol=1e-5
        )
