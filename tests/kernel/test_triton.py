import torch
import pytest
from src.kernel.triton.rmsnorm import triton_rmsnorm_fwd, triton_rmsnorm_bwd, triton_rmsnorm
from src.kernel.torch.rmsnorm import torch_rmsnorm


torch.manual_seed(42)


def test_rmsnorm_fwd():
    M, N = 32, 768
    eps = 1e-6
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)

    y_ref = torch_rmsnorm(x, w, eps)
    y_triton = triton_rmsnorm_fwd(x, w, eps)

    assert y_ref.dtype == y_triton.dtype == x.dtype
    assert torch.allclose(y_triton, y_ref, atol=1e-2, rtol=1e-2)


def test_rmsnorm_bwd():
    M, N = 32, 768
    eps = 1e-6

    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    y_ref = torch_rmsnorm(x, w, eps)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()
    dw_ref = w.grad.clone()

    dx_triton, dw_triton = triton_rmsnorm_bwd(dy, x.detach(), w.detach(), eps)

    assert dx_ref.dtype == dx_triton.dtype == x.dtype
    assert dw_ref.dtype == dw_triton.dtype == w.dtype
    assert torch.allclose(dx_triton, dx_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dw_triton, dw_ref, atol=1e-2, rtol=1e-2)


def test_rmsnorm_autograd():
    M, N = 32, 768
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    y = triton_rmsnorm(x, w, 1e-6)
    y.sum().backward()

    assert x.grad is not None
    assert w.grad is not None
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
