import torch
import pytest
from src.kernel.triton.rmsnorm import triton_rmsnorm_fwd, triton_rmsnorm_bwd, triton_rmsnorm
from src.kernel.triton.swiglu import triton_swiglu_fwd, triton_swiglu_bwd, triton_swiglu
from src.kernel.torch.rmsnorm import torch_rmsnorm
from src.kernel.torch.swiglu import torch_swiglu


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


def test_swiglu_fwd():
    M, N = 32, 768
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    y_triton = triton_swiglu_fwd(gate, up)
    y_ref = torch_swiglu(gate, up)

    assert y_triton.dtype == y_ref.dtype == gate.dtype
    assert torch.allclose(y_triton, y_ref, atol=1e-2, rtol=1e-2)


def test_swiglu_bwd():
    M, N = 32, 768
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    y_ref = torch_swiglu(gate, up)
    y_ref.backward(dy)
    dgate_ref = gate.grad.clone()
    dup_ref = up.grad.clone()

    dgate_triton, dup_triton = triton_swiglu_bwd(dy, gate.detach(), up.detach())

    assert torch.allclose(dgate_triton, dgate_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dup_triton, dup_ref, atol=1e-2, rtol=1e-2)


def test_swiglu_autograd():
    M, N = 32, 768
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    y = triton_swiglu(gate, up)
    y.sum().backward()

    assert gate.grad is not None
    assert up.grad is not None
    assert gate.grad.shape == gate.shape
    assert up.grad.shape == up.shape
