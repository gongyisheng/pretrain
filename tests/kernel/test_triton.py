import torch
import pytest
from src.kernel.triton.rmsnorm import triton_rmsnorm_fwd, triton_rmsnorm_bwd, triton_rmsnorm
from src.kernel.triton.swiglu import triton_swiglu_fwd, triton_swiglu_bwd, triton_swiglu
from src.kernel.triton.rope import triton_rope_fwd, triton_rope_bwd, triton_rope
from src.kernel.torch.rmsnorm import torch_rmsnorm
from src.kernel.torch.swiglu import torch_swiglu
from src.kernel.torch.rope import torch_rope


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


def _make_rope_inputs(B=2, n_heads=4, S=32, d_head=64, dtype=torch.bfloat16):
    """Helper to create RoPE test inputs with precomputed cos/sin."""
    x = torch.randn(B, n_heads, S, d_head, device='cuda', dtype=dtype)
    # Build cos/sin the same way as RoPE module
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device='cuda').float() / d_head))
    positions = torch.arange(S, device='cuda').float()
    angles = positions[:, None] * freqs[None, :]       # (S, d_head//2)
    angles = torch.cat([angles, angles], dim=-1)        # (S, d_head)
    cos = torch.cos(angles)[None, None, :, :].to(dtype) # (1, 1, S, d_head)
    sin = torch.sin(angles)[None, None, :, :].to(dtype) # (1, 1, S, d_head)
    return x, cos, sin


def test_rope_fwd():
    x, cos, sin = _make_rope_inputs()

    y_triton = triton_rope_fwd(x, cos, sin)
    y_ref = torch_rope(x, cos, sin)

    assert y_triton.dtype == y_ref.dtype == x.dtype
    assert y_triton.shape == y_ref.shape
    assert torch.allclose(y_triton, y_ref, atol=1e-2, rtol=1e-2)


def test_rope_bwd():
    x, cos, sin = _make_rope_inputs()
    x.requires_grad_(True)
    dy = torch.randn_like(x)

    y_ref = torch_rope(x, cos, sin)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()

    dx_triton = triton_rope_bwd(dy, cos, sin)

    assert dx_triton.dtype == dx_ref.dtype
    assert dx_triton.shape == dx_ref.shape
    assert torch.allclose(dx_triton, dx_ref, atol=1e-2, rtol=1e-2)


def test_rope_autograd():
    x, cos, sin = _make_rope_inputs()
    x.requires_grad_(True)

    y = triton_rope(x, cos, sin)
    y.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
