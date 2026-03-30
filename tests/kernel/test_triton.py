import torch
import pytest
import torch.nn.functional as F
from src.kernel.triton.rmsnorm import triton_rmsnorm_fwd, triton_rmsnorm_bwd, triton_rmsnorm
from src.kernel.triton.swiglu import triton_swiglu_fwd, triton_swiglu_bwd, triton_swiglu
from src.kernel.triton.rope import triton_rope_fwd, triton_rope_bwd, triton_rope
from src.kernel.triton.layernorm import triton_layernorm_fwd, triton_layernorm_bwd, triton_layernorm
from src.kernel.triton.flashattn import triton_flash_attn_fwd, triton_flash_attn_bwd, triton_flash_attn
from src.kernel.triton.cross_entropy import triton_cross_entropy
from src.kernel.torch.rmsnorm import torch_rmsnorm
from src.kernel.torch.swiglu import torch_swiglu
from src.kernel.torch.rope import torch_rope
from src.kernel.torch.layernorm import torch_layernorm
from src.kernel.torch.flashattn import torch_flash_attn
from src.kernel.torch.cross_entropy import torch_cross_entropy


torch.manual_seed(42)


@pytest.fixture
def ce_inputs():
    torch.manual_seed(42)
    M, V = 128, 1024
    logits = torch.randn(M, V, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, V, (M,), device="cuda")
    return logits, targets


@pytest.fixture
def ce_inputs_large():
    torch.manual_seed(42)
    M, V = 4096, 50257
    logits = torch.randn(M, V, device="cuda", dtype=torch.bfloat16)
    targets = torch.randint(0, V, (M,), device="cuda")
    return logits, targets


# --- Torch cross-entropy tests ---

def test_torch_ce_fwd(ce_inputs):
    logits, targets = ce_inputs
    ref = F.cross_entropy(logits.float(), targets)
    out = torch_cross_entropy(logits, targets)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_torch_ce_bwd(ce_inputs):
    logits, targets = ce_inputs
    logits_ref = logits.clone().float().requires_grad_(True)
    logits_test = logits.clone().requires_grad_(True)

    F.cross_entropy(logits_ref, targets).backward()
    torch_cross_entropy(logits_test, targets).backward()

    torch.testing.assert_close(logits_test.grad.float(), logits_ref.grad.float(), atol=1e-2, rtol=1e-2)


# --- Triton cross-entropy tests ---

def test_triton_ce_fwd(ce_inputs):
    logits, targets = ce_inputs
    ref = F.cross_entropy(logits.float(), targets)
    out = triton_cross_entropy(logits, targets)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_triton_ce_fwd_large(ce_inputs_large):
    logits, targets = ce_inputs_large
    ref = F.cross_entropy(logits.float(), targets)
    out = triton_cross_entropy(logits, targets)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_triton_ce_bwd(ce_inputs):
    logits, targets = ce_inputs
    logits_ref = logits.clone().float().requires_grad_(True)
    logits_fused = logits.clone().requires_grad_(True)

    F.cross_entropy(logits_ref, targets).backward()
    triton_cross_entropy(logits_fused, targets).backward()

    torch.testing.assert_close(logits_fused.grad.float(), logits_ref.grad.float(), atol=1e-2, rtol=1e-2)


def test_triton_ce_autograd(ce_inputs):
    logits, targets = ce_inputs
    logits.requires_grad_(True)
    loss = triton_cross_entropy(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


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


def test_layernorm_fwd():
    M, N = 32, 768
    eps = 1e-5
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(N, device='cuda', dtype=torch.bfloat16)

    y_triton = triton_layernorm_fwd(x, w, b, eps)
    y_ref = torch_layernorm(x, w, b, eps)

    assert y_triton.dtype == y_ref.dtype == x.dtype
    assert torch.allclose(y_triton, y_ref, atol=1e-2, rtol=1e-2)


def test_layernorm_bwd():
    M, N = 32, 768
    eps = 1e-5
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    y_ref = torch_layernorm(x, w, b, eps)
    y_ref.backward(dy)
    dx_ref = x.grad.clone()
    dw_ref = w.grad.clone()
    db_ref = b.grad.clone()

    dx_tri, dw_tri, db_tri = triton_layernorm_bwd(dy, x.detach(), w.detach(), eps)

    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-1, rtol=1e-1)
    assert torch.allclose(db_tri, db_ref, atol=1e-1, rtol=1e-1)


def test_layernorm_autograd():
    M, N = 32, 768
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    b = torch.randn(N, device='cuda', dtype=torch.bfloat16, requires_grad=True)

    y = triton_layernorm(x, w, b)
    y.sum().backward()

    assert x.grad is not None
    assert w.grad is not None
    assert b.grad is not None
    assert x.grad.shape == x.shape
    assert w.grad.shape == w.shape
    assert b.grad.shape == b.shape


def _make_flashattn_inputs(B=2, n_heads=4, seq=64, d_head=64, dtype=torch.bfloat16):
    q = torch.randn(B, n_heads, seq, d_head, device='cuda', dtype=dtype)
    k = torch.randn(B, n_heads, seq, d_head, device='cuda', dtype=dtype)
    v = torch.randn(B, n_heads, seq, d_head, device='cuda', dtype=dtype)
    return q, k, v


def test_flashattn_fwd():
    q, k, v = _make_flashattn_inputs()

    o_triton, L = triton_flash_attn_fwd(q, k, v, causal=True)
    o_ref = torch_flash_attn(q, k, v, causal=True)

    assert o_triton.dtype == o_ref.dtype == q.dtype
    assert o_triton.shape == o_ref.shape
    assert torch.allclose(o_triton, o_ref, atol=1e-2, rtol=1e-2)


def test_flashattn_fwd_noncausal():
    q, k, v = _make_flashattn_inputs()

    o_triton, L = triton_flash_attn_fwd(q, k, v, causal=False)
    o_ref = torch_flash_attn(q, k, v, causal=False)

    assert torch.allclose(o_triton, o_ref, atol=1e-2, rtol=1e-2)


def test_flashattn_bwd():
    q, k, v = _make_flashattn_inputs()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    o_ref = torch_flash_attn(q, k, v, causal=True)
    do = torch.randn_like(o_ref)
    o_ref.backward(do)
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()

    o_triton, L = triton_flash_attn_fwd(q.detach(), k.detach(), v.detach(), causal=True)
    dq_tri, dk_tri, dv_tri = triton_flash_attn_bwd(
        q.detach(), k.detach(), v.detach(), o_triton, L, do, causal=True
    )

    assert torch.allclose(dq_tri, dq_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dk_tri, dk_ref, atol=1e-2, rtol=1e-2)
    assert torch.allclose(dv_tri, dv_ref, atol=1e-2, rtol=1e-2)


def test_flashattn_autograd():
    q, k, v = _make_flashattn_inputs()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    o = triton_flash_attn(q, k, v, causal=True)
    o.sum().backward()

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None
    assert q.grad.shape == q.shape
    assert k.grad.shape == k.shape
    assert v.grad.shape == v.shape
