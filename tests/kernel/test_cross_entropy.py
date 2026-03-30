import pytest
import torch
import torch.nn.functional as F

from src.kernel.triton.cross_entropy import triton_cross_entropy
from src.kernel.torch.cross_entropy import torch_cross_entropy


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
