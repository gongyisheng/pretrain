import pytest
import torch

from src.quant import moe
from src.quant.quantize import fake_quantize_operand
from src.quant.utils import is_supported
from src.utils.config import QuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="scaled grouped GEMM is CUDA only"
)


def _cfg(recipe="fp8", scaling="rowwise"):
    return QuantConfig(
        enabled=True, dtype={"recipe": recipe}, scaling={"recipe": scaling}
    )


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


def _fwd_oracle(a, b, offs, cfg):
    """Dequant-both fp32 grouped matmul (fake-quant reference)."""
    gran = cfg.scaling["granularity"]
    bs = cfg.scaling.get("block_size", 0)
    a_fq = fake_quantize_operand(a, -1, gran, bs, cfg.dtype["act"]).float()
    out = torch.zeros(a.shape[0], b.shape[2], device=a.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offs.tolist()):
        if end > start:
            bg = fake_quantize_operand(b[g], 0, gran, bs, cfg.dtype["weight"]).float()
            out[start:end] = a_fq[start:end] @ bg
        start = end
    return out


@pytest.mark.parametrize("scaling", ["rowwise", "tensorwise"])
def test_scaled_grouped_gemm_forward_matches_oracle(scaling):
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", scaling)
    got = moe.scaled_grouped_gemm(a, b, offs, cfg)
    ref = _fwd_oracle(a, b, offs, cfg)
    assert got.shape == (a.shape[0], b.shape[2])
    rel = (got.float() - ref).norm() / ref.norm().clamp_min(1e-12)
    assert rel < 5e-2, rel


def _bwd_oracle(a, b, offs, cfg):
    """Analytic fully-quantized backward reference (mirrors _fwd_oracle's manual
    style — no reliance on autograd through the quantize op, which severs the
    graph for int8 and never fake-quantizes the gradient stream anyway).
    Upstream grad is ones (the test does `.sum().backward()`); matches production
    `backward`/`_wgrad` operand quantization exactly: dgrad quantizes grad_y along
    N (input_grad) and b along N (weight); wgrad quantizes a and grad_y along the
    whole ragged M axis (act / weight_grad) before slicing per group.
    """
    gran, bs, dt = (
        cfg.scaling["granularity"],
        cfg.scaling.get("block_size", 0),
        cfg.dtype,
    )
    R, K = a.shape
    E, _, N = b.shape
    gy = torch.ones(R, N, device=a.device, dtype=a.dtype)  # d(sum)/dy
    # dgrad: grad_a = fq(gy, input_grad) @ fq(b, weight)^T, contracting N
    gy_ig = fake_quantize_operand(gy, -1, gran, bs, dt["input_grad"])
    # wgrad: grad_b[g] = fq(a, act)[g]^T @ fq(gy, weight_grad)[g], contracting ragged M
    a_act = fake_quantize_operand(a, 0, gran, bs, dt["act"])
    gy_wg = fake_quantize_operand(gy, 0, gran, bs, dt["weight_grad"])
    grad_a = torch.zeros(R, K, device=a.device, dtype=torch.float32)
    grad_b = torch.zeros(E, K, N, device=a.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offs.tolist()):
        if end > start:
            b_w = fake_quantize_operand(
                b[g], 1, gran, bs, dt["weight"]
            )  # (K,N), contract N
            grad_a[start:end] = gy_ig[start:end].float() @ b_w.float().t()
            grad_b[g] = a_act[start:end].float().t() @ gy_wg[start:end].float()
        start = end
    return grad_a, grad_b


@pytest.mark.parametrize("recipe", ["fp8", "int8"])
def test_scaled_grouped_gemm_backward_matches_oracle(recipe):
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg(recipe, "rowwise")
    a1 = a.detach().clone().requires_grad_(True)
    b1 = b.detach().clone().requires_grad_(True)
    y = moe.scaled_grouped_gemm(a1, b1, offs, cfg)
    y.sum().backward()
    ga_ref, gb_ref = _bwd_oracle(a, b, offs, cfg)
    ra = (a1.grad.float() - ga_ref.float()).norm() / ga_ref.float().norm().clamp_min(
        1e-12
    )
    rb = (b1.grad.float() - gb_ref.float()).norm() / gb_ref.float().norm().clamp_min(
        1e-12
    )
    assert ra < 8e-2, ra
    assert rb < 8e-2, rb


def test_scaled_grouped_gemm_compiles_fullgraph():
    # Forward compiles with no graph break; the custom Function's backward runs
    # eager (Dynamo cannot trace the autograd engine under fullgraph — a
    # categorical limitation), so we compile forward and call backward outside
    # the traced region, checking grads match eager.
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", "rowwise")

    def fwd(a, b):
        return moe.scaled_grouped_gemm(a, b, offs, cfg)

    a_e = a.clone().requires_grad_(True)
    b_e = b.clone().requires_grad_(True)
    a_c = a.clone().requires_grad_(True)
    b_c = b.clone().requires_grad_(True)

    ey = fwd(a_e, b_e)
    ey.sum().backward()

    cy = torch.compile(fwd, fullgraph=True)(a_c, b_c)
    cy.sum().backward()

    torch.testing.assert_close(cy.float(), ey.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(a_c.grad.float(), a_e.grad.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(b_c.grad.float(), b_e.grad.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not is_supported("fp8"), reason="fp8 needs SM >= 8.9")
def test_moe_block_quantized_forward_backward_runs():
    import torch.nn as nn

    from src.kernel.gemm import grouped_gemm
    from src.layers.mlp import SparseMoEBlock
    from src.quant.convert import apply_quantization
    from src.utils.config import (
        ModelConfig,
        TrainConfig,
        TrainingConfig,
    )

    torch.manual_seed(0)  # deterministic init regardless of prior RNG state

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe = SparseMoEBlock(
                d_model=32,
                intermediate_size=48,
                n_routed_experts=4,
                n_routed_experts_per_token=2,
            )

    m = M().cuda().bfloat16()
    with torch.no_grad():
        nn.init.normal_(m.moe.expert_gate_up, mean=0.0, std=0.02)
        nn.init.normal_(m.moe.expert_down, mean=0.0, std=0.02)
    cfg = TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(
            quant={
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
            }
        ),
    )
    apply_quantization(m, cfg)
    assert m.moe.expert_mm is not grouped_gemm  # confirms the quantized seam installed
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = m.moe(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(m.moe.expert_gate_up.grad).all()
    assert torch.isfinite(m.moe.expert_down.grad).all()
