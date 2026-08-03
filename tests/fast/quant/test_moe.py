import pytest
import torch

from src.quant import moe
from src.quant.utils import is_supported
from src.utils.config import QuantizationConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="scaled grouped GEMM is CUDA only"
)


def _cfg(recipe="fp8", scaling="rowwise"):
    return QuantizationConfig(
        enabled=True, dtype={"recipe": recipe}, scaling={"recipe": scaling}
    )


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    E, R = len(counts), sum(counts)
    a = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


def test_scaled_grouped_gemm_compiles_fullgraph():
    # Forward compiles with no graph break; the custom Function's backward runs
    # eager (Dynamo cannot trace the autograd engine under fullgraph — a
    # categorical limitation), so we compile forward and call backward outside
    # the traced region, checking grads match eager.
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", "rowwise")

    def fwd(a, b):
        return moe.quantized_expert_mm(cfg)(a, b, offs)

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
            self.mlp = SparseMoEBlock(
                d_model=32,
                intermediate_size=48,
                n_routed_experts=4,
                n_routed_experts_per_token=2,
            )

    m = M().cuda().bfloat16()
    with torch.no_grad():
        nn.init.normal_(m.mlp.expert_gate_up, mean=0.0, std=0.02)
        nn.init.normal_(m.mlp.expert_down, mean=0.0, std=0.02)
    cfg = TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 2}}],
        ),
        training=TrainingConfig(
            quantization={
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scaling": {"recipe": "rowwise"},
            }
        ),
    )
    apply_quantization(m, cfg)
    assert m.mlp.expert_mm is not grouped_gemm  # confirms the quantized seam installed
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = m.mlp(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(m.mlp.expert_gate_up.grad).all()
    assert torch.isfinite(m.mlp.expert_down.grad).all()


@pytest.mark.skipif(not is_supported("fp8"), reason="fp8 needs SM >= 8.9")
@pytest.mark.parametrize("scaling", ["blockwise", "mxfp8"])
def test_blockwise_fuses_and_matches_fake_quant(scaling, monkeypatch):
    # counts give R=299: with block_size 128 the last scale block [256,299) straddles
    # the expert-2/expert-3 boundary at 258, and expert 1 is empty.
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", scaling)

    wgrad_calls = []
    real_wgrad = moe.scaled_grouped_gemm_wgrad

    def spy(*args, **kwargs):
        wgrad_calls.append(args[-1])
        return real_wgrad(*args, **kwargs)

    monkeypatch.setattr(moe, "scaled_grouped_gemm_wgrad", spy)

    a_q = a.clone().requires_grad_(True)
    b_q = b.clone().requires_grad_(True)
    y_q = moe.quantized_expert_mm(cfg)(a_q, b_q, offs)
    gy = torch.randn_like(y_q)
    y_q.backward(gy)

    assert wgrad_calls, "wgrad fell back off the fused path"
    assert wgrad_calls[0] == (128 if scaling == "blockwise" else 32)

    # Reference: identical quantization, but dequantized and multiplied in bf16.
    # Comparing against this isolates the kernel from the quantization error itself.
    monkeypatch.setattr(moe, "is_fp8", lambda _: False)
    monkeypatch.setattr(moe, "is_int8s", lambda _: False)
    a_f = a.clone().requires_grad_(True)
    b_f = b.clone().requires_grad_(True)
    y_f = moe.quantized_expert_mm(cfg)(a_f, b_f, offs)
    y_f.backward(gy)

    assert len(wgrad_calls) == 1, "reference run did not fall back off the fused path"

    # mxfp8 scales are powers of two (torch.exp2(int)): scaling is exact in fp32 and
    # distributes exactly over addition, and fp8x fp8 products are exact in fp32, so
    # the fused and fake-quant paths are bitwise identical there. blockwise keeps a
    # real (if tight) tolerance since fp8 rounding differs between the two paths.
    bound = 0.0 if scaling == "mxfp8" else 2e-2
    for name, got, ref in (
        ("y", y_q, y_f),
        ("grad_a", a_q.grad, a_f.grad),
        ("grad_b", b_q.grad, b_f.grad),
    ):
        rel = (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)
        assert rel <= bound, (scaling, name, rel)
