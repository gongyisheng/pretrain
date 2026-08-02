import pytest
import torch

from src.quant import moe
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
            quant={
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
