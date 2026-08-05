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


@pytest.mark.skipif(not is_supported("fp8_e4m3"), reason="fp8 needs SM >= 8.9")
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


@pytest.mark.skipif(not is_supported("fp8_e4m3"), reason="fp8 needs SM >= 8.9")
@pytest.mark.parametrize("scaling", ["blockwise", "mxfp8"])
def test_blockwise_fuses_and_matches_fake_quant(scaling, monkeypatch):
    # counts give R=299 with expert 1 empty. Scale blocks restart at each expert, so
    # expert 2 (rows 128..257) owns two 128-row blocks and expert 3 (rows 258..298)
    # its own short one — no block straddles the boundary at 258 any more.
    a, b, offs = _make([128, 0, 130, 41], K=64, N=48)
    cfg = _cfg("fp8", scaling)

    wgrad_calls = []
    real_gemm = moe.scaled_grouped_gemm

    def spy(aq, bq, sa, sb, offs_, out_dtype, block_size):
        # one op serves every layout now; 2D x 2D is ragged-K, i.e. the wgrad
        if bq.ndim == 2:
            wgrad_calls.append(block_size)
        return real_gemm(aq, bq, sa, sb, offs_, out_dtype, block_size)

    monkeypatch.setattr(moe, "scaled_grouped_gemm", spy)

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


@pytest.mark.parametrize("scaling", ["tensorwise", "rowwise", "blockwise"])
def test_grad_b_quality_is_independent_across_experts(scaling):
    """A 1e5x hotter expert must not degrade a cold expert's grad_b.

    This is the whole point of per-expert scales: with one amax shared across the
    ragged token axis a big enough hot/cold spread pushes the cold expert's rows
    into the bottom of the fp8 range and its grad_b becomes mostly quantization
    noise (100x is absorbed by fp8's exponent range at ~no cost; 1e5x is not).
    """
    counts = [64, 64]
    a, b, offs = _make(counts, K=128, N=96)
    a[: counts[0]] *= 1e5  # expert 0 hot, expert 1 cold
    cfg = _cfg("fp8", scaling)

    a_q = a.clone().requires_grad_(True)
    b_q = b.clone().requires_grad_(True)
    moe.quantized_expert_mm(cfg)(a_q, b_q, offs).sum().backward()

    a_ref = a.float().clone().requires_grad_(True)
    b_ref = b.float().clone().requires_grad_(True)
    lo = 0
    for gi, hi in enumerate(offs.tolist()):
        (a_ref[lo:hi] @ b_ref[gi]).sum().backward()
        lo = hi

    rel = [
        (
            (b_q.grad[gi].float() - b_ref.grad[gi]).norm()
            / b_ref.grad[gi].norm().clamp_min(1e-12)
        ).item()
        for gi in range(len(counts))
    ]
    # Observed max(rel): tensorwise 0.0272, rowwise/blockwise 0.0268 (~4x headroom
    # on the 0.1 bound); max/min ratio ~1.1-1.2 for all three (~3.4x headroom).
    assert max(rel) < 0.1, rel
    assert max(rel) / min(rel) < 4.0, rel


def test_wgrad_receives_per_expert_block_table(monkeypatch):
    """The kernel re-derives the per-expert block table from offs and block_size, so
    it must get kernel_block_size's width — not an element count, which would collapse
    rowwise back to one block — and a scale row per expert to match."""
    a, b, offs = _make([8, 0, 24], K=64, N=48)
    seen = {}
    real = moe.scaled_grouped_gemm

    def spy(aq, gq, sa, sg, offs_, out_dtype, block_size):
        if gq.ndim == 2:  # ragged-K layout, i.e. the wgrad
            seen["block_size"] = block_size
            # sa arrives transposed (K,nrb) with the contraction axis last
            seen["n_scale_rows"] = sa.shape[-1]
        return real(aq, gq, sa, sg, offs_, out_dtype, block_size)

    monkeypatch.setattr(moe, "scaled_grouped_gemm", spy)
    a_q = a.clone().requires_grad_(True)
    moe.quantized_expert_mm(_cfg("fp8", "rowwise"))(
        a_q, b.clone(), offs
    ).sum().backward()
    assert seen["block_size"] == 0  # rowwise -> one block per expert
    assert seen["n_scale_rows"] == 3  # including the empty expert
