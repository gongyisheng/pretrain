import pytest
import torch
import torch.nn as nn

from src.kernel.ops import gemm as gemm_module
from src.layers.mlp import SparseMoEBlock
from src.quant import moe
from src.quant.convert import apply_quantization
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="scaled grouped GEMM is CUDA only"
)

SCALE_RECIPES = ["tensorwise", "rowwise", "blockwise", "mxfp8"]

# The width the wgrad's per-expert block table must be built from. row/tensorwise
# carry no numeric width, so 0 means one block per ragged segment.
WGRAD_BLOCK_SIZE = {"tensorwise": 0, "rowwise": 0, "blockwise": 128, "mxfp8": 32}

# R=299 with expert 1 empty. Scale blocks restart at each expert, so under
# blockwise-128 expert 2 (rows 128..257) owns two blocks and expert 3 (rows
# 258..298) its own short one — no block straddles the boundary at 258.
COUNTS = [128, 0, 130, 41]


def _cfg(scale="rowwise", recipe="fp8"):
    return QuantizationConfig(
        enabled=True, dtype={"recipe": recipe}, scale={"recipe": scale}
    )


def _expert_mm(cfg, a, b, offs, bias=None):
    """Drive the quantized expert GEMM directly, with no block and no stats."""
    return moe.ScaledGroupedGemmFn.apply(a, b, bias, offs, cfg, {})


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    n_experts, n_rows = len(counts), sum(counts)
    a = torch.randn(n_rows, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_experts, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


def _rel(got, ref):
    return (got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12)


# --- quantized_grouped_mm ---


def test_quantized_grouped_mm_raise_error():
    # (E,M,K) x (K,N) -> (M,N) with N ragged: the one layout the kernel takes but the
    # quantizer does not, since B's scale would need a column-wise ragged mapping.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    offs = torch.tensor([8, 32], device=device, dtype=torch.int32)
    a = torch.randn(2, 16, 48, device=device, dtype=torch.bfloat16)
    b = torch.randn(48, 32, device=device, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="ragged-N"):
        moe.quantized_grouped_mm(
            a, b, offs, "fp8_e4m3", "fp8_e4m3", a.dtype, _cfg().scale
        )


# --- ScaledGroupedGemmFn ---


@cuda_only
def test_scaled_grouped_gemm_fn_compiles_fullgraph():
    # Forward compiles with no graph break; the custom Function's backward runs
    # eager (Dynamo cannot trace the autograd engine under fullgraph — a
    # categorical limitation), so we compile forward and call backward outside
    # the traced region, checking grads match eager.
    a, b, offs = _make(COUNTS, K=64, N=48)
    cfg = _cfg("rowwise")

    def fwd(a, b):
        return _expert_mm(cfg, a, b, offs)

    def run(fn):
        a_, b_ = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
        y = fn(a_, b_)
        y.sum().backward()
        return y, a_.grad, b_.grad

    eager = run(fwd)
    compiled = run(torch.compile(fwd, fullgraph=True))
    for got, ref in zip(compiled, eager):
        torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


@cuda_only
@pytest.mark.parametrize("scale", SCALE_RECIPES)
def test_scaled_grouped_gemm_fn_bias_is_additive(scale):
    """The fused bias must stay purely additive: same output and same grad_a/grad_b as
    running the GEMM without it and adding it per row afterwards.
    """
    a, b, offs = _make(COUNTS, K=64, N=48)
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    gy = torch.randn(a.shape[0], 48, device="cuda", dtype=torch.bfloat16)
    group_of_row = torch.searchsorted(
        offs, torch.arange(a.shape[0], device=offs.device), right=True
    )
    cfg = _cfg(scale)

    def run(fused):
        a_ = a.clone().requires_grad_(True)
        b_ = b.clone().requires_grad_(True)
        bias = bias0.clone().requires_grad_(True)
        y = (
            _expert_mm(cfg, a_, b_, offs, bias=bias)
            if fused
            else _expert_mm(cfg, a_, b_, offs) + bias[group_of_row]
        )
        y.backward(gy)
        return y, a_.grad, b_.grad, bias.grad

    got, ref = run(True), run(False)
    for name, g, r in zip(("y", "grad_a", "grad_b", "grad_bias"), got, ref):
        assert _rel(g, r) < 2e-2, (name, _rel(g, r))

    # grad_bias sums each group's rows. Pin it against an exact fp32 segment sum: it
    # is a reduction over hundreds of rows, which the unfused path only approximates
    # because autograd's index backward accumulates in grad_out's dtype (bf16).
    exact = torch.zeros(offs.shape[0], 48, device="cuda", dtype=torch.float32)
    exact.index_add_(0, group_of_row, gy.float())
    assert _rel(got[3], exact) < 5e-3


@cuda_only
@pytest.mark.parametrize("scale", SCALE_RECIPES)
def test_scaled_grouped_gemm_fn_matches_fake_quant(scale, monkeypatch):
    a, b, offs = _make(COUNTS, K=64, N=48)
    cfg = _cfg(scale)

    wgrad_calls = []
    real_dispatch = gemm_module.dispatch

    def spy(op, args, kwargs, backend=None, *, device):
        # one op serves every layout now; 2D x 2D is ragged-K, i.e. the wgrad. Plain
        # gemm.grouped_mm (the dequant fallback's bf16 reference path) shares this
        # dispatch name but takes a shorter (a, b, offs, bias) args tuple -- skip it.
        if op.endswith("scaled_grouped_mm"):
            bq, block_size = args[1], args[6]
            if bq.ndim == 2:
                wgrad_calls.append(block_size)
        return real_dispatch(op, args, kwargs, backend, device=device)

    monkeypatch.setattr(gemm_module, "dispatch", spy)

    a_q = a.clone().requires_grad_(True)
    b_q = b.clone().requires_grad_(True)
    y_q = _expert_mm(cfg, a_q, b_q, offs)
    gy = torch.randn_like(y_q)
    y_q.backward(gy)

    assert wgrad_calls, "wgrad fell back off the fused path"
    assert wgrad_calls[0] == WGRAD_BLOCK_SIZE[scale]

    # Reference: identical quantization, but dequantized and multiplied in bf16.
    # Comparing against this isolates the kernel from the quantization error itself.
    monkeypatch.setattr(moe, "scaled_grouped_mm_op", lambda *a, **k: None)
    a_f = a.clone().requires_grad_(True)
    b_f = b.clone().requires_grad_(True)
    y_f = _expert_mm(cfg, a_f, b_f, offs)
    y_f.backward(gy)

    assert len(wgrad_calls) == 1, "reference run did not fall back off the fused path"

    # mxfp8 scales are powers of two (torch.exp2(int)): scaling is exact in fp32 and
    # distributes exactly over addition, and fp8x fp8 products are exact in fp32, so
    # the fused and fake-quant paths are bitwise identical there. blockwise keeps a
    # real (if tight) tolerance since fp8 rounding differs between the two paths.
    bound = 0.0 if scale == "mxfp8" else 2e-2
    for name, got, ref in (
        ("y", y_q, y_f),
        ("grad_a", a_q.grad, a_f.grad),
        ("grad_b", b_q.grad, b_f.grad),
    ):
        assert _rel(got, ref) <= bound, (scale, name, _rel(got, ref))


@cuda_only
@pytest.mark.parametrize("scale", SCALE_RECIPES)
def test_scaled_grouped_gemm_fn_grad_b_is_independent_across_experts(scale):
    """A 1e5x hotter expert must not degrade a cold expert's grad_b.

    This is the whole point of per-expert scales: with one amax shared across the
    ragged token axis a big enough hot/cold spread pushes the cold expert's rows
    into the bottom of the fp8 range and its grad_b becomes mostly quantization
    noise (100x is absorbed by fp8's exponent range at ~no cost; 1e5x is not).

    Observed max(rel): tensorwise 0.0272, rowwise/blockwise 0.0268 (~4x headroom on
    the 0.1 bound); max/min ratio ~1.1-1.2 for all three (~3.4x headroom).
    """
    counts = [64, 64]
    a, b, offs = _make(counts, K=128, N=96)
    a[: counts[0]] *= 1e5  # expert 0 hot, expert 1 cold

    a_q = a.clone().requires_grad_(True)
    b_q = b.clone().requires_grad_(True)
    _expert_mm(_cfg(scale), a_q, b_q, offs).sum().backward()

    a_ref = a.float().clone().requires_grad_(True)
    b_ref = b.float().clone().requires_grad_(True)
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        (a_ref[lo:hi] @ b_ref[group]).sum().backward()
        lo = hi

    rel = [
        _rel(b_q.grad[group], b_ref.grad[group]).item() for group in range(len(counts))
    ]
    assert max(rel) < 0.1, rel
    assert max(rel) / min(rel) < 4.0, rel


@cuda_only
def test_scaled_grouped_gemm_fn_wgrad_receives_per_expert_block_table(monkeypatch):
    """The kernel re-derives the per-expert block table from offs and block_size, so
    it must get the 0-sentinel width — not an element count, which would collapse
    rowwise back to one block — and a scale row per expert to match."""
    a, b, offs = _make([8, 0, 24], K=64, N=48)
    seen = {}
    real_dispatch = gemm_module.dispatch

    def spy(op, args, kwargs, backend=None, *, device):
        bq, scale_a, block_size = args[1], args[2], args[6]
        if bq.ndim == 2:  # ragged-K layout, i.e. the wgrad
            seen["block_size"] = block_size
            # scale_a arrives transposed (K,nrb) with the contraction axis last
            seen["n_scale_rows"] = scale_a.shape[-1]
        return real_dispatch(op, args, kwargs, backend, device=device)

    monkeypatch.setattr(gemm_module, "dispatch", spy)
    a_q = a.clone().requires_grad_(True)
    _expert_mm(_cfg("rowwise"), a_q, b.clone(), offs).sum().backward()
    assert seen["block_size"] == 0  # rowwise -> one block per expert
    assert seen["n_scale_rows"] == 3  # including the empty expert


# --- QuantizedSparseMoEBlock ---


def test_quantized_sparse_moe_block_expert_mm(monkeypatch):
    """The seam quantizes in train and falls through to the base block in eval."""
    with torch.device("cpu"):
        source = SparseMoEBlock(
            d_model=4,
            intermediate_size=8,
            n_routed_experts=2,
            n_routed_experts_per_token=1,
            aux_loss=False,
        )
    source.eval()
    block = moe.QuantizedSparseMoEBlock.from_module(
        source, QuantizationConfig(enabled=True, dtype={"weight": "int4"})
    )
    assert not block.training  # from_module preserves the source's mode

    # Which GEMM the seam dispatches to is the whole question, so spy on both: the
    # base block's seam is what eval falls through to.
    used = []
    monkeypatch.setattr(
        SparseMoEBlock, "expert_mm", lambda *a, **kw: used.append("plain")
    )
    monkeypatch.setattr(
        moe.ScaledGroupedGemmFn,
        "apply",
        staticmethod(lambda *a: used.append("quantized")),
    )
    args = (torch.randn(4, 4), torch.randn(2, 4, 8), torch.tensor([2, 4]))

    block.expert_mm(*args, projection="gate_up")
    block.train()
    block.expert_mm(*args, projection="gate_up")
    block.eval()
    block.expert_mm(*args, projection="gate_up")

    assert used == ["plain", "quantized", "plain"]


@cuda_only
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_sparse_moe_block_forward_backward(bias):
    torch.manual_seed(0)  # deterministic init regardless of prior RNG state

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = SparseMoEBlock(
                d_model=32,
                intermediate_size=48,
                n_routed_experts=4,
                n_routed_experts_per_token=2,
                bias=bias,
            )

    m = M().cuda().bfloat16()
    with torch.no_grad():
        nn.init.normal_(m.mlp.expert_gate_up, mean=0.0, std=0.02)
        nn.init.normal_(m.mlp.expert_down, mean=0.0, std=0.02)
    config = TrainConfig(
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
                "scale": {"recipe": "rowwise"},
            }
        ),
    )
    apply_quantization(m, config)
    # confirms the quantized seam installed
    assert m.mlp.expert_mm.__func__ is moe.QuantizedSparseMoEBlock.expert_mm

    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out, _ = m.mlp(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(m.mlp.expert_gate_up.grad).all()
    assert torch.isfinite(m.mlp.expert_down.grad).all()
    if bias:
        # the per-expert bias rides the fused epilogue, so its grad comes from the
        # quantized Function rather than from a separate add
        for p in (m.mlp.expert_gate_up_bias, m.mlp.expert_down_bias):
            assert p.grad is not None and torch.isfinite(p.grad).all()
            assert p.grad.abs().sum() > 0
