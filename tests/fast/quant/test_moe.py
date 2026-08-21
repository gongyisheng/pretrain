import pytest
import torch
import torch.nn as nn

from src.layers.mlp import SparseMoEBlock
from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.convert import apply_quantization
from src.quant.moe import (
    QuantizedSparseMoEBlock,
    ScaledGroupedGemmFn,
    quantized_grouped_mm,
)
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)
from tests.fast.quant.helper import (
    ALL_FORMATS,
    FORWARD_DTYPES,
    BACKWARD_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    ALL_SCALES,
    ROWWISE,
    SCALE_DTYPE_NAMES,
    fp8_only,
    mm_ref,
    operand_fmt,
    rel,
    rule,
    skip_unsupported_dtype_scale,
    skip_unsupported_ragged_k_scale,
    skip_unsupported_fmt_scale,
)


# One bf16 tolerance covers all format, scale, and layout combinations.
PRECISION_BOUND = 1e-3


# 299 rows across four experts; expert 1 is empty to test expert boundaries.
COUNTS = [128, 0, 130, 41]


def _cfg(scale_cfg=ROWWISE, dtype=None):
    """Build a resolved rule, translating the scale dtype key to config vocabulary."""
    return QuantizationConfig(
        enabled=True,
        dtype=dict(dtype or FP8_E4M3_W8A8_E5M2_G8_DTYPES),
        scale={**scale_cfg, "scale_dtype": SCALE_DTYPE_NAMES[scale_cfg["scale_dtype"]]},
    )


def _expert_mm(cfg, a, b, offs, bias=None):
    """Run the quantized expert GEMM without block metadata or statistics."""
    return ScaledGroupedGemmFn.apply(a, b, bias, offs, cfg, {})


def _make(counts, K, N, seed=0):
    torch.manual_seed(seed)
    n_experts, n_rows = len(counts), sum(counts)
    a = torch.randn(n_rows, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n_experts, K, N, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    return a, b, offs


# --- quantized_grouped_mm ---


# Exercise ragged-M, ragged-K, and ragged-N layouts.
GROUPED_LAYOUTS = ["ragged_m", "ragged_k", "ragged_n"]


@fp8_only
@pytest.mark.parametrize("a_fmt", ALL_FORMATS)
@pytest.mark.parametrize("b_fmt", ALL_FORMATS)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("layout", GROUPED_LAYOUTS)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_grouped_mm_precision(a_fmt, b_fmt, scale_cfg, layout, bias):
    """Compare each layout with per-expert dequantized GEMM, including optional bias."""
    skip_unsupported_fmt_scale(a_fmt, scale_cfg)
    skip_unsupported_fmt_scale(b_fmt, scale_cfg)
    if layout == "ragged_k":
        skip_unsupported_ragged_k_scale(scale_cfg)
    if bias and layout != "ragged_m":
        pytest.skip(
            "bias is defined per expert over the row axis, which only ragged-M has"
        )
    a, b, offs = _make(COUNTS, K=64, N=48)

    if layout == "ragged_m":
        # Ragged-M: (R,K) x (E,K,N) -> (R,N)
        bias0 = (
            torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
            if bias
            else None
        )
        out = quantized_grouped_mm(
            a, b, offs, a_fmt, b_fmt, a.dtype, scale_cfg, bias=bias0
        )
        ref = torch.empty_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                fwd = mm_ref(a[lo:hi], a_fmt, b[group], b_fmt, scale_cfg)
                if bias:
                    fwd = fwd + bias0[group].to(fwd.dtype)
                ref[lo:hi] = fwd.to(out.dtype)
            lo = hi
    elif layout == "ragged_n":
        # Ragged-N: (E,M,K) x (K,N) -> (M,N)
        slabs = torch.randn(offs.shape[0], 32, 64, device="cuda", dtype=torch.bfloat16)
        cols = torch.randn(64, a.shape[0], device="cuda", dtype=torch.bfloat16) * 0.1
        out = quantized_grouped_mm(
            slabs, cols, offs, a_fmt, b_fmt, slabs.dtype, scale_cfg
        )
        ref = torch.empty_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                ref[:, lo:hi] = mm_ref(
                    slabs[group], a_fmt, cols[:, lo:hi], b_fmt, scale_cfg
                ).to(out.dtype)
            lo = hi
    else:
        # Ragged-K: (K,R) x (R,N) -> (E,K,N)
        gy = torch.randn(a.shape[0], 48, device="cuda", dtype=torch.bfloat16)
        out = quantized_grouped_mm(a.mT, gy, offs, a_fmt, b_fmt, a.dtype, scale_cfg)
        ref = torch.zeros_like(out)
        lo = 0
        for group, hi in enumerate(offs.tolist()):
            if hi > lo:
                ref[group] = mm_ref(
                    a[lo:hi].t(), a_fmt, gy[lo:hi], b_fmt, scale_cfg
                ).to(out.dtype)
            lo = hi

    assert rel(out, ref) < PRECISION_BOUND, rel(out, ref)


# (a_fmt, b_fmt, monitoring, a_folded, b_folded)
GROUPED_STATS_CASES = [
    ("int8", "int8", False, False, False),  # no monitoring
    ("int8", "bf16", True, True, False),
    ("int8", "int8", True, True, True),
]


@fp8_only
@pytest.mark.parametrize(
    "a_fmt,b_fmt,with_stats,a_folded,b_folded", GROUPED_STATS_CASES
)
def test_quantized_grouped_mm_records_stats(
    a_fmt, b_fmt, with_stats, a_folded, b_folded
):
    a, b, offs = _make(COUNTS, K=64, N=48)
    # Allocate one stats slot per expert to catch cold experts.
    experts = len(COUNTS)
    a_stats = QuantizationStats("act/x", experts, a.device) if with_stats else None
    b_stats = QuantizationStats("weight/x", experts, b.device) if with_stats else None
    set_quantization_monitoring_status(True)
    try:
        out = quantized_grouped_mm(
            a,
            b,
            offs,
            a_fmt,
            b_fmt,
            a.dtype,
            ROWWISE,
            a_stats=a_stats,
            b_stats=b_stats,
        )
    finally:
        set_quantization_monitoring_status(False)
    assert torch.isfinite(out).all()
    if with_stats:
        assert a_stats.numel.shape == (experts,)
        assert a_stats.numel.sum().item() == (a.numel() if a_folded else 0)
        assert b_stats.numel.sum().item() == (b.numel() if b_folded else 0)


# --- ScaledGroupedGemmFn ---


@fp8_only
def test_scaled_grouped_gemm_fn_compiles_fullgraph():
    # Fullgraph covers forward; autograd backward remains eager.
    a, b, offs = _make(COUNTS, K=64, N=48)
    cfg = _cfg(ROWWISE)

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


@fp8_only
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("dtype", FORWARD_DTYPES)
def test_scaled_grouped_gemm_fn_forward_precision(dtype, scale_cfg, bias):
    """Check forward precision across formats, scales, bias, and empty experts."""
    skip_unsupported_dtype_scale(dtype, scale_cfg)
    a, b, offs = _make(COUNTS, K=64, N=48)
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    y = _expert_mm(
        _cfg(scale_cfg, dtype), a, b, offs, bias=bias0.clone() if bias else None
    )

    y_ref = torch.empty_like(y)
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        if hi > lo:
            fwd = mm_ref(
                a[lo:hi],
                operand_fmt(dtype, "act", "fwd"),
                b[group],
                operand_fmt(dtype, "weight", "fwd"),
                scale_cfg,
            )
            if bias:
                fwd = fwd + bias0[group].to(fwd.dtype)
            y_ref[lo:hi] = fwd.to(y.dtype)
        lo = hi
    assert rel(y, y_ref) < PRECISION_BOUND


@fp8_only
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
def test_scaled_grouped_gemm_fn_backward_precision(dtype, scale_cfg, bias):
    """Check backward precision and fp32 bias-gradient accumulation."""
    skip_unsupported_dtype_scale(dtype, scale_cfg)
    # The wgrad of a grouped GEMM contracts over the ragged axis.
    skip_unsupported_ragged_k_scale(scale_cfg)
    a, b, offs = _make(COUNTS, K=64, N=48)
    bias0 = torch.randn(offs.shape[0], 48, device="cuda", dtype=torch.bfloat16) * 0.1
    a_q, b_q = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    bias_q = bias0.clone().requires_grad_(True) if bias else None
    y = _expert_mm(_cfg(scale_cfg, dtype), a_q, b_q, offs, bias=bias_q)
    gy = torch.randn_like(y)
    y.backward(gy)

    ga_ref, gb_ref = torch.empty_like(a), torch.zeros_like(b)
    lo = 0
    for group, hi in enumerate(offs.tolist()):
        if hi > lo:
            ga_ref[lo:hi] = mm_ref(
                gy[lo:hi],
                operand_fmt(dtype, "grad_out", "dgrad"),
                b[group].t().contiguous(),
                operand_fmt(dtype, "weight", "dgrad"),
                scale_cfg,
            ).to(y.dtype)
            gb_ref[group] = mm_ref(
                a[lo:hi].t(),
                operand_fmt(dtype, "act", "wgrad"),
                gy[lo:hi],
                operand_fmt(dtype, "grad_out", "wgrad"),
                scale_cfg,
            ).to(y.dtype)
        lo = hi

    expected = [("grad_a", a_q.grad, ga_ref), ("grad_b", b_q.grad, gb_ref)]
    if bias:
        rows = torch.arange(y.shape[0], device=offs.device)
        acc = torch.zeros_like(bias0, dtype=torch.float32)
        acc.index_add_(0, torch.searchsorted(offs, rows, right=True), gy.float())
        expected.append(("grad_bias", bias_q.grad, acc.to(bias0.dtype)))
    for name, got, ref in expected:
        assert rel(got, ref) < PRECISION_BOUND, (name, rel(got, ref))


# --- QuantizedSparseMoEBlock ---


def test_quantized_sparse_moe_block_only_quantizes_during_training():
    """Check quantized training and exact unquantized evaluation."""
    torch.manual_seed(0)
    source = SparseMoEBlock(
        d_model=4,
        intermediate_size=8,
        n_routed_experts=2,
        n_routed_experts_per_token=1,
        aux_loss=False,
    )
    block = QuantizedSparseMoEBlock.from_module(source, rule(INT4_W8A16_DTYPES))
    args = (torch.randn(4, 4), torch.randn(2, 4, 8), torch.tensor([2, 4]))
    plain = SparseMoEBlock.expert_mm(block, *args, projection="gate_up")

    block.train()
    assert not torch.equal(block.expert_mm(*args, projection="gate_up"), plain)
    block.eval()
    assert torch.equal(block.expert_mm(*args, projection="gate_up"), plain)


@fp8_only
def test_quantized_sparse_moe_block_autocast():
    """Check autocast GEMMs, caller-dtype outputs, and fp32-master gradients."""
    torch.manual_seed(0)
    source = SparseMoEBlock(
        d_model=32,
        intermediate_size=48,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=True,
        aux_loss_coef=1e-3,
    ).cuda()  # Keep the master weights in fp32.
    with torch.no_grad():  # Initialize the expert weights for a nonzero signal.
        nn.init.normal_(source.expert_gate_up, std=0.02)
        nn.init.normal_(source.expert_down, std=0.02)
    block = QuantizedSparseMoEBlock.from_module(
        source, rule(FP8_E4M3_W8A8_E5M2_G8_DTYPES, ROWWISE)
    )
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out, _ = block(x)
    assert out.dtype == torch.float32  # The block preserves the caller's dtype.
    out.square().mean().backward()
    assert block.expert_gate_up.grad.dtype == torch.float32  # Matches the fp32 master.
    assert torch.isfinite(block.expert_gate_up.grad).all()
    assert torch.isfinite(block.expert_down.grad).all()
    assert torch.isfinite(x.grad).all()


@fp8_only
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_sparse_moe_block_trains_a_full_model(bias):
    """Check one converted FP8 MoE step and fused per-expert bias gradients."""
    config = TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=128,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "moe",
                    "mlp_kwargs": {
                        "intermediate_size": 48,
                        "n_routed_experts": 4,
                        "n_routed_experts_per_token": 2,
                        "bias": bias,
                        "aux_loss": True,
                        "aux_loss_coef": 1e-3,
                        "activation": "silu",
                        "gated": True,
                    },
                }
            ],
            norm_cls="rmsnorm",
            pos_emb_cls="rope",
        ),
        training=TrainingConfig(
            mixed_precision="bf16",
            quantization={
                "enabled": True,
                "dtype": {"recipe": "fp8"},
                "scale": {"recipe": "rowwise"},
            },
        ),
    )
    model = build_model(config).cuda().to(torch.bfloat16)
    apply_quantization(model, config)
    blocks = [m for m in model.modules() if isinstance(m, QuantizedSparseMoEBlock)]
    assert blocks  # Conversion installed the quantized seam.

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()

    assert torch.isfinite(loss).item()
    for block in blocks:
        for name, p in block.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
            assert p.grad.abs().sum() > 0, name
