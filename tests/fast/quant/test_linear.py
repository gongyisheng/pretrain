import pytest
import torch
import torch.nn as nn

from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.convert import apply_quantization
from src.quant.linear import QuantizedLinear, quantized_mm
from src.quant.rotation import build_rotation
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig
from tests.fast.quant.helper import (
    ALL_FORMATS,
    FORWARD_DTYPES,
    BACKWARD_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    ALL_SCALES,
    TENSORWISE,
    BLOCKWISE1D_128,
    fp8_only,
    mm_ref,
    operand_fmt,
    rel,
    roundtrip,
    rule,
    skip_unsupported_dtype_scale,
    skip_unsupported_fmt_scale,
)


PASSTHROUGH_DTYPES = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
]
OUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
# Canonical rotation config shared by quantized modules and their oracles.
ROTATION_CFG = {
    "rotation_cls": "hadamard",
    "rotation_kwargs": {"block_size": 32, "random_sign": True, "seed": 42},
}

ROTATED_REL_TOL = 1e-2


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("fmt,dtype", PASSTHROUGH_DTYPES)
def test_quantized_mm_passthrough(fmt, dtype, out_dtype):
    """Unquantized matmul preserves operand dtype and adds bias before casting."""
    torch.manual_seed(0)
    a = torch.randn(20, 32, dtype=dtype)
    b = torch.randn(32, 40, dtype=dtype)
    bias = torch.randn(40, dtype=dtype)
    out = quantized_mm(a, b, fmt, fmt, out_dtype, {})
    assert out.dtype == out_dtype
    torch.testing.assert_close(out, (a @ b).to(out_dtype), atol=0, rtol=0)
    torch.testing.assert_close(
        quantized_mm(a, b, fmt, fmt, out_dtype, {}, bias=bias),
        torch.addmm(bias, a, b).to(out_dtype),
        atol=0,
        rtol=0,
    )


@fp8_only
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("rotation_cfg", [None, ROTATION_CFG])
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("b_fmt", ALL_FORMATS)
@pytest.mark.parametrize("a_fmt", ALL_FORMATS)
def test_quantized_mm_precision(a_fmt, b_fmt, scale_cfg, bias, rotation_cfg):
    """Compare each format pair with its dequantization oracle."""
    skip_unsupported_fmt_scale(a_fmt, scale_cfg)
    skip_unsupported_fmt_scale(b_fmt, scale_cfg)
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 128, device="cuda")
    bias_t = torch.randn(128, device="cuda") if bias else None
    # Share one rotation so the oracle and GEMM use the same baked-in sign vector.
    rotation = build_rotation(rotation_cfg)
    out = quantized_mm(
        a,
        b,
        a_fmt,
        b_fmt,
        torch.float32,
        scale_cfg,
        bias=bias_t,
        rotation=rotation,
    )
    ref = roundtrip(a, -1, a_fmt, scale_cfg, rotation=rotation) @ roundtrip(
        b, -2, b_fmt, scale_cfg, rotation=rotation
    )
    if bias:
        ref = ref + bias_t
    # Use 3e-4 absolute tolerance: worst error is 3e-5 and reordering dominates near zero.
    torch.testing.assert_close(
        out,
        ref,
        rtol=0,
        atol=3e-4,
    )


STATS_CASES = [
    ("int8", "int8", False, False, False),  # no stats: folding is skipped
    ("int8", "bf16", True, True, False),  # only `a` is quantized
    ("int8", "int8", True, True, True),
]


@pytest.mark.parametrize("a_fmt,b_fmt,with_stats,a_folded,b_folded", STATS_CASES)
def test_quantized_mm_records_stats(a_fmt, b_fmt, with_stats, a_folded, b_folded):
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    a_stats = QuantizationStats("act/x", 1, a.device) if with_stats else None
    b_stats = QuantizationStats("weight/x", 1, b.device) if with_stats else None
    set_quantization_monitoring_status(True)
    try:
        out = quantized_mm(
            a,
            b,
            a_fmt,
            b_fmt,
            torch.float32,
            TENSORWISE,
            a_stats=a_stats,
            b_stats=b_stats,
        )
    finally:
        set_quantization_monitoring_status(False)
    assert torch.isfinite(out).all()
    if with_stats:
        assert a_stats.numel.item() == (a.numel() if a_folded else 0)
        assert b_stats.numel.item() == (b.numel() if b_folded else 0)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("eval_mode", [False, True])
def test_quantized_linear_from_module(bias, eval_mode):
    source = nn.Linear(64, 32, bias=bias)
    if eval_mode:
        source.eval()
    q = QuantizedLinear.from_module(source, rule({"recipe": "fp8"}))
    assert torch.equal(q.weight, source.weight) and q.weight.requires_grad
    assert q.training is not eval_mode  # preserve the source mode
    if bias:
        assert torch.equal(q.bias, source.bias)
    else:
        assert q.bias is None


@fp8_only
@pytest.mark.parametrize("rotation_cfg", [None, ROTATION_CFG])
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("dtype", FORWARD_DTYPES)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_forward_precision(dtype, scale_cfg, bias, rotation_cfg):
    """Match the quantized result to its rounded bf16 oracle."""
    skip_unsupported_dtype_scale(dtype, scale_cfg)
    act_fmt, weight_fmt = (
        operand_fmt(dtype, "act", "fwd"),
        operand_fmt(dtype, "weight", "fwd"),
    )
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=bias).cuda().to(torch.bfloat16)
    cfg = rule(dtype, scale_cfg, rotation=rotation_cfg)
    rotation = build_rotation(rotation_cfg)
    q = QuantizedLinear.from_module(lin, cfg, rotation=rotation)
    x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)

    out = q(x)
    # Flatten batch dimensions to match the GEMM.
    ref2d = mm_ref(
        x.flatten(0, -2),
        act_fmt,
        lin.weight.t(),
        weight_fmt,
        scale_cfg,
        rotation=rotation,
    )
    if bias:
        # Bias is added in the fp32 accumulator before the final round.
        ref2d = ref2d + lin.bias.float()
    ref = ref2d.to(torch.bfloat16).unflatten(0, x.shape[:-1])
    assert out.shape == (2, 128, 128) and out.dtype == torch.bfloat16
    tol = ROTATED_REL_TOL if rotation_cfg is not None else 3e-4
    assert rel(out, ref) < tol


# 250 exercises the wgrad contraction pad.
N_TOKENS = [256, 250]
BACKWARD_ROTATION_GEMMS = [
    None,
    ("wgrad",),
    ("fwd", "dgrad", "wgrad"),
]


@fp8_only
@pytest.mark.parametrize("rotation_gemms", BACKWARD_ROTATION_GEMMS)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_backward_precision(
    dtype, scale_cfg, n_tokens, bias, rotation_gemms
):
    """Backward matches unrotated, Wgrad-only, and all-GEMM oracles.

    The unrotated cells retain non-divisible token counts to exercise Wgrad padding.
    """
    skip_unsupported_dtype_scale(dtype, scale_cfg)
    if (
        rotation_gemms is not None
        and n_tokens % ROTATION_CFG["rotation_kwargs"]["block_size"]
    ):
        pytest.skip(
            f"rotation block {ROTATION_CFG['rotation_kwargs']['block_size']} does not divide "
            f"the {n_tokens}-token contraction"
        )
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=bias).cuda().to(torch.bfloat16)
    rotation = (
        None
        if rotation_gemms is None
        else {**ROTATION_CFG, "gemms": list(rotation_gemms)}
    )
    cfg = rule(dtype, scale_cfg, rotation=rotation)
    rotation = build_rotation(rotation)
    q = QuantizedLinear.from_module(lin, cfg, rotation=rotation)

    x = torch.randn(
        2, n_tokens // 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    out = q(x)
    g = torch.randn_like(out)
    out.backward(g)

    # dx = g @ W and dW = gᵀ @ X, with separate GEMM formats. Each GEMM's oracle
    # is rotated only when the config scopes the rotation to it.
    g2d, x2d = g.flatten(0, -2), x.detach().flatten(0, -2)
    effective = set(rotation_gemms or ())
    rotated = {"dgrad": "dgrad" in effective, "wgrad": "wgrad" in effective}
    dx_ref = mm_ref(
        g2d,
        operand_fmt(dtype, "grad_out", "dgrad"),
        lin.weight,
        operand_fmt(dtype, "weight", "dgrad"),
        scale_cfg,
        rotation=rotation if rotated["dgrad"] else None,
    )
    dw_ref = mm_ref(
        g2d.t(),
        operand_fmt(dtype, "grad_out", "wgrad"),
        x2d,
        operand_fmt(dtype, "act", "wgrad"),
        scale_cfg,
        rotation=rotation if rotated["wgrad"] else None,
    )
    expected = [
        (x.grad, dx_ref.unflatten(0, x.shape[:-1]), rotated["dgrad"]),
        (q.weight.grad, dw_ref, rotated["wgrad"]),
    ]
    if bias:
        expected.append((q.bias.grad, g2d.sum(0), False))  # db bypasses GEMM
    likes = (x, q.weight) + ((q.bias,) if bias else ())
    for (grad, ref, is_rotated), like in zip(expected, likes):
        assert torch.isfinite(grad).all()
        assert grad.dtype == like.dtype
        assert grad.shape == like.shape
        # Compare in the gradient's master dtype.
        tol = ROTATED_REL_TOL if is_rotated else 3e-4
        assert rel(grad, ref.to(grad.dtype)) < tol


def test_quantized_linear_only_quantizes_during_training():
    """Training quantizes; evaluation matches the unquantized linear exactly."""
    torch.manual_seed(0)
    q = QuantizedLinear.from_module(nn.Linear(4, 3), rule(INT4_W8A16_DTYPES))
    x = torch.randn(2, 4)
    plain = nn.functional.linear(x, q.weight, q.bias)

    q.train()
    assert not torch.equal(q(x), plain)
    q.eval()
    assert torch.equal(q(x), plain)


# Stochastic-rounding targets and consuming GEMMs.
SR_CASES = [
    ("weight", ("fwd", "dgrad")),
    ("act", ("fwd", "wgrad")),
    ("grad_out", ("dgrad", "wgrad")),
]


@fp8_only
@pytest.mark.parametrize("enable_sr", [False, True])
@pytest.mark.parametrize("tensor,gemms", SR_CASES)
def test_quantized_linear_stochastic_rounding(tensor, gemms, enable_sr):
    """SR affects only the GEMMs consuming the selected tensor."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    cfg = rule(
        FP8_E4M3_W8A8_E5M2_G8_DTYPES, None, {tensor: "SR" if enable_sr else "RNE"}
    )
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16)

    def step():
        q.zero_grad(set_to_none=True)
        x.grad = None
        out = q(x)
        out.backward(g)
        return {
            "fwd": out.detach().clone(),
            "wgrad": q.weight.grad.clone(),
            "dgrad": x.grad.clone(),
        }

    first, second = step(), step()
    stochastic = gemms if enable_sr else ()
    for gemm, got in first.items():
        assert torch.equal(got, second[gemm]) is (gemm not in stochastic)


@fp8_only
def test_quantized_linear_autocast():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.float32)  # fp32 master
    q = QuantizedLinear.from_module(lin, rule({"recipe": "fp8"}))
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = q(x)
    assert out.dtype == torch.bfloat16  # follows autocast
    out.square().mean().backward()
    assert q.weight.grad.dtype == torch.float32  # grad matches the fp32 master
    assert torch.isfinite(q.weight.grad).all()


@fp8_only
@pytest.mark.parametrize("rotation_cfg", [None, ROTATION_CFG])
def test_quantized_linear_compiles_fullgraph(rotation_cfg):
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    cfg = rule(
        {"recipe": "fp8"},
        BLOCKWISE1D_128,
        rotation=rotation_cfg,
    )
    q = QuantizedLinear.from_module(lin, cfg, rotation=build_rotation(rotation_cfg))
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = torch.compile(q, fullgraph=True)(x)
    out.square().mean().backward()

    assert out.shape == (64, 128) and torch.isfinite(out).all()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@fp8_only
def test_quantized_linear_trains_a_full_model():
    """Run a training step through a quantized TransformerLM."""
    config = TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=128,
            attn=[{"attn_cls": "gqa", "attn_kwargs": {"n_heads": 4, "n_kv_heads": 2}}],
            mlp=[
                {
                    "mlp_cls": "dense",
                    "mlp_kwargs": {
                        "activation_cls": "swiglu",
                    },
                }
            ],
            norm_cls="rmsnorm",
            pos_emb_cls="rope",
        ),
        training=TrainingConfig(
            mixed_precision="bf16",
            quantization={"enabled": True, "dtype": {"recipe": "fp8"}},
        ),
    )
    model = build_model(config)
    apply_quantization(model, config)
    model.cuda().to(torch.bfloat16)
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()

    assert torch.isfinite(loss).item()
    weight = next(m.weight for m in model.modules() if isinstance(m, QuantizedLinear))
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
