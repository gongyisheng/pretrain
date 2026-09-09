import pytest
import torch
import torch.nn as nn

from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.convert import apply_quantization
from src.quant.linear import QuantizedLinear, quantized_mm
from src.quant.rotation import build_rotation
from src.quant.utils import is_fp4
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig
from tests.fast.helper import cuda_capability_at_least, cuda_sm89_or_newer
from tests.fast.quant.helper import (
    ALL_FORMATS,
    FORWARD_DTYPES,
    BACKWARD_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    SCALE_PAIRS,
    SCALE_TRIPLES,
    TENSORWISE,
    BLOCKWISE1D_128,
    BLOCKWISE2D_128,
    mm_ref,
    operand_fmt,
    rel,
    roundtrip,
    rule,
    skip_unsupported_dtype_scale,
    skip_unsupported_fmt_scale,
    scale_combinations,
    uses_fp4_gemm,
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
# worst errors 3.04e-4 forward, 9.12e-4 dgrad, and 7.09e-4 wgrad
LINEAR_REL_TOL = 3e-3
# worst errors 1.316e-3
FP4_4OVER6_FORWARD_REL_TOL = 5e-3
MM_PRECISION_SHAPE = (256, 512, 128)
NVFP4_MM_PRECISION_SHAPE = (32, 32, 16)
MM_PRECISION_DEVICES = ["cpu", "cuda"]
MM_PRECISION_BIASES = [False, True]
MM_PRECISION_ROTATIONS = [None, ROTATION_CFG]
COMPILE_SCALE_TRIPLES = scale_combinations([BLOCKWISE1D_128, BLOCKWISE2D_128], 3)


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("fmt,dtype", PASSTHROUGH_DTYPES)
def test_quantized_mm_passthrough(fmt, dtype, out_dtype):
    """Unquantized matmul preserves operand dtype and adds bias before casting."""
    torch.manual_seed(0)
    a = torch.randn(20, 32, dtype=dtype)
    b = torch.randn(32, 40, dtype=dtype)
    bias = torch.randn(40, dtype=dtype)
    out = quantized_mm(a, b, fmt, fmt, out_dtype, TENSORWISE, TENSORWISE)
    assert out.dtype == out_dtype
    torch.testing.assert_close(out, (a @ b).to(out_dtype), atol=0, rtol=0)
    torch.testing.assert_close(
        quantized_mm(a, b, fmt, fmt, out_dtype, TENSORWISE, TENSORWISE, bias=bias),
        torch.addmm(bias, a, b).to(out_dtype),
        atol=0,
        rtol=0,
    )


def test_quantized_mm_raise_error():
    a = torch.ones(2, 4, device="cpu", dtype=torch.bfloat16)
    b = torch.ones(4, 3, device="cpu", dtype=torch.float16)

    with pytest.raises(ValueError):
        quantized_mm(a, b, "bf16", "fp16", torch.bfloat16, TENSORWISE, TENSORWISE)


@pytest.mark.parametrize("device", MM_PRECISION_DEVICES)
@pytest.mark.parametrize("a_fmt", ALL_FORMATS)
@pytest.mark.parametrize("b_fmt", ALL_FORMATS)
@pytest.mark.parametrize("a_scale,b_scale", SCALE_PAIRS)
@pytest.mark.parametrize("bias", MM_PRECISION_BIASES)
@pytest.mark.parametrize("rotation_cfg", MM_PRECISION_ROTATIONS)
def test_quantized_mm_precision(
    device,
    a_fmt,
    b_fmt,
    a_scale,
    b_scale,
    bias,
    rotation_cfg,
):
    """Compare each format pair with its dequantization oracle."""
    has_fp4 = is_fp4(a_fmt) or is_fp4(b_fmt)
    if device == "cpu" and not has_fp4:
        pytest.skip("CPU precision coverage is limited to FP4")
    capability = (10, 0) if uses_fp4_gemm(a_fmt, b_fmt, a_scale) else (8, 9)
    if device == "cuda" and not cuda_capability_at_least(capability):
        pytest.skip(f"CUDA SM{capability[0]}{capability[1]} or newer required")
    shape = NVFP4_MM_PRECISION_SHAPE if has_fp4 else MM_PRECISION_SHAPE
    atol = 2e-5 if has_fp4 else 3e-4
    skip_unsupported_fmt_scale(a_fmt, a_scale)
    skip_unsupported_fmt_scale(b_fmt, b_scale)
    torch.manual_seed(0)
    n_rows, contraction_size, n_columns = shape
    a = torch.randn(n_rows, contraction_size, device=device)
    b = torch.randn(contraction_size, n_columns, device=device)
    bias_t = torch.randn(n_columns, device=device) if bias else None
    # Share one rotation so the oracle and GEMM use the same baked-in sign vector.
    rotation = build_rotation(rotation_cfg)
    out = quantized_mm(
        a,
        b,
        a_fmt,
        b_fmt,
        torch.float32,
        a_scale,
        b_scale,
        bias=bias_t,
        rotation=rotation,
    )
    ref = roundtrip(a, -1, a_fmt, a_scale, rotation=rotation) @ roundtrip(
        b, -2, b_fmt, b_scale, rotation=rotation
    )
    if bias:
        ref = ref + bias_t
    # Full grids peak at 4.58e-5 generic and 5.73e-6 FP4 (6.55x and 3.50x).
    torch.testing.assert_close(
        out,
        ref,
        rtol=0,
        atol=atol,
    )


# fmt: off
# Format pairs bind the expected statistics.
STATS_CASES = [
    ("int8", "int8", False, False, False),
    ("int8", "bf16", True, True, False),
    ("int8", "int8", True, True, True),
    ("fp4_e2m1_4over6", "fp4_e2m1_4over6", True, True, True),
]
# fmt: on


@pytest.mark.parametrize("a_fmt,b_fmt,with_stats,a_folded,b_folded", STATS_CASES)
@pytest.mark.parametrize("a_scale,b_scale", SCALE_PAIRS)
def test_quantized_mm_records_stats(
    a_fmt,
    b_fmt,
    with_stats,
    a_folded,
    b_folded,
    a_scale,
    b_scale,
):
    skip_unsupported_fmt_scale(a_fmt, a_scale)
    skip_unsupported_fmt_scale(b_fmt, b_scale)
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    a_stats = QuantizationStats("act/x", 1, a.device) if with_stats else None
    b_stats = QuantizationStats("weight/x", 1, b.device) if with_stats else None
    unmonitored = quantized_mm(a, b, a_fmt, b_fmt, torch.float32, a_scale, b_scale)
    set_quantization_monitoring_status(True)
    try:
        out = quantized_mm(
            a,
            b,
            a_fmt,
            b_fmt,
            torch.float32,
            a_scale,
            b_scale,
            a_stats=a_stats,
            b_stats=b_stats,
        )
    finally:
        set_quantization_monitoring_status(False)
    torch.testing.assert_close(out, unmonitored, rtol=0, atol=0)
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


@cuda_sm89_or_newer
@pytest.mark.parametrize("rotation_cfg", [None, ROTATION_CFG])
@pytest.mark.parametrize("act_scale,weight_scale", SCALE_PAIRS)
@pytest.mark.parametrize("dtype", FORWARD_DTYPES)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_forward_precision(
    dtype,
    act_scale,
    weight_scale,
    bias,
    rotation_cfg,
):
    """Match the quantized result to its rounded bf16 oracle."""
    skip_unsupported_dtype_scale(dtype, act_scale)
    skip_unsupported_dtype_scale(dtype, weight_scale)
    act_fmt, weight_fmt = (
        operand_fmt(dtype, "act", "fwd"),
        operand_fmt(dtype, "weight", "fwd"),
    )
    if uses_fp4_gemm(act_fmt, weight_fmt, act_scale) and not cuda_capability_at_least(
        (10, 0)
    ):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=bias).cuda().to(torch.bfloat16)
    cfg = rule(
        dtype,
        {
            **act_scale,
            "block_shape": {
                "weight": weight_scale["block_shape"],
                "act": act_scale["block_shape"],
                "grad_out": act_scale["block_shape"],
            },
        },
        rotation=rotation_cfg,
    )
    rotation = build_rotation(rotation_cfg)
    q = QuantizedLinear.from_module(lin, cfg, rotation=rotation)
    x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)

    out = q(x)
    # Flatten batch dimensions to match the GEMM.
    ref2d = mm_ref(
        x.flatten(0, -2),
        lin.weight.t(),
        act_fmt,
        weight_fmt,
        act_scale,
        weight_scale,
        rotation=rotation,
    )
    if bias:
        ref2d = ref2d + lin.bias.float()
    ref = ref2d.to(torch.bfloat16).unflatten(0, x.shape[:-1])
    assert out.shape == (2, 128, 128) and out.dtype == torch.bfloat16
    tolerance = LINEAR_REL_TOL
    if "fp4_e2m1_4over6" in (act_fmt, weight_fmt) and uses_fp4_gemm(
        act_fmt, weight_fmt, act_scale
    ):
        tolerance = FP4_4OVER6_FORWARD_REL_TOL
    assert rel(out, ref) < tolerance


# 250 exercises the wgrad contraction pad.
N_TOKENS = [256, 250]
BACKWARD_ROTATION_GEMMS = [
    None,
    ("wgrad",),
    ("fwd", "dgrad", "wgrad"),
]


@cuda_sm89_or_newer
@pytest.mark.parametrize("rotation_gemms", BACKWARD_ROTATION_GEMMS)
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("act_scale,weight_scale,grad_out_scale", SCALE_TRIPLES)
@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_backward_precision(
    dtype,
    act_scale,
    weight_scale,
    grad_out_scale,
    n_tokens,
    bias,
    rotation_gemms,
):
    """Backward matches unrotated, Wgrad-only, and all-GEMM oracles.

    The unrotated cells retain non-divisible token counts to exercise Wgrad padding.
    """
    skip_unsupported_dtype_scale(dtype, act_scale)
    skip_unsupported_dtype_scale(dtype, weight_scale)
    skip_unsupported_dtype_scale(dtype, grad_out_scale)
    fwd_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "act", "fwd"), operand_fmt(dtype, "weight", "fwd"), act_scale
    )
    dgrad_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "grad_out", "dgrad"),
        operand_fmt(dtype, "weight", "dgrad"),
        grad_out_scale,
    )
    wgrad_uses_fused_fp4 = uses_fp4_gemm(
        operand_fmt(dtype, "grad_out", "wgrad"),
        operand_fmt(dtype, "act", "wgrad"),
        grad_out_scale,
    )
    if (
        fwd_uses_fused_fp4 or dgrad_uses_fused_fp4 or wgrad_uses_fused_fp4
    ) and not cuda_capability_at_least((10, 0)):
        pytest.skip("fused FP4 requires CUDA SM100 or newer")
    wgrad_uses_fp4 = is_fp4(operand_fmt(dtype, "act", "wgrad")) or is_fp4(
        operand_fmt(dtype, "grad_out", "wgrad")
    )
    if wgrad_uses_fp4 and n_tokens % 16:
        pytest.skip("FP4 Wgrad requires a contraction extent divisible by 16")
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
    if bias:
        lin.bias = nn.Parameter(lin.bias.float())
    rotation = (
        None
        if rotation_gemms is None
        else {**ROTATION_CFG, "gemms": list(rotation_gemms)}
    )
    cfg = rule(
        dtype,
        {
            **act_scale,
            "block_shape": {
                "weight": weight_scale["block_shape"],
                "act": act_scale["block_shape"],
                "grad_out": grad_out_scale["block_shape"],
            },
        },
        rotation=rotation,
    )
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
        lin.weight,
        operand_fmt(dtype, "grad_out", "dgrad"),
        operand_fmt(dtype, "weight", "dgrad"),
        grad_out_scale,
        weight_scale,
        rotation=rotation if rotated["dgrad"] else None,
    )
    dw_ref = mm_ref(
        g2d.t(),
        x2d,
        operand_fmt(dtype, "grad_out", "wgrad"),
        operand_fmt(dtype, "act", "wgrad"),
        grad_out_scale,
        act_scale,
        rotation=rotation if rotated["wgrad"] else None,
    )
    expected = [
        (x.grad, dx_ref.unflatten(0, x.shape[:-1])),
        (q.weight.grad, dw_ref),
    ]
    likes = (x, q.weight)
    for (grad, ref), like in zip(expected, likes):
        assert torch.isfinite(grad).all()
        assert grad.dtype == like.dtype
        assert grad.shape == like.shape
        # Compare in the gradient's master dtype.
        assert rel(grad, ref.to(grad.dtype)) < LINEAR_REL_TOL
    if bias:
        assert q.bias.grad.dtype == torch.float32
        assert q.bias.grad.shape == q.bias.shape
        torch.testing.assert_close(
            q.bias.grad,
            g2d.sum(0, dtype=torch.float32),
            atol=0,
            rtol=0,
        )


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


@cuda_sm89_or_newer
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


@cuda_sm89_or_newer
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


@cuda_sm89_or_newer
@pytest.mark.parametrize("rotation_cfg", [None, ROTATION_CFG])
@pytest.mark.parametrize("act_scale,weight_scale,grad_out_scale", COMPILE_SCALE_TRIPLES)
def test_quantized_linear_compiles_fullgraph(
    rotation_cfg,
    act_scale,
    weight_scale,
    grad_out_scale,
):
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    cfg = rule(
        {"recipe": "fp8"},
        {
            **act_scale,
            "block_shape": {
                "weight": weight_scale["block_shape"],
                "act": act_scale["block_shape"],
                "grad_out": grad_out_scale["block_shape"],
            },
        },
        rotation=rotation_cfg,
    )
    q = QuantizedLinear.from_module(lin, cfg, rotation=build_rotation(rotation_cfg))
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    torch.compiler.reset()
    try:
        out = torch.compile(q, fullgraph=True)(x)
        out.square().mean().backward()
    finally:
        torch.compiler.reset()

    assert out.shape == (64, 128) and torch.isfinite(out).all()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@cuda_sm89_or_newer
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
