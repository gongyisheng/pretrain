import pytest
import torch
import torch.nn as nn

from src.metrics.quant import QuantizationStats, set_quantization_monitoring_status
from src.model import build_model
from src.quant.constants import QUANT_PASSTHROUGH, _FP8_FORMATS, _INT8_FORMATS
import src.quant.linear as quant_linear
from src.quant.convert import apply_quantization
from src.quant.linear import QuantizedLinear, quantized_mm
from src.quant.quantize import dequantize_operand, quantize_operand
from src.quant.utils import is_fp8, is_int8s, is_quantized
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig


def _fp8_capable():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")
cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the scaled GEMM kernels need CUDA"
)


E4M3 = "fp8_e4m3"
FP8_FORMATS = sorted(_FP8_FORMATS)
INT_FORMATS = sorted(_INT8_FORMATS)
ALL_QUANT_FORMATS = FP8_FORMATS + INT_FORMATS
PASSTHROUGH_FORMATS = sorted(QUANT_PASSTHROUGH)
ALL_FORMATS = ALL_QUANT_FORMATS + PASSTHROUGH_FORMATS

FP8_E4M3_W8A8_DTYPES = {"weight": "fp8_e4m3", "act": "fp8_e4m3", "grad_out": "bf16"}
FP8_E4M3_W16A8_DTYPES = {"weight": "bf16", "act": "fp8_e4m3", "grad_out": "bf16"}
FP8_E4M3_W8A16_DTYPES = {"weight": "fp8_e4m3", "act": "bf16", "grad_out": "bf16"}
FP8_E5M2_W8A8_DTYPES = {"weight": "fp8_e5m2", "act": "fp8_e5m2", "grad_out": "bf16"}
FP8_E5M2_W16A8_DTYPES = {"weight": "bf16", "act": "fp8_e5m2", "grad_out": "bf16"}
FP8_E5M2_W8A16_DTYPES = {"weight": "fp8_e5m2", "act": "bf16", "grad_out": "bf16"}
FP8_E4M3_W8A8G8_DTYPES = {
    "weight": "fp8_e4m3",
    "act": "fp8_e4m3",
    "grad_out": "fp8_e4m3",
}
FP8_E4M3_W8A8_E5M2_G8_DTYPES = {
    "weight": "fp8_e4m3",
    "act": "fp8_e4m3",
    "grad_out": "fp8_e5m2",
}
FP8_E5M2_W8A8G8_DTYPES = {
    "weight": "fp8_e5m2",
    "act": "fp8_e5m2",
    "grad_out": "fp8_e5m2",
}
FP8_E5M2_W8A8_E4M3_G8_DTYPES = {
    "weight": "fp8_e5m2",
    "act": "fp8_e5m2",
    "grad_out": "fp8_e4m3",
}
INT8_W8A8_DTYPES = {"weight": "int8", "act": "int8", "grad_out": "bf16"}
INT8_W8A8G8_DTYPES = {"weight": "int8", "act": "int8", "grad_out": "int8"}
INT8_W16A8_DTYPES = {"weight": "bf16", "act": "int8", "grad_out": "bf16"}
INT8_W8A16_DTYPES = {"weight": "int8", "act": "bf16", "grad_out": "bf16"}
INT7_W8A16_DTYPES = {"weight": "int7", "act": "bf16", "grad_out": "bf16"}
INT6_W8A16_DTYPES = {"weight": "int6", "act": "bf16", "grad_out": "bf16"}
INT5_W8A16_DTYPES = {"weight": "int5", "act": "bf16", "grad_out": "bf16"}
INT4_W8A16_DTYPES = {"weight": "int4", "act": "bf16", "grad_out": "bf16"}

FP8_E4M3_W8A8G8_GWHP_DTYPES = {
    "weight": "fp8_e4m3",
    "act": {"fwd": "fp8_e4m3", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"},
}
FP8_E4M3_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e4m3", "dgrad": "bf16"},
    "act": "fp8_e4m3",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e4m3"},
}
FP8_E4M3_W8A8_E5M2_G8_GWHP_DTYPES = {
    "weight": "fp8_e4m3",
    "act": {"fwd": "fp8_e4m3", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "bf16"},
}
FP8_E4M3_W8A8_E5M2_G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e4m3", "dgrad": "bf16"},
    "act": "fp8_e4m3",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e5m2"},
}
FP8_E5M2_W8A8G8_GWHP_DTYPES = {
    "weight": "fp8_e5m2",
    "act": {"fwd": "fp8_e5m2", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e5m2", "wgrad": "bf16"},
}
FP8_E5M2_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e5m2", "dgrad": "bf16"},
    "act": "fp8_e5m2",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e5m2"},
}
FP8_E5M2_W8A8_E4M3_G8_GWHP_DTYPES = {
    "weight": "fp8_e5m2",
    "act": {"fwd": "fp8_e5m2", "wgrad": "bf16"},
    "grad_out": {"dgrad": "fp8_e4m3", "wgrad": "bf16"},
}
FP8_E5M2_W8A8_E4M3_G8_GIHP_DTYPES = {
    "weight": {"fwd": "fp8_e5m2", "dgrad": "bf16"},
    "act": "fp8_e5m2",
    "grad_out": {"dgrad": "bf16", "wgrad": "fp8_e4m3"},
}
INT8_W8A8G8_GWHP_DTYPES = {
    "weight": "int8",
    "act": {"fwd": "int8", "wgrad": "bf16"},
    "grad_out": {"dgrad": "int8", "wgrad": "bf16"},
}
INT8_W8A8G8_GIHP_DTYPES = {
    "weight": {"fwd": "int8", "dgrad": "bf16"},
    "act": "int8",
    "grad_out": {"dgrad": "bf16", "wgrad": "int8"},
}

FORWARD_DTYPES = [
    FP8_E4M3_W8A8_DTYPES,
    FP8_E4M3_W16A8_DTYPES,
    FP8_E4M3_W8A16_DTYPES,
    INT8_W8A8_DTYPES,
    INT8_W16A8_DTYPES,
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
]

BACKWARD_DTYPES = [
    FP8_E4M3_W8A8_DTYPES,
    FP8_E4M3_W16A8_DTYPES,
    FP8_E4M3_W8A16_DTYPES,
    FP8_E5M2_W8A8_DTYPES,
    FP8_E5M2_W16A8_DTYPES,
    FP8_E5M2_W8A16_DTYPES,
    FP8_E4M3_W8A8G8_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_DTYPES,
    FP8_E5M2_W8A8G8_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_DTYPES,
    INT8_W8A8_DTYPES,
    INT8_W16A8_DTYPES,
    INT8_W8A16_DTYPES,
    INT7_W8A16_DTYPES,
    INT6_W8A16_DTYPES,
    INT5_W8A16_DTYPES,
    INT4_W8A16_DTYPES,
    FP8_E4M3_W8A8G8_GWHP_DTYPES,
    FP8_E4M3_W8A8G8_GIHP_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_GWHP_DTYPES,
    FP8_E4M3_W8A8_E5M2_G8_GIHP_DTYPES,
    FP8_E5M2_W8A8G8_GWHP_DTYPES,
    FP8_E5M2_W8A8G8_GIHP_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_GWHP_DTYPES,
    FP8_E5M2_W8A8_E4M3_G8_GIHP_DTYPES,
    INT8_W8A8G8_GWHP_DTYPES,
    INT8_W8A8G8_GIHP_DTYPES,
]


def _scale(granularity, block_shape=(0, 0), scale_dtype=torch.float32):
    return {
        "granularity": granularity,
        "block_shape": block_shape,
        "scale_dtype": scale_dtype,
    }


TENSORWISE = _scale("tensorwise")
ROWWISE = _scale("rowwise")
# A blockwise contract extent must be a multiple of 16, and the outer extent is either
# 1 (a K-vector) or the same extent (a square tile).
BLOCKWISE1D_16 = _scale("blockwise", (1, 16))
BLOCKWISE1D_32 = _scale("blockwise", (1, 32))
BLOCKWISE1D_64 = _scale("blockwise", (1, 64))
BLOCKWISE1D_128 = _scale("blockwise", (1, 128))
BLOCKWISE2D_16 = _scale("blockwise", (16, 16))
BLOCKWISE2D_32 = _scale("blockwise", (32, 32))
BLOCKWISE2D_64 = _scale("blockwise", (64, 64))
BLOCKWISE2D_128 = _scale("blockwise", (128, 128))
# e8m0 scales are the mx grid, so their extent must be a multiple of 32. The 2D-64 tile
# composes the outer-axis expansion with the mxfp8 K-replication fast path (rep_k == 2).
MXFP8 = _scale("blockwise", (1, 32), torch.float8_e8m0fnu)
BLOCKWISE2D_64_E8M0 = _scale("blockwise", (64, 64), torch.float8_e8m0fnu)


ALL_SCALES = [
    TENSORWISE,
    ROWWISE,
    BLOCKWISE1D_16,
    BLOCKWISE1D_32,
    BLOCKWISE1D_64,
    BLOCKWISE1D_128,
    BLOCKWISE2D_16,
    BLOCKWISE2D_32,
    BLOCKWISE2D_64,
    BLOCKWISE2D_128,
    MXFP8,
    BLOCKWISE2D_64_E8M0,
]

SCALE_DTYPE_NAMES = {torch.float32: "fp32", torch.float8_e8m0fnu: "fp8_e8m0"}


def _roundtrip(x, contract_dim, fmt, scale_cfg):
    """dequant(quant(x)) — the operand the quantized GEMM actually multiplies."""
    if not is_quantized(fmt):
        return x
    xq, scale = quantize_operand(x, contract_dim, fmt, scale_cfg)
    return dequantize_operand(xq, scale, contract_dim, scale_cfg)


def _mm_ref(a, a_fmt, b, b_fmt, scale_cfg):
    """The GEMM quantized_mm actually performs, dequantized where that path dequantizes:
    a same-family pair lands on a fused kernel that works in its fp32 accumulator, any
    other pairing falls back and rounds the dequantized operand to bf16 first."""
    dtype = (
        torch.float32 if is_quantized(a_fmt) and is_quantized(b_fmt) else torch.bfloat16
    )
    return _roundtrip(a, -1, a_fmt, scale_cfg).to(dtype) @ _roundtrip(
        b, -2, b_fmt, scale_cfg
    ).to(dtype)


def _fmts(dtype):
    """Every format a dtype cell names, flattening the per-GEMM specs."""
    for spec in dtype.values():
        yield from spec.values() if isinstance(spec, dict) else (spec,)


def _fmt(dtype, tensor, gemm):
    """The format `dtype` names for one tensor in one GEMM; a plain string covers both
    of that tensor's GEMMs."""
    spec = dtype[tensor]
    return spec[gemm] if isinstance(spec, dict) else spec


def _rel(got, ref):
    return (got.float() - ref.float()).norm() / ref.float().norm()


def _rule(dtype, scale_cfg=None, rounding=None):
    """Build one resolved rule. Goes through TrainingConfig because that is what fills
    the tensors a partial `dtype` dict leaves out. `scale_cfg` is a resolved config, the
    same constant the kernels take, restated here in the config's vocabulary -- one key
    renamed, not a call into the resolver, so a test comparing against `scale_cfg` does
    not lean on the resolution it is checking."""
    spec = {"enabled": True, "dtype": dict(dtype)}
    if scale_cfg is not None:
        spec["scale"] = {
            **scale_cfg,
            "scale_dtype": SCALE_DTYPE_NAMES[scale_cfg["scale_dtype"]],
        }
    if rounding is not None:
        spec["rounding"] = dict(rounding)
    return TrainingConfig(mixed_precision="no", quantization=spec).quantization[0]


# --- quantized_mm ---


# The passthrough formats, paired with the tensor dtype each one names.
PASSTHROUGH_DTYPES = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
]
OUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


@pytest.mark.parametrize("out_dtype", OUT_DTYPES)
@pytest.mark.parametrize("fmt,dtype", PASSTHROUGH_DTYPES)
def test_quantized_mm_passthrough(fmt, dtype, out_dtype):
    """An unquantized pair never reaches a kernel: the matmul runs in the operand
    dtype, the unfused branch is the one that has to add the bias, and only the
    finished sum is cast to out_dtype. Exact compares, since no kernel is involved."""
    torch.manual_seed(0)
    a = torch.randn(20, 32, dtype=dtype)
    b = torch.randn(32, 40, dtype=dtype)
    bias = torch.randn(40, dtype=dtype)
    out = quantized_mm(a, b, fmt, fmt, out_dtype, {})
    assert out.dtype == out_dtype
    torch.testing.assert_close(out, (a @ b).to(out_dtype), atol=0, rtol=0)
    torch.testing.assert_close(
        quantized_mm(a, b, fmt, fmt, out_dtype, {}, bias=bias),
        (a @ b + bias).to(out_dtype),
        atol=0,
        rtol=0,
    )


@cuda_only
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("b_fmt", ALL_FORMATS)
@pytest.mark.parametrize("a_fmt", ALL_FORMATS)
def test_quantized_mm_precision(a_fmt, b_fmt, scale_cfg):
    """Every format pair stays within tolerance of its own dequant oracle on every
    scale config: what is left is accumulation order, not quantization. `bias` only
    shifts that result, so it never passes through the scales."""
    if (is_fp8(a_fmt) or is_fp8(b_fmt)) and not _fp8_capable():
        pytest.skip("fp8 needs SM >= 8.9")
    torch.manual_seed(0)
    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 128, device="cuda")
    bias = torch.randn(128, device="cuda")
    ref = _roundtrip(a, -1, a_fmt, scale_cfg) @ _roundtrip(b, -2, b_fmt, scale_cfg)

    torch.testing.assert_close(
        quantized_mm(a, b, a_fmt, b_fmt, torch.float32, scale_cfg),
        ref,
        rtol=0,
        atol=3e-4,
    )
    torch.testing.assert_close(
        quantized_mm(a, b, a_fmt, b_fmt, torch.float32, scale_cfg, bias=bias),
        ref + bias,
        rtol=0,
        atol=3e-4,
    )


STATS_CASES = [
    ("int8", "int8", False, False, False),  # training passes none: fold skipped
    ("int8", "bf16", True, True, False),  # only `a` is a quantized format
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


# --- QuantizedLinear ---


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("eval_mode", [False, True])
def test_quantized_linear_from_module(bias, eval_mode):
    source = nn.Linear(64, 32, bias=bias)
    if eval_mode:
        source.eval()
    q = QuantizedLinear.from_module(source, _rule({"recipe": "fp8"}))
    assert torch.equal(q.weight, source.weight) and q.weight.requires_grad
    assert q.training is not eval_mode  # from_module preserves the source's mode
    if bias:
        assert torch.equal(q.bias, source.bias)
    else:
        assert q.bias is None


@fp8_only
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("dtype", FORWARD_DTYPES)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_forward_precision(dtype, scale_cfg, bias):
    """A scale config has to reach the kernel intact, so the oracle rebuilds the same
    operands; rounded to bf16 as the accumulator does, only one-ulp disagreement is
    left. A loose bound would pass against an unquantized GEMM, 3.6e-2 away."""
    if scale_cfg["scale_dtype"] is torch.float8_e8m0fnu and any(
        is_int8s(fmt) for fmt in _fmts(dtype)
    ):
        pytest.skip("an e8m0 scale is defined only over fp8 elements")
    act_fmt, weight_fmt = _fmt(dtype, "act", "fwd"), _fmt(dtype, "weight", "fwd")
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=bias).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _rule(dtype, scale_cfg))
    x = torch.randn(2, 128, 256, device="cuda", dtype=torch.bfloat16)

    out = q(x)
    # forward blocks x along in-features (-1) and w along in-features (-1); the batch
    # dims are flattened before the GEMM, so the oracle flattens them too
    ref2d = _mm_ref(x.flatten(0, -2), act_fmt, lin.weight.t(), weight_fmt, scale_cfg)
    if bias:
        # both paths add it before the downcast: the kernel in its epilogue, the
        # fallback right after the matmul
        ref2d = ref2d + lin.bias.to(ref2d.dtype)
    ref = ref2d.to(torch.bfloat16).unflatten(0, x.shape[:-1])
    assert out.shape == (2, 128, 128) and out.dtype == torch.bfloat16
    assert _rel(out, ref) < 3e-4


# 250 is not a multiple of 32, so it exercises the wgrad contraction pad
N_TOKENS = [256, 250]


@fp8_only
@pytest.mark.parametrize("dtype", BACKWARD_DTYPES)
@pytest.mark.parametrize("scale_cfg", ALL_SCALES)
@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize("bias", [False, True])
def test_quantized_linear_backward_precision(dtype, scale_cfg, n_tokens, bias):
    if scale_cfg["scale_dtype"] is torch.float8_e8m0fnu and any(
        is_int8s(fmt) for fmt in _fmts(dtype)
    ):
        pytest.skip("an e8m0 scale is defined only over fp8 elements")
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=bias).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _rule(dtype, scale_cfg))

    x = torch.randn(
        2, n_tokens // 2, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    out = q(x)
    g = torch.randn_like(out)
    out.backward(g)

    # dx = g @ W and dW = gᵀ @ X, each quantized with its own GEMM's formats
    g2d, x2d = g.flatten(0, -2), x.detach().flatten(0, -2)
    dx_ref = _mm_ref(
        g2d,
        _fmt(dtype, "grad_out", "dgrad"),
        lin.weight,
        _fmt(dtype, "weight", "dgrad"),
        scale_cfg,
    )
    dw_ref = _mm_ref(
        g2d.t(),
        _fmt(dtype, "grad_out", "wgrad"),
        x2d,
        _fmt(dtype, "act", "wgrad"),
        scale_cfg,
    )
    expected = [(x.grad, dx_ref.unflatten(0, x.shape[:-1])), (q.weight.grad, dw_ref)]
    if bias:
        expected.append((q.bias.grad, g2d.sum(0)))  # db never passes through a GEMM
    for (grad, ref), like in zip(expected, (x, q.weight) + ((q.bias,) if bias else ())):
        assert torch.isfinite(grad).all()
        assert grad.dtype == like.dtype  # grads follow the master parameter's dtype
        assert grad.shape == like.shape  # dx comes back in the input's batched shape
        # the oracle lands in the master dtype the grads are stored in
        assert _rel(grad, ref.to(grad.dtype)) < 3e-4


def test_quantized_linear_only_quantizes_during_training(monkeypatch):
    calls = []
    original = quant_linear.quantized_mm
    monkeypatch.setattr(
        quant_linear,
        "quantized_mm",
        lambda *a, **kw: (calls.append(None), original(*a, **kw))[1],
    )
    q = QuantizedLinear.from_module(nn.Linear(4, 3), _rule({"weight": "int4"}))
    x = torch.randn(2, 4)

    q.train()
    q(x)
    q.eval()
    actual = q(x)

    assert len(calls) == 1
    torch.testing.assert_close(actual, nn.functional.linear(x, q.weight, q.bias))


# (the tensor whose rounding is set, the GEMMs that consume it)
SR_CASES = [
    ("weight", ("fwd", "dgrad")),
    ("act", ("fwd", "wgrad")),
    ("grad_out", ("dgrad", "wgrad")),
]


@fp8_only
@pytest.mark.parametrize("enable_sr", [False, True])
@pytest.mark.parametrize("tensor,gemms", SR_CASES)
def test_quantized_linear_stochastic_rounding(tensor, gemms, enable_sr):
    """SR draws a fresh random per call, so it has to reach exactly the GEMMs that
    consume the tensor it is set on: those stop reproducing across a repeated step,
    while every GEMM still rounding to nearest comes back bit-identical -- and under an
    explicit RNE, all three do."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    rule = _rule(
        FP8_E4M3_W8A8_E5M2_G8_DTYPES, None, {tensor: "SR" if enable_sr else "RNE"}
    )
    q = QuantizedLinear.from_module(lin, rule)
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
    q = QuantizedLinear.from_module(lin, _rule({"recipe": "fp8"}))
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32, requires_grad=True)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = q(x)
    assert out.dtype == torch.bfloat16  # follows autocast
    out.square().mean().backward()
    assert q.weight.grad.dtype == torch.float32  # grad matches the fp32 master
    assert torch.isfinite(q.weight.grad).all()


@fp8_only
def test_quantized_linear_compiles_fullgraph():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _rule({"recipe": "fp8"}, BLOCKWISE1D_128))
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    out = torch.compile(q, fullgraph=True)(x)
    out.square().mean().backward()

    assert out.shape == (64, 128) and torch.isfinite(out).all()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@fp8_only
def test_quantized_linear_trains_a_full_model():
    """End to end through the converter: a built TransformerLM takes one step."""
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
                    "mlp_kwargs": {"activation": "silu", "gated": True},
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
    model = build_model(config).cuda().to(torch.bfloat16)
    apply_quantization(model, config)
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
