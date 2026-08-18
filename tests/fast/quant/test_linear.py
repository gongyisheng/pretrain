import pytest
import torch
import torch.nn as nn

from src.metrics.quant import (
    QuantizationStats,
    set_quantization_monitoring_status,
)
from src.model import build_model
from src.quant.constants import _INT8_FORMATS
import src.quant.linear as quant_linear
from src.quant.linear import QuantizedLinear, quantized_mm
from src.quant.convert import apply_quantization
from src.quant.quantize import dequantize_operand, quantize_operand
from src.utils.config import (
    ModelConfig,
    QuantizationConfig,
    TrainConfig,
    TrainingConfig,
)

_E4M3 = "fp8_e4m3"


def _scaling(gran, bs=0, scale_dtype=torch.float32):
    return {
        "granularity": gran,
        "block_shape": (1, bs) if bs else (0, 0),
        "scale_dtype": scale_dtype,
    }


_MXFP8 = _scaling("blockwise", 32, torch.float8_e8m0fnu)


def test_mxfp8_oracle_scaling_uses_power_of_two_scales():
    _, scale = quantize_operand(torch.ones(1, 32), -1, _E4M3, _MXFP8)
    log2_scale = torch.log2(scale)
    assert torch.equal(log2_scale, log2_scale.round())


def _roundtrip(x, contract_dim, fmt, scaling):
    """dequant(quant(x)) — the operand the quantized GEMM actually multiplies."""
    xq, scale = quantize_operand(x, contract_dim, fmt, scaling)
    return dequantize_operand(xq, scale, contract_dim, scaling)


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")
# mxfp8 is a scaling scheme (power-of-two blockwise), not Blackwell-only anymore;
# it now runs through the same per-format scaled GEMM ops as any other CUDA fp8/int path.
mxfp8_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mxfp8 needs CUDA"
)


def _cfg(grad="bf16"):
    # recipe fills weight/act=fp8_e4m3; grad_out overridden in both its GEMMs
    return QuantizationConfig(enabled=True, dtype={"recipe": "fp8", "grad_out": grad})


def _record_rounding(monkeypatch):
    """Collect (fmt, stochastic_rounding) for every operand the linear quantizes."""
    seen = []
    original = quant_linear.quantize_operand

    def record(
        x,
        contract_dim,
        fmt,
        scaling,
        offs=None,
        ragged_dim=None,
        stochastic_rounding=False,
    ):
        seen.append((fmt, stochastic_rounding))
        return original(
            x, contract_dim, fmt, scaling, offs, ragged_dim, stochastic_rounding
        )

    monkeypatch.setattr(quant_linear, "quantize_operand", record)
    return seen


def test_rounding_is_scoped_to_the_grad_out_tensor(monkeypatch):
    # The recipe gives weight/act e4m3 and grad_out e5m2, so the format identifies
    # which tensor each quantize call is for.
    seen = _record_rounding(monkeypatch)
    cfg = TrainingConfig(
        mixed_precision="no",
        quantization={
            "enabled": True,
            "dtype": {"recipe": "fp8"},
            "rounding": {"grad_out": "SR"},
        },
    ).quantization[0]
    linear = QuantizedLinear.from_module(nn.Linear(4, 3, bias=False), cfg)
    linear(torch.randn(2, 4, requires_grad=True)).sum().backward()

    # fwd quantizes act and weight; dgrad grad_out and weight; wgrad grad_out and act
    assert [fmt for fmt, _ in seen] == [
        "fp8_e4m3",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e4m3",
        "fp8_e5m2",
        "fp8_e4m3",
    ]
    assert all(sr == (fmt == "fp8_e5m2") for fmt, sr in seen)


def test_rounding_is_supported_on_the_forward_tensors(monkeypatch):
    seen = _record_rounding(monkeypatch)
    cfg = TrainingConfig(
        mixed_precision="no",
        quantization={
            "enabled": True,
            "dtype": {"weight": "int4"},
            "rounding": {"weight": "SR"},
        },
    ).quantization[0]
    linear = QuantizedLinear.from_module(nn.Linear(4, 3, bias=False), cfg)
    linear(torch.randn(2, 4))
    # act/grad_out stay at the compute dtype, so the weight is the only operand
    assert seen == [("int4", True)]


def test_quantized_linear_only_quantizes_during_training(monkeypatch):
    calls = []
    original = quant_linear.quantized_mm

    def record(
        a,
        b,
        a_fmt,
        b_fmt,
        out_dtype,
        scaling_cfg,
        bias=None,
        a_stochastic_rounding=False,
        b_stochastic_rounding=False,
        a_stats=None,
        b_stats=None,
    ):
        calls.append(None)
        return original(
            a,
            b,
            a_fmt,
            b_fmt,
            out_dtype,
            scaling_cfg,
            bias=bias,
            a_stochastic_rounding=a_stochastic_rounding,
            b_stochastic_rounding=b_stochastic_rounding,
            a_stats=a_stats,
            b_stats=b_stats,
        )

    monkeypatch.setattr(quant_linear, "quantized_mm", record)
    linear = QuantizedLinear.from_module(
        nn.Linear(4, 3, device="cpu"),
        TrainingConfig(
            quantization={"enabled": True, "dtype": {"weight": "int4"}}
        ).quantization[0],
    )
    x = torch.randn(2, 4, device="cpu")

    linear.train()
    linear(x)
    linear.eval()
    actual = linear(x)

    assert len(calls) == 1
    torch.testing.assert_close(
        actual, nn.functional.linear(x, linear.weight, linear.bias)
    )


def test_quantized_linear_preserves_source_eval_mode():
    source = nn.Linear(4, 3, device="cpu")
    source.eval()
    linear = QuantizedLinear.from_module(
        source,
        TrainingConfig(
            quantization={"enabled": True, "dtype": {"weight": "int4"}}
        ).quantization[0],
    )

    assert not linear.training


@fp8_only
def test_from_linear_copies_weights():
    lin = nn.Linear(64, 32, bias=True).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg())
    assert torch.equal(q.weight, lin.weight)
    assert torch.equal(q.bias, lin.bias)
    assert q.weight.requires_grad


@fp8_only
def test_forward_matches_roundtrip_oracle():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    out = q(x)
    ref = (
        _roundtrip(x, -1, _E4M3, _scaling("tensorwise"))
        @ _roundtrip(lin.weight, -1, _E4M3, _scaling("tensorwise")).t()
    )
    assert out.shape == (64, 96) and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05


@fp8_only
def test_backward_finite_grads_and_dtypes():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=True).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert q.weight.grad.dtype == q.weight.dtype
    assert q.bias.grad.dtype == q.bias.dtype
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.dtype == x.dtype


@fp8_only
def test_rowwise_forward_and_backward():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    cfg = QuantizationConfig(
        enabled=True, dtype={"recipe": "fp8"}, scaling={"granularity": "rowwise"}
    )
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = q(x)
    # rowwise oracle: per-row act, per-column weight (reduce over K)
    ref = (
        _roundtrip(x, -1, _E4M3, _scaling("rowwise"))
        @ _roundtrip(lin.weight, -1, _E4M3, _scaling("rowwise")).t()
    )
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05
    out.square().mean().backward()
    assert torch.isfinite(q.weight.grad).all() and torch.isfinite(x.grad).all()


@fp8_only
def test_high_precision_wgrad_changes_the_weight_gradient():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    def grad_weight(hp):
        dtype = {"grad_out": {"wgrad": "bf16"}} if hp else {}
        cfg = QuantizationConfig(enabled=True, dtype={"recipe": "fp8", **dtype})
        q = QuantizedLinear.from_module(lin, cfg)
        q(x.clone()).square().mean().backward()
        return q.weight.grad

    g_hp = grad_weight(True)
    g_fp8 = grad_weight(False)
    assert torch.isfinite(g_hp).all() and torch.isfinite(g_fp8).all()
    # hp wgrad skips fp8 rounding of grad/act, so it differs from the fp8 wgrad
    rel = (g_hp.float() - g_fp8.float()).norm() / g_fp8.float().norm()
    assert rel > 1e-3


@fp8_only
def test_preserves_batch_dims():
    lin = nn.Linear(64, 48, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg())
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    assert q(x).shape == (2, 16, 48)


@fp8_only
@pytest.mark.parametrize("grad", ["bf16", "fp8_e5m2"])
def test_grad_fmt_both_train(grad):
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg(grad=grad))
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert torch.isfinite(q.weight.grad).all()


@fp8_only
def test_runs_under_bf16_autocast():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.float32)  # fp32 master
    q = QuantizedLinear.from_module(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = q(x)
    assert out.dtype == torch.bfloat16  # follows autocast
    out.square().mean().backward()
    assert q.weight.grad.dtype == torch.float32  # grad matches fp32 master
    assert torch.isfinite(q.weight.grad).all()


# --- int8-family quantized_mm dispatch (migrated from the old test_int8.py) ---


int_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="int gemm kernel needs CUDA"
)


def test_gemm_passthrough_is_plain_matmul():
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    out = quantized_mm(a, b, "bf16", "bf16", torch.float32, {})
    assert torch.allclose(out, a @ b, atol=1e-4)


def test_gemm_records_nothing_without_stats():
    """Training passes no accumulators, so the fold is skipped outright."""
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    set_quantization_monitoring_status(True)
    try:
        out = quantized_mm(a, b, "int8", "int8", torch.float32, _scaling("tensorwise"))
    finally:
        set_quantization_monitoring_status(False)
    assert torch.isfinite(out).all()


def test_gemm_records_only_the_operands_whose_format_is_quantized():
    """int x bf16: only `a` is a quantized format, so only its accumulator moves."""
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    a_stats = QuantizationStats("act/x", 1, a.device)
    b_stats = QuantizationStats("weight/x", 1, b.device)
    set_quantization_monitoring_status(True)
    try:
        out = quantized_mm(
            a,
            b,
            "int8",
            "bf16",
            torch.float32,
            _scaling("tensorwise"),
            a_stats=a_stats,
            b_stats=b_stats,
        )
    finally:
        set_quantization_monitoring_status(False)
    assert torch.isfinite(out).all()
    assert a_stats.numel.item() == a.numel()
    assert b_stats.numel.item() == 0


def test_gemm_applies_bias_on_the_fallback_path():
    """bf16 x bf16 never reaches the kernel, so the unfused branch must add it too."""
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    bias = torch.randn(40)
    out = quantized_mm(a, b, "bf16", "bf16", torch.float32, {}, bias=bias)
    assert torch.allclose(out, a @ b + bias, atol=1e-4)


@fp8_only
def test_gemm_applies_bias_in_the_fused_kernel():
    """fp8 x fp8 on CUDA takes fp8_scaled_mm, so the bias rides its epilogue."""
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(128, 96, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(96, device="cuda", dtype=torch.bfloat16)
    cfg = _scaling("rowwise")
    with_bias = quantized_mm(
        a, b, "fp8_e4m3", "fp8_e4m3", torch.bfloat16, cfg, bias=bias
    )
    without = quantized_mm(a, b, "fp8_e4m3", "fp8_e4m3", torch.bfloat16, cfg)
    torch.testing.assert_close(
        with_bias.float(), without.float() + bias.float(), rtol=2e-2, atol=2e-2
    )


@fp8_only
def test_quantized_linear_adds_bias_exactly_once():
    """Guards the fused wiring: a leftover post-GEMM add would double the bias."""
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=True).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    with_bias = q(x)
    with torch.no_grad():
        bias = q.bias.clone()
        q.bias.zero_()
    torch.testing.assert_close(
        (with_bias - q(x)).float(),
        bias.float().expand(64, 96),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int8s_gemm_mixed_family_uses_fake_quant(fmt):
    a, b = torch.randn(20, 32), torch.randn(32, 40)  # int x bf16 -> fallback
    out = quantized_mm(a, b, fmt, "bf16", torch.float32, _scaling("tensorwise"))
    assert out.shape == (20, 40) and torch.isfinite(out).all()


@int_gpu
@pytest.mark.parametrize("fmt", sorted(_INT8_FORMATS))
def test_int8s_gemm_dispatches_to_kernel(fmt):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 96, device="cuda")
    out = quantized_mm(a, b, fmt, fmt, torch.float32, _scaling("rowwise"))
    ref = _roundtrip(a, -1, fmt, _scaling("rowwise")) @ _roundtrip(
        b, -2, fmt, _scaling("rowwise")
    )
    assert (out - ref).norm() / ref.norm() < 0.02


# --- mxfp8 (blockwise) on all three GEMMs ---


@mxfp8_only
def test_quant_linear_mxfp8_forward_matches_oracle():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(
        lin, QuantizationConfig(enabled=True, dtype={"recipe": "mxfp8"})
    )
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    out = q(x)
    # forward blocks x along in-features (-1) and w along in-features (-1).
    ref = (
        _roundtrip(x, -1, _E4M3, _MXFP8) @ _roundtrip(lin.weight, -1, _E4M3, _MXFP8).t()
    )
    assert out.shape == (256, 256) and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05


@mxfp8_only
def test_quant_linear_mxfp8_backward_finite():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=True).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(
        lin, QuantizationConfig(enabled=True, dtype={"recipe": "mxfp8"})
    )
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert q.weight.grad.dtype == q.weight.dtype
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert q.bias.grad is not None and torch.isfinite(q.bias.grad).all()


@mxfp8_only
def test_mxfp8_wgrad_unaligned_tokens():
    # M=250 tokens (not a multiple of 32) exercises the wgrad contraction pad.
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(
        lin, QuantizationConfig(enabled=True, dtype={"recipe": "mxfp8"})
    )
    x = torch.randn(250, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert q.weight.grad.shape == (256, 256)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mxfp8_fake_quant_fallback_on_cpu():
    # A one-sided mxfp8 rule (weight quantized, act passthrough) exercises the
    # mxfp8 fake-quant path inside quantized_mm without touching the real GEMM — on CPU.
    torch.manual_seed(0)
    cfg = QuantizationConfig(
        enabled=True,
        dtype={"weight": "fp8_e4m3", "act": "bf16", "grad_out": "bf16"},
        scaling={"recipe": "mxfp8"},
    )
    lin = nn.Linear(64, 64, bias=False).to(torch.float32)
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(32, 64, requires_grad=True)
    out = q(x)
    assert out.shape == (32, 64) and torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(q.weight.grad).all() and torch.isfinite(x.grad).all()


# --- end-to-end: a full model trains one step through the converter ---


def _model_cfg():
    return TrainConfig(
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


@fp8_only
def test_quantized_model_trains_one_step():
    cfg = _model_cfg()
    model = build_model(cfg).cuda().to(torch.bfloat16)
    apply_quantization(model, cfg)
    assert any(isinstance(m, QuantizedLinear) for m in model.modules())

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()
    assert torch.isfinite(loss).item()
    qw = next(m.weight for m in model.modules() if isinstance(m, QuantizedLinear))
    assert qw.grad is not None and torch.isfinite(qw.grad).all()


@fp8_only
def test_compiled_quant_linear_blockwise_fwd_bwd():
    torch.manual_seed(0)
    lin = nn.Linear(256, 128, bias=False).cuda().to(torch.bfloat16)
    cfg = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        scaling={"granularity": "blockwise", "block_shape": (1, 128)},
    )
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(q, fullgraph=True)
    out = compiled(x)
    out.square().mean().backward()
    assert out.shape == (64, 128) and torch.isfinite(out).all()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@fp8_only
def test_quantized_linear_forward_backward_with_square_tiles():
    """End to end: the module runs and produces finite grads at a 2D block shape."""
    cfg = QuantizationConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        scaling={
            "granularity": "blockwise",
            "block_shape": (32, 32),
            "scale_dtype": "fp32",
        },
    )
    layer = QuantizedLinear.from_module(nn.Linear(128, 64), cfg)
    x = torch.randn(16, 128, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert y.shape == (16, 64)
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


@fp8_only
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.float8_e8m0fnu])
@pytest.mark.parametrize("tile", [32, 64])
def test_quantized_mm_2d_matches_fp32_dequant_reference(tile, scale_dtype):
    """The real kernel path against an fp32 oracle, both scale dtypes.

    tile=64 with `torch.float8_e8m0fnu` also exercises rep_k == 2, composing the 2D
    outer-axis expansion with the mxfp8 K-replication fast path.
    """
    scaling = {
        "granularity": "blockwise",
        "block_shape": (tile, tile),
        "scale_dtype": scale_dtype,
    }
    a = torch.randn(256, 512, device="cuda")
    b = torch.randn(512, 128, device="cuda")
    out = quantized_mm(a, b, _E4M3, _E4M3, torch.float32, scaling)
    ref = _roundtrip(a, -1, _E4M3, scaling) @ _roundtrip(b, -2, _E4M3, scaling)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
