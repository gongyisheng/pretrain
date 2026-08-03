import pytest
import torch
import torch.nn as nn

from src.model import build_model
from src.quant.constants import _INT8_FORMATS
from src.quant.linear import QuantizedLinear, quantized_gemm
from src.quant.convert import apply_quantization
from src.quant.quantize import dequantize_operand, quantize_operand
from src.utils.config import ModelConfig, QuantConfig, TrainConfig, TrainingConfig

_MXFP8_KW = dict(fmt="fp8_e4m3", scale_dtype="fp8_e8m0")


def _roundtrip(x, contract_dim, granularity, block_size, fmt, *, scale_dtype=None):
    """dequant(quant(x)) — the operand the quantized GEMM actually multiplies."""
    xq, scale = quantize_operand(
        x, contract_dim, granularity, block_size, fmt, scale_dtype=scale_dtype
    )
    return dequantize_operand(xq, scale, contract_dim, granularity, block_size)


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")
# mxfp8 is a scaling scheme (power-of-two blockwise), not Blackwell-only anymore;
# it now runs through the generic scaled_gemm kernel like any other CUDA fp8/int path.
mxfp8_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="mxfp8 needs CUDA"
)


def _cfg(grad="bf16"):
    # recipe fills weight/act=fp8_e4m3; both backward grads overridden explicitly
    return QuantConfig(
        enabled=True,
        dtype={"recipe": "fp8", "grad_input": grad, "grad_weight": grad},
    )


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
        _roundtrip(x, -1, "tensorwise", 0, fmt="fp8_e4m3")
        @ _roundtrip(lin.weight, -1, "tensorwise", 0, fmt="fp8_e4m3").t()
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
    cfg = QuantConfig(
        enabled=True, dtype={"recipe": "fp8"}, scaling={"granularity": "rowwise"}
    )
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = q(x)
    # rowwise oracle: per-row act, per-column weight (reduce over K)
    ref = (
        _roundtrip(x, -1, "rowwise", 0, fmt="fp8_e4m3")
        @ _roundtrip(lin.weight, -1, "rowwise", 0, fmt="fp8_e4m3").t()
    )
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05
    out.square().mean().backward()
    assert torch.isfinite(q.weight.grad).all() and torch.isfinite(x.grad).all()


@fp8_only
def test_high_precision_grad_weight_changes_wgrad():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    def grad_weight(hp):
        dtype = {"grad_weight": "bf16"} if hp else {}
        cfg = QuantConfig(enabled=True, dtype={"recipe": "fp8", **dtype})
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


# --- int8-family quantized_gemm dispatch (migrated from the old test_int8.py) ---


int_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="int gemm kernel needs CUDA"
)


def test_gemm_passthrough_is_plain_matmul():
    a, b = torch.randn(20, 32), torch.randn(32, 40)
    out, a_snap, b_snap = quantized_gemm(a, b, "bf16", "bf16", torch.float32, {})
    assert torch.allclose(out, a @ b, atol=1e-4)
    assert a_snap is None and b_snap is None  # nothing was quantized


@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int8s_gemm_mixed_family_uses_fake_quant(fmt):
    a, b = torch.randn(20, 32), torch.randn(32, 40)  # int x bf16 -> fallback
    out, a_snap, b_snap = quantized_gemm(a, b, fmt, "bf16", torch.float32, {})
    assert out.shape == (20, 40) and torch.isfinite(out).all()
    assert a_snap is not None and b_snap is None  # only a is a quantized format


@int_gpu
@pytest.mark.parametrize("fmt", _INT8_FORMATS)
def test_int8s_gemm_dispatches_to_kernel(fmt):
    torch.manual_seed(0)
    a = torch.randn(64, 128, device="cuda")
    b = torch.randn(128, 96, device="cuda")
    out, _, _ = quantized_gemm(
        a, b, fmt, fmt, torch.float32, {"granularity": "rowwise"}
    )
    ref = _roundtrip(a, -1, "rowwise", 0, fmt) @ _roundtrip(b, 0, "rowwise", 0, fmt)
    assert (out - ref).norm() / ref.norm() < 0.02


# --- mxfp8 (blockwise) on all three GEMMs ---


@mxfp8_only
def test_quant_linear_mxfp8_forward_matches_oracle():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=False).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(
        lin, QuantConfig(enabled=True, dtype={"recipe": "mxfp8"})
    )
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    out = q(x)
    # forward blocks x along in-features (-1) and w along in-features (-1).
    ref = (
        _roundtrip(x, -1, "blockwise", 32, **_MXFP8_KW)
        @ _roundtrip(lin.weight, -1, "blockwise", 32, **_MXFP8_KW).t()
    )
    assert out.shape == (256, 256) and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05


@mxfp8_only
def test_quant_linear_mxfp8_backward_finite():
    torch.manual_seed(0)
    lin = nn.Linear(256, 256, bias=True).cuda().to(torch.bfloat16)
    q = QuantizedLinear.from_module(
        lin, QuantConfig(enabled=True, dtype={"recipe": "mxfp8"})
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
        lin, QuantConfig(enabled=True, dtype={"recipe": "mxfp8"})
    )
    x = torch.randn(250, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert q.weight.grad.shape == (256, 256)
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_mxfp8_fake_quant_fallback_on_cpu():
    # A one-sided mxfp8 rule (weight quantized, act passthrough) exercises the
    # mxfp8 fake-quant path inside quantized_gemm without touching the real GEMM — on CPU.
    torch.manual_seed(0)
    cfg = QuantConfig(
        enabled=True,
        dtype={
            "weight": "fp8_e4m3",
            "act": "bf16",
            "grad_input": "bf16",
            "grad_weight": "bf16",
        },
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
            quant={"enabled": True, "dtype": {"recipe": "fp8"}},
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
    cfg = QuantConfig(
        enabled=True,
        dtype={"recipe": "fp8"},
        scaling={"granularity": "blockwise", "block_size": 128},
    )
    q = QuantizedLinear.from_module(lin, cfg)
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(q, fullgraph=True)
    out = compiled(x)
    out.square().mean().backward()
    assert out.shape == (64, 128) and torch.isfinite(out).all()
    assert q.weight.grad is not None and torch.isfinite(q.weight.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
