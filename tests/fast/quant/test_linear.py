import pytest
import torch
import torch.nn as nn

from src.model import build_model
from src.quant.linear import QuantLinear
from src.quant.convert import apply_quantization
from src.quant.fp8 import fake_quantize_fp8
from src.utils.config import ModelConfig, QuantConfig, TrainConfig, TrainingConfig

FP8_FORWARD_DTYPE = torch.float8_e4m3fn


def _fp8_capable():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() >= (8, 9)


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="fp8 needs SM >= 8.9")


def _cfg(grad="bf16"):
    # recipe fills weight/act=fp8_e4m3; both backward grads overridden explicitly
    return QuantConfig(
        enabled=True,
        dtype_recipe="fp8",
        dtype={"input_grad": grad, "weight_grad": grad},
    )


@fp8_only
def test_from_linear_copies_weights():
    lin = nn.Linear(64, 32, bias=True).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, _cfg())
    assert torch.equal(q.weight, lin.weight)
    assert torch.equal(q.bias, lin.bias)
    assert q.weight.requires_grad


@fp8_only
def test_forward_matches_fake_quant_oracle():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    out = q(x)
    ref = (
        fake_quantize_fp8(x, FP8_FORWARD_DTYPE)
        @ fake_quantize_fp8(lin.weight, FP8_FORWARD_DTYPE).t()
    )
    assert out.shape == (64, 96) and out.dtype == torch.bfloat16
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05


@fp8_only
def test_backward_finite_grads_and_dtypes():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=True).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, _cfg())
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
        enabled=True, dtype_recipe="fp8", scaling={"granularity": "rowwise"}
    )
    q = QuantLinear.from_linear(lin, cfg)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = q(x)
    # rowwise oracle: per-row act, per-column weight (reduce over K)
    ref = (
        fake_quantize_fp8(x, FP8_FORWARD_DTYPE, dim=-1)
        @ fake_quantize_fp8(lin.weight, FP8_FORWARD_DTYPE, dim=-1).t()
    )
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    assert rel < 0.05
    out.square().mean().backward()
    assert torch.isfinite(q.weight.grad).all() and torch.isfinite(x.grad).all()


@fp8_only
def test_high_precision_weight_grad_changes_wgrad():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

    def weight_grad(hp):
        dtype = {"weight_grad": "bf16"} if hp else {}
        cfg = QuantConfig(enabled=True, dtype_recipe="fp8", dtype=dtype)
        q = QuantLinear.from_linear(lin, cfg)
        q(x.clone()).square().mean().backward()
        return q.weight.grad

    g_hp = weight_grad(True)
    g_fp8 = weight_grad(False)
    assert torch.isfinite(g_hp).all() and torch.isfinite(g_fp8).all()
    # hp wgrad skips fp8 rounding of grad/act, so it differs from the fp8 wgrad
    rel = (g_hp.float() - g_fp8.float()).norm() / g_fp8.float().norm()
    assert rel > 1e-3


@fp8_only
def test_preserves_batch_dims():
    lin = nn.Linear(64, 48, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, _cfg())
    x = torch.randn(2, 16, 64, device="cuda", dtype=torch.bfloat16)
    assert q(x).shape == (2, 16, 48)


@fp8_only
@pytest.mark.parametrize("grad", ["bf16", "fp8_e5m2"])
def test_grad_fmt_both_train(grad):
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.bfloat16)
    q = QuantLinear.from_linear(lin, _cfg(grad=grad))
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q(x).square().mean().backward()
    assert torch.isfinite(q.weight.grad).all()


@fp8_only
def test_runs_under_bf16_autocast():
    torch.manual_seed(0)
    lin = nn.Linear(128, 96, bias=False).cuda().to(torch.float32)  # fp32 master
    q = QuantLinear.from_linear(lin, _cfg())
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32, requires_grad=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = q(x)
    assert out.dtype == torch.bfloat16  # follows autocast
    out.square().mean().backward()
    assert q.weight.grad.dtype == torch.float32  # grad matches fp32 master
    assert torch.isfinite(q.weight.grad).all()


# --- end-to-end: a full model trains one step through the converter ---


def _model_cfg():
    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            d_model=64,
            n_layers=2,
            vocab_size=128,
            attn_cls="gqa",
            attn_kwargs={"n_heads": 4, "n_kv_heads": 2},
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
            quant={"enabled": True, "dtype_recipe": "fp8"},
        ),
    )


@fp8_only
def test_quantized_model_trains_one_step():
    cfg = _model_cfg()
    model = build_model(cfg).cuda().to(torch.bfloat16)
    apply_quantization(model, cfg)
    assert any(isinstance(m, QuantLinear) for m in model.modules())

    ids = torch.randint(0, 128, (2, 64), device="cuda")
    position_ids = torch.arange(64, device="cuda").unsqueeze(0).expand(2, 64)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits, _ = model(ids, position_ids)
        loss = logits.float().log_softmax(-1).mean().neg()
    loss.backward()
    assert torch.isfinite(loss).item()
    qw = next(m.weight for m in model.modules() if isinstance(m, QuantLinear))
    assert qw.grad is not None and torch.isfinite(qw.grad).all()
