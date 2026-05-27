"""Tests for MetricsTracker."""

import math

import pytest
import torch

from src.model.registry import build_model
from src.training.metrics import MetricsTracker, compute_flops_per_token
from src.training.optimizer import LionOptimizer
from src.utils.config import LoggingConfig, ModelConfig, TrainConfig, TrainingConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported

# ---------------------------------------------------------------------------
# Small model configs matching real architectures. Factories rather than
# constants so they can pick up the parametrized attn_implementation.
# ---------------------------------------------------------------------------


def _gpt2_cfg(impl):
    return ModelConfig(
        arch="gpt2",
        n_layers=2,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        attn_implementation=impl,
    )


def _qwen3_cfg(impl):
    return ModelConfig(
        arch="qwen3",
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_model=64,
        vocab_size=256,
        qk_norm=True,
        attn_implementation=impl,
    )


def _qwen3_moe_cfg(impl):
    return ModelConfig(
        arch="qwen3_moe",
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_model=64,
        vocab_size=256,
        qk_norm=True,
        moe_n_experts=4,
        moe_n_experts_per_token=2,
        moe_intermediate_size=128,
        attn_implementation=impl,
    )


def _gpt2_attn_res_cfg(impl):
    return ModelConfig(
        arch="gpt2",
        n_layers=4,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        residual_cls="attn_res",
        residual_kwargs={"seal_block_size": 2},
        attn_implementation=impl,
    )


def _qwen3_attn_res_cfg(impl):
    return ModelConfig(
        arch="qwen3",
        n_layers=4,
        n_heads=2,
        n_kv_heads=1,
        d_model=64,
        vocab_size=256,
        qk_norm=True,
        residual_cls="attn_res",
        residual_kwargs={"seal_block_size": 2},
        attn_implementation=impl,
    )


_CFG_FACTORIES = {
    "gpt2": _gpt2_cfg,
    "qwen3": _qwen3_cfg,
    "qwen3_moe": _qwen3_moe_cfg,
    "gpt2_attn_res": _gpt2_attn_res_cfg,
    "qwen3_attn_res": _qwen3_attn_res_cfg,
}


class _FakeTrainConfig:
    """Minimal object accepted by build_model (needs .model and .max_seq_len)."""

    def __init__(self, model_cfg: ModelConfig):
        self.model = model_cfg
        self.max_seq_len = 128


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _populate_grads(model, vocab_size: int, impl: str):
    """Run a forward/backward pass to populate .grad on all parameters."""
    idx = torch.randint(0, vocab_size, (1, 16))
    position_ids = torch.arange(16).unsqueeze(0)
    attn_mask, _ = make_attn_mask("causal", impl, position_ids, torch.float32)
    out = model(idx, position_ids=position_ids, attn_mask=attn_mask)
    # MoE models return (logits, aux_loss)
    logits = out[0] if isinstance(out, tuple) else out
    logits.sum().backward()


def _expected_keys(model: torch.nn.Module) -> set[str]:
    """Derive expected grad_norm keys from model's named_parameters."""
    keys = set()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        name = name.removeprefix("_orig_mod.")
        keys.add(f"grad_norm/{name}")
    return keys


def _assert_keys_match(result: dict, model: torch.nn.Module):
    """Check that result keys exactly match expected per-module keys."""
    expected = _expected_keys(model)
    actual = set(result.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"Missing keys: {missing}"
    assert not extra, f"Unexpected keys: {extra}"
    # All values should be non-negative finite floats
    for k, v in result.items():
        assert v >= 0, f"{k} has negative grad norm: {v}"
        assert torch.isfinite(torch.tensor(v)), f"{k} is not finite: {v}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_tracker():
    """Build a MetricsTracker with a minimal non-MoE TrainConfig."""
    cfg = TrainConfig()
    cfg.model = ModelConfig(
        arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256
    )
    return MetricsTracker(cfg, device="cpu")


# ==================== optimizer step diagnostics ====================


def test_logging_config_default_log_optimizer_step_norms_is_true():
    """Optimizer step-norm logging is opt-in (extra memory + compute)."""
    assert LoggingConfig().log_optimizer_step_norms is True


def test_snapshot_and_param_step_norm_match_l2_of_diff():
    """compute_param_step_norm = sqrt(sum((p_after - p_before)^2))."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    snapshot = MetricsTracker.snapshot_params(model)
    # Add a known delta to every parameter.
    delta = 0.1
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, delta))
    n_params = sum(p.numel() for p in model.parameters())
    expected = delta * math.sqrt(n_params)
    actual = MetricsTracker.compute_param_step_norm(model, snapshot)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_param_step_norm_zero_when_unchanged():
    """No update → step norm exactly 0."""
    model = torch.nn.Linear(4, 2)
    snapshot = MetricsTracker.snapshot_params(model)
    assert MetricsTracker.compute_param_step_norm(model, snapshot) == 0.0


def test_momentum_norm_reads_exp_avg_buffers():
    """compute_momentum_norm = sqrt(sum_p ||state[p]['exp_avg']||^2)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    # Lion populates state on first step; do one no-op step to seed exp_avg=0 buffers.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    # Now set the buffers to a known value.
    fill = 0.5
    for p in model.parameters():
        opt.state[p]["exp_avg"].fill_(fill)
    n_params = sum(p.numel() for p in model.parameters())
    expected = fill * math.sqrt(n_params)
    actual = MetricsTracker.compute_momentum_norm(opt)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_momentum_norm_none_when_no_state():
    """Before any optimizer.step(), no state buffers exist → None (no W&B series)."""
    model = torch.nn.Linear(4, 2)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    assert MetricsTracker.compute_momentum_norm(opt) is None


def test_build_train_log_dict_omits_step_norms_when_none():
    """When param_step_norm/momentum_norm aren't passed, log_dict skips those keys."""
    tracker = _make_tracker()
    model = torch.nn.Linear(4, 2)
    scaler = torch.amp.GradScaler(enabled=False)
    d = tracker.build_train_log_dict(
        loss=2.0,
        total_tokens=1000,
        lr=1e-4,
        grad_norm=0.5,
        tokens_per_sec=1000.0,
        elapsed=0.1,
        model=model,
        scaler=scaler,
    )
    assert "optim/param_step_norm" not in d
    assert "optim/momentum_norm" not in d


def test_build_train_log_dict_includes_step_norms_when_passed():
    """When values are passed, they appear under optim/ keys."""
    tracker = _make_tracker()
    model = torch.nn.Linear(4, 2)
    scaler = torch.amp.GradScaler(enabled=False)
    d = tracker.build_train_log_dict(
        loss=2.0,
        total_tokens=1000,
        lr=1e-4,
        grad_norm=0.5,
        tokens_per_sec=1000.0,
        elapsed=0.1,
        model=model,
        scaler=scaler,
        param_step_norm=0.03,
        momentum_norm=0.7,
        variance_norm=0.02,
    )
    assert d["optim/param_step_norm"] == 0.03
    assert d["optim/momentum_norm"] == 0.7
    assert d["optim/variance_norm"] == 0.02


def test_variance_norm_reads_exp_avg_sq_buffers():
    """compute_variance_norm = sqrt(sum_p ||state[p]['exp_avg_sq']||^2)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # AdamW seeds exp_avg + exp_avg_sq on first step.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    fill = 0.25
    for p in model.parameters():
        opt.state[p]["exp_avg_sq"].fill_(fill)
    n_params = sum(p.numel() for p in model.parameters())
    expected = fill * math.sqrt(n_params)
    actual = MetricsTracker.compute_variance_norm(opt)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_variance_norm_none_for_lion():
    """Lion stores only exp_avg (no second moment) → None (no W&B series)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert "exp_avg" in opt.state[next(iter(opt.state))]  # sanity
    assert MetricsTracker.compute_variance_norm(opt) is None


def test_variance_norm_none_before_first_step():
    """No optimizer state yet → None (no W&B series)."""
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert MetricsTracker.compute_variance_norm(opt) is None


# ==================== existing tests ====================


def test_build_eval_log_dict_no_bpb_without_tokens_per_byte():
    tracker = _make_tracker()
    d = tracker.build_eval_log_dict(avg_loss=2.0)
    assert "val/loss" in d and d["val/loss"] == 2.0
    assert "val/perplexity" in d
    assert "val/bpb" not in d


def test_build_eval_log_dict_emits_bpb():
    tracker = _make_tracker()
    avg_loss = 2.5
    tpb = 0.4  # tokens per byte
    d = tracker.build_eval_log_dict(avg_loss=avg_loss, tokens_per_byte=tpb)
    expected = avg_loss * tpb / math.log(2)
    assert d["val/bpb"] == pytest.approx(expected)


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_layer_grad_norms_plain_model(arch_id, impl, device):
    """Per-module grad norms should have one key per module on a plain model."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    _populate_grads(model, model_cfg.vocab_size, impl)
    result = MetricsTracker.compute_layer_grad_norms(model)
    _assert_keys_match(result, model)


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_layer_grad_norms_compiled_model(arch_id, impl, device):
    """Per-module grad norms must work after torch.compile (which prepends _orig_mod.)."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    compiled = torch.compile(model, backend="eager")
    _populate_grads(compiled, model_cfg.vocab_size, impl)
    result = MetricsTracker.compute_layer_grad_norms(compiled)
    _assert_keys_match(result, compiled)


def test_eval_log_dict_pretrain_includes_perplexity_and_bpb():
    cfg = TrainConfig(model=_qwen3_cfg("flex_attention"))
    cfg.task = "pretrain"
    tracker = MetricsTracker(cfg, device="cpu")
    d = tracker.build_eval_log_dict(avg_loss=0.5, tokens_per_byte=0.25)
    assert "val/loss" in d
    assert "val/perplexity" in d
    assert "val/bpb" in d
    assert "val/val_acc" not in d


def test_eval_log_dict_sft_includes_acc_only():
    cfg = TrainConfig(model=_qwen3_cfg("flex_attention"))
    cfg.task = "sft"
    tracker = MetricsTracker(cfg, device="cpu")
    d = tracker.build_eval_log_dict(avg_loss=0.5, avg_acc=0.87)
    assert "val/loss" in d
    assert "val/val_acc" in d
    assert d["val/val_acc"] == pytest.approx(0.87)
    assert "val/perplexity" not in d
    assert "val/bpb" not in d


def test_eval_log_dict_sft_with_train_acc():
    cfg = TrainConfig(model=_qwen3_cfg("flex_attention"))
    cfg.task = "sft"
    tracker = MetricsTracker(cfg, device="cpu")
    d = tracker.build_eval_log_dict(avg_loss=0.5, avg_acc=0.87, train_avg_acc=0.99)
    assert d["val/val_acc"] == pytest.approx(0.87)
    assert d["val/train_acc"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# compute_flops_per_token
# ---------------------------------------------------------------------------


def test_compute_flops_per_token_dense_gpt2_mha_ungated():
    """GPT-2 style: MHA, ungated FFN with biases. Hand-computed expected values."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        arch="gpt2",
        n_layers=2,
        n_heads=2,
        d_model=64,
        vocab_size=256,
        intermediate_size=256,
        mlp_activation="gelu",
        mlp_gated=False,
        attn_bias=True,
        mlp_bias=True,
    )
    f = compute_flops_per_token(cfg)
    # head_dim=32, n_kv_heads=n_heads=2 (MHA), L=2, T=128
    # qkv_proj per layer: 2*64*(2+2*2)*32 + (2+2*2)*32 = 24576+192 = 24768
    assert f["qkv_proj"] == 2 * 24768
    # o_proj per layer: 2*64*64 + 64 = 8256
    assert f["o_proj"] == 2 * 8256
    # attn_matmul per layer (full T, PaLM): 4 * n_heads*head_dim*seq_len = 4*2*32*128 = 32768
    assert f["attn_matmul"] == 2 * 32768
    # ffn ungated per layer: 4*64*256 + (256+64) = 65856
    assert f["ffn"] == 2 * 65856
    # lm_head: 2*64*256 = 32768 (no bias by default)
    assert f["lm_head"] == 32768
    # norm: per-layer (2 RMSNorms) = 2*3*64 = 384; final = 3*64 = 192
    # qk_norm=False (default), so no extra term. Total = 2*384 + 192 = 960
    assert f["norm"] == 2 * 384 + 192
    expected_fwd = 2 * 24768 + 2 * 8256 + 2 * 32768 + 2 * 65856 + 960 + 32768
    assert f["fwd_total"] == expected_fwd
    # default activation_checkpointing=False -> 3x
    assert f["total"] == 3 * expected_fwd


def test_compute_flops_per_token_gqa_gated_qwen3():
    """GQA halves KV proj vs MHA; gated FFN is 6*d*d_ff (vs 4 ungated)."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        arch="qwen3",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # GQA: 2x reduction in KV proj cost
        d_model=64,
        vocab_size=256,
        intermediate_size=256,
        mlp_activation="silu",
        mlp_gated=True,
        attn_bias=False,
        mlp_bias=False,
    )
    f = compute_flops_per_token(cfg)
    # head_dim=16, L=2, T=128
    # qkv per layer: 2*64*(4+2*2)*16 = 16384 (no bias)
    assert f["qkv_proj"] == 2 * 16384
    # o per layer: 2*64*64 = 8192
    assert f["o_proj"] == 2 * 8192
    # attn matmul (full T): 4 * 4*16*128 = 32768 per layer
    assert f["attn_matmul"] == 2 * 32768
    # gated ffn: 6*64*256 = 98304 per layer
    assert f["ffn"] == 2 * 98304
    assert f["lm_head"] == 2 * 64 * 256
    # GQA invariant: with n_kv_heads = n_heads/2, KV proj = Q proj
    # qkv = Q + K + V; Q = 2*d*n_heads*head_dim; K = V = 2*d*n_kv_heads*head_dim
    # For n_kv_heads = n_heads/2, total = 2*d*n_heads*head_dim * 2 = 2*Q
    # vs MHA which would be 2*d*n_heads*head_dim * 3 = 3*Q
    q_only = 2 * 64 * 4 * 16
    assert f["qkv_proj"] // cfg.model.n_layers == 2 * q_only


def test_compute_flops_per_token_moe_uses_k_active():
    """MoE ffn cost = router + k active experts, NOT all N experts."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        arch="qwen3_moe",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=64,
        vocab_size=256,
        intermediate_size=256,
        moe_n_experts=8,
        moe_n_experts_per_token=2,
        moe_intermediate_size=128,
        mlp_gated=True,
        mlp_bias=False,
    )
    f = compute_flops_per_token(cfg)
    # router: 2*64*8 = 1024 per layer
    # active experts (k=2, gated): 2 * 6*64*128 = 98304 per layer
    expected_ffn_per_layer = 1024 + 98304
    assert f["ffn"] == 2 * expected_ffn_per_layer
    # Sanity: if MoE used all 8 experts, ffn would be ~4x larger
    full_expert_ffn = 1024 + 8 * 6 * 64 * 128
    assert f["ffn"] < 2 * full_expert_ffn // 3  # well below full count


def test_compute_flops_per_token_ckpt_multiplier():
    """activation_checkpointing flips backward multiplier 3 -> 4."""
    base_kwargs = dict(
        arch="qwen3",
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_model=64,
        vocab_size=256,
    )
    cfg_off = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(**base_kwargs),
        training=TrainingConfig(activation_checkpointing=False),
    )
    cfg_on = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(**base_kwargs),
        training=TrainingConfig(activation_checkpointing=True),
    )
    f_off = compute_flops_per_token(cfg_off)
    f_on = compute_flops_per_token(cfg_on)
    assert f_off["fwd_total"] == f_on["fwd_total"]  # forward unchanged
    assert f_off["total"] == 3 * f_off["fwd_total"]
    assert f_on["total"] == 4 * f_on["fwd_total"]


def test_compute_flops_per_token_lm_head_bias():
    """LM head bias adds vocab_size FLOPs when lm_head_bias=True."""
    base = dict(arch="qwen3", n_layers=1, n_heads=2, d_model=32, vocab_size=128)
    cfg_off = TrainConfig(max_seq_len=64, model=ModelConfig(**base, lm_head_bias=False))
    cfg_on = TrainConfig(max_seq_len=64, model=ModelConfig(**base, lm_head_bias=True))
    f_off = compute_flops_per_token(cfg_off)
    f_on = compute_flops_per_token(cfg_on)
    assert f_on["lm_head"] - f_off["lm_head"] == 128


def test_compute_flops_per_token_qk_norm():
    """qk_norm adds 3 * (n_heads + n_kv_heads) * head_dim per layer."""
    base = dict(
        arch="qwen3", n_layers=2, n_heads=4, n_kv_heads=2, d_model=64, vocab_size=256
    )
    cfg_off = TrainConfig(max_seq_len=128, model=ModelConfig(**base, qk_norm=False))
    cfg_on = TrainConfig(max_seq_len=128, model=ModelConfig(**base, qk_norm=True))
    f_off = compute_flops_per_token(cfg_off)
    f_on = compute_flops_per_token(cfg_on)
    # head_dim=16; per-layer extra = 3 * (4+2) * 16 = 288; x n_layers=2 = 576
    expected_delta = 2 * 3 * (4 + 2) * 16
    assert f_on["norm"] - f_off["norm"] == expected_delta
    assert f_on["fwd_total"] - f_off["fwd_total"] == expected_delta
