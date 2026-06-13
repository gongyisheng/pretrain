"""Tests for src.utils.metric_utils — the pure metric-computation layer.

These are stateless functions (numbers/tensors in, numbers/dicts out), so the
tests assert directly on returned values with no logger or tracker involved.
"""

import math

import pytest
import torch

from src.model import build_model
from src.model.transformer import TransformerLM
from src.training.optimizer import AdamWOptimizer, LionOptimizer
from src.utils import metric_utils
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig
from tests.fast.helpers import ATTN_IMPLEMENTATION, make_attn_mask, skip_if_unsupported

# ---------------------------------------------------------------------------
# Small model configs matching real architectures. Factories rather than
# constants so they can pick up the parametrized attn_implementation.
# ---------------------------------------------------------------------------


def _gpt2_cfg(impl):
    return ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="mha",
        attn_kwargs={"n_heads": 2, "attn_implementation": impl},
        mlp_cls="dense",
        mlp_kwargs={"gated": False, "bias": True, "activation": "gelu"},
    )


def _qwen3_cfg(impl):
    return ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": 2,
            "n_kv_heads": 1,
            "qk_norm": True,
            "attn_implementation": impl,
        },
        mlp_cls="dense",
        mlp_kwargs={},
    )


def _qwen3_moe_cfg(impl):
    return ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": 2,
            "n_kv_heads": 1,
            "qk_norm": True,
            "attn_implementation": impl,
        },
        mlp_cls="moe",
        mlp_kwargs={
            "n_experts": 4,
            "n_experts_per_token": 2,
            "intermediate_size": 128,
        },
    )


def _gpt2_attn_res_cfg(impl):
    return ModelConfig(
        n_layers=4,
        d_model=64,
        vocab_size=256,
        attn_cls="mha",
        attn_kwargs={"n_heads": 2, "attn_implementation": impl},
        mlp_cls="dense",
        mlp_kwargs={"gated": False, "bias": True, "activation": "gelu"},
        residual_cls="attn_res",
        residual_kwargs={"seal_block_size": 2},
    )


def _qwen3_attn_res_cfg(impl):
    return ModelConfig(
        n_layers=4,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={
            "n_heads": 2,
            "n_kv_heads": 1,
            "qk_norm": True,
            "attn_implementation": impl,
        },
        mlp_cls="dense",
        mlp_kwargs={},
        residual_cls="attn_res",
        residual_kwargs={"seal_block_size": 2},
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


def _expected_grad_keys(model: torch.nn.Module) -> set[str]:
    """Derive expected raw (un-prefixed) grad-norm keys from named_parameters."""
    keys = set()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        keys.add(name.removeprefix("_orig_mod."))
    return keys


# ---------------------------------------------------------------------------
# Optimizer step diagnostics: snapshot / param-step / momentum / variance
# ---------------------------------------------------------------------------


def test_snapshot_and_param_step_norm_match_l2_of_diff():
    """compute_param_step_norm = sqrt(sum((p_after - p_before)^2))."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    snapshot = metric_utils.snapshot_params(model)
    delta = 0.1
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, delta))
    n_params = sum(p.numel() for p in model.parameters())
    expected = delta * math.sqrt(n_params)
    actual = metric_utils.compute_param_step_norm(model, snapshot)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_param_step_norm_zero_when_unchanged():
    """No update → step norm exactly 0."""
    model = torch.nn.Linear(4, 2)
    snapshot = metric_utils.snapshot_params(model)
    assert metric_utils.compute_param_step_norm(model, snapshot) == 0.0


def test_momentum_norm_reads_exp_avg_buffers():
    """compute_momentum_norm = sqrt(sum_p ||state[p]['exp_avg']||^2)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    # Lion populates state on first step; one no-op step seeds exp_avg=0 buffers.
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    fill = 0.5
    for p in model.parameters():
        opt.state[p]["exp_avg"].fill_(fill)
    n_params = sum(p.numel() for p in model.parameters())
    expected = fill * math.sqrt(n_params)
    actual = metric_utils.compute_momentum_norm(opt)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_momentum_norm_none_when_no_state():
    """Before any optimizer.step(), no state buffers exist → None."""
    model = torch.nn.Linear(4, 2)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    assert metric_utils.compute_momentum_norm(opt) is None


def test_variance_norm_reads_exp_avg_sq_buffers():
    """compute_variance_norm = sqrt(sum_p ||state[p]['exp_avg_sq']||^2)."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    opt = AdamWOptimizer(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    fill = 0.25
    for p in model.parameters():
        opt.state[p]["exp_avg_sq"].fill_(fill)
    n_params = sum(p.numel() for p in model.parameters())
    expected = fill * math.sqrt(n_params)
    actual = metric_utils.compute_variance_norm(opt)
    assert math.isclose(actual, expected, rel_tol=1e-5)


def test_variance_norm_none_for_lion():
    """Lion stores only exp_avg (no second moment) → None."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    opt = LionOptimizer(model.parameters(), lr=1e-3)
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    assert "exp_avg" in opt.state[next(iter(opt.state))]  # sanity
    assert metric_utils.compute_variance_norm(opt) is None


def test_variance_norm_none_before_first_step():
    """No optimizer state yet → None."""
    model = torch.nn.Linear(4, 2)
    opt = AdamWOptimizer(model.parameters(), lr=1e-3)
    assert metric_utils.compute_variance_norm(opt) is None


# ---------------------------------------------------------------------------
# Per-layer gradient norms (raw, un-prefixed keys)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_layer_grad_norms_plain_model(arch_id, impl, device):
    """One key per parameter with a grad; keys are raw names (no prefix)."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    _populate_grads(model, model_cfg.vocab_size, impl)
    result = metric_utils.compute_layer_grad_norms(model)
    assert set(result) == _expected_grad_keys(model)
    assert all(not k.startswith("grad_norm/") for k in result)
    for k, v in result.items():
        assert v >= 0 and math.isfinite(v), f"{k}={v}"


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
@pytest.mark.parametrize("impl", ATTN_IMPLEMENTATION)
def test_layer_grad_norms_compiled_model(arch_id, impl, device):
    """Must work after torch.compile (which prepends _orig_mod.)."""
    skip_if_unsupported(impl, device)
    model_cfg = _CFG_FACTORIES[arch_id](impl)
    model = build_model(_FakeTrainConfig(model_cfg))
    compiled = torch.compile(model, backend="eager")
    _populate_grads(compiled, model_cfg.vocab_size, impl)
    result = metric_utils.compute_layer_grad_norms(compiled)
    assert set(result) == _expected_grad_keys(compiled)
    assert all(not k.startswith("_orig_mod.") for k in result)


# ---------------------------------------------------------------------------
# Per-2D-weight spectral metrics (SVD)
# ---------------------------------------------------------------------------


def _expected_svd_keys(model: torch.nn.Module) -> set[str]:
    """2D float params, raw names, minus rope buffers and embeddings."""
    keys = set()
    for name, param in model.named_parameters():
        if param.ndim != 2 or not param.is_floating_point():
            continue
        name = name.removeprefix("_orig_mod.")
        if name.startswith("rope.") or "emb" in name:
            continue
        keys.add(name)
    return keys


def test_svd_metrics_orthogonal_is_full_rank():
    """Orthogonal matrix: all σ equal → srank=pr=n."""
    torch.manual_seed(0)
    q, _ = torch.linalg.qr(torch.randn(8, 8))  # σ all == 1
    m = metric_utils._svd_metrics(q)
    assert m["srank"] == pytest.approx(8.0, rel=1e-4)
    assert m["pr"] == pytest.approx(8.0, rel=1e-4)


def test_svd_metrics_rank_one_is_minimal():
    """Rank-1 matrix: one σ dominates → srank≈pr≈1."""
    w = torch.outer(torch.arange(1.0, 6.0), torch.arange(1.0, 4.0))
    m = metric_utils._svd_metrics(w)
    assert m["srank"] == pytest.approx(1.0, abs=1e-4)
    assert m["pr"] == pytest.approx(1.0, abs=1e-4)


def test_svd_metrics_zero_matrix():
    """All-zero weight (no positive σ) → every metric 0."""
    assert metric_utils._svd_metrics(torch.zeros(4, 4)) == {
        "srank": 0.0,
        "pr": 0.0,
    }


@pytest.mark.parametrize("arch_id", list(_CFG_FACTORIES))
def test_layer_svd_metrics_keys_and_bounds(arch_id):
    """One entry per 2D weight (no rope/embedding); metrics within bounds."""
    model_cfg = _CFG_FACTORIES[arch_id]("sdpa")
    model = build_model(_FakeTrainConfig(model_cfg))
    result = metric_utils.compute_layer_svd_metrics(model)
    assert set(result) == _expected_svd_keys(model)
    assert not any("emb" in k or k.startswith("rope.") for k in result)

    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    for name, m in result.items():
        n = min(shapes[name])
        assert 1.0 - 1e-4 <= m["srank"] <= n + 1e-4
        assert 1.0 - 1e-4 <= m["pr"] <= n + 1e-4


def test_layer_svd_metrics_compiled_model_strips_prefix():
    """Works after torch.compile (which prepends _orig_mod.)."""
    model = build_model(_FakeTrainConfig(_qwen3_cfg("sdpa")))
    compiled = torch.compile(model, backend="eager")
    result = metric_utils.compute_layer_svd_metrics(compiled)
    assert set(result) == _expected_svd_keys(compiled)
    assert all(not k.startswith("_orig_mod.") for k in result)


# ---------------------------------------------------------------------------
# Parameter counts
# ---------------------------------------------------------------------------


def _gpt2_layernorm_learned_cfg(impl):
    """GPT-2-style: layernorm (weight + bias) + learned positional embedding +
    untied lm_head with bias. Exercises the norm-bias, learned-pos-emb, and
    untied-head param paths the rmsnorm+rope factories don't."""
    return ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="mha",
        attn_kwargs={"n_heads": 2, "bias": True, "attn_implementation": impl},
        mlp_cls="dense",
        mlp_kwargs={"gated": False, "bias": True, "activation": "gelu"},
        norm_cls="layernorm",
        pos_emb_cls="learned",
        tie_word_embeddings=False,
        lm_head_bias=True,
    )


@pytest.mark.parametrize(
    "factory",
    [*_CFG_FACTORIES.values(), _gpt2_layernorm_learned_cfg],
    ids=[*_CFG_FACTORIES, "gpt2_layernorm_learned"],
)
def test_count_parameters_matches_real_model(factory):
    """Analytic count_parameters reproduces the live model's param counts exactly
    for every architecture variant."""
    cfg = TrainConfig(max_seq_len=128, model=factory("sdpa"))
    model = build_model(cfg)
    c = metric_utils.count_parameters(cfg)

    real_total = sum(p.numel() for p in model.parameters())
    real_non_emb = real_total - sum(
        p.numel() for name, p in model.named_parameters() if "emb" in name
    )
    assert c["total"] == real_total
    assert c["non_emb"] == real_non_emb
    assert c["non_emb"] < c["total"]  # embeddings present

    if cfg.model.mlp_cls == "moe":
        assert c["active_non_emb"] < c["non_emb"]  # k experts < all experts
    else:
        assert c["active_non_emb"] == c["non_emb"]


# ---------------------------------------------------------------------------
# Eval primitives
# ---------------------------------------------------------------------------


def test_count_correct_basic():
    """argmax==labels counted; ignore_index masked out."""
    logits = torch.zeros(1, 3, 5)
    logits[0, torch.arange(3), torch.tensor([1, 2, 3])] = 10.0  # argmax = 1, 2, 3
    labels = torch.tensor([[1, 9, 3]])  # position 1 wrong (pred 2 != 9)
    correct, total = metric_utils.count_correct(logits, labels)
    assert (correct, total) == (2, 3)


def test_count_correct_ignore_and_exclude():
    """ignore_index and exclude_id both drop positions from the denominator."""
    logits = torch.zeros(1, 3, 5)
    logits[0, torch.arange(3), torch.tensor([1, 2, 3])] = 10.0
    labels = torch.tensor([[1, -100, 3]])  # middle ignored
    assert metric_utils.count_correct(logits, labels) == (2, 2)
    # exclude the EOT id (3 here) → only position 0 remains
    assert metric_utils.count_correct(logits, labels, exclude_id=3) == (1, 1)


def test_compute_statistics():
    assert metric_utils.compute_statistics([1.0, 2.0, 3.0]) == {
        "mean": 2.0,
        "median": 2.0,
        "max": 3.0,
        "min": 1.0,
    }


def test_compute_statistics_empty():
    assert metric_utils.compute_statistics([]) == {}


def test_compute_perplexity():
    assert metric_utils.compute_perplexity(0.0) == pytest.approx(1.0)
    assert metric_utils.compute_perplexity(1.0) == pytest.approx(math.e)


def test_compute_perplexity_caps_overflow():
    """Huge loss saturates to the cap instead of returning inf."""
    assert metric_utils.compute_perplexity(1e9) == 1e6


def test_compute_bits_per_byte():
    loss, tpb = 2.5, 0.4
    assert metric_utils.compute_bits_per_byte(loss, tpb) == pytest.approx(
        loss * tpb / math.log(2)
    )


# ---------------------------------------------------------------------------
# compute_flops_per_token
# ---------------------------------------------------------------------------


def test_compute_flops_per_token_dense_gpt2_mha_ungated():
    """GPT-2 style: MHA, ungated FFN with biases. Hand-computed expected values."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="mha",
        attn_kwargs={"n_heads": 2, "bias": True},
        mlp_cls="dense",
        mlp_kwargs={
            "intermediate_size": 256,
            "activation": "gelu",
            "gated": False,
            "bias": True,
        },
    )
    f = metric_utils.compute_flops_per_token(cfg)
    # head_dim=32, n_kv_heads=n_heads=2 (MHA), L=2, T=128. Per-layer components:
    #   attn = qkv + o + matmul = 24768 + 8256 + 32768   (qkv has bias +192, o +64)
    #   mlp (ungated, bias) = 4*64*256 + (256+64) = 65856
    #   norm (2 LayerNorms) = 2*3*64 = 384
    # final norm = 3*64 = 192; lm_head = 2*64*256 = 32768 (no bias by default)
    attn = 24768 + 8256 + 32768
    mlp = 65856
    norm_per_layer = 384
    expected_fwd = 2 * (attn + mlp + norm_per_layer) + 192 + 32768
    # compute_flops_per_token returns the total (fwd × 3 by default)
    assert f == 3 * expected_fwd


def test_compute_flops_per_token_gqa_gated_qwen3():
    """GQA halves KV proj vs MHA; gated FFN is 6*d*d_ff (vs 4 ungated)."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 4, "n_kv_heads": 2, "bias": False},
        mlp_cls="dense",
        mlp_kwargs={
            "intermediate_size": 256,
            "activation": "silu",
            "gated": True,
            "bias": False,
        },
    )
    f = metric_utils.compute_flops_per_token(cfg)
    # head_dim=16. Per-layer: attn = qkv(16384) + o(8192) + matmul(32768);
    # gated mlp = 6*64*256 = 98304; norm = 2*3*64 = 384.
    attn = 16384 + 8192 + 32768
    mlp = 98304
    norm_per_layer = 384
    expected_fwd = 2 * (attn + mlp + norm_per_layer) + 192 + 2 * 64 * 256
    assert f == 3 * expected_fwd


def test_compute_flops_per_token_moe_uses_k_active():
    """MoE mlp cost = router + k active experts, NOT all N experts."""
    cfg = TrainConfig(max_seq_len=128)
    cfg.model = ModelConfig(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 4, "n_kv_heads": 2},
        mlp_cls="moe",
        mlp_kwargs={
            "n_experts": 8,
            "n_experts_per_token": 2,
            "intermediate_size": 128,
            "gated": True,
            "bias": False,
        },
    )
    f = metric_utils.compute_flops_per_token(cfg)
    # head_dim=16. attn = 16384 + 8192 + 32768; norm = 384.
    # moe mlp = router(2*64*8=1024) + k=2 gated experts (2*6*64*128=98304) = 99328.
    attn = 16384 + 8192 + 32768
    mlp_k_active = 1024 + 98304
    expected_fwd = 2 * (attn + mlp_k_active + 384) + 192 + 2 * 64 * 256
    assert f == 3 * expected_fwd
    # k-active is far cheaper than activating all N=8 experts
    mlp_all_experts = 1024 + 8 * 6 * 64 * 128
    assert mlp_k_active < mlp_all_experts // 3


def test_compute_flops_per_token_ckpt_multiplier():
    """activation_checkpointing flips backward multiplier 3 -> 4."""
    base_kwargs = dict(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 4, "n_kv_heads": 2},
        mlp_cls="dense",
        mlp_kwargs={},
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
    fwd = TransformerLM.compute_flops(cfg_off.model, cfg_off.max_seq_len)
    f_off = metric_utils.compute_flops_per_token(cfg_off)
    f_on = metric_utils.compute_flops_per_token(cfg_on)
    assert f_off == 3 * fwd
    assert f_on == 4 * fwd


def test_compute_flops_per_token_lm_head_bias():
    """LM head bias adds vocab_size FLOPs when lm_head_bias=True."""
    base = dict(
        n_layers=1,
        d_model=32,
        vocab_size=128,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 2},
        mlp_cls="dense",
        mlp_kwargs={},
    )
    cfg_off = TrainConfig(max_seq_len=64, model=ModelConfig(**base, lm_head_bias=False))
    cfg_on = TrainConfig(max_seq_len=64, model=ModelConfig(**base, lm_head_bias=True))
    f_off = metric_utils.compute_flops_per_token(cfg_off)
    f_on = metric_utils.compute_flops_per_token(cfg_on)
    # lm_head bias adds vocab_size (128) FLOPs to fwd; total is ×3
    assert f_on - f_off == 3 * 128


def test_compute_flops_per_token_qk_norm():
    """qk_norm adds 3 * (n_heads + n_kv_heads) * head_dim per layer."""
    base = dict(
        n_layers=2,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        mlp_cls="dense",
        mlp_kwargs={},
    )
    cfg_off = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(
            **base, attn_kwargs={"n_heads": 4, "n_kv_heads": 2, "qk_norm": False}
        ),
    )
    cfg_on = TrainConfig(
        max_seq_len=128,
        model=ModelConfig(
            **base, attn_kwargs={"n_heads": 4, "n_kv_heads": 2, "qk_norm": True}
        ),
    )
    f_off = metric_utils.compute_flops_per_token(cfg_off)
    f_on = metric_utils.compute_flops_per_token(cfg_on)
    # qk_norm adds 3*(n_heads+n_kv_heads)*head_dim per layer to fwd; total is ×3
    expected_delta = 2 * 3 * (4 + 2) * 16
    assert f_on - f_off == 3 * expected_delta


def test_compute_flops_per_token_counts_attn_res_residual():
    """attn_res adds depth-dependent residual FLOPs; standard residual adds 0."""
    from src.layers.residual import AttnResidual, StandardResidual

    base = dict(
        n_layers=4,
        d_model=64,
        vocab_size=256,
        attn_cls="gqa",
        attn_kwargs={"n_heads": 4, "n_kv_heads": 2},
        mlp_cls="dense",
        mlp_kwargs={},
    )
    std = TrainConfig(max_seq_len=64, model=ModelConfig(**base))
    ar = TrainConfig(
        max_seq_len=64,
        model=ModelConfig(
            **base, residual_cls="attn_res", residual_kwargs={"seal_block_size": 2}
        ),
    )
    # per-slot residual cost: 0 for standard, 7 * n_ctx * d_model for attn_res
    assert StandardResidual.compute_flops(std.model, 64, layer_idx=3) == 0
    assert (
        AttnResidual.compute_flops(ar.model, 64, layer_idx=3) == 7 * (3 // 2 + 1) * 64
    )
    # model-level: attn_res strictly higher than standard residual
    assert metric_utils.compute_flops_per_token(
        ar
    ) > metric_utils.compute_flops_per_token(std)
