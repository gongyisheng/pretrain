"""Tests for src/training/optimizer.py.

Organized in source-file order:
  1. LionOptimizer class
  2. build_optimizer function
  3. Scheduler classes (Constant / Cosine)
  4. build_scheduler function
"""

import math

import pytest
import torch

from src.utils.config import TrainConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from src.model.registry import build_model
from src.training.optimizer import (
    LionOptimizer,
    MuonAdamWOptimizer,
    build_optimizer,
    build_scheduler,
    ConstantWarmupScheduler,
    CosineWarmupScheduler,
)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #


def _make_cfg(tie: bool = False, lr_mult: dict | None = None) -> TrainConfig:
    cfg = TrainConfig()
    cfg.max_seq_len = 128
    cfg.model = ModelConfig(
        arch="qwen3",
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=256,
        qk_norm=True,
        tie_word_embeddings=tie,
    )
    cfg.optimizer = OptimizerConfig(
        lr=1e-3,
        lr_mult=lr_mult if lr_mult is not None else {"lm_head": 1.0},
        weight_decay=0.1,
    )
    cfg.scheduler = SchedulerConfig(warmup_steps=10, min_lr=1e-4)
    cfg.training.max_steps = 100
    return cfg


def _group_by_mult(opt):
    """Map lr_mult -> total parameter count across groups with that mult."""
    out = {}
    for pg in opt.param_groups:
        out.setdefault(pg["lr_mult"], 0)
        out[pg["lr_mult"]] += len(pg["params"])
    return out


def _dummy_optimizer():
    p = torch.nn.Parameter(torch.zeros(4))
    return torch.optim.SGD([p], lr=1.0)


# --------------------------------------------------------------------------- #
# 1. LionOptimizer class                                                      #
# --------------------------------------------------------------------------- #


def test_lion_one_step_matches_reference_formula():
    """Hand-compute the Lion update for one step and compare element-wise."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5, 3.0]))
    g = torch.tensor([0.1, -0.4, 0.2, -0.05])
    lr, beta1, beta2, wd = 1e-3, 0.9, 0.99, 0.1

    opt = LionOptimizer([p], lr=lr, betas=(beta1, beta2), weight_decay=wd)
    p.grad = g.clone()

    # Reference: m_{t-1} = 0 initially.
    m0 = torch.zeros_like(p)
    c = beta1 * m0 + (1 - beta1) * g
    update = torch.sign(c)
    expected_p = p.detach() - lr * (update + wd * p.detach())
    expected_m = beta2 * m0 + (1 - beta2) * g

    opt.step()

    assert torch.allclose(p.detach(), expected_p, atol=1e-7)
    assert torch.allclose(opt.state[p]["exp_avg"], expected_m, atol=1e-7)


def test_lion_two_steps_use_momentum_buffer():
    """Second step must use m updated from the first step, not zero."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    lr, beta1, beta2, wd = 1e-3, 0.9, 0.99, 0.0

    opt = LionOptimizer([p], lr=lr, betas=(beta1, beta2), weight_decay=wd)

    g1 = torch.tensor([0.3, -0.7])
    p.grad = g1.clone()
    opt.step()

    g2 = torch.tensor([-0.1, 0.2])
    p.grad = g2.clone()

    # Reference after step 1: m_1 = (1 - β2) * g1.
    m_after_1 = (1 - beta2) * g1
    # Step 2: c = β1 * m_1 + (1 - β1) * g2; sign(c) gives the update direction.
    c2 = beta1 * m_after_1 + (1 - beta1) * g2
    update2 = torch.sign(c2)
    expected_p_after_2 = p.detach() - lr * update2

    opt.step()
    assert torch.allclose(p.detach(), expected_p_after_2, atol=1e-7)


def test_lion_decoupled_weight_decay_with_zero_grad():
    """With grad=0, Lion's update is purely the wd shrink (sign(0) == 0)."""
    p = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.0]))
    lr, wd = 1e-2, 0.5
    opt = LionOptimizer([p], lr=lr, betas=(0.9, 0.99), weight_decay=wd)
    p.grad = torch.zeros_like(p)

    expected = p.detach() * (1 - lr * wd)
    opt.step()
    assert torch.allclose(p.detach(), expected, atol=1e-7)


def test_lion_rejects_invalid_hyperparams():
    p = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match="lr must be positive"):
        LionOptimizer([p], lr=0.0)
    with pytest.raises(ValueError, match="betas"):
        LionOptimizer([p], lr=1e-3, betas=(0.9, 1.0))


def test_lion_foreach_matches_single_tensor():
    """Foreach path must produce bitwise-identical results to the per-param loop."""
    torch.manual_seed(0)
    shapes = [(4,), (3, 5), (2,), (6, 2)]
    init = [torch.randn(s) for s in shapes]

    params_a = [torch.nn.Parameter(t.clone()) for t in init]
    params_b = [torch.nn.Parameter(t.clone()) for t in init]

    opt_a = LionOptimizer(
        params_a, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1, foreach=True
    )
    opt_b = LionOptimizer(
        params_b, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.1, foreach=False
    )

    for _ in range(5):
        torch.manual_seed(1)
        grads = [torch.randn_like(p) for p in params_a]
        for p, g in zip(params_a, grads):
            p.grad = g.clone()
        for p, g in zip(params_b, grads):
            p.grad = g.clone()
        opt_a.step()
        opt_b.step()
        for pa, pb in zip(params_a, params_b):
            assert torch.allclose(pa.detach(), pb.detach(), atol=0.0, rtol=0.0), (
                "foreach and single-tensor paths diverged"
            )


def test_lion_foreach_handles_mixed_dtype_within_group():
    """foreach groups by (device, dtype); mixed dtypes inside one param group must still work."""
    p_fp32 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p_fp64 = torch.nn.Parameter(torch.tensor([0.5, 3.0], dtype=torch.float64))
    opt = LionOptimizer(
        [p_fp32, p_fp64], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0, foreach=True
    )
    p_fp32.grad = torch.tensor([0.1, -0.4])
    p_fp64.grad = torch.tensor([0.2, -0.05], dtype=torch.float64)
    opt.step()  # must not raise
    assert p_fp32.dtype == torch.float32
    assert p_fp64.dtype == torch.float64


def test_lion_state_dict_roundtrip():
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    opt = LionOptimizer([p], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    p.grad = torch.tensor([0.1, -0.2, 0.05])
    opt.step()
    p.grad = torch.tensor([-0.3, 0.4, -0.1])
    opt.step()
    state = opt.state_dict()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = LionOptimizer([p2], lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0)
    opt2.load_state_dict(state)

    # Subsequent step should match between original and reloaded optimizer.
    p.grad = torch.tensor([0.2, 0.1, -0.4])
    p2.grad = p.grad.clone()
    opt.step()
    opt2.step()
    assert torch.allclose(p.detach(), p2.detach(), atol=1e-7)


# --------------------------------------------------------------------------- #
# 2. build_optimizer function                                                 #
# --------------------------------------------------------------------------- #


def test_default_lr_mult_produces_two_groups_untied():
    """Default {lm_head: 1.0} with untied: all params effectively at mult=1.0."""
    cfg = _make_cfg(tie=False)
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    mults = _group_by_mult(opt)
    assert set(mults.keys()) == {1.0}
    total = sum(len(pg["params"]) for pg in opt.param_groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)


def test_lm_head_mult_untied_creates_lm_head_group():
    cfg = _make_cfg(tie=False, lr_mult={"lm_head": 0.3})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    lm_head_groups = [pg for pg in opt.param_groups if pg["lr_mult"] == 0.3]
    assert len(lm_head_groups) == 1
    assert len(lm_head_groups[0]["params"]) == 1  # only lm_head.weight

    # The lm_head param is the correct tensor (same shape as lm_head.weight).
    lm_head_param = lm_head_groups[0]["params"][0]
    assert lm_head_param is model.lm_head.weight

    # Weight decay applied (lm_head.weight is 2D).
    assert lm_head_groups[0]["weight_decay"] == cfg.optimizer.weight_decay


def test_lm_head_mult_tied_is_noop():
    """In tied mode, lm_head.weight is enumerated as token_emb.weight — no lm_head group."""
    cfg = _make_cfg(tie=True, lr_mult={"lm_head": 0.3})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    # No group should have mult=0.3 since no param name contains 'lm_head'
    # when the param is enumerated under token_emb.weight.
    assert all(pg["lr_mult"] == 1.0 for pg in opt.param_groups)


def test_regex_pattern_matches_multiple_params():
    """A regex like 'bias' should catch every param whose name contains 'bias'."""
    cfg = _make_cfg(tie=False, lr_mult={"bias": 0.5})
    # Enable some biases so we have params matching.
    cfg.model.attn_bias = True
    cfg.model.mlp_bias = True
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    bias_groups = [pg for pg in opt.param_groups if pg["lr_mult"] == 0.5]
    assert len(bias_groups) >= 1
    bias_params = [p for pg in bias_groups for p in pg["params"]]

    expected = [
        p for n, p in model.named_parameters() if "bias" in n and p.requires_grad
    ]
    assert len(bias_params) == len(expected)
    assert {id(p) for p in bias_params} == {id(p) for p in expected}


def test_first_pattern_wins_on_overlap():
    """If two patterns can match a param, the first one in insertion order wins."""
    cfg = _make_cfg(tie=False, lr_mult={"lm_head": 0.3, "head": 0.1})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    # lm_head.weight should match 'lm_head' (first), not 'head'.
    mults = _group_by_mult(opt)
    assert 0.3 in mults
    assert mults[0.3] == 1  # just lm_head.weight


def test_unmatched_params_get_default_mult():
    cfg = _make_cfg(tie=False, lr_mult={"lm_head": 0.3})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    default_groups = [pg for pg in opt.param_groups if pg["lr_mult"] == 1.0]
    lm_head_groups = [pg for pg in opt.param_groups if pg["lr_mult"] == 0.3]

    n_default = sum(len(pg["params"]) for pg in default_groups)
    n_lm_head = sum(len(pg["params"]) for pg in lm_head_groups)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_default + n_lm_head == total


def test_standard_decay_rule_within_matched_group():
    """A prefix like 'blocks\\.0' should split into decay and no_decay sub-groups."""
    cfg = _make_cfg(tie=False, lr_mult={r"blocks\.0": 2.0})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    matched = [pg for pg in opt.param_groups if pg["lr_mult"] == 2.0]
    assert len(matched) == 2  # one decay group, one no_decay group
    wds = {pg["weight_decay"] for pg in matched}
    assert wds == {0.0, cfg.optimizer.weight_decay}

    # Every param in these groups comes from blocks.0.*
    for pg in matched:
        for p in pg["params"]:
            matching_names = [n for n, q in model.named_parameters() if q is p]
            assert any(n.startswith("blocks.0.") for n in matching_names)


def test_all_params_covered_exactly_once():
    """Every trainable parameter ends up in exactly one group."""
    cfg = _make_cfg(tie=False, lr_mult={"lm_head": 0.3, "token_emb": 0.5})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    seen = []
    for pg in opt.param_groups:
        seen.extend(id(p) for p in pg["params"])
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable)
    assert len(seen) == len(set(seen))  # no duplicates


def test_build_optimizer_dispatches_to_lion():
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "lion"
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, LionOptimizer)
    # Param grouping (wd filter + lr_mult) still applies.
    total = sum(len(pg["params"]) for pg in opt.param_groups)
    assert total == sum(1 for p in model.parameters() if p.requires_grad)
    # At least one decay group and one no-decay group exist.
    wds = {pg["weight_decay"] for pg in opt.param_groups}
    assert 0.0 in wds and cfg.optimizer.weight_decay in wds


def test_build_optimizer_dispatches_to_muon():
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "muon"
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, MuonAdamWOptimizer)
    # Every trainable param appears exactly once across the two subsystems.
    seen = [id(p) for pg in opt.param_groups for p in pg["params"]]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(trainable)
    assert len(seen) == len(set(seen))


def test_muon_routes_only_2d_hidden_weights():
    """2D weights -> Muon; embeddings, lm_head, and 1D params -> AdamW."""
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "muon"
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)

    name_by_id = {id(p): n for n, p in model.named_parameters()}
    muon_ids = {id(p) for pg in opt.muon.param_groups for p in pg["params"]}
    adamw_ids = {id(p) for pg in opt.adamw.param_groups for p in pg["params"]}

    # Muon only ever sees 2D non-embedding, non-head weights.
    for pid in muon_ids:
        p = next(p for _, p in model.named_parameters() if id(p) == pid)
        n = name_by_id[pid]
        assert p.ndim == 2 and "emb" not in n and "lm_head" not in n

    # Embeddings and the output head route to AdamW; a hidden projection to Muon.
    assert id(model.token_emb.weight) in adamw_ids
    assert id(model.lm_head.weight) in adamw_ids
    q_proj = dict(model.named_parameters())["blocks.0.attn.q_proj.weight"]
    assert id(q_proj) in muon_ids


def test_muon_step_updates_params(device):
    if device != "cuda":
        pytest.skip("Muon hybrid uses fused AdamW (CUDA-only)")
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "muon"
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt.step()
    for n, p in model.named_parameters():
        assert not torch.allclose(p, before[n]), f"{n} did not update"


def test_muon_state_dict_roundtrip(device):
    if device != "cuda":
        pytest.skip("Muon hybrid uses fused AdamW (CUDA-only)")
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "muon"
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    opt.step()

    state = opt.state_dict()
    assert set(state) == {"muon", "adamw"}

    opt2 = build_optimizer(model, cfg)
    opt2.load_state_dict(state)  # must not raise
    # Muon momentum buffers were restored.
    assert any("momentum_buffer" in s for s in opt2.muon.state.values())


def test_build_optimizer_unknown_name_raises():
    cfg = _make_cfg(tie=False)
    cfg.optimizer.name = "shampoo"
    model = build_model(cfg)
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(model, cfg)


# --------------------------------------------------------------------------- #
# 3. Scheduler classes (ConstantWarmupScheduler / CosineWarmupScheduler)      #
# --------------------------------------------------------------------------- #


def test_constant_scheduler_warmup_ramps_linearly():
    opt = _dummy_optimizer()
    sched = ConstantWarmupScheduler(opt, warmup_steps=10, max_lr=1e-3)
    # Step 1..10: linear ramp from 0 to max_lr
    sched.step()  # step 1
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 1 / 10)
    for _ in range(9):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_constant_scheduler_plateau_stays_at_max_lr():
    opt = _dummy_optimizer()
    sched = ConstantWarmupScheduler(opt, warmup_steps=5, max_lr=2e-3)
    for _ in range(100):
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(2e-3)


def test_constant_scheduler_state_dict_roundtrip():
    opt = _dummy_optimizer()
    sched = ConstantWarmupScheduler(opt, warmup_steps=5, max_lr=1e-3)
    for _ in range(7):
        sched.step()
    state = sched.state_dict()

    opt2 = _dummy_optimizer()
    sched2 = ConstantWarmupScheduler(opt2, warmup_steps=5, max_lr=1e-3)
    sched2.load_state_dict(state)
    assert sched2.current_step == 7
    # Verify the resumed scheduler produces the same LR as the original at the next step.
    sched.step()
    sched2.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])


def test_constant_scheduler_zero_warmup_immediately_at_max_lr():
    opt = _dummy_optimizer()
    sched = ConstantWarmupScheduler(opt, warmup_steps=0, max_lr=5e-4)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_scheduler_scales_each_group_by_lr_mult():
    """Scheduler applies the cosine/warmup shape times each group's lr_mult."""
    cfg = _make_cfg(tie=False, lr_mult={"lm_head": 0.2})
    model = build_model(cfg)
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)

    # Full warmup: base lr should be at cfg.optimizer.lr.
    for _ in range(cfg.scheduler.warmup_steps):
        sched.step()
    base = sched.get_lr()
    for pg in opt.param_groups:
        expected = base * pg["lr_mult"]
        assert math.isclose(pg["lr"], expected, rel_tol=1e-6), (
            f"group with lr_mult={pg['lr_mult']}: got lr={pg['lr']}, expected {expected}"
        )

    # Cosine phase: base lr decayed, ratio preserved.
    for _ in range(50):
        sched.step()
    base = sched.get_lr()
    for pg in opt.param_groups:
        expected = base * pg["lr_mult"]
        assert math.isclose(pg["lr"], expected, rel_tol=1e-6)


# --------------------------------------------------------------------------- #
# 4. build_scheduler function                                                 #
# --------------------------------------------------------------------------- #


def test_build_scheduler_dispatch_cosine():
    cfg = _make_cfg()
    cfg.scheduler.name = "cosine"
    cfg.scheduler.warmup_steps = 5
    cfg.training.max_steps = 100
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(4))], lr=1.0)
    sched = build_scheduler(opt, cfg)
    assert isinstance(sched, CosineWarmupScheduler)


def test_build_scheduler_dispatch_constant():
    cfg = _make_cfg()
    cfg.scheduler.name = "constant"
    cfg.scheduler.warmup_steps = 5
    cfg.training.max_steps = 100
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(4))], lr=1.0)
    sched = build_scheduler(opt, cfg)
    assert isinstance(sched, ConstantWarmupScheduler)


def test_build_scheduler_unknown_name_raises():
    cfg = _make_cfg()
    cfg.scheduler.name = "linear"
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(4))], lr=1.0)
    with pytest.raises(ValueError, match="unknown scheduler"):
        build_scheduler(opt, cfg)
