"""Tests for build_optimizer, focused on lr_mult regex-based param grouping."""
import math
import pytest
import torch

from src.utils.config import TrainConfig, ModelConfig, OptimizerConfig, SchedulerConfig
from src.model.registry import build_model
from src.training.optimizer import build_optimizer, build_scheduler


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

    expected = [p for n, p in model.named_parameters() if "bias" in n and p.requires_grad]
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
        assert math.isclose(pg["lr"], expected, rel_tol=1e-6), \
            f"group with lr_mult={pg['lr_mult']}: got lr={pg['lr']}, expected {expected}"

    # Cosine phase: base lr decayed, ratio preserved.
    for _ in range(50):
        sched.step()
    base = sched.get_lr()
    for pg in opt.param_groups:
        expected = base * pg["lr_mult"]
        assert math.isclose(pg["lr"], expected, rel_tol=1e-6)


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
