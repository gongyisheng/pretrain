"""Tests for FP8 training support (src/training/fp8.py).

These tests check the module-swap plumbing and config validation. They do NOT
exercise the FP8 GEMM itself — that requires SM 9.0+ hardware and is covered
by the end-to-end smoke test when run on a supported GPU.
"""

import pytest
import torch
import torch.nn as nn

from src.training.fp8 import maybe_convert_to_fp8
from src.utils.config import ModelConfig, TrainConfig, TrainingConfig


pytest.importorskip("torchao.float8")
from torchao.float8.float8_linear import Float8Linear  # noqa: E402


def _fp8_capable() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9


fp8_only = pytest.mark.skipif(not _fp8_capable(), reason="FP8 requires SM 9.0+ GPU")


# --- helpers ---


class _TinyModel(nn.Module):
    """Stand-in for a real model: an Linear-tagged 'lm_head' plus other Linears."""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(64, 32)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(32, 32, bias=False),
                        "down_proj": nn.Linear(32, 32, bias=False),
                    }
                )
            ]
        )
        self.lm_head = nn.Linear(32, 64, bias=False)


def _make_config(
    mlp_cls: str = "dense",
    fp8: bool = True,
    fp8_recipe: str = "tensorwise",
    fp8_exclude_lm_head: bool = True,
) -> TrainConfig:
    return TrainConfig(
        model=ModelConfig(
            d_model=32,
            n_layers=1,
            vocab_size=64,
            attn_kwargs={"n_heads": 2},
            mlp_cls=mlp_cls,
            mlp_kwargs=(
                {"aux_loss": True, "aux_loss_coef": 1e-3} if mlp_cls == "moe" else {}
            ),
        ),
        training=TrainingConfig(
            fp8=fp8,
            fp8_recipe=fp8_recipe,
            fp8_exclude_lm_head=fp8_exclude_lm_head,
        ),
    )


# --- behavior ---


def test_disabled_is_noop():
    model = _TinyModel()
    cfg = _make_config(fp8=False)
    out = maybe_convert_to_fp8(model, cfg)
    assert out is model
    for m in model.modules():
        assert not isinstance(m, Float8Linear)


def test_moe_arch_raises():
    model = _TinyModel()
    cfg = _make_config(mlp_cls="moe", fp8=True)
    with pytest.raises(ValueError, match="moe"):
        maybe_convert_to_fp8(model, cfg)


def test_unknown_recipe_raises():
    # Recipe is validated at config construction (hardware-independent).
    with pytest.raises(ValueError, match="fp8_recipe"):
        _make_config(fp8=True, fp8_recipe="not_a_real_recipe")


@pytest.mark.skipif(_fp8_capable(), reason="testing non-FP8-capable branch")
def test_requires_capable_gpu():
    model = _TinyModel()
    cfg = _make_config(fp8=True)
    with pytest.raises(RuntimeError, match="compute capability"):
        maybe_convert_to_fp8(model, cfg)


@fp8_only
def test_swaps_linears_and_excludes_lm_head():
    model = _TinyModel().cuda()
    cfg = _make_config(fp8=True, fp8_exclude_lm_head=True)
    maybe_convert_to_fp8(model, cfg)
    # Inner projections are swapped
    assert isinstance(model.blocks[0]["q_proj"], Float8Linear)
    assert isinstance(model.blocks[0]["down_proj"], Float8Linear)
    # lm_head stays plain nn.Linear
    assert isinstance(model.lm_head, nn.Linear)
    assert not isinstance(model.lm_head, Float8Linear)
    # Embeddings are never linears, never touched
    assert isinstance(model.token_emb, nn.Embedding)


@fp8_only
def test_swaps_lm_head_when_not_excluded():
    model = _TinyModel().cuda()
    cfg = _make_config(fp8=True, fp8_exclude_lm_head=False)
    maybe_convert_to_fp8(model, cfg)
    assert isinstance(model.lm_head, Float8Linear)


@fp8_only
def test_forward_backward_runs():
    """Sanity: swapped model still trains end-to-end on a single step.

    All GEMM dims must be divisible by 16 — including B*S, which becomes the
    contracting dim in the weight-grad GEMM (grad_w = grad_out.T @ x).
    """
    model = _TinyModel().cuda()
    cfg = _make_config(fp8=True)
    maybe_convert_to_fp8(model, cfg)

    x = torch.randn(4, 16, 32, device="cuda", dtype=torch.bfloat16)
    model = model.to(torch.bfloat16)
    h = model.blocks[0]["down_proj"](model.blocks[0]["q_proj"](x))
    loss = h.float().square().mean()
    loss.backward()
    assert model.blocks[0]["q_proj"].weight.grad is not None
    assert torch.isfinite(loss).item()
