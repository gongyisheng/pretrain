"""Tests for src.metrics.convert — the install pass that decides which modules get
accumulators, on what device, and that installing does not thrash recompiles.

`apply_quantization_monitoring`, the module's other public function, has no test
coverage yet -- out of scope for this move, tracked separately.
"""

import pytest
import torch

from src.layers.block import TransformerBlock
from src.metrics.activation import set_activation_monitoring_status
from src.metrics.convert import apply_activation_monitoring
from src.metrics.functional import compute_activation_norm
from src.utils.config import ModelConfig
from tests.fast.helpers import make_attn_mask


D_MODEL = 64


@pytest.fixture(autouse=True)
def _disarm_recording():
    """_RECORDING is module-global: without this, one failing test leaves recording
    armed for every later test in the same xdist worker."""
    yield
    set_activation_monitoring_status(False)


def _block_cfg(n_layers=1):
    return ModelConfig(
        d_model=D_MODEL,
        n_layers=n_layers,
        vocab_size=256,
        attn=[
            {
                "attn_cls": "mha",
                "attn_kwargs": {"n_heads": 4, "attn_implementation": "sdpa"},
            }
        ],
        mlp=[{"mlp_cls": "dense", "mlp_kwargs": {"activation": "silu", "gated": True}}],
        norm_cls="rmsnorm",
        pos_emb_cls="rope",
        pos_emb_kwargs={},
    )


def _run_block(block, x=None):
    x = torch.randn(2, 16, D_MODEL) if x is None else x
    pos = torch.arange(x.shape[1]).unsqueeze(0).expand(x.shape[0], -1)
    attn_mask, _ = make_attn_mask("causal", "sdpa", pos, x.dtype)
    return block(x, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Sites and naming
# ---------------------------------------------------------------------------


def test_sites_are_exactly_the_linear_inputs():
    """Every site is one Linear's GEMM operand, named by its module path. Anything
    that is not a Linear input -- the residual stream, the SwiGLU product -- is not
    reachable by a hook and is deliberately absent."""
    block = TransformerBlock(_block_cfg(), layer_idx=0)
    apply_activation_monitoring(block)
    set_activation_monitoring_status(True)
    _run_block(block)
    names = set(compute_activation_norm(block))
    assert names == {
        n for n, m in block.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert "mlp.down_proj" in names


def test_embedding_is_not_a_site_but_tied_lm_head_is():
    """torch.nn.Embedding consumes token ids, not an activation; the tied lm_head still
    contracts a real activation against the same weight."""
    model = torch.nn.Module()
    model.token_emb = torch.nn.Embedding(32, D_MODEL)
    model.lm_head = torch.nn.Linear(D_MODEL, 32, bias=False)
    model.lm_head.weight = model.token_emb.weight
    apply_activation_monitoring(model)
    set_activation_monitoring_status(True)
    model.lm_head(model.token_emb(torch.randint(0, 32, (2, 8))))
    assert set(compute_activation_norm(model)) == {"lm_head"}


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------


def test_compiled_model_records_and_does_not_thrash_recompiles(device):
    """Hooks must be installed before compile, and must not close over per-module
    data -- a per-name closure specializes per site and blows the recompile limit."""
    if device != "cuda":
        pytest.skip("compile parity checked on CUDA")
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()
    model = torch.nn.Sequential(
        *[torch.nn.Linear(D_MODEL, D_MODEL) for _ in range(6)]
    ).to(device)
    apply_activation_monitoring(model)
    compiled = torch.compile(model)
    x = torch.randn(4, D_MODEL, device=device)
    compiled(x)
    set_activation_monitoring_status(True)
    compiled(x)
    norms = compute_activation_norm(model)
    assert len(norms) == 6
    assert counters["stats"]["unique_graphs"] <= 3, counters["stats"]
    assert not counters["graph_break"], dict(counters["graph_break"])


# ---------------------------------------------------------------------------
# Device placement
# ---------------------------------------------------------------------------


def test_buffers_follow_the_model_not_the_ambient_default_device():
    """Accumulators must be created on the module's own weight device. conftest sets
    torch.set_default_device('cuda') autouse, so a bare torch.zeros(()) would take
    the process default instead -- and eager quietly cross-device syncs it, leaving
    torch.compile to be the first thing that fails, on a fake tensor mismatch."""
    block = TransformerBlock(_block_cfg(), layer_idx=0).to("cpu")
    apply_activation_monitoring(block)
    assert block.mlp.down_proj.act_stats.energy.device.type == "cpu"
    assert block.attn.q_proj.act_stats.count.device.type == "cpu"
