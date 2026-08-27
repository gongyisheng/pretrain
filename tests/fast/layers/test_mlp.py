"""MLP tests: DenseMLPBlock + SparseMoEBlock behavior, numerics, and registry."""

import pytest
import torch

from src.kernel.ops.gemm import grouped_mm
from src.layers.activation import GATED_ACTIVATIONS, UNGATED_ACTIVATIONS
from src.layers.mlp import (
    DenseMLPBlock,
    ExpertBias,
    ExpertLoad,
    GroupedGemmFn,
    MLP_REGISTRY,
    MoERouter,
    MOE_ROUTER_SCORE_FNS,
    SparseMoEBlock,
    grouped_mlp,
)
from tests.fast.layers._refs import (
    COMPOUND_DTYPES,
    SIMPLE_DTYPES,
    dense_mlp_ref,
    moe_router_ref,
    sparse_moe_block_ref,
)


def grouped_mm_eager(a, b, offs, bias=None):
    return grouped_mm(a, b, offs, bias=bias, backend="eager")


ACT_NAMES = list(UNGATED_ACTIVATIONS.keys())
GATED_ACT_NAMES = list(GATED_ACTIVATIONS.keys())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_mlp_registry():
    assert MLP_REGISTRY["dense"] is DenseMLPBlock
    assert MLP_REGISTRY["moe"] is SparseMoEBlock


# ---------------------------------------------------------------------------
# DenseMLPBlock — behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gated", [False, True])
def test_dense_output_shape(gated):
    blk = DenseMLPBlock(
        d_model=64, intermediate_size=128, activation="silu", gated=gated, dropout=0.0
    )
    x = torch.randn(2, 16, 64)
    out, aux = blk(x)
    assert out.shape == (2, 16, 64)


def test_dense_returns_none_aux():
    blk = DenseMLPBlock(
        d_model=64, intermediate_size=128, activation="silu", gated=True
    )
    out, aux = blk(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64) and aux is None


@pytest.mark.parametrize("gated", [False, True])
def test_dense_default_bias_is_false(gated):
    blk = DenseMLPBlock(
        d_model=64, intermediate_size=128, activation="silu", gated=gated
    )
    w1 = blk.gate_up_proj if gated else blk.up_proj
    assert w1.bias is None
    assert blk.down_proj.bias is None


@pytest.mark.parametrize("gated", [False, True])
def test_dense_bias_true_adds_biases(gated):
    blk = DenseMLPBlock(
        d_model=64, intermediate_size=128, activation="silu", gated=gated, bias=True
    )
    w1 = blk.gate_up_proj if gated else blk.up_proj
    assert w1.bias is not None
    assert blk.down_proj.bias is not None


ACT_LIMIT_BLOCKS = [
    (DenseMLPBlock, {}),
    (
        SparseMoEBlock,
        {
            "n_routed_experts": 4,
            "n_routed_experts_per_token": 2,
            "aux_loss": True,
            "n_shared_experts": 1,
        },
    ),
]


@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("block_cls,extra", ACT_LIMIT_BLOCKS)
def test_activation_limit(block_cls, extra, gated):
    """A limit above the data is a no-op; a tight one bounds the pre-activation."""
    torch.manual_seed(0)
    common = dict(d_model=64, intermediate_size=128, gated=gated, **extra)
    # x is scaled so the projection output reliably exceeds the tight limit
    x = torch.randn(2, 16, 64) * 8
    unbounded = block_cls(**common)
    loose = block_cls(**common, activation_limit=1e6)
    tight = block_cls(**common, activation_limit=1.0)
    loose.load_state_dict(unbounded.state_dict())
    tight.load_state_dict(unbounded.state_dict())

    base = unbounded(x)[0]
    # a limit the data never reaches leaves the forward bit-identical
    assert torch.equal(base, loose(x)[0])
    out = tight(x)[0]
    assert out.shape == base.shape
    assert torch.isfinite(out).all()
    assert not torch.allclose(base, out)


@pytest.mark.parametrize("gated", [False, True])
def test_dense_activation_limit_precision(gated):
    """Clamping the projection output reproduces the block's forward exactly."""
    torch.manual_seed(0)
    blk = DenseMLPBlock(
        d_model=64, intermediate_size=128, gated=gated, activation_limit=1.0
    )
    x = torch.randn(2, 16, 64) * 8
    proj = blk.gate_up_proj if gated else blk.up_proj
    h = proj(x).clamp(-1.0, 1.0)
    ref = blk.down_proj(blk.act_fn(*h.chunk(2, dim=-1)) if gated else blk.act_fn(h))
    # same ops in the same order, so the match is exact
    assert torch.equal(blk(x)[0], ref)


# ---------------------------------------------------------------------------
# DenseMLPBlock — compute_flops
# ---------------------------------------------------------------------------


def test_dense_compute_flops_gated():
    f = DenseMLPBlock.compute_flops(64, intermediate_size=128, gated=True, bias=False)
    assert f == 6 * 64 * 128


def test_dense_compute_flops_ungated():
    f = DenseMLPBlock.compute_flops(64, intermediate_size=128, gated=False, bias=False)
    assert f == 4 * 64 * 128


def test_dense_compute_flops_gated_bias():
    f = DenseMLPBlock.compute_flops(64, intermediate_size=128, gated=True, bias=True)
    assert f == 6 * 64 * 128 + 2 * 128 + 64


# ---------------------------------------------------------------------------
# DenseMLPBlock — numerical parity vs dense_mlp_ref
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_dense_ungated_matches_ref(activation, dtype, atol):
    torch.manual_seed(0)
    blk = DenseMLPBlock(
        d_model=64,
        intermediate_size=256,
        activation=activation,
        gated=False,
        dropout=0.0,
    ).to(dtype)
    blk.eval()
    x = torch.randn(2, 16, 64, dtype=dtype)
    out, _ = blk(x)
    out_ref = dense_mlp_ref(
        x, blk.down_proj, activation=activation, up_proj=blk.up_proj
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


@pytest.mark.parametrize("activation", GATED_ACT_NAMES)
@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_dense_gated_matches_ref(activation, dtype, atol):
    torch.manual_seed(0)
    blk = DenseMLPBlock(
        d_model=64,
        intermediate_size=128,
        activation=activation,
        gated=True,
        dropout=0.0,
    ).to(dtype)
    blk.eval()
    x = torch.randn(2, 16, 64, dtype=dtype)
    out, _ = blk(x)
    out_ref = dense_mlp_ref(
        x, blk.down_proj, activation=activation, gate_up_proj=blk.gate_up_proj
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)


# ---------------------------------------------------------------------------
# MoERouter — behavior
# ---------------------------------------------------------------------------


def test_moe_router_output_shapes():
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    x = torch.randn(4 * 16, 64)
    top_indices, top_weights, router_probs = router(x)
    assert top_indices.shape == (64, 2)
    assert top_weights.shape == (64, 2)
    assert router_probs.shape == (64, 8)


def test_moe_router_indices_in_range():
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    x = torch.randn(32, 64)
    top_indices, _, _ = router(x)
    assert top_indices.min() >= 0
    assert top_indices.max() < 8


def test_moe_router_weights_normalized():
    router = MoERouter(
        d_model=64, n_routed_experts=8, n_routed_experts_per_token=2, normalize=True
    )
    x = torch.randn(32, 64)
    _, top_weights, _ = router(x)
    sums = top_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_moe_router_weights_unnormalized():
    router = MoERouter(
        d_model=64, n_routed_experts=8, n_routed_experts_per_token=2, normalize=False
    )
    x = torch.randn(32, 64)
    _, top_weights, _ = router(x)
    assert (top_weights > 0).all()
    sums = top_weights.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_moe_router_default_score_fn_is_sigmoid():
    torch.manual_seed(0)
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    x = torch.randn(32, 64)
    _, _, router_probs = router(x)
    # sigmoid scores each expert independently in (0, 1); no cross-expert normalization.
    assert (router_probs > 0).all() and (router_probs < 1).all()
    sums = router_probs.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_moe_router_sigmoid_scores_are_independent():
    torch.manual_seed(0)
    router = MoERouter(
        d_model=64,
        n_routed_experts=8,
        n_routed_experts_per_token=2,
        router_score_fn="sigmoid",
    )
    x = torch.randn(32, 64)
    _, _, router_probs = router(x)
    # sigmoid scores each expert in (0, 1) and does not normalize across experts.
    assert (router_probs > 0).all() and (router_probs < 1).all()
    sums = router_probs.sum(dim=-1)
    assert not torch.allclose(sums, torch.ones_like(sums), atol=1e-3)


def test_moe_router_sigmoid_matches_logits():
    torch.manual_seed(0)
    router = MoERouter(
        d_model=64,
        n_routed_experts=8,
        n_routed_experts_per_token=2,
        router_score_fn="sigmoid",
    )
    x = torch.randn(32, 64)
    _, _, router_probs = router(x)
    expected = router.gate(x.float()).sigmoid()
    assert torch.allclose(router_probs, expected, atol=1e-6)


def test_moe_router_sigmoid_weights_normalized():
    torch.manual_seed(0)
    router = MoERouter(
        d_model=64,
        n_routed_experts=8,
        n_routed_experts_per_token=2,
        normalize=True,
        router_score_fn="sigmoid",
    )
    x = torch.randn(32, 64)
    _, top_weights, _ = router(x)
    sums = top_weights.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_moe_router_unknown_score_fn_raises():
    with pytest.raises(KeyError):
        MoERouter(
            d_model=64,
            n_routed_experts=8,
            n_routed_experts_per_token=2,
            router_score_fn="argmax",
        )


def test_sparse_moe_block_forwards_score_fn_to_router():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=True,
        router_score_fn="sigmoid",
    )
    assert block.router.score_fn is MOE_ROUTER_SCORE_FNS["sigmoid"]


def test_moe_router_gate_weight_stays_fp32_across_dtype_casts():
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    assert router.gate.weight.dtype == torch.float32
    for cast in [
        lambda m: m.to(torch.bfloat16),
        lambda m: m.bfloat16(),
        lambda m: m.half(),
        lambda m: m.to(dtype=torch.bfloat16),
        lambda m: m.float(),
    ]:
        cast(router)
        assert router.gate.weight.dtype == torch.float32


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16])
def test_moe_router_gate_weight_loaded_as_fp32(source_dtype):
    torch.manual_seed(0)
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    state_dict = router.state_dict()
    state_dict["gate.weight"] = state_dict["gate.weight"].to(source_dtype)
    src_values = state_dict["gate.weight"].clone()
    router.load_state_dict(state_dict)
    assert router.gate.weight.dtype == torch.float32
    assert torch.equal(router.gate.weight, src_values.float())


@pytest.mark.parametrize("amp_dtype", [torch.bfloat16, torch.float16])
def test_moe_router_runs_in_fp32_under_autocast(amp_dtype):
    if not torch.cuda.is_available():
        pytest.skip("autocast fp16/bf16 path is CUDA-only in this codebase")
    torch.manual_seed(0)
    d_model, n_routed_experts, k = 512, 64, 4
    router = MoERouter(d_model, n_routed_experts, k).cuda()
    router.eval()
    x = torch.randn(256, d_model, device="cuda", dtype=amp_dtype) * 30
    top_idx_a, _, _ = router(x)
    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
        top_idx_b, _, _ = router(x)
    assert torch.equal(top_idx_a, top_idx_b)


# ---------------------------------------------------------------------------
# MoERouter — auxiliary-loss-free balancing bias (arXiv:2408.15664)
# ---------------------------------------------------------------------------


def test_moe_router_no_expert_bias_by_default():
    router = MoERouter(d_model=64, n_routed_experts=8, n_routed_experts_per_token=2)
    assert router.expert_bias is None


def test_moe_router_expert_bias_module_created():
    router = MoERouter(
        d_model=64, n_routed_experts=8, n_routed_experts_per_token=2, expert_bias=True
    )
    assert isinstance(router.expert_bias, ExpertBias)
    assert router.expert_bias.bias.shape == (8,)
    assert router.expert_bias.bias.dtype == torch.float32
    assert torch.count_nonzero(router.expert_bias.bias) == 0


def test_moe_router_bias_drives_selection_not_combine_weights():
    # normalize=False so returned weights are the raw gathered probs.
    router = MoERouter(
        d_model=64,
        n_routed_experts=8,
        n_routed_experts_per_token=2,
        normalize=False,
        expert_bias=True,
    )
    router.expert_bias.bias[3] = 100.0  # force expert 3 into every token's top-k
    x = torch.randn(32, 64)
    top_indices, top_weights, router_probs = router(x)
    # expert 3 selected for every token
    assert (top_indices == 3).any(dim=-1).all()
    # combine weights are the ORIGINAL probs (bias must not leak in)
    expected = router_probs.gather(-1, top_indices)
    assert torch.allclose(top_weights, expected)


def test_moe_router_bias_stays_fp32_across_dtype_casts():
    router = MoERouter(
        d_model=64, n_routed_experts=8, n_routed_experts_per_token=2, expert_bias=True
    )
    for cast in [
        lambda m: m.to(torch.bfloat16),
        lambda m: m.half(),
        lambda m: m.float(),
    ]:
        cast(router)
        assert router.expert_bias.bias.dtype == torch.float32


def test_expert_bias_compute_cost():
    assert ExpertBias.compute_flops(8) == 8
    # The bias is a non-trainable buffer, not an nn.Parameter, so it's excluded
    # from the parameter count (matches sum(p.numel()) on the live module).
    assert ExpertBias.compute_parameters(8) == 0


def test_moe_router_compute_cost_with_expert_bias():
    base_f = MoERouter.compute_flops(64, 8)
    base_p = MoERouter.compute_parameters(64, 8)
    assert base_f == 2 * 64 * 8
    assert base_p == 64 * 8
    assert MoERouter.compute_flops(64, 8, expert_bias=True) == base_f + 8
    # expert_bias adds a real FLOP (the bias add) but no trainable parameter.
    assert MoERouter.compute_parameters(64, 8, expert_bias=True) == base_p


def test_expert_bias_update_moves_toward_balance():
    eb = ExpertBias(n_experts=4, update_rate=0.01)
    counts = torch.tensor([10, 0, 5, 5])  # mean 5: expert 0 overloaded, 1 underloaded
    eb.update(counts)
    assert eb.bias[0].item() == pytest.approx(-0.01)  # overloaded -> down
    assert eb.bias[1].item() == pytest.approx(+0.01)  # underloaded -> up
    assert eb.bias[2].item() == 0.0 and eb.bias[3].item() == 0.0  # balanced


# ---------------------------------------------------------------------------
# SparseMoEBlock — behavior
# ---------------------------------------------------------------------------


def test_sparse_moe_block_output_shape():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    assert out.shape == (2, 8, 64)


def test_sparse_moe_block_aux_loss_is_scalar_and_nonneg():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    assert aux_loss.ndim == 0
    assert aux_loss.item() >= 0.0


def test_sparse_moe_sigmoid_aux_loss_normalized():
    """Sigmoid scores are unnormalized (each in (0,1), summing to ~E/2), so the
    Switch balance loss must normalize per-token probs before computing P.
    Without it, aux ~ E*k/2 at init and the router can minimize it by shrinking
    all probs (aux -> 0, logged value goes negative) instead of balancing load.
    With normalization the loss is ~k, matching softmax."""
    torch.manual_seed(0)
    k, E = 8, 64
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        router_score_fn="sigmoid",
    )
    x = torch.randn(8, 32, 64)
    _, aux_loss = block(x)
    # Normalized balance loss is ~k at init; unnormalized would be ~E*k/2 = 256.
    assert aux_loss.item() < 4 * k


def test_sparse_moe_block_aux_loss_has_grad():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    aux_loss.backward()
    assert block.router.gate.weight.grad is not None


def test_expert_load_record_and_reset():
    load = ExpertLoad(n_experts=4)
    counts = torch.tensor([1, 2, 3, 4])
    # training routes into train_load (accumulates), leaving eval_load untouched.
    load.record_load(counts, training=True)
    load.record_load(counts, training=True)
    assert load.train_load.tolist() == [2, 4, 6, 8]
    assert load.eval_load.tolist() == [0, 0, 0, 0]
    # eval overwrites eval_load with the current batch, leaving train_load.
    load.record_load(torch.tensor([5, 6, 7, 8]), training=False)
    assert load.eval_load.tolist() == [5, 6, 7, 8]
    assert load.train_load.tolist() == [2, 4, 6, 8]
    load.reset_train_load()
    load.reset_eval_load()
    assert load.train_load.tolist() == [0, 0, 0, 0]
    assert load.eval_load.tolist() == [0, 0, 0, 0]


def test_sparse_moe_block_records_expert_load():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    block.train()
    x = torch.randn(2, 8, 64)  # T=16 tokens, k=2 -> 32 routings
    block(x)
    load = block.expert_load.train_load
    assert load.shape == (4,)
    assert load.sum().item() == 16 * 2
    assert not load.requires_grad
    # Buffer is non-persistent (excluded from state_dict).
    assert "expert_load.train_load" not in block.state_dict()


def test_sparse_moe_block_aux_loss_coef_stored():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss_coef=0.05,
    )
    assert block.aux_loss_coef == 0.05


def test_moe_block_scales_aux_by_its_coef():
    torch.manual_seed(0)
    common = dict(
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=True,
        expert_bias=False,
    )
    x = torch.randn(2, 8, 64)
    b1 = SparseMoEBlock(64, aux_loss_coef=1e-3, **common)
    b2 = SparseMoEBlock(64, aux_loss_coef=1e-2, **common)
    b2.load_state_dict(b1.state_dict())  # identical weights -> identical unscaled aux
    _, a1 = b1(x)
    _, a2 = b2(x)
    assert torch.allclose(a2, a1 * 10, rtol=1e-4)


def test_sparse_moe_block_aux_loss_false_returns_none():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=False,
    )
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    assert aux_loss is None


def test_sparse_moe_block_expert_bias_returns_no_aux_loss():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=False,
        expert_bias=True,
    )
    x = torch.randn(2, 8, 64)
    _, aux_loss = block(x)
    assert aux_loss is None


def test_sparse_moe_block_aux_loss_and_expert_bias_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        SparseMoEBlock(
            d_model=64,
            intermediate_size=128,
            n_routed_experts=4,
            n_routed_experts_per_token=2,
            aux_loss=True,
            expert_bias=True,
        )


def test_sparse_moe_block_expert_bias_updates_in_train_only():
    torch.manual_seed(0)
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=False,
        expert_bias=True,
    )
    x = torch.randn(4, 16, 64)

    block.eval()
    block(x)
    block.post_step()
    assert (
        torch.count_nonzero(block.router.expert_bias.bias) == 0
    )  # eval forward doesn't accumulate load, so post_step is a no-op for the bias

    block.train()
    block(x)
    assert (
        torch.count_nonzero(block.router.expert_bias.bias) == 0
    )  # forward only accumulates load; bias unchanged until post_step
    block.post_step()
    assert (
        torch.count_nonzero(block.router.expert_bias.bias) > 0
    )  # imbalanced load moves bias at the step boundary


def test_sparse_moe_block_post_step_accumulates_across_microbatches():
    torch.manual_seed(0)
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        aux_loss=False,
        expert_bias=True,
    )
    block.train()
    x = torch.randn(4, 16, 64)  # 64 tokens, k=2 -> 128 routings per micro-batch

    block(x)
    block(x)  # two micro-batches accumulate before the step boundary
    assert block.expert_load.train_load.sum().item() == 2 * 64 * 2

    block.post_step()
    # post_step consumes the accumulated load (bias update) and resets it.
    assert block.expert_load.train_load.sum().item() == 0
    assert "expert_load.train_load" not in block.state_dict()  # non-persistent


# ---------------------------------------------------------------------------
# SparseMoEBlock — shared experts (DeepSeekMoE)
# ---------------------------------------------------------------------------


def test_sparse_moe_no_shared_experts_by_default():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    assert block.shared_expert is None


@pytest.mark.parametrize("n_shared_experts", [1, 2])
@pytest.mark.parametrize("gated", [True, False])
def test_sparse_moe_shared_expert_width_and_shape(n_shared_experts, gated):
    inter = 128
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=inter,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        n_shared_experts=n_shared_experts,
        gated=gated,
    )
    assert block.shared_expert is not None
    w1 = block.shared_expert.gate_up_proj if gated else block.shared_expert.up_proj
    expected = (2 if gated else 1) * n_shared_experts * inter
    assert w1.weight.shape[0] == expected
    out, aux = block(torch.randn(2, 8, 64))
    assert out.shape == (2, 8, 64)
    assert aux.ndim == 0


def test_sparse_moe_shared_expert_adds_to_routed_output():
    """Output with shared experts = routed output + shared FFN(x)."""
    torch.manual_seed(0)
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        n_shared_experts=2,
        dropout=0.0,
    )
    block.eval()
    x = torch.randn(2, 8, 64)
    out, _ = block(x)
    shared_out, _ = block.shared_expert(x)
    block.shared_expert = None
    routed_out, _ = block(x)
    assert torch.allclose(out, routed_out + shared_out, atol=1e-5)


def test_sparse_moe_shared_expert_in_param_count():
    kwargs = dict(
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
    )
    base = SparseMoEBlock.compute_parameters(64, **kwargs)
    with_shared = SparseMoEBlock.compute_parameters(64, n_shared_experts=2, **kwargs)
    dense = DenseMLPBlock.compute_parameters(64, intermediate_size=2 * 128, gated=True)
    assert with_shared - base == dense
    # Shared experts count in active params too (always run).
    base_active = SparseMoEBlock.compute_parameters(64, active=True, **kwargs)
    shared_active = SparseMoEBlock.compute_parameters(
        64, n_shared_experts=2, active=True, **kwargs
    )
    assert shared_active - base_active == dense


def test_sparse_moe_shared_expert_in_flops():
    kwargs = dict(
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
    )
    base = SparseMoEBlock.compute_flops(64, **kwargs)
    with_shared = SparseMoEBlock.compute_flops(64, n_shared_experts=2, **kwargs)
    dense = DenseMLPBlock.compute_flops(64, intermediate_size=2 * 128, gated=True)
    assert with_shared - base == dense


# ---------------------------------------------------------------------------
# SparseMoEBlock — compute_flops
# ---------------------------------------------------------------------------


def test_sparse_moe_compute_flops_gated():
    f = SparseMoEBlock.compute_flops(
        64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
        bias=False,
    )
    router = 2 * 64 * 4
    expert = 6 * 64 * 128
    assert f == router + 2 * expert


def test_sparse_moe_compute_flops_ungated():
    f = SparseMoEBlock.compute_flops(
        64,
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=False,
        bias=False,
    )
    router = 2 * 64 * 4
    expert = 4 * 64 * 128
    assert f == router + 2 * expert


def test_sparse_moe_compute_flops_expert_bias():
    kwargs = dict(
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
    )
    base = SparseMoEBlock.compute_flops(64, **kwargs)
    with_bias = SparseMoEBlock.compute_flops(64, expert_bias=True, **kwargs)
    assert with_bias == base + 4  # +n_routed_experts for the pre-top-k bias add


def test_sparse_moe_compute_parameters_expert_bias():
    kwargs = dict(
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
    )
    base = SparseMoEBlock.compute_parameters(64, **kwargs)
    with_bias = SparseMoEBlock.compute_parameters(64, expert_bias=True, **kwargs)
    # expert_bias adds no trainable parameter (its bias is a non-trainable buffer).
    assert with_bias == base
    # active count is likewise unaffected by expert_bias
    base_active = SparseMoEBlock.compute_parameters(64, active=True, **kwargs)
    with_bias_active = SparseMoEBlock.compute_parameters(
        64, expert_bias=True, active=True, **kwargs
    )
    assert with_bias_active == base_active


def test_sparse_moe_compute_parameters_expert_bias_matches_live_module():
    """Regression: analytic compute_parameters must equal the live module's
    sum(p.numel()) for an expert_bias MoE block (the ExpertBias bias buffer is
    excluded from both, since it's registered as a buffer, not a parameter)."""
    kwargs = dict(
        intermediate_size=128,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        gated=True,
        expert_bias=True,
        aux_loss=False,
    )
    block = SparseMoEBlock(64, **kwargs)
    live = sum(p.numel() for p in block.parameters())
    analytic = SparseMoEBlock.compute_parameters(64, **kwargs)
    assert analytic == live


# ---------------------------------------------------------------------------
# MoERouter — numerical parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_moe_router_matches_ref(normalize, dtype, atol):
    torch.manual_seed(0)
    d_model, n_routed_experts, k = 64, 8, 2
    router = MoERouter(
        d_model, n_routed_experts, k, normalize=normalize, router_score_fn="softmax"
    ).to(dtype)
    router.eval()
    x = torch.randn(32, d_model, dtype=dtype)
    top_idx, top_w, probs = router(x)
    top_idx_ref, top_w_ref, probs_ref = moe_router_ref(
        x, router.gate.weight, k, normalize=normalize
    )
    assert torch.equal(top_idx, top_idx_ref)
    assert torch.allclose(top_w, top_w_ref, atol=atol)
    assert torch.allclose(probs, probs_ref, atol=atol)


@pytest.mark.parametrize("dtype,atol", SIMPLE_DTYPES)
def test_moe_router_matches_ref_under_saturation(dtype, atol):
    torch.manual_seed(0)
    d_model, n_routed_experts, k = 512, 64, 4
    router = MoERouter(d_model, n_routed_experts, k, router_score_fn="softmax").to(
        dtype
    )
    router.eval()
    x = torch.randn(256, d_model, dtype=dtype) * 30
    top_idx, top_w, probs = router(x)
    top_idx_ref, top_w_ref, probs_ref = moe_router_ref(
        x, router.gate.weight, k, normalize=True
    )
    assert torch.equal(top_idx, top_idx_ref)
    assert torch.allclose(top_w, top_w_ref, atol=atol)
    assert torch.allclose(probs, probs_ref, atol=atol)


# ---------------------------------------------------------------------------
# SparseMoEBlock — numerical parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gated,activation",
    [
        (True, "silu"),
        (False, "gelu"),
        (False, "relu"),
    ],
)
@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
def test_sparse_moe_block_matches_ref(gated, activation, dtype, atol):
    torch.manual_seed(0)
    d_model, inter, n_routed_experts, k = 64, 32, 4, 2
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=n_routed_experts,
        n_routed_experts_per_token=k,
        dropout=0.0,
        gated=gated,
        activation=activation,
        router_score_fn="softmax",
        aux_loss_coef=1.0,  # ref returns unscaled aux; coef=1.0 keeps parity
    )
    with torch.no_grad():
        w1 = block.expert_gate_up if gated else block.expert_up
        torch.nn.init.normal_(w1, std=0.02)
        torch.nn.init.normal_(block.expert_down, std=0.02)
    block.to(dtype)
    block.eval()

    x = torch.randn(2, 8, d_model, dtype=dtype)
    out, aux = block(x)
    out_ref, aux_ref = sparse_moe_block_ref(
        x,
        block.router.gate.weight,
        block.expert_down,
        n_routed_experts_per_token=k,
        activation=activation,
        normalize=True,
        expert_gate_up=block.expert_gate_up if gated else None,
        expert_up=None if gated else block.expert_up,
    )

    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)
    assert torch.allclose(aux, aux_ref, atol=atol)


def test_sparse_moe_block_matches_hf_qwen3_moe():
    """SparseMoEBlock output matches HF Qwen3MoeSparseMoeBlock with copied weights."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeConfig,
        Qwen3MoeSparseMoeBlock,
    )

    torch.manual_seed(0)
    d_model, inter, n_routed_experts, top_k = 64, 32, 4, 2

    ours = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=n_routed_experts,
        n_routed_experts_per_token=top_k,
        dropout=0.0,
        router_score_fn="softmax",
    )
    with torch.no_grad():
        torch.nn.init.normal_(ours.expert_gate_up, std=0.02)
        torch.nn.init.normal_(ours.expert_down, std=0.02)
    ours.eval()

    hf_cfg = Qwen3MoeConfig(
        hidden_size=d_model,
        moe_intermediate_size=inter,
        num_experts=n_routed_experts,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
    )
    hf = Qwen3MoeSparseMoeBlock(hf_cfg)
    hf.eval()

    with torch.no_grad():
        hf.gate.weight.copy_(ours.router.gate.weight)
        hf.experts.gate_up_proj.copy_(ours.expert_gate_up)
        hf.experts.down_proj.copy_(ours.expert_down)

    x = torch.randn(2, 8, d_model)
    our_out, _ = ours(x)
    hf_out = hf(x)

    assert torch.allclose(our_out, hf_out, atol=1e-5)


# ---------------------------------------------------------------------------
# grouped_mlp — numerical parity vs per-group loop
# ---------------------------------------------------------------------------


# counts cover an empty group in interior, leading, and trailing positions.
@pytest.mark.parametrize("counts", [[5, 0, 7, 4], [0, 5, 7, 4], [5, 7, 4, 0]])
@pytest.mark.parametrize("gated,activation", [(True, "silu"), (False, "gelu")])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
def test_grouped_mlp_matches_per_group_loop(
    gated, activation, use_bias, dtype, atol, counts
):
    torch.manual_seed(0)
    E, D, inter = 4, 16, 32
    R = sum(counts)
    act = (GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS)[activation]

    x = torch.randn(R, D, dtype=dtype)
    out_dim = 2 * inter if gated else inter
    w_in = torch.randn(E, out_dim, D, dtype=dtype) * 0.1
    w_down = torch.randn(E, D, inter, dtype=dtype) * 0.1
    b_in = torch.randn(E, out_dim, dtype=dtype) * 0.1 if use_bias else None
    b_down = torch.randn(E, D, dtype=dtype) * 0.1 if use_bias else None

    offs = torch.tensor(counts).cumsum(0).to(torch.int32)

    got = grouped_mlp(
        x,
        w_in,
        w_down,
        act,
        offs,
        gated,
        b_in=b_in,
        b_down=b_down,
    )

    # Oracle: run each group through the existing 2D fused op.
    ref = torch.empty_like(got)
    start = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        xs = x[start : start + c]
        h = torch.addmm(b_in[e], xs, w_in[e].mT) if use_bias else xs @ w_in[e].mT
        if gated:
            gate, up = h.chunk(2, dim=-1)
            h = act(gate, up)
        else:
            h = act(h)
        out = torch.addmm(b_down[e], h, w_down[e].mT) if use_bias else h @ w_down[e].mT
        ref[start : start + c] = out
        start += c

    assert got.dtype == dtype
    assert got.shape == (R, D)
    assert torch.allclose(got, ref, atol=atol)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped_mm compile is CUDA+bf16 only"
)
@pytest.mark.parametrize("gated,activation", [(True, "silu"), (False, "gelu")])
def test_grouped_mlp_compiled_matches_eager_cuda_bf16(gated, activation):
    torch.manual_seed(0)
    E, D, inter = 4, 16, 32
    counts = [5, 0, 7, 4]  # includes an empty group
    R = sum(counts)
    act = (GATED_ACTIVATIONS if gated else UNGATED_ACTIVATIONS)[activation]
    x = torch.randn(R, D, device="cuda", dtype=torch.bfloat16)
    out_dim = 2 * inter if gated else inter
    w_in = torch.randn(E, out_dim, D, device="cuda", dtype=torch.bfloat16) * 0.1
    w_down = torch.randn(E, D, inter, device="cuda", dtype=torch.bfloat16) * 0.1
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    eager = grouped_mlp(x, w_in, w_down, act, offs, gated)
    compiled = torch.compile(grouped_mlp)(x, w_in, w_down, act, offs, gated)
    assert compiled.dtype == torch.bfloat16
    assert torch.allclose(compiled, eager, atol=1e-2)


@pytest.mark.parametrize("gated", [True, False])
def test_sparse_moe_dropless_handles_empty_expert_and_bias(gated):
    torch.manual_seed(0)
    d_model, inter, E, k = 32, 16, 4, 2
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        gated=gated,
        activation="silu" if gated else "gelu",
        bias=True,
        router_score_fn="softmax",
        aux_loss_coef=1.0,  # ref returns unscaled aux; coef=1.0 keeps parity
    )
    with torch.no_grad():
        w1 = block.expert_gate_up if gated else block.expert_up
        torch.nn.init.normal_(w1, std=0.02)
        torch.nn.init.normal_(block.expert_down, std=0.02)
        torch.nn.init.normal_(
            block.expert_gate_up_bias if gated else block.expert_up_bias, std=0.02
        )
        torch.nn.init.normal_(block.expert_down_bias, std=0.02)
        # expert 3 gets logit = gate_weight[3] · x. Using all-negative weight row and
        # all-positive x (abs) ensures logit_3 << 0 reliably → expert 3 never in top-k.
        block.router.gate.weight.data[3] = -1e4
    block.eval()

    # All-positive x guarantees logit_3 = -1e4 * sum(|x_features|) << other logits.
    x = torch.randn(2, 8, d_model).abs()
    out, aux = block(x)

    out_ref, aux_ref = sparse_moe_block_ref(
        x,
        block.router.gate.weight,
        block.expert_down,
        n_routed_experts_per_token=k,
        activation="silu" if gated else "gelu",
        normalize=True,
        expert_gate_up=block.expert_gate_up if gated else None,
        expert_up=None if gated else block.expert_up,
        expert_gate_up_bias=block.expert_gate_up_bias if gated else None,
        expert_up_bias=None if gated else block.expert_up_bias,
        expert_down_bias=block.expert_down_bias,
    )
    assert torch.allclose(out, out_ref, atol=1e-4)
    assert torch.allclose(aux, aux_ref, atol=1e-4)


def test_grouped_mlp_casts_to_autocast_dtype():
    # Tensors created on CPU explicitly so the test is device-agnostic
    # (conftest sets default device to cuda when available).
    torch.manual_seed(0)
    E, D, inter = 4, 16, 32
    counts = [5, 0, 7, 4]
    R = sum(counts)
    act = GATED_ACTIVATIONS["silu"]
    x = torch.randn(R, D, device="cpu")  # fp32 on CPU
    w_in = torch.randn(E, 2 * inter, D, device="cpu") * 0.1  # fp32 on CPU
    w_down = torch.randn(E, D, inter, device="cpu") * 0.1  # fp32 on CPU
    offs = torch.tensor(counts, device="cpu").cumsum(0).to(torch.int32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = grouped_mlp(x, w_in, w_down, act, offs, True)
    assert out.dtype == torch.bfloat16
    # eager (no autocast) preserves fp32
    out_eager = grouped_mlp(x, w_in, w_down, act, offs, True)
    assert out_eager.dtype == torch.float32


def test_sparse_moe_latent_off_by_default():
    block = SparseMoEBlock(
        d_model=64,
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
    )
    assert block.latent_moe is False
    assert block.latent_down_proj is None
    assert block.latent_up_proj is None
    # expert I/O axis is d_model when latent is off
    assert block.expert_down.shape == (4, 64, 32)
    assert block.expert_gate_up.shape == (4, 2 * 32, 64)


def test_sparse_moe_latent_weight_shapes():
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    assert block.latent_down_proj.weight.shape == (ell, d_model)
    assert block.latent_up_proj.weight.shape == (d_model, ell)
    assert block.expert_gate_up.shape == (E, 2 * inter, ell)
    assert block.expert_down.shape == (E, ell, inter)
    # router still scores at d_model
    assert block.router.gate.weight.shape == (E, d_model)


def test_sparse_moe_latent_output_shape():
    d_model, ell = 64, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=32,
        n_routed_experts=4,
        n_routed_experts_per_token=2,
        latent_moe=True,
        latent_dim=ell,
    )
    block.eval()
    x = torch.randn(2, 8, d_model)
    out, aux = block(x)
    assert out.shape == (2, 8, d_model)
    assert aux.ndim == 0


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("gated,activation", [(True, "silu"), (False, "gelu")])
@pytest.mark.parametrize("dtype,atol", COMPOUND_DTYPES)
def test_sparse_moe_latent_matches_ref(gated, activation, dtype, atol, bias):
    torch.manual_seed(0)
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    aux_loss_coef = 0.05
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        dropout=0.0,
        gated=gated,
        activation=activation,
        router_score_fn="softmax",
        aux_loss_coef=aux_loss_coef,
        bias=bias,
        latent_moe=True,
        latent_dim=ell,
    )
    with torch.no_grad():
        w1 = block.expert_gate_up if gated else block.expert_up
        torch.nn.init.normal_(w1, std=0.02)
        torch.nn.init.normal_(block.expert_down, std=0.02)
        torch.nn.init.normal_(block.latent_down_proj.weight, std=0.02)
        torch.nn.init.normal_(block.latent_up_proj.weight, std=0.02)
        if bias:
            b1 = block.expert_gate_up_bias if gated else block.expert_up_bias
            torch.nn.init.normal_(b1, std=0.02)
            torch.nn.init.normal_(block.expert_down_bias, std=0.02)
    block.to(dtype)
    block.eval()

    x = torch.randn(2, 8, d_model, dtype=dtype)
    out, aux = block(x)
    out_ref, aux_ref = sparse_moe_block_ref(
        x,
        block.router.gate.weight,
        block.expert_down,
        n_routed_experts_per_token=k,
        activation=activation,
        normalize=True,
        expert_gate_up=block.expert_gate_up if gated else None,
        expert_up=None if gated else block.expert_up,
        expert_gate_up_bias=(block.expert_gate_up_bias if gated and bias else None),
        expert_up_bias=(block.expert_up_bias if not gated and bias else None),
        expert_down_bias=block.expert_down_bias if bias else None,
        latent_down_weight=block.latent_down_proj.weight,
        latent_up_weight=block.latent_up_proj.weight,
        aux_loss_coef=aux_loss_coef,
    )
    assert out.dtype == dtype
    assert torch.allclose(out, out_ref, atol=atol)
    assert torch.allclose(aux, aux_ref, atol=atol)


def test_sparse_moe_latent_with_shared_experts_matches_ref_plus_shared():
    """LatentMoE + shared experts: shared experts run at d_model and are not
    modeled by sparse_moe_block_ref, so verify the coupling directly — total
    output must equal the ref's routed-only output plus the shared expert's
    own output on the same input."""
    torch.manual_seed(0)
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    aux_loss_coef = 0.05
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        n_shared_experts=1,
        dropout=0.0,
        gated=True,
        activation="silu",
        router_score_fn="softmax",
        aux_loss_coef=aux_loss_coef,
        latent_moe=True,
        latent_dim=ell,
    )
    with torch.no_grad():
        torch.nn.init.normal_(block.expert_gate_up, std=0.02)
        torch.nn.init.normal_(block.expert_down, std=0.02)
        torch.nn.init.normal_(block.latent_down_proj.weight, std=0.02)
        torch.nn.init.normal_(block.latent_up_proj.weight, std=0.02)
    block.eval()

    x = torch.randn(2, 8, d_model)
    out, aux = block(x)

    routed_ref, aux_ref = sparse_moe_block_ref(
        x,
        block.router.gate.weight,
        block.expert_down,
        n_routed_experts_per_token=k,
        activation="silu",
        normalize=True,
        expert_gate_up=block.expert_gate_up,
        latent_down_weight=block.latent_down_proj.weight,
        latent_up_weight=block.latent_up_proj.weight,
        aux_loss_coef=aux_loss_coef,
    )
    shared_out, _ = block.shared_expert(x)

    assert torch.allclose(out, routed_ref + shared_out, atol=1e-5)
    assert torch.allclose(aux, aux_ref, atol=1e-5)


# ---------------------------------------------------------------------------
# SparseMoEBlock — latent cost accounting
# ---------------------------------------------------------------------------


def test_sparse_moe_latent_param_count_matches_module():
    d_model, inter, E, k, ell = 64, 32, 4, 2, 16
    block = SparseMoEBlock(
        d_model=d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    counted = SparseMoEBlock.compute_parameters(
        d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    actual = sum(p.numel() for p in block.parameters())
    assert counted == actual


def test_sparse_moe_latent_compute_flops_gated():
    d_model, inter, E, k, ell = 64, 32, 8, 2, 16
    f = SparseMoEBlock.compute_flops(
        d_model,
        intermediate_size=inter,
        n_routed_experts=E,
        n_routed_experts_per_token=k,
        latent_moe=True,
        latent_dim=ell,
    )
    router = 2 * d_model * E
    expert = 6 * ell * inter  # expert I/O axis is ℓ, not d_model
    proj = 4 * d_model * ell  # W↓ + W↑, once per token
    assert f == router + k * expert + proj


def test_sparse_moe_latent_flops_less_than_dense_moe():
    kwargs = dict(
        intermediate_size=128, n_routed_experts=8, n_routed_experts_per_token=2
    )
    dense = SparseMoEBlock.compute_flops(256, **kwargs)
    latent = SparseMoEBlock.compute_flops(256, latent_moe=True, latent_dim=64, **kwargs)
    assert latent < dense


# ---------------------------------------------------------------------------
# SparseMoEBlock — compiled auto-path smoke test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped GEMM is CUDA only")
@pytest.mark.parametrize("gated", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_sparse_moe_block_compiles_fullgraph(gated, dtype):
    torch.manual_seed(0)
    blk = (
        SparseMoEBlock(
            d_model=64,
            intermediate_size=32,
            n_routed_experts=4,
            n_routed_experts_per_token=2,
            gated=gated,
            activation="silu" if gated else "gelu",
            router_score_fn="softmax",
            aux_loss=False,
            expert_bias=True,
        )
        .cuda()
        .to(dtype)
    )
    for p in blk.parameters():
        if p.ndim >= 2:
            torch.nn.init.normal_(p, std=0.02)
    x = torch.randn(2, 8, 64, device="cuda", dtype=dtype, requires_grad=True)
    compiled = torch.compile(blk, fullgraph=True)
    out, _ = compiled(x)
    out.float().sum().backward()
    assert out.shape == x.shape and x.grad is not None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
def test_moe_expert_mm_seam_is_pluggable():
    torch.manual_seed(0)
    blk = (
        SparseMoEBlock(
            d_model=32,
            intermediate_size=48,
            n_routed_experts=4,
            n_routed_experts_per_token=2,
        )
        .cuda()
        .bfloat16()
    )
    # default seam is the bf16 grouped GEMM wrapped for autograd
    assert blk.expert_mm.__func__ is SparseMoEBlock.expert_mm

    seen = []

    def spy(a, b, offs, bias=None, projection=None):
        seen.append(projection)
        return grouped_mm(a, b, offs, bias=bias)

    blk.expert_mm = spy
    x = torch.randn(2, 8, 32, device="cuda", dtype=torch.bfloat16)
    out, _ = blk(x)
    assert out.shape == x.shape
    # one seam, called once per projection, each tagged so a swapped-in
    # implementation can tell the block input from the post-activation hidden state
    assert seen == ["gate_up", "down"]


_GG_LAYOUTS = ("ragged_m", "ragged_k", "ragged_n")


def _make_grouped_layout(layout, counts=(64, 0, 130, 46), M=32, K=64, N=48, seed=0):
    """Build operands for one grouped ragged layout."""
    torch.manual_seed(seed)
    G, R = len(counts), sum(counts)
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)

    def rand(dim0, dim1, dim2=None):
        shape = (dim0, dim1) if dim2 is None else (dim0, dim1, dim2)
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.1

    if layout == "ragged_m":  # (R,K) x (G,K,N) -> (R,N)
        return rand(R, K), rand(G, N, K).mT, offs
    if layout == "ragged_k":  # (M,R) x (R,N) -> (G,M,N)
        return rand(R, M).mT, rand(R, N), offs
    return rand(G, M, K), rand(K, R), offs  # ragged_n: (G,M,K) x (K,R) -> (M,R)


def _run_grouped_mm_apply(a0, b0, offs, grad, backend, include_backend):
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    if include_backend:
        out = GroupedGemmFn.apply(a, b, None, offs, backend)
    else:
        out = GroupedGemmFn.apply(a, b, None, offs)
    out.backward(grad)
    return out, a.grad, b.grad


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
@pytest.mark.parametrize(
    ("backend", "include_backend"),
    [(None, False), ("triton", True)],
    ids=["four_inputs", "forced_backend"],
)
def test_grouped_mm_fn_apply_backward_arity(backend, include_backend):
    a0, b0, offs = _make_grouped_layout("ragged_m")
    grad = torch.randn(a0.shape[0], b0.shape[-1], device="cuda", dtype=torch.bfloat16)

    out, grad_a, grad_b = _run_grouped_mm_apply(
        a0, b0, offs, grad, backend, include_backend
    )

    assert torch.isfinite(out).all()
    assert torch.isfinite(grad_a).all()
    assert torch.isfinite(grad_b).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
def test_grouped_mm_fn_explicit_none_backend_matches_omitted_backend():
    a0, b0, offs = _make_grouped_layout("ragged_m")
    grad = torch.randn(a0.shape[0], b0.shape[-1], device="cuda", dtype=torch.bfloat16)

    omitted = _run_grouped_mm_apply(a0, b0, offs, grad, None, False)
    explicit_none = _run_grouped_mm_apply(a0, b0, offs, grad, None, True)

    for got, expected in zip(explicit_none, omitted):
        torch.testing.assert_close(got, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
@pytest.mark.parametrize("layout", _GG_LAYOUTS)
def test_grouped_mm_fn_grads_match_torch(layout):
    """Each layout's dgrad/wgrad lands in another layout of the same set, so
    GroupedGemmFn is only correct once all three are."""
    a0, b0, offs = _make_grouped_layout(layout)

    def run(fn):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = fn(a, b)
        (out * out).sum().backward()
        return out, a.grad, b.grad

    ref = run(lambda a, b: grouped_mm_eager(a, b, offs))
    got = run(lambda a, b: GroupedGemmFn.apply(a, b, None, offs))
    for g, r in zip(got, ref):
        torch.testing.assert_close(g.float(), r.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
@pytest.mark.parametrize("layout", ("ragged_m", "ragged_k"))
def test_grouped_mm_fn_bias_grads_match_eager(layout):
    a0, b0, offs = _make_grouped_layout(layout)
    G, N = offs.shape[0], 48
    bias0 = torch.randn(G, N, device="cuda", dtype=torch.bfloat16)
    grad_out = torch.randn_like(grouped_mm_eager(a0, b0, offs, bias0))

    def run(fn):
        a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
        bias = bias0.clone().requires_grad_(True)
        out = fn(a, b, bias)
        out.backward(grad_out)
        return out, a.grad, b.grad, bias.grad

    ref = run(lambda a, b, bias: grouped_mm_eager(a, b, offs, bias))
    got = run(lambda a, b, bias: GroupedGemmFn.apply(a, b, bias, offs))
    for index, (g, r) in enumerate(zip(got, ref)):
        if layout == "ragged_m" and index == 3:
            torch.testing.assert_close(g.float(), r.float(), rtol=8e-2, atol=1.0)
        else:
            torch.testing.assert_close(g.float(), r.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="grouped GEMM is CUDA+bf16 only"
)
def test_grouped_mm_fn_compiles_fullgraph_and_matches_eager():
    # Forward compiles with no graph break; the Function's backward runs eager
    # (Dynamo cannot trace the autograd engine under fullgraph), so we compile
    # forward and call backward outside the traced region, matching eager.
    torch.manual_seed(0)
    counts = [64, 0, 130, 41]
    R, K, N = sum(counts), 64, 48
    offs = torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)
    a0 = torch.randn(R, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(len(counts), N, K, device="cuda", dtype=torch.bfloat16) * 0.1

    def fwd(a, b):
        return GroupedGemmFn.apply(a, b, None, offs)

    def run(f):
        a = a0.clone().requires_grad_(True)
        wv = w.clone().requires_grad_(True)
        c = f(a, wv.mT)
        c.sum().backward()
        return c, a.grad, wv.grad

    eager = run(fwd)
    comp = run(torch.compile(fwd, fullgraph=True))
    for e, c in zip(eager, comp):
        torch.testing.assert_close(c.float(), e.float(), rtol=2e-2, atol=2e-2)
