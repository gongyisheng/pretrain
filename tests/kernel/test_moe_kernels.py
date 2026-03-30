"""Tests for MoE kernel implementations (scatter and expert FFN)."""
import pytest
import torch
import torch.nn.functional as F

from src.kernel.torch.moe_ffn import torch_moe_expert_ffn
from src.kernel.torch.moe_scatter import torch_moe_scatter_in, torch_moe_scatter_out
from src.kernel.triton.moe_ffn import triton_moe_expert_ffn
from src.kernel.triton.moe_scatter import triton_moe_scatter_in, triton_moe_scatter_out

torch.manual_seed(42)


# --- Fixtures ---

@pytest.fixture
def routing_data():
    """Generate realistic MoE routing data."""
    torch.manual_seed(42)
    T, D, E, k = 256, 128, 8, 2
    x_flat = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

    top_indices = torch.randint(0, E, (T, k), device="cuda")
    flat_expert_ids = top_indices.reshape(-1)
    flat_token_ids = torch.arange(T, device="cuda").unsqueeze(1).expand(T, k).reshape(-1)

    sorted_expert_ids, sorted_order = flat_expert_ids.sort(stable=True)
    sorted_token_ids = flat_token_ids[sorted_order]

    expert_counts = torch.bincount(sorted_expert_ids.long(), minlength=E)
    capacity = int(T * k * 1.5 / E)
    offsets = torch.zeros(E, dtype=torch.long, device="cuda")
    offsets[1:] = expert_counts[:-1].cumsum(0)
    positions = torch.arange(T * k, device="cuda") - offsets[sorted_expert_ids]
    keep_mask = positions < capacity
    se = sorted_expert_ids[keep_mask]
    st = sorted_token_ids[keep_mask]
    pos = positions[keep_mask]
    weights = torch.rand(se.shape[0], device="cuda", dtype=torch.bfloat16)

    return {
        "x_flat": x_flat, "T": T, "D": D, "E": E, "capacity": capacity,
        "expert_ids": se, "token_ids": st, "positions": pos, "weights": weights,
    }


@pytest.fixture
def expert_ffn_data():
    """Generate expert FFN input data."""
    torch.manual_seed(42)
    E, C, D, I = 8, 64, 128, 64
    padded_input = torch.randn(E, C, D, device="cuda", dtype=torch.bfloat16) * 0.02
    expert_gate_up = torch.randn(E, 2 * I, D, device="cuda", dtype=torch.bfloat16) * 0.02
    expert_down = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.02
    return padded_input, expert_gate_up, expert_down


# --- Scatter IN tests ---

def test_triton_scatter_in_matches_pytorch(routing_data):
    d = routing_data
    ref = torch_moe_scatter_in(d["x_flat"], d["expert_ids"], d["token_ids"], d["positions"], d["E"], d["capacity"])
    out = triton_moe_scatter_in(d["x_flat"], d["expert_ids"], d["token_ids"], d["positions"], d["E"], d["capacity"])
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=0, rtol=0)


def test_scatter_in_output_shape(routing_data):
    d = routing_data
    out = triton_moe_scatter_in(d["x_flat"], d["expert_ids"], d["token_ids"], d["positions"], d["E"], d["capacity"])
    assert out.shape == (d["E"], d["capacity"], d["D"])


def test_scatter_in_zeros_unassigned(routing_data):
    """Unassigned positions in padded tensor should remain zero."""
    d = routing_data
    out = triton_moe_scatter_in(d["x_flat"], d["expert_ids"], d["token_ids"], d["positions"], d["E"], d["capacity"])
    # Create mask of assigned positions
    mask = torch.zeros(d["E"], d["capacity"], dtype=torch.bool, device="cuda")
    mask[d["expert_ids"], d["positions"]] = True
    # Unassigned positions should be zero
    assert (out[~mask] == 0).all()


# --- Scatter OUT tests ---

def test_triton_scatter_out_matches_pytorch(routing_data):
    d = routing_data
    expert_out = torch.randn(d["E"], d["capacity"], d["D"], device="cuda", dtype=torch.bfloat16)
    ref = torch_moe_scatter_out(expert_out, d["expert_ids"], d["token_ids"], d["positions"], d["weights"], d["T"])
    out = triton_moe_scatter_out(expert_out, d["expert_ids"], d["token_ids"], d["positions"], d["weights"], d["T"])
    assert out.shape == ref.shape
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-2)


def test_scatter_out_output_shape(routing_data):
    d = routing_data
    expert_out = torch.randn(d["E"], d["capacity"], d["D"], device="cuda", dtype=torch.bfloat16)
    out = triton_moe_scatter_out(expert_out, d["expert_ids"], d["token_ids"], d["positions"], d["weights"], d["T"])
    assert out.shape == (d["T"], d["D"])


def test_scatter_out_accumulates_for_topk(routing_data):
    """Tokens routed to multiple experts should have accumulated outputs."""
    d = routing_data
    expert_out = torch.ones(d["E"], d["capacity"], d["D"], device="cuda", dtype=torch.bfloat16)
    out = triton_moe_scatter_out(expert_out, d["expert_ids"], d["token_ids"], d["positions"], d["weights"], d["T"])
    # At least some tokens should have non-zero output (routed to at least one expert)
    assert (out.abs().sum(-1) > 0).any()


# --- Expert FFN tests ---

def test_torch_expert_ffn_output_shape(expert_ffn_data):
    padded_input, expert_gate_up, expert_down = expert_ffn_data
    out = torch_moe_expert_ffn(padded_input, expert_gate_up, expert_down)
    assert out.shape == padded_input.shape


def test_triton_expert_ffn_output_shape(expert_ffn_data):
    padded_input, expert_gate_up, expert_down = expert_ffn_data
    out = triton_moe_expert_ffn(padded_input, expert_gate_up, expert_down)
    assert out.shape == padded_input.shape


def test_triton_expert_ffn_matches_torch(expert_ffn_data):
    padded_input, expert_gate_up, expert_down = expert_ffn_data
    ref = torch_moe_expert_ffn(padded_input, expert_gate_up, expert_down)
    out = triton_moe_expert_ffn(padded_input, expert_gate_up, expert_down)
    # Manual reference (no torch.compile)
    gate_up = torch.bmm(padded_input, expert_gate_up.mT)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = F.silu(gate) * up
    manual_ref = torch.bmm(hidden, expert_down.mT)
    # Both should match the manual reference within bf16 precision
    torch.testing.assert_close(out, manual_ref, atol=1e-2, rtol=1e-2)


def test_expert_ffn_preserves_dtype(expert_ffn_data):
    padded_input, expert_gate_up, expert_down = expert_ffn_data
    for fn in [torch_moe_expert_ffn, triton_moe_expert_ffn]:
        out = fn(padded_input, expert_gate_up, expert_down)
        assert out.dtype == padded_input.dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_expert_ffn_dtype_support(dtype):
    """Expert FFN should work with float32, float16, and bfloat16."""
    torch.manual_seed(42)
    E, C, D, I = 4, 16, 32, 16
    padded_input = torch.randn(E, C, D, device="cuda", dtype=dtype) * 0.02
    expert_gate_up = torch.randn(E, 2 * I, D, device="cuda", dtype=dtype) * 0.02
    expert_down = torch.randn(E, D, I, device="cuda", dtype=dtype) * 0.02

    out = triton_moe_expert_ffn(padded_input, expert_gate_up, expert_down)
    assert out.dtype == dtype
    assert out.shape == (E, C, D)
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_scatter_dtype_support(dtype):
    """Scatter kernels should work with float32, float16, and bfloat16."""
    torch.manual_seed(42)
    T, D, E, C = 32, 16, 4, 16
    x_flat = torch.randn(T, D, device="cuda", dtype=dtype)
    expert_ids = torch.tensor([0, 0, 1, 1, 2, 3], device="cuda")
    token_ids = torch.tensor([0, 5, 10, 15, 20, 25], device="cuda")
    positions = torch.tensor([0, 1, 0, 1, 0, 0], device="cuda")
    weights = torch.rand(6, device="cuda", dtype=dtype)

    padded = triton_moe_scatter_in(x_flat, expert_ids, token_ids, positions, E, C)
    assert padded.dtype == dtype

    expert_out = torch.randn(E, C, D, device="cuda", dtype=dtype)
    output = triton_moe_scatter_out(expert_out, expert_ids, token_ids, positions, weights, T)
    assert output.dtype == dtype
