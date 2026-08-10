"""Bitwise parity of quantize_operand against an explicit scale-group enumeration.

The reference assigns every element a scale-group id with Python loops, amaxes each
group, and divides. `amax` is exact and every other step is elementwise float32, so
equality here is bitwise, not approximate.
"""

import pytest
import torch

from src.quant.constants import EPS
from src.quant.quantize import quantize_operand
from src.quant.utils import is_int8s, str_to_dtype, str_to_emax, str_to_qmax

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _call(x, contract_dim, fmt, scaling, offs=None, ragged_dim=None):
    """Adapter onto the API under test. Task 3 re-points this and nothing else."""
    return quantize_operand(
        x, contract_dim, fmt, scaling, offs=offs, ragged_dim=ragged_dim
    )


def _segments(n, offs, is_ragged):
    """[(lo, hi)] along one axis: the group bounds when ragged, else the whole axis."""
    if not is_ragged:
        return [(0, n)]
    bounds = [0] + offs.tolist()
    return list(zip(bounds[:-1], bounds[1:]))


def _axis_blocks(n, offs, is_ragged, block_size, padded_height=None):
    """(block id per index, number of blocks) for one axis.

    Blocks are numbered in order and restart at every segment boundary -- the same
    convention `ragged_scale_blocks` builds, and the one its own tests in
    test_quantize.py pin. `padded_height` reproduces the implementation's static
    upper bound, whose unused rows hold the scale of amax 0.

    The two block_size cases differ on empty segments, and COUNTS has one. With
    block_size 0 the implementation numbers blocks by group id, so an empty group
    still owns a block; with a positive block_size it numbers by
    cumsum(ceil(counts/bs)), so an empty group contributes none.
    """
    ids, nxt = [0] * n, 0
    for lo, hi in _segments(n, offs, is_ragged):
        if not block_size:
            for i in range(lo, hi):
                ids[i] = nxt
            nxt += 1
            continue
        for start in range(lo, hi, block_size):
            for i in range(start, min(start + block_size, hi)):
                ids[i] = nxt
            nxt += 1
    return ids, padded_height or nxt


def _scale_of(amax, fmt, scale_dtype):
    """The reference's own copy of the scale formula (src/quant/quantize.py)."""
    if scale_dtype == "fp8_e8m0":
        emax = str_to_emax("fp8_e8m0")
        exp = min(
            max(
                torch.floor(torch.log2(torch.clamp(amax, min=EPS))).item()
                - str_to_emax(fmt),
                -emax,
            ),
            emax,
        )
        return 2.0**exp
    return max((amax / str_to_qmax(fmt)).item(), EPS)


def _ref_quantize(x, contract_dim, fmt, scaling, offs=None, ragged_dim=None):
    granularity, block_size = scaling["granularity"], scaling["block_size"]
    scale_dtype = scaling.get("scale_dtype")
    xf = x.float()
    batched = xf.reshape(-1, *xf.shape[-2:])  # (B, R, C), B == 1 when 2D
    # work in (outer, contract) order so the loops below are orientation-free
    view = batched if contract_dim == -1 else batched.transpose(-2, -1)
    B, n_outer, n_contract = view.shape

    n_groups = 0 if offs is None else offs.shape[0]
    c_ids, n_cb = _axis_blocks(
        n_contract,
        offs,
        ragged_dim == contract_dim,
        block_size,
        padded_height=(
            n_contract // block_size + n_groups
            if (ragged_dim == contract_dim and block_size)
            else None
        ),
    )
    outer_dim = -1 if contract_dim == -2 else -2
    if granularity == "tensorwise":
        o_ids, n_ob = _axis_blocks(n_outer, offs, ragged_dim == outer_dim, 0)
    else:
        o_ids, n_ob = list(range(n_outer)), n_outer

    amax = torch.zeros(B, n_ob, n_cb)
    for b in range(B):
        for i in range(n_outer):
            for j in range(n_contract):
                v = view[b, i, j].abs().cpu()
                amax[b, o_ids[i], c_ids[j]] = torch.maximum(
                    amax[b, o_ids[i], c_ids[j]], v
                )

    scale = torch.tensor(
        [
            [
                [_scale_of(amax[b, i, j], fmt, scale_dtype) for j in range(n_cb)]
                for i in range(n_ob)
            ]
            for b in range(B)
        ],
        dtype=torch.float32,
        device=x.device,
    )

    codes = torch.empty_like(view)
    for b in range(B):
        for i in range(n_outer):
            for j in range(n_contract):
                q = view[b, i, j] / scale[b, o_ids[i], c_ids[j]]
                if is_int8s(fmt):
                    q = torch.round(q)
                qmax = str_to_qmax(fmt)
                codes[b, i, j] = torch.clamp(q, -qmax, qmax)
    codes = codes.to(str_to_dtype(fmt))

    # the invariant: outer restored to full length, contract left at n_blocks
    out_scale = scale[:, o_ids, :]  # (B, n_outer, n_cb)
    if contract_dim == -2:
        codes, out_scale = codes.transpose(-2, -1), out_scale.transpose(-2, -1)
    if x.ndim == 2:
        codes, out_scale = codes[0], out_scale[0]
    else:
        codes = codes.reshape(*x.shape[:-2], *codes.shape[-2:])
        out_scale = out_scale.reshape(*x.shape[:-2], *out_scale.shape[-2:])
    return codes.contiguous(), out_scale.contiguous()


def _offs(counts):
    return torch.tensor(counts, device="cuda").cumsum(0).to(torch.int32)


# (label, shape, contract_dim, ragged_dim) -- the spec's truth table
CASES = [
    ("A_dense", (24, 32), -1, None),
    ("B_dense", (32, 16), -2, None),
    ("B_experts", (3, 32, 16), -2, None),
    ("A_ragged_outer", (24, 32), -1, -2),
    ("A_ragged_last", (32, 24), -1, -1),
    ("B_ragged_first", (24, 16), -2, -2),
]
SCALINGS = [
    {"granularity": "tensorwise", "block_size": 0},
    {"granularity": "rowwise", "block_size": 0},
    {"granularity": "blockwise", "block_size": 16},
    {"granularity": "blockwise", "block_size": 16, "scale_dtype": "fp8_e8m0"},
]
COUNTS = [8, 0, 10, 6]  # sums to 24; the empty group is the degenerate case


@pytest.mark.parametrize("fmt", ["fp8_e4m3", "int8"])
@pytest.mark.parametrize("scaling", SCALINGS, ids=lambda s: s["granularity"])
@pytest.mark.parametrize(
    "label,shape,contract_dim,ragged_dim",
    CASES,
    ids=[getattr(c, "values", c)[0] for c in CASES],
)
def test_matches_enumeration_reference(
    label, shape, contract_dim, ragged_dim, scaling, fmt
):
    if scaling.get("scale_dtype") == "fp8_e8m0" and not fmt.startswith("fp8"):
        pytest.skip("mxfp8 scaling is defined only over fp8 elements")
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    offs = None if ragged_dim is None else _offs(COUNTS)

    codes, scale = _call(x, contract_dim, fmt, scaling, offs, ragged_dim)
    ref_codes, ref_scale = _ref_quantize(
        x, contract_dim, fmt, scaling, offs, ragged_dim
    )

    assert scale.shape == ref_scale.shape
    assert torch.equal(scale, ref_scale)
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))


@pytest.mark.parametrize("scaling", SCALINGS, ids=lambda s: s["granularity"])
def test_expert_stack_matches_per_expert_quantize(scaling):
    """One rank-agnostic call on an (E,K,N) stack == moe.py's deleted per-expert loop."""
    torch.manual_seed(0)
    b = torch.randn(3, 32, 16, device="cuda", dtype=torch.bfloat16)
    codes, scale = quantize_operand(b, -2, "fp8_e4m3", scaling)
    per = [quantize_operand(b[g], -2, "fp8_e4m3", scaling) for g in range(b.shape[0])]
    assert torch.equal(
        codes.view(torch.uint8), torch.stack([q for q, _ in per]).view(torch.uint8)
    )
    assert torch.equal(scale, torch.stack([s for _, s in per]))
