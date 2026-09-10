"""Blackwell FP4 conversion parity tests."""

import pytest
import torch

from src.kernel.backends.triton.quantize import has_e2m1_rne, pack_e2m1_rne
from src.quant.constants import _FP4_E2M1_VALUES


DEVICE = torch.device("cuda")
CUDA_E2M1_RNE = torch.cuda.is_available() and has_e2m1_rne(DEVICE)
pytestmark = pytest.mark.skipif(
    not CUDA_E2M1_RNE, reason="Blackwell E2M1 RNE conversion required"
)


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
DIMS = (-2, -1)
REPEATS = (1, 33)
COMPILED = (False, True)
LAYOUTS = ("dense", "strided", "broadcast")
MIDPOINTS = tuple(
    (lower + upper) / 2 for lower, upper in zip(_FP4_E2M1_VALUES, _FP4_E2M1_VALUES[1:])
)
PRECISION_CASES = (
    ("values", (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE)),
    ("midpoints", (0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE)),
    ("below", (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE)),
    ("above", (0x21, 0x43, 0x65, 0x77, 0xA9, 0xCB, 0xED, 0xFF)),
    ("nonfinite", (0x80, 0xF7, 0x77, 0xF7)),
    ("empty", ()),
)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("case", PRECISION_CASES)
@pytest.mark.parametrize("repeats", REPEATS)
@pytest.mark.parametrize("compiled", COMPILED)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_pack_e2m1_rne_precision(dtype, dim, case, repeats, compiled, layout):
    case, packed_bytes = case
    if case == "values":
        values = torch.tensor(_FP4_E2M1_VALUES, dtype=dtype, device=DEVICE)
    elif case in ("midpoints", "below", "above"):
        values = torch.tensor(MIDPOINTS, dtype=dtype, device=DEVICE)
        if case != "midpoints":
            direction = -torch.inf if case == "below" else torch.inf
            values = torch.nextafter(values, torch.full_like(values, direction))
        values = torch.cat((values, values.new_tensor((6.0,))))
    elif case == "nonfinite":
        values = torch.tensor(
            (0.0, -0.0, torch.inf, -torch.inf, torch.nan, -torch.nan, 7.0, -7.0),
            dtype=dtype,
            device=DEVICE,
        )
    else:
        values = torch.empty(0, dtype=dtype, device=DEVICE)

    if case not in ("nonfinite", "empty"):
        values = torch.cat((values, -values))
    source = values.repeat(2, repeats)
    if layout == "strided":
        source = source.repeat_interleave(2, dim=-1)[..., ::2]
    elif layout == "broadcast":
        source = source[:1].expand(2, -1)
    expected = torch.tensor(packed_bytes, dtype=torch.uint8, device=DEVICE).repeat(
        2, repeats
    )
    if dim == -2:
        source, expected = source.mT, expected.mT

    if compiled:
        torch.compiler.reset()
        try:
            actual = torch.compile(pack_e2m1_rne, fullgraph=True)(source, dim)
        finally:
            torch.compiler.reset()
    else:
        actual = pack_e2m1_rne(source, dim)

    assert torch.equal(actual, expected)
