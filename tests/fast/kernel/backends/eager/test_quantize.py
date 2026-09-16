"""Eager E2M1 decoding contracts."""

import torch

from src.kernel.ops import unpack_e2m1


def test_unpack_e2m1_precision():
    codes = torch.tensor(
        ((0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE),), dtype=torch.uint8
    )
    expected = torch.tensor(
        (
            (
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ),
        )
    )

    actual = unpack_e2m1(codes, backend="eager")

    assert torch.equal(actual, expected)
