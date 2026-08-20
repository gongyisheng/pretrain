import pytest
import torch

from src.kernel.utils import to_column_major, to_swizzle_32_4_4

COLUMN_MAJOR_SHAPES = [(2, 3), (4, 2, 3)]

SWIZZLE_SHAPES = [(1, 1), (128, 4), (129, 5)]


@pytest.mark.parametrize("shape", COLUMN_MAJOR_SHAPES, ids=["2d", "batched"])
def test_to_column_major_returns_existing_column_major_tensor(shape):
    tensor = torch.randn(*shape).transpose(-1, -2).contiguous().transpose(-1, -2)

    assert to_column_major(tensor) is tensor


@pytest.mark.parametrize("shape", COLUMN_MAJOR_SHAPES, ids=["2d", "batched"])
def test_to_column_major_converts_row_major_tensor(shape):
    tensor = torch.randn(*shape)

    actual = to_column_major(tensor)

    assert actual.stride(-2) == 1
    torch.testing.assert_close(actual, tensor)


@pytest.mark.parametrize("rows,blocks", SWIZZLE_SHAPES)
def test_to_swizzle_32_4_4_pads_and_maps_logical_scale_bytes(rows, blocks):
    scale = torch.tensor(
        [2.0 ** (index % 10 - 5) for index in range(rows * blocks)], dtype=torch.float32
    ).reshape(rows, blocks)
    padded_rows = -(-rows // 128) * 128
    padded_blocks = -(-blocks // 4) * 4

    actual = to_swizzle_32_4_4(scale)

    assert actual.dtype is torch.float8_e8m0fnu
    assert actual.is_contiguous()
    assert actual.numel() == padded_rows * padded_blocks
    logical = scale.to(torch.float8_e8m0fnu).view(torch.uint8)
    actual_bytes = actual.view(torch.uint8)
    zero_byte = torch.tensor(0.0, dtype=torch.float8_e8m0fnu).view(torch.uint8).item()
    for row in range(padded_rows):
        for block in range(padded_blocks):
            row_group, row_within_group = divmod(row, 128)
            row_tile, row_inner = divmod(row_within_group, 32)
            block_group, block_inner = divmod(block, 4)
            offset = (
                ((row_group * (padded_blocks // 4) + block_group) * 32 + row_inner) * 4
                + row_tile
            ) * 4 + block_inner
            in_bounds = row < rows and block < blocks
            expected = logical[row, block].item() if in_bounds else zero_byte
            assert actual_bytes[offset].item() == expected
