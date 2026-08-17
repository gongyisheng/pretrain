import pytest
import torch

from src.kernel.backends.triton import gemm as triton_gemm
from src.kernel.utils import to_column_major, to_swizzle_32_4_4


def test_to_column_major_returns_existing_column_major_tensor():
    tensor = torch.empty(2, 3).t().contiguous().t()

    assert to_column_major(tensor) is tensor


def test_to_column_major_converts_row_major_tensor():
    tensor = torch.randn(2, 3)

    actual = to_column_major(tensor)

    assert actual.stride(0) == 1
    torch.testing.assert_close(actual, tensor)


def test_to_column_major_returns_existing_batched_column_major_tensor():
    tensor = torch.empty(4, 2, 3).transpose(-1, -2).contiguous().transpose(-1, -2)

    assert to_column_major(tensor) is tensor


def test_to_column_major_converts_batched_row_major_tensor():
    tensor = torch.randn(4, 2, 3)

    actual = to_column_major(tensor)

    assert actual.stride(-2) == 1
    torch.testing.assert_close(actual, tensor)


def test_to_swizzle_32_4_4_pads_and_maps_logical_scale_bytes():
    powers = torch.tensor(
        [2.0 ** (index % 10 - 5) for index in range(129 * 5)], dtype=torch.float32
    ).reshape(129, 5)

    actual = to_swizzle_32_4_4(powers)
    logical = powers.to(torch.float8_e8m0fnu).view(torch.uint8)
    actual_bytes = actual.view(torch.uint8)
    zero_byte = torch.tensor(0.0, dtype=torch.float8_e8m0fnu).view(torch.uint8)

    assert actual.dtype is torch.float8_e8m0fnu
    assert actual.is_contiguous()
    assert actual.numel() == 256 * 8
    for outer in range(128):
        for inner in range(4):
            offset = (outer % 32) * 16 + (outer // 32) * 4 + inner
            assert actual_bytes[offset].item() == logical[outer, inner].item()
    for row in range(256):
        for block in range(8):
            row_group, row_within_group = divmod(row, 128)
            row_tile, row_inner = divmod(row_within_group, 32)
            block_group, block_inner = divmod(block, 4)
            offset = (
                ((row_group * 2 + block_group) * 32 + row_inner) * 4 + row_tile
            ) * 4 + block_inner
            if row >= 129 or block >= 5:
                assert actual_bytes[offset].item() == zero_byte.item()


def _prune_scaled_configs(configs, named_args):
    return triton_gemm._early_prune_scaled_configs(
        configs, named_args, named_args["SCALE_BLOCK_SIZE"], False, None, False
    )


def _prune_scaled_grouped_configs(configs, named_args):
    return triton_gemm._early_prune_scaled_grouped_configs(
        configs,
        named_args,
        36,
        True,
        False,
        named_args["SCALE_BLOCK_SIZE"],
        False,
        None,
        False,
    )


@pytest.mark.parametrize(
    "prune,configs",
    [
        (_prune_scaled_configs, triton_gemm._SCALED_CONFIGS),
        (_prune_scaled_grouped_configs, triton_gemm._SCALED_GROUPED_CONFIGS),
    ],
    ids=["scaled", "scaled_grouped"],
)
def test_triton_autotune_pruning_caps_block_k_at_scale_block(prune, configs):
    for scale_block in (16, 32, 128, 4096):
        kept = prune(list(configs), {"SCALE_BLOCK_SIZE": scale_block})
        assert kept
        assert all(config.kwargs["BLOCK_K"] <= max(32, scale_block) for config in kept)
        assert {id(config) for config in kept} == {
            id(config)
            for config in configs
            if config.kwargs["BLOCK_K"] <= max(32, scale_block)
        }

    assert prune(list(configs), {"SCALE_BLOCK_SIZE": 0}) == list(configs)
