import torch
import torch.nn.functional as F


def to_column_major(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(-2) == 1:
        return tensor
    return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)


def to_swizzle_32_4_4(scale: torch.Tensor) -> torch.Tensor:
    rows, blocks = scale.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_blocks = (blocks + 3) // 4 * 4
    pad = (0, padded_blocks - blocks, 0, padded_rows - rows)
    if scale.dtype not in {torch.float8_e4m3fn, torch.float8_e8m0fnu}:
        raise ValueError(
            "32x4x4 swizzle scales must have float8_e4m3fn or float8_e8m0fnu dtype"
        )
    # F.pad cannot construct float8 pad values.
    padded = F.pad(scale.view(torch.uint8), pad).view(scale.dtype)
    return (
        padded.reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def to_hadamard_scales(hadamard_block: int) -> tuple[float, float]:
    """
    Split 1/sqrt(block) into an exact pre-transform factor and a residual
    """
    log_block = hadamard_block.bit_length() - 1
    pre = 2.0 ** -((log_block + 1) // 2)
    return pre, 2.0**0.5 if log_block % 2 else 1.0
