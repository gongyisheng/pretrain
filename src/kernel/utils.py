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
    # F.pad cannot build E8M0's pad value
    bytes_ = scale.view(torch.uint8) if scale.dtype is torch.float8_e8m0fnu else scale
    padded = F.pad(bytes_, (0, padded_blocks - blocks, 0, padded_rows - rows))
    return (
        (
            padded.view(torch.float8_e8m0fnu)
            if padded.dtype is torch.uint8
            else padded.to(torch.float8_e8m0fnu)
        )
        .reshape(padded_rows // 128, 4, 32, padded_blocks // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )
