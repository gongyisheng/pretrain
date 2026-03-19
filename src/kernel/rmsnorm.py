"""Triton fused RMSNorm kernel.

Fuses 7 separate PyTorch ops into 1 GPU kernel for ~3.5x speedup.

Usage:
    from src.kernel import triton_rmsnorm
    y = triton_rmsnorm(x, weight, eps=1e-6)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = X_ptr + row_idx * stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)

    variance = tl.sum(x * x, axis=0) / D
    rrms = 1.0 / tl.sqrt(variance + eps)
    x_normed = x * rrms

    w = tl.load(W_ptr + offsets, mask=mask, other=0.0)
    y = x_normed * w

    out_start = Y_ptr + row_idx * stride
    tl.store(out_start + offsets, y, mask=mask)


@triton.jit
def _rmsnorm_kernel_tiled(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride,
    D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = X_ptr + row_idx * stride

    # Pass 1: accumulate sum of squares in chunks
    sum_sq = 0.0
    for start in range(0, D, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    rrms = 1.0 / tl.sqrt(sum_sq / D + eps)

    # Pass 2: normalize and scale in chunks
    out_start = Y_ptr + row_idx * stride
    for start in range(0, D, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < D
        x = tl.load(row_start + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offsets, mask=mask, other=0.0)
        y = x * rrms * w
        tl.store(out_start + offsets, y, mask=mask)


@triton.jit
def _rmsnorm_kernel_2d(
    X_ptr,
    Y_ptr,
    W_ptr,
    stride,
    N_ROWS,
    D,
    eps,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    seq_blk_id = tl.program_id(0)
    seq_start = seq_blk_id * BLOCK_SEQ

    seq_offset = seq_start + tl.arange(0, BLOCK_SEQ)[:, None]  # (BLOCK_SEQ, 1)
    d_offset = tl.arange(0, BLOCK_DIM)[None, :]                # (1, BLOCK_DIM)
    s_mask = seq_offset < N_ROWS
    d_mask = d_offset < D
    mask = s_mask & d_mask

    x_ptrs = X_ptr + seq_offset * stride + d_offset
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=1, keep_dims=True) / D   # (BLOCK_SEQ, 1)
    rrms = tl.math.rsqrt(mean_sq + eps)

    w = tl.load(W_ptr + d_offset, mask=d_mask)
    y = x * rrms * w

    y_ptrs = Y_ptr + seq_offset * stride + d_offset
    tl.store(y_ptrs, y, mask=mask)


def _get_block_seq(n_rows: int, D: int, device) -> int:
    """Pick BLOCK_SEQ based on SM count, row count, and D to avoid register spilling.

    Register budget per SM: ~65536 registers.
    Each worker needs ~3 * BLOCK_SEQ * BLOCK_DIM registers (x, w, y live simultaneously).
    We target ~4 workers per SM, so each worker gets ~16384 registers.
    """
    n_sms = torch.cuda.get_device_properties(device).multi_processor_count
    target_workers = n_sms * 4

    # Max BLOCK_SEQ that fits in registers: 16384 / (3 * BLOCK_DIM)
    block_dim = triton.next_power_of_2(D)
    max_seq_for_registers = max(1, 16384 // (3 * block_dim))
    max_seq = triton.next_power_of_2(max_seq_for_registers) // 2 or 1  # round down to power of 2

    if n_rows <= target_workers:
        return 1
    block_seq = triton.next_power_of_2(max(1, n_rows // target_workers))
    return min(block_seq, max_seq)


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in replacement for PyTorch RMSNorm forward pass.

    Args:
        x: Input tensor of shape (..., D)
        weight: Learnable scale of shape (D,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor, same shape and dtype as x
    """
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, D = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(D)

    _rmsnorm_kernel[(n_rows,)](
        x, y, weight,
        x.stride(0),
        D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.view(orig_shape)


def triton_rmsnorm_2d(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Adaptive 2D RMSNorm — dynamically picks rows-per-worker based on GPU SM count.

    Args:
        x: Input tensor of shape (..., D)
        weight: Learnable scale of shape (D,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor, same shape and dtype as x
    """
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, D = x.shape

    y = torch.empty_like(x)
    BLOCK_DIM = triton.next_power_of_2(D)
    BLOCK_SEQ = _get_block_seq(n_rows, D, x.device)
    grid = (triton.cdiv(n_rows, BLOCK_SEQ),)

    _rmsnorm_kernel_2d[grid](
        x, y, weight,
        x.stride(0),
        n_rows,
        D,
        eps,
        BLOCK_SEQ=BLOCK_SEQ,
        BLOCK_DIM=BLOCK_DIM,
    )

    return y.view(orig_shape)


def triton_rmsnorm_tiled(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, block_size: int = 4096,
) -> torch.Tensor:
    """Tiled RMSNorm for large D. Processes each row in chunks to avoid register spilling.

    Args:
        x: Input tensor of shape (..., D)
        weight: Learnable scale of shape (D,)
        eps: Epsilon for numerical stability
        block_size: Tile size per chunk (default 4096)

    Returns:
        Normalized tensor, same shape and dtype as x
    """
    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])
    n_rows, D = x.shape

    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(min(D, block_size))

    _rmsnorm_kernel_tiled[(n_rows,)](
        x, y, weight,
        x.stride(0),
        D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.view(orig_shape)
