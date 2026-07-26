"""Standalone Triton grouped (ragged) GEMM prototype.

Matches `torch._grouped_mm(a, b, offs=offs)` for the MoE 2D x 3D case:
  a (R, K) x per-group b (E, K, N) -> c (R, N), rows expert-sorted, `offs` the
  (E,) int32 cumulative END-offsets (offs[-1] == R). One persistent kernel walks
  the ragged tile space; `offs` is read on-device (no offs.cpu() sync). fp32
  accumulation, bf16 in/out. Forward only — no autograd, no torch.compile wiring.
"""

import functools

import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) autotune candidates.
# autotune keys on (N, K) and picks the best candidate per shape. The set below
# targets the expert-GEMM shapes used by experiments/latent_moe/* (all d_model=512,
# intermediate_size=192; latent_dim in {64,128,256} or non-latent d=512):
#   gate_up:  K in {64,128,256,512}, N = 2*d_ff = 384
#   down:     K = d_ff = 192,        N in {64,128,256,512}
# So K in {64,128,192,256,512}, N in {64,128,256,384,512}. Small-N (down, N=64)
# wants small BLOCK_N + tall BLOCK_M; large-N (gate_up, N=384) wants wide BLOCK_N.
# BLOCK_K in {32,64} evenly divides every K here (192 = 6*32 = 3*64). Ragged small
# groups keep BLOCK_M in {32,64,128}. Configs that exceed shared memory on this arch
# are auto-pruned by Triton's autotuner.
_CFG = [
    # small N (down, N=64/128): narrow tiles, taller M
    (32, 64, 64, 4, 3),
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 4),
    (128, 64, 32, 4, 4),
    (128, 64, 64, 8, 4),
    (32, 128, 64, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 128, 64, 8, 4),
    (128, 128, 32, 8, 4),
    (128, 128, 64, 8, 3),
    # wide N (gate_up N=384, down N=256/512): wide BLOCK_N
    (64, 256, 32, 8, 3),
    (64, 256, 64, 8, 4),
    (128, 256, 64, 8, 3),
]

_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _CFG
]


@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    offs_ptr,
    E,
    N,
    K,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile_idx = pid  # this persistent program handles tiles pid, pid+NUM_SMS, ...
    last_end = 0  # running count of tiles before the current group
    m_start = 0  # first row of the current group (prev group's end-offset)
    for g in range(E):
        m_end = tl.load(offs_ptr + g)
        m_g = m_end - m_start
        num_m_tiles = tl.cdiv(m_g, BLOCK_M)
        num_tiles_g = num_m_tiles * num_n_tiles
        # process every tile of group g that this program owns
        while (tile_idx >= last_end) and (tile_idx < last_end + num_tiles_g):
            local = tile_idx - last_end
            tile_m = local // num_n_tiles
            tile_n = local % num_n_tiles
            offs_m = m_start + tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = (
                b_ptr
                + g * stride_be
                + offs_k[:, None] * stride_bk
                + offs_n[None, :] * stride_bn
            )
            m_mask = offs_m < m_end
            n_mask = offs_n < N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k_mask = offs_k < K - k0
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                acc += tl.dot(a, b)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            tile_idx += NUM_SMS
        last_end += num_tiles_g
        m_start = m_end


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def triton_grouped_mm(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Grouped GEMM matching torch._grouped_mm(a, b, offs=offs) (2D x 3D).

    a: (R, K); b: (E, K, N) (may be a transposed/strided view); offs: (E,) int32
    cumulative end-offsets on device. Returns (R, N) in a.dtype, fp32 accumulation.
    """
    assert a.ndim == 2 and b.ndim == 3 and offs.ndim == 1
    R, K = a.shape
    E, K_b, N = b.shape
    assert K == K_b, f"K mismatch: a K={K}, b K={K_b}"
    assert offs.shape[0] == E, f"offs len {offs.shape[0]} != E {E}"
    assert offs.dtype == torch.int32
    c = torch.empty((R, N), device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    _grouped_gemm_kernel[(num_sms,)](
        a,
        b,
        c,
        offs,
        E,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        b.stride(2),
        c.stride(0),
        c.stride(1),
        NUM_SMS=num_sms,
    )
    return c


@triton.autotune(configs=_CONFIGS, key=["N", "K"])
@triton.jit
def _grouped_gemm_wgrad_kernel(
    a_ptr,
    gc_ptr,
    gb_ptr,
    offs_ptr,
    E,
    N,
    K,
    stride_am,
    stride_ak,
    stride_gm,
    stride_gn,
    stride_ge,
    stride_gk,
    stride_gn2,
    NUM_SMS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_k_tiles = tl.cdiv(K, BLOCK_K)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tiles_per_group = num_k_tiles * num_n_tiles
    total_tiles = E * tiles_per_group
    for tile_id in range(pid, total_tiles, NUM_SMS):
        g = tile_id // tiles_per_group
        local = tile_id % tiles_per_group
        tile_k = local // num_n_tiles
        tile_n = local % num_n_tiles
        # group g owns rows [m_start, m_end); offs are END-offsets, so prev end is
        # the start. mask the g-1 load so g==0 does not read out of bounds.
        m_start = tl.load(offs_ptr + g - 1, mask=g > 0, other=0)
        m_end = tl.load(offs_ptr + g)
        offs_k = tile_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        k_mask = offs_k < K
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
        m = m_start
        while m < m_end:
            offs_m = m + tl.arange(0, BLOCK_M)
            m_mask = offs_m < m_end
            # load A as (BLOCK_K, BLOCK_M) so tl.dot contracts over M
            a_ptrs = a_ptr + offs_k[:, None] * stride_ak + offs_m[None, :] * stride_am
            gc_ptrs = gc_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
            a = tl.load(a_ptrs, mask=k_mask[:, None] & m_mask[None, :], other=0.0)
            gc = tl.load(gc_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(a, gc)  # (BLOCK_K, BLOCK_M) @ (BLOCK_M, BLOCK_N)
            m += BLOCK_M
        gb_ptrs = (
            gb_ptr
            + g * stride_ge
            + offs_k[:, None] * stride_gk
            + offs_n[None, :] * stride_gn2
        )
        tl.store(
            gb_ptrs,
            acc.to(gb_ptr.dtype.element_ty),
            mask=k_mask[:, None] & n_mask[None, :],
        )


@triton_op("pretrain::grouped_mm_wgrad", mutates_args={})
def grouped_mm_wgrad(
    a: torch.Tensor, grad_c: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Weight gradient of grouped_mm: grad_b[g] = a[g].T @ grad_c[g], shape (E,K,N)."""
    R, K = a.shape
    N = grad_c.shape[1]
    E = offs.shape[0]
    grad_b = torch.empty((E, K, N), device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_gemm_wgrad_kernel)[(num_sms,)](
        a,
        grad_c,
        grad_b,
        offs,
        E,
        N,
        K,
        a.stride(0),
        a.stride(1),
        grad_c.stride(0),
        grad_c.stride(1),
        grad_b.stride(0),
        grad_b.stride(1),
        grad_b.stride(2),
        NUM_SMS=num_sms,
    )
    return grad_b
