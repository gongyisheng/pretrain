"""Standalone Triton grouped (ragged) GEMM prototype.

Matches `torch._grouped_mm(a, b, offs=offs)` for the MoE 2D x 3D case:
  a (R, K) x per-group b (E, K, N) -> c (R, N), rows expert-sorted, `offs` the
  (E,) int32 cumulative END-offsets (offs[-1] == R). One persistent kernel walks
  the ragged tile space; `offs` is read on-device (no offs.cpu() sync). fp32
  accumulation, bf16 in/out. Forward only — no autograd, no torch.compile wiring.
"""

import torch
import triton
import triton.language as tl

# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — curated to cover K=64 and K=512.
_CFG = [
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 32, 4, 3),
    (64, 128, 64, 8, 4),
    (128, 64, 32, 4, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 32, 8, 4),
    (128, 128, 64, 8, 3),
    (64, 256, 64, 8, 4),
    (128, 256, 64, 8, 3),
    (32, 64, 64, 4, 3),
    (64, 32, 32, 4, 3),
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


_NUM_SMS = None


def _num_sms() -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
    return _NUM_SMS


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
    num_sms = _num_sms()
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
