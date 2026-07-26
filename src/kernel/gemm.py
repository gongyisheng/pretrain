"""Triton grouped (ragged) GEMM, matching `torch._grouped_mm(a, b, offs=offs)`.

MoE 2D x 3D case: a (R, K) x per-group b (E, K, N) -> c (R, N), rows expert-sorted,
`offs` the (E,) int32 cumulative END-offsets (offs[-1] == R). One persistent kernel
walks the ragged tile space; `offs` is read on-device (no offs.cpu() sync). fp32
accumulation, bf16 in/out. `_grouped_gemm` is differentiable: dgrad reuses the forward
kernel (`grad_a = _grouped_gemm(grad_c, b.T, offs)`), wgrad uses the dedicated wgrad
kernel (`grad_b = _grouped_gemm_wgrad(a, grad_c, offs)`).
"""

import functools

import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------

_CFG = [
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


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Torch wrappers / ops
# ---------------------------------------------------------------------------


@triton_op("jit_kernel::grouped_gemm", mutates_args={})
def _grouped_gemm(a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    """Differentiable grouped GEMM matching torch._grouped_mm(a, b, offs=offs)."""
    R, K = a.shape
    N = b.shape[2]
    c = torch.empty((R, N), device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_gemm_kernel)[(num_sms,)](
        a,
        b,
        c,
        offs,
        b.shape[0],
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


def _grouped_gemm_setup_context(ctx, inputs, output):
    a, b, offs = inputs
    ctx.save_for_backward(a, b, offs)


def _grouped_gemm_backward(ctx, grad_c):
    a, b, offs = ctx.saved_tensors
    grad_a = _grouped_gemm(grad_c, b.transpose(-2, -1), offs)  # dgrad reuses forward
    grad_b = _grouped_gemm_wgrad(a, grad_c, offs)  # wgrad
    return grad_a, grad_b, None


_grouped_gemm.register_autograd(
    _grouped_gemm_backward, setup_context=_grouped_gemm_setup_context
)


@triton_op("jit_kernel::grouped_gemm_wgrad", mutates_args={})
def _grouped_gemm_wgrad(
    a: torch.Tensor, grad_c: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    """Weight gradient of _grouped_gemm: grad_b[g] = a[g].T @ grad_c[g], shape (E,K,N)."""
    assert a.dtype == grad_c.dtype, (
        f"dtype mismatch: a {a.dtype}, grad_c {grad_c.dtype}"
    )
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


def grouped_gemm(
    a: torch.Tensor, b: torch.Tensor, offs: torch.Tensor, impl: str = "auto"
) -> torch.Tensor:

    if impl not in ("auto", "triton", "torch"):
        raise ValueError(f"impl must be auto|triton|torch, got {impl!r}")

    if impl == "auto":
        impl = (
            "triton"
            if (
                a.is_cuda
                and a.dtype == torch.bfloat16
                and torch.cuda.get_device_capability(a.device)[0] not in (9, 10)
            )
            else "torch"
        )

    if impl == "torch":
        return torch._grouped_mm(a, b, offs=offs)

    assert a.ndim == 2 and b.ndim == 3 and offs.ndim == 1
    assert a.shape[1] == b.shape[1], f"K mismatch: a K={a.shape[1]}, b K={b.shape[1]}"
    assert offs.shape[0] == b.shape[0], f"offs len {offs.shape[0]} != E {b.shape[0]}"
    assert offs.dtype == torch.int32
    assert a.dtype == b.dtype, f"dtype mismatch: a {a.dtype}, b {b.dtype}"
    assert a.device == b.device == offs.device, "a, b, offs must share same device"

    if not (a.is_cuda and a.dtype == torch.bfloat16):
        raise ValueError("impl='triton' requires bf16 CUDA tensors")
    return _grouped_gemm(a, b, offs)
