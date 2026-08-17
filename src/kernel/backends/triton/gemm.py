import functools
import torch
import triton
import triton.language as tl
from torch._library.triton import triton_op, wrap_triton

from src.kernel.registry import register_kernel
from src.kernel.spec import cuda


_FLOAT_DTYPES = {torch.float32, torch.bfloat16, torch.float16}
_QUANTIZED_DTYPES = {torch.int8, torch.float8_e4m3fn, torch.float8_e5m2}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


# ---------------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _num_sms(device: torch.device | None = None) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Grouped GEMM (bf16 ragged MoE): ragged over M, N, or K
# ---------------------------------------------------------------------------


_GROUPED_CFG = [
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 256, 32, 8, 3),
    (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 4),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (64, 256, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
]

_GROUPED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _GROUPED_CFG
]


@triton.autotune(configs=_GROUPED_CONFIGS, key=["M", "N", "K", "A_IS_2D", "B_IS_2D"])
@triton.jit
def _grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged grouped GEMM over any one of M, N, K, with an optional (G,N) bias.

    Which dim `offs` partitions follows from the operand ranks, as in
    torch._grouped_mm. The tile grid is always (M, N) and the reduction always over
    K, so one tile body serves all three layouts. The ragged dim's extent is passed
    as 0 (its bounds come from `offs`) to keep the autotune key stable.

    The bias is per group, broadcast over the output's row dim -- a superset of
    torch._grouped_mm, which rejects bias.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # this persistent program owns pid, pid+NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = 0  # flat index of the current group's first tile
    m_end = 0  # offs are END-offsets, so the previous end is the next start
    n_end = 0
    k_end = 0
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
        else:
            k_start = 0
            k_size = K

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        # an empty ragged M/N group has no tiles; an empty ragged K group still
        # stores its zero slice
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, k_size, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < k_size
                a_ptrs = (
                    a_ptr
                    + (0 if A_IS_2D else g * stride_ag)
                    + (m_start + offs_m)[:, None] * stride_am
                    + (k_start + offs_k)[None, :] * stride_ak
                )
                b_ptrs = (
                    b_ptr
                    + (0 if B_IS_2D else g * stride_bg)
                    + (k_start + offs_k)[:, None] * stride_bk
                    + (n_start + offs_n)[None, :] * stride_bn
                )
                a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
                acc += tl.dot(a, b)
            if HAS_BIAS:
                acc += tl.load(
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn,
                    mask=n_mask,
                    other=0.0,
                )[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


@triton_op("jit_kernel::grouped_gemm", mutates_args={})
def _grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    a_is_2d, b_is_2d = a.ndim == 2, b.ndim == 2
    G = offs.shape[0]
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (G,K,N) -> (M,N), ragged M
        M, N, K = 0, b.shape[2], a.shape[1]
        output_size = (a.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (G,M,N), ragged K
        M, N, K = a.shape[0], b.shape[1], 0
        output_size = (G, M, N)
    else:  # (G,M,K) x (K,N) -> (M,N), ragged N
        if bias is not None:
            raise NotImplementedError("bias not supported for ragged-N")
        M, N, K = a.shape[1], 0, a.shape[2]
        output_size = (M, b.shape[1])

    # pad the last dim to 16 bytes like torch._grouped_mm, so the output is itself a
    # legal torch._grouped_mm operand when backward feeds it back
    align = 16 // a.dtype.itemsize
    padded = (output_size[-1] + align - 1) // align * align
    stride = (M * padded, padded, 1) if len(output_size) == 3 else (padded, 1)
    c = torch.empty_strided(output_size, stride, device=a.device, dtype=a.dtype)
    num_sms = _num_sms(a.device)
    wrap_triton(_grouped_gemm_kernel)[(num_sms,)](
        a,
        b,
        c,
        bias,
        offs,
        G,
        M,
        N,
        K,
        0 if a_is_2d else a.stride(0),
        a.stride(-2),
        a.stride(-1),
        0 if b_is_2d else b.stride(0),
        b.stride(-2),
        b.stride(-1),
        c.stride(0) if c.ndim == 3 else 0,
        c.stride(-2),
        c.stride(-1),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        HAS_BIAS=bias is not None,
    )
    return c


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged grouped GEMM matching torch._grouped_mm's layout convention.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (G,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (G,M,N)  ragged K
        (G,M,K) x (K,N) -> (M,N)    ragged N

    `bias` is an optional (G,N) tensor broadcast over the output's row dim, fused
    into the Triton epilogue.
    """
    _require(
        all(isinstance(tensor, torch.Tensor) for tensor in (a, b, offs)),
        "a, b, and offs must be torch.Tensor inputs",
    )
    _require(a.ndim in (2, 3) and b.ndim in (2, 3), "a and b must be 2-D or 3-D")
    _require(offs.ndim == 1 and offs.numel() > 0, "offs must be a nonempty 1-D tensor")
    _require(a.ndim == 2 or b.ndim == 2, "3D x 3D has no ragged dim; use torch.bmm")
    _require(
        a.shape[-1] == b.shape[-2],
        f"contraction mismatch: a {a.shape[-1]}, b {b.shape[-2]}",
    )
    if a.ndim == 3:
        _require(
            offs.shape[0] == a.shape[0],
            f"offs len {offs.shape[0]} != G {a.shape[0]}",
        )
    if b.ndim == 3:
        _require(
            offs.shape[0] == b.shape[0],
            f"offs len {offs.shape[0]} != G {b.shape[0]}",
        )
    _require(offs.dtype == torch.int32, f"offs must be int32, got {offs.dtype}")
    _require(offs.is_contiguous(), "offs must be contiguous")
    _require(a.dtype == b.dtype == torch.bfloat16, "a and b must both be bfloat16")
    _require(
        a.is_cuda and b.is_cuda and offs.is_cuda, "a, b, and offs must be CUDA tensors"
    )
    _require(
        a.device == b.device == offs.device, "a, b, and offs must share one device"
    )

    if bias is not None:
        _require(isinstance(bias, torch.Tensor), "bias must be a torch.Tensor")
        _require(a.ndim != 3, "bias not supported for ragged-N")
        _require(
            bias.shape == (offs.shape[0], b.shape[-1]),
            f"bias must be (G,N)=({offs.shape[0]},{b.shape[-1]}), got {tuple(bias.shape)}",
        )
        _require(bias.dtype == a.dtype, f"bias dtype {bias.dtype} != a {a.dtype}")
        _require(bias.device == a.device, "bias must be on the same device as a")

    capability = torch.cuda.get_device_capability(a.device)
    _require(
        capability >= (8, 0),
        f"grouped GEMM requires SM80 or newer, got SM{capability[0]}{capability[1]}",
    )

    return _grouped_gemm_unchecked(a, b, offs, bias)


# Dispatch reaches this entry directly, gated only by device/SM capability plus the
# bf16-only dtype routing in src/kernel/ops/gemm.py. `grouped_gemm` above is a
# separate, unused-by-dispatch reference kept for direct callers and tests.
@register_kernel(
    op="gemm.grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
def _grouped_gemm_unchecked(
    a: torch.Tensor,
    b: torch.Tensor,
    offs: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _grouped_gemm(a, b, offs, bias)


# ---------------------------------------------------------------------------
# Scaled GEMM (quantized): fp8 / int8, tensorwise|rowwise|blockwise scaling
# ---------------------------------------------------------------------------

_SCALED_CFG = [
    (32, 128, 32, 4, 4),
    (64, 128, 32, 4, 3),
    (128, 128, 32, 4, 3),
    (128, 128, 32, 8, 4),
    (64, 64, 64, 4, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 4, 4),
    (128, 256, 64, 8, 4),
    (256, 128, 64, 8, 3),
    (64, 128, 128, 4, 3),
    (128, 128, 128, 8, 3),
]
_SCALED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _SCALED_CFG
]


def _early_prune_scaled_configs(
    configs, named_args, SCALE_BLOCK_SIZE, HAS_BIAS, grid=None, warmup=None
):
    del named_args, HAS_BIAS, grid, warmup
    scale_block = SCALE_BLOCK_SIZE
    if not scale_block:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_K"] <= max(32, scale_block)]


@triton.autotune(
    configs=_SCALED_CONFIGS,
    key=["M", "N", "K", "SCALE_BLOCK_SIZE"],
    prune_configs_by={"early_config_prune": _early_prune_scaled_configs},
)
@triton.jit
def _scaled_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    n_scale_blocks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbk,
    stride_sbn,
    stride_biasn,
    SCALE_BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """C[m,n] = Σ_b sa[m,b] · sb[b,n] · (Σ_{k∈b} aq[m,k] · bq[k,n]) + bias[n]."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for scale_block_idx in range(n_scale_blocks):
        # BLOCK_K floors at 32 (fp8/int8 tl.dot rejects K < 32), so a 16-wide scale
        # block runs one 32-wide tile; block_end, not the trip count, keeps the extra
        # lanes out of this block's scale.
        block_end = tl.minimum(K, scale_block_idx * SCALE_BLOCK_SIZE + SCALE_BLOCK_SIZE)
        block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_in_block in range(0, SCALE_BLOCK_SIZE, BLOCK_K):
            offs_k = (
                scale_block_idx * SCALE_BLOCK_SIZE + k_in_block + tl.arange(0, BLOCK_K)
            )
            k_mask = offs_k < block_end
            a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
            b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            block_acc += tl.dot(a, b).to(tl.float32)
        sa = tl.load(
            sa_ptr + offs_m * stride_sam + scale_block_idx * stride_sak,
            mask=m_mask,
            other=0.0,
        )
        sb = tl.load(
            sb_ptr + scale_block_idx * stride_sbk + offs_n * stride_sbn,
            mask=n_mask,
            other=0.0,
        )
        acc += sa[:, None] * block_acc * sb[None, :]
    # on acc, not block_acc: the bias is unscaled, so folding it into a per-block
    # partial would scale it and add it once per block
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_biasn, mask=n_mask, other=0.0)
        acc += bias[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


_SCALED_MXFP8_CFG = [
    (64, 64, 64, 4, 3),
    (64, 128, 64, 4, 3),
    (64, 128, 128, 4, 3),
    (64, 256, 64, 4, 3),
    (128, 128, 64, 4, 3),
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 4),
    (256, 128, 64, 8, 3),
]
_SCALED_MXFP8_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=w,
        num_stages=s,
    )
    for (bm, bn, bk, w, s) in _SCALED_MXFP8_CFG
]

# tl.dot_scaled names its operand format; the e8m0 scale width is fixed at 32
_MXFP8_FORMAT = {torch.float8_e4m3fn: "e4m3", torch.float8_e5m2: "e5m2"}
_MXFP8_BLOCK_SIZE = 32


@triton.autotune(configs=_SCALED_MXFP8_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _scaled_gemm_mxfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    M,
    N,
    K,
    n_scale_blocks,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_sam,
    stride_sak,
    stride_sbk,
    stride_sbn,
    stride_biasn,
    A_FORMAT: tl.constexpr,
    B_FORMAT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[m,n] = Σ_k sa[m,k/32] · sb[k/32,n] · aq[m,k] · bq[k,n] + bias[n], mxfp8 only.

    The scales never reach the accumulator here: `tl.dot_scaled` lowers to
    `mma.sync...kind::mxf8f6f4.block_scale` (SASS `QMMA.SF`), which applies the e8m0
    factors per 32-element group inside the MMA. That is why there is no scale-block
    loop -- and why this runs on the faster of the two fp8 tensor-core paths, ~2.4x
    the epilogue-scaling kernel next door on the same shapes.

    `sa`/`sb` are the e8m0 exponent bytes viewed as uint8. `sb` is read transposed
    through its strides -- the instruction wants B's scales indexed (N, K/32) while
    `quantize_operand` produces (K/32, N) -- which costs nothing on a tensor this small.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        # a masked-off scale rides along with a zeroed operand, so its value is moot
        sk = k0 // 32 + offs_sk
        sk_mask = sk < n_scale_blocks
        sa = tl.load(
            sa_ptr + offs_m[:, None] * stride_sam + sk[None, :] * stride_sak,
            mask=m_mask[:, None] & sk_mask[None, :],
            other=0,
        )
        sb = tl.load(
            sb_ptr + offs_n[:, None] * stride_sbn + sk[None, :] * stride_sbk,
            mask=n_mask[:, None] & sk_mask[None, :],
            other=0,
        )
        acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc=acc)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n * stride_biasn, mask=n_mask, other=0.0)[
            None, :
        ]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(
        c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


def _check_scaled_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
    operand_formats: dict | set,
) -> tuple[int, int, int]:
    """Checks shared by `epilogue_scaled_gemm` and `mxfp8_scaled_gemm`. `operand_formats`
    is the allowed aq/bq dtype set -- `_QUANTIZED_DTYPES` or `_MXFP8_FORMAT` -- since
    that is the one thing the two kernels genuinely disagree on. Returns (M, K, N) so
    callers don't re-derive them.
    """
    tensors = (aq, bq, sa, sb) if bias is None else (aq, bq, sa, sb, bias)
    _require(
        all(isinstance(tensor, torch.Tensor) for tensor in tensors),
        "aq, bq, sa, sb, and bias must be torch.Tensor inputs",
    )
    _require(
        all(tensor.ndim == 2 for tensor in (aq, bq, sa, sb)),
        "aq, bq, sa, and sb must be 2-D",
    )
    _require(isinstance(out_dtype, torch.dtype), "out_dtype must be torch.dtype")
    _require(
        type(block_size) is int and block_size >= 0,
        "block_size must be a non-negative integer",
    )
    _require(
        all(tensor.is_cuda for tensor in tensors), "scaled GEMM requires CUDA tensors"
    )
    _require(
        all(tensor.device == aq.device for tensor in tensors),
        "all scaled GEMM tensors must share one device",
    )
    _require(aq.dtype in operand_formats, f"unsupported aq dtype {aq.dtype}")
    _require(bq.dtype in operand_formats, f"unsupported bq dtype {bq.dtype}")
    _require(sa.dtype == sb.dtype == torch.float32, "sa and sb must both be float32")
    _require(out_dtype in _FLOAT_DTYPES, f"unsupported out_dtype {out_dtype}")
    _require(
        aq.dtype.is_floating_point == bq.dtype.is_floating_point,
        f"aq {aq.dtype} and bq {bq.dtype} are not the same family",
    )
    M, K = aq.shape
    N = bq.shape[1]
    _require(bq.shape[0] == K, f"contraction mismatch: aq {K}, bq {bq.shape[0]}")
    expected_blocks = triton.cdiv(K, block_size) if block_size else 1
    _require(
        sa.shape == (M, expected_blocks),
        f"sa must be (M,B)=({M},{expected_blocks}), got {tuple(sa.shape)}",
    )
    _require(
        sb.shape == (expected_blocks, N),
        f"sb must be (B,N)=({expected_blocks},{N}), got {tuple(sb.shape)}",
    )
    if bias is not None:
        _require(bias.ndim == 1 and bias.shape[0] == N, f"bias must be (N,)=({N},)")
        _require(bias.dtype in _FLOAT_DTYPES, f"unsupported bias dtype {bias.dtype}")
    return M, K, N


def epilogue_scaled_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _, K, _ = _check_scaled_gemm(
        aq, bq, sa, sb, out_dtype, block_size, bias, _QUANTIZED_DTYPES
    )
    expected_blocks = triton.cdiv(K, block_size) if block_size else 1
    if expected_blocks > 1:
        _require(
            block_size >= 16 and block_size & (block_size - 1) == 0,
            f"block_size must be a power of two >= 16 when it tiles K, got {block_size}",
        )
    capability = torch.cuda.get_device_capability(aq.device)
    minimum = (8, 9) if aq.dtype.is_floating_point else (8, 0)
    _require(
        capability >= minimum,
        f"scaled GEMM requires SM{minimum[0]}{minimum[1]} or newer, got SM{capability[0]}{capability[1]}",
    )
    return _epilogue_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


def mxfp8_scaled_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_scaled_gemm(aq, bq, sa, sb, out_dtype, block_size, bias, _MXFP8_FORMAT)
    # block_size 0 is the "one block spans all of K" sentinel, not a real width -- 0 %
    # 32 == 0 in Python, so it must be excluded explicitly or it would collapse rep_k
    # to 0 in `_mxfp8_scaled_mm` and hand the kernel a zero-width scale tensor.
    _require(
        block_size != 0 and block_size % _MXFP8_BLOCK_SIZE == 0,
        f"mxfp8 block_size must be a nonzero multiple of {_MXFP8_BLOCK_SIZE}, got {block_size}",
    )
    capability = torch.cuda.get_device_capability(aq.device)
    _require(
        capability >= (10, 0),
        f"mxfp8 scaled GEMM requires SM100 or newer, got SM{capability[0]}{capability[1]}",
    )
    return _mxfp8_scaled_mm(aq, bq, sa, sb, out_dtype, block_size, bias)


# Dispatch reaches this entry directly, gated only by device/SM capability; see the
# entry's own docstring for what it still checks and why.
@register_kernel(
    op="gemm.int8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
@register_kernel(
    op="gemm.fp8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
@triton_op("jit_kernel::epilogue_scaled_mm", mutates_args={})
def _epilogue_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled GEMM, (M,K) x (K,N) -> (M,N), with per-scale-block accumulation.

    `block_size` 0 means one scale block spanning the whole contraction.
    `SCALE_BLOCK_SIZE` below is derived from `n_scale_blocks` rather than from that
    sentinel because a blockwise width >= K also collapses to one block, and would
    otherwise leave the kernel restarting its accumulator on a boundary K never
    reaches.

    `bias` is an optional (N,) tensor broadcast over the rows, added to the fp32
    accumulator in the epilogue so it never passes through the scales.

    Dispatch reaches this entry directly, gated only by device/SM capability; the
    `epilogue_scaled_gemm` function above is a separate, unused-by-dispatch reference
    kept for direct callers and tests. Only checks whose violation is silent or
    numerically wrong live here (see the `block_size` check below); everything else is
    the caller's contract to get right.
    """
    M, K = aq.shape
    N = bq.shape[1]
    n_scale_blocks = sa.shape[1]
    # SCALE_BLOCK_SIZE tiles K in the epilogue-scaling kernel below; a non-power-of-two
    # or sub-16 width is not rejected by Triton, it just computes the wrong answer.
    if n_scale_blocks > 1 and (block_size < 16 or block_size & (block_size - 1) != 0):
        raise ValueError(
            f"block_size must be a power of two >= 16 when it tiles K, got {block_size}"
        )
    SCALE_BLOCK_SIZE = block_size if n_scale_blocks > 1 else K
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    wrap_triton(_scaled_gemm_kernel)[grid](
        aq,
        bq,
        c,
        sa,
        sb,
        bias,
        M,
        N,
        K,
        n_scale_blocks,
        aq.stride(0),
        aq.stride(1),
        bq.stride(0),
        bq.stride(1),
        c.stride(0),
        c.stride(1),
        sa.stride(0),
        sa.stride(1),
        sb.stride(0),
        sb.stride(1),
        0 if bias is None else bias.stride(0),
        SCALE_BLOCK_SIZE=SCALE_BLOCK_SIZE,
        HAS_BIAS=bias is not None,
    )
    return c


@register_kernel(
    op="gemm.mxfp8_scaled_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
@triton_op("jit_kernel::mxfp8_scaled_mm", mutates_args={})
def _mxfp8_scaled_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled GEMM, (M,K) x (K,N) -> (M,N), mxfp8 only; see `mxfp8_scaled_gemm` for the
    checked entry point.

    Dispatch reaches this entry directly, gated only by device/SM capability; the
    `mxfp8_scaled_gemm` function above is a separate, unused-by-dispatch reference kept
    for direct callers and tests.
    """
    M, K = aq.shape
    N = bq.shape[1]
    c = torch.empty((M, N), device=aq.device, dtype=out_dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))

    # The MMA's scale vector is fixed at 32. A wider scale block is the same scale
    # repeated across consecutive 32-wide groups, so replicating here reaches the
    # block-scaled MMA rather than falling back to epilogue scaling. block_size 32
    # leaves both repeats as no-ops.
    rep_k = block_size // _MXFP8_BLOCK_SIZE
    # ceil, not floor: the quantizer pads a trailing partial block whenever
    # block_size does not divide K, and sa/sb legitimately carry that column.
    n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
    sa_mx = sa.repeat_interleave(rep_k, 1)[:, :n_mx]
    sb_mx = sb.repeat_interleave(rep_k, 0)[:n_mx]
    # e8m0 holds a bare exponent, so `_compute_scale` already produced exact powers
    # of two and this cast is lossless; the kernel wants those bytes, not floats.
    sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    wrap_triton(_scaled_gemm_mxfp8_kernel)[grid](
        aq,
        bq,
        c,
        sa8,
        sb8,
        bias,
        M,
        N,
        K,
        n_mx,
        aq.stride(0),
        aq.stride(1),
        bq.stride(0),
        bq.stride(1),
        c.stride(0),
        c.stride(1),
        sa8.stride(0),
        sa8.stride(1),
        sb8.stride(0),
        sb8.stride(1),
        0 if bias is None else bias.stride(0),
        A_FORMAT=_MXFP8_FORMAT[aq.dtype],
        B_FORMAT=_MXFP8_FORMAT[bq.dtype],
        HAS_BIAS=bias is not None,
    )
    return c


# ---------------------------------------------------------------------------
# Scaled grouped GEMM (quantized ragged MoE): fp8 / int8 with block scaling
# ---------------------------------------------------------------------------


_SCALED_GROUPED_CFG = [
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 256, 32, 8, 3),
    (128, 64, 32, 4, 4),
    (128, 128, 32, 8, 2),
    (128, 128, 32, 8, 3),
    (128, 128, 32, 8, 4),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 4, 3),
    (64, 64, 64, 4, 4),
    (64, 128, 64, 8, 4),
    (64, 256, 64, 8, 4),
    (128, 64, 64, 8, 4),
    (128, 128, 64, 8, 3),
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 3),
    (64, 64, 128, 4, 3),
    (64, 128, 128, 8, 4),
    (128, 64, 128, 8, 4),
    (128, 128, 128, 8, 3),
]

_SCALED_GROUPED_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
    )
    for (bm, bn, bk, w, s) in _SCALED_GROUPED_CFG
]


def _early_prune_scaled_grouped_configs(
    configs,
    named_args,
    NUM_SMS,
    A_IS_2D,
    B_IS_2D,
    SCALE_BLOCK_SIZE,
    HAS_BIAS,
    grid=None,
    warmup=None,
):
    del named_args, NUM_SMS, A_IS_2D, B_IS_2D, HAS_BIAS, grid, warmup
    scale_block = SCALE_BLOCK_SIZE
    if not scale_block:
        return configs
    return [c for c in configs if c.kwargs["BLOCK_K"] <= max(32, scale_block)]


@triton.autotune(
    configs=_SCALED_GROUPED_CONFIGS,
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D", "SCALE_BLOCK_SIZE"],
    prune_configs_by={"early_config_prune": _early_prune_scaled_grouped_configs},
)
@triton.jit
def _scaled_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_sag,
    stride_sam,
    stride_sak,
    stride_sbg,
    stride_sbk,
    stride_sbn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    SCALE_BLOCK_SIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged scaled grouped GEMM over any one of M, N, K, with block scaling.

    Same layout model as _grouped_gemm_kernel. The quantized addition is
    `scale_block_idx`, the index along the contraction axis -- global when the
    contraction is dense, per-group re-tiled and carried across the group loop when
    the contraction is itself ragged.

    SCALE_BLOCK_SIZE is the scale block's width along the contraction axis; 0 means
    one block spanning the whole segment. row/tensorwise are that case and cannot
    state a width instead: a ragged contraction has a per-group segment length known
    only at runtime, while SCALE_BLOCK_SIZE is a constexpr. An empty group also owns
    one scale row, which cdiv(0, width) would skip, leaving the cursor a row behind
    for every group after it.

    The optional bias is per group, broadcast over the output's row dim, and added
    to the fp32 accumulator after the scales so it never passes through them.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    # this persistent program owns pid, pid+NUM_SMS, ...
    global_tile_idx = tl.program_id(0)
    group_tile_start = 0  # flat index of the current group's first tile
    m_end = 0  # offs are END-offsets, so the previous end is the next start
    n_end = 0
    k_end = 0
    scale_block_end = 0  # cursor, only advanced when the contraction is ragged
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
            # blocks re-tile inside each group, so the cursor carries across g; the
            # group loop visits g in order, so no prefix scan is needed
            scale_block_start = scale_block_end
            scale_block_end = scale_block_start + (
                1 if SCALE_BLOCK_SIZE == 0 else tl.cdiv(k_size, SCALE_BLOCK_SIZE)
            )
        else:
            k_start = 0
            k_size = K
            scale_block_start = 0
            scale_block_end = (
                1 if SCALE_BLOCK_SIZE == 0 else tl.cdiv(K, SCALE_BLOCK_SIZE)
            )

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for scale_block_idx in range(scale_block_start, scale_block_end):
                if SCALE_BLOCK_SIZE == 0:
                    r0 = k_start
                    r1 = k_start + k_size
                else:
                    r0 = (
                        k_start
                        + (scale_block_idx - scale_block_start) * SCALE_BLOCK_SIZE
                    )
                    r1 = tl.minimum(r0 + SCALE_BLOCK_SIZE, k_start + k_size)
                # one scale block: accumulate in fp32, then apply both scales once
                block_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # a computed trip count rather than `while k < r1`: Triton's software
                # pipeliner runs on scf.for, so a while loop would serialize the
                # global loads instead of overlapping them with the dots
                for step in range(tl.cdiv(r1 - r0, BLOCK_K)):
                    offs_k = r0 + step * BLOCK_K + tl.arange(0, BLOCK_K)  # absolute
                    k_mask = offs_k < r1
                    a_ptrs = (
                        a_ptr
                        + (0 if A_IS_2D else g * stride_ag)
                        + (m_start + offs_m)[:, None] * stride_am
                        + offs_k[None, :] * stride_ak
                    )
                    b_ptrs = (
                        b_ptr
                        + (0 if B_IS_2D else g * stride_bg)
                        + offs_k[:, None] * stride_bk
                        + (n_start + offs_n)[None, :] * stride_bn
                    )
                    a = tl.load(
                        a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0
                    )
                    b = tl.load(
                        b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0
                    )
                    block_acc += tl.dot(a, b).to(tl.float32)
                sa = tl.load(
                    sa_ptr
                    + (0 if A_IS_2D else g * stride_sag)
                    + (m_start + offs_m) * stride_sam
                    + scale_block_idx * stride_sak,
                    mask=m_mask,
                    other=0.0,
                )
                sb = tl.load(
                    sb_ptr
                    + (0 if B_IS_2D else g * stride_sbg)
                    + scale_block_idx * stride_sbk
                    + (n_start + offs_n) * stride_sbn,
                    mask=n_mask,
                    other=0.0,
                )
                acc += sa[:, None] * block_acc * sb[None, :]
            # on acc, not block_acc: the bias is unscaled, so folding it into a
            # per-block partial would scale it and add it once per block
            if HAS_BIAS:
                bias_ptrs = (
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn
                )
                acc += tl.load(bias_ptrs, mask=n_mask, other=0.0)[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


_SCALED_GROUPED_MXFP8_CFG = [
    (bm, bn, bk, w, s) for (bm, bn, bk, w, s) in _SCALED_GROUPED_CFG if bk >= 64
]
_SCALED_GROUPED_MXFP8_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}, num_warps=w, num_stages=s
    )
    for (bm, bn, bk, w, s) in _SCALED_GROUPED_MXFP8_CFG
]


@triton.autotune(
    configs=_SCALED_GROUPED_MXFP8_CONFIGS,
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D"],
)
@triton.jit
def _scaled_grouped_gemm_mxfp8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sa_ptr,
    sb_ptr,
    bias_ptr,
    offs_ptr,
    G,
    M,
    N,
    K,
    stride_ag,
    stride_am,
    stride_ak,
    stride_bg,
    stride_bk,
    stride_bn,
    stride_cg,
    stride_cm,
    stride_cn,
    stride_sag,
    stride_sam,
    stride_sak,
    stride_sbg,
    stride_sbk,
    stride_sbn,
    stride_biasg,
    stride_biasn,
    NUM_SMS: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    A_FORMAT: tl.constexpr,
    B_FORMAT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ragged scaled grouped GEMM, mxfp8 only: same layout model as the kernel below,
    but the e8m0 scales go into the MMA (`QMMA.SF`) instead of the accumulator.

    The scale-block index is `scale_block_start + k0 // 32 + i` for the i-th 32-wide
    group of the tile. `scale_block_start` is 0 whenever the contraction is dense, and
    otherwise carries across the group loop exactly as it does next door -- a ragged
    contraction re-tiles its scale blocks inside each group, so group g's blocks begin
    where g-1's ended. BLOCK_K is a multiple of 32, so a tile's 32-wide groups never
    straddle a scale block.
    """
    M_VARY: tl.constexpr = A_IS_2D and not B_IS_2D
    N_VARY: tl.constexpr = not A_IS_2D and B_IS_2D
    K_VARY: tl.constexpr = A_IS_2D and B_IS_2D

    global_tile_idx = tl.program_id(0)
    group_tile_start = 0
    m_end = 0
    n_end = 0
    k_end = 0
    scale_block_end = 0
    offs_sk = tl.arange(0, BLOCK_K // 32)
    for g in range(G):
        if M_VARY:
            m_start = m_end
            m_end = tl.load(offs_ptr + g)
            m_size = m_end - m_start
        else:
            m_start = 0
            m_size = M
        if N_VARY:
            n_start = n_end
            n_end = tl.load(offs_ptr + g)
            n_size = n_end - n_start
        else:
            n_start = 0
            n_size = N
        if K_VARY:
            k_start = k_end
            k_end = tl.load(offs_ptr + g)
            k_size = k_end - k_start
            scale_block_start = scale_block_end
            scale_block_end = scale_block_start + tl.cdiv(k_size, 32)
        else:
            k_start = 0
            k_size = K
            scale_block_start = 0
            scale_block_end = tl.cdiv(K, 32)

        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        group_tile_count = num_m_tiles * num_n_tiles
        while global_tile_idx < group_tile_start + group_tile_count:
            group_tile_idx = global_tile_idx - group_tile_start
            tile_m = group_tile_idx // num_n_tiles
            tile_n = group_tile_idx % num_n_tiles
            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
            m_mask = offs_m < m_size
            n_mask = offs_n < n_size
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, k_size, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_mask = offs_k < k_size
                a = tl.load(
                    a_ptr
                    + (0 if A_IS_2D else g * stride_ag)
                    + (m_start + offs_m)[:, None] * stride_am
                    + (k_start + offs_k)[None, :] * stride_ak,
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                b = tl.load(
                    b_ptr
                    + (0 if B_IS_2D else g * stride_bg)
                    + (k_start + offs_k)[:, None] * stride_bk
                    + (n_start + offs_n)[None, :] * stride_bn,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
                sbi = scale_block_start + k0 // 32 + offs_sk
                sk_mask = sbi < scale_block_end
                sa = tl.load(
                    sa_ptr
                    + (0 if A_IS_2D else g * stride_sag)
                    + (m_start + offs_m)[:, None] * stride_sam
                    + sbi[None, :] * stride_sak,
                    mask=m_mask[:, None] & sk_mask[None, :],
                    other=0,
                )
                sb = tl.load(
                    sb_ptr
                    + (0 if B_IS_2D else g * stride_sbg)
                    + (n_start + offs_n)[:, None] * stride_sbn
                    + sbi[None, :] * stride_sbk,
                    mask=n_mask[:, None] & sk_mask[None, :],
                    other=0,
                )
                acc = tl.dot_scaled(a, sa, A_FORMAT, b, sb, B_FORMAT, acc=acc)
            if HAS_BIAS:
                acc += tl.load(
                    bias_ptr + g * stride_biasg + (n_start + offs_n) * stride_biasn,
                    mask=n_mask,
                    other=0.0,
                )[None, :]
            c_ptrs = (
                c_ptr
                + (g * stride_cg if K_VARY else 0)
                + (m_start + offs_m)[:, None] * stride_cm
                + (n_start + offs_n)[None, :] * stride_cn
            )
            tl.store(
                c_ptrs,
                acc.to(c_ptr.dtype.element_ty),
                mask=m_mask[:, None] & n_mask[None, :],
            )
            global_tile_idx += NUM_SMS
        group_tile_start += group_tile_count


def _check_scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None,
    operand_formats: dict | set,
) -> None:
    """Checks shared by `epilogue_scaled_grouped_gemm` and `mxfp8_scaled_grouped_gemm`.
    `operand_formats` is the allowed aq/bq dtype set, the one thing the two kernels
    genuinely disagree on.
    """
    tensors = (aq, bq, sa, sb, offs) if bias is None else (aq, bq, sa, sb, offs, bias)
    _require(
        all(isinstance(tensor, torch.Tensor) for tensor in tensors),
        "scaled grouped GEMM inputs must be torch.Tensor objects",
    )
    _require(aq.ndim in (2, 3) and bq.ndim in (2, 3), "aq and bq must be 2-D or 3-D")
    _require(sa.ndim in (2, 3) and sb.ndim in (2, 3), "sa and sb must be 2-D or 3-D")
    _require(offs.ndim == 1 and offs.numel() > 0, "offs must be a nonempty 1-D tensor")
    _require(isinstance(out_dtype, torch.dtype), "out_dtype must be torch.dtype")
    _require(
        type(block_size) is int and block_size >= 0,
        "block_size must be a non-negative integer",
    )
    _require(
        all(tensor.is_cuda for tensor in tensors),
        "scaled grouped GEMM requires CUDA tensors",
    )
    _require(
        all(tensor.device == aq.device for tensor in tensors),
        "all scaled grouped GEMM tensors must share one device",
    )
    _require(aq.dtype in operand_formats, f"unsupported aq dtype {aq.dtype}")
    _require(bq.dtype in operand_formats, f"unsupported bq dtype {bq.dtype}")
    _require(sa.dtype == sb.dtype == torch.float32, "sa and sb must both be float32")
    _require(offs.dtype == torch.int32, f"offs must be int32, got {offs.dtype}")
    _require(offs.is_contiguous(), "offs must be contiguous")
    _require(out_dtype in _FLOAT_DTYPES, f"unsupported out_dtype {out_dtype}")
    _require(
        aq.dtype.is_floating_point == bq.dtype.is_floating_point,
        "aq and bq must use the same dtype family",
    )
    _require(aq.ndim == 2 or bq.ndim == 2, "3D x 3D has no ragged dim; use torch.bmm")
    _require(
        aq.shape[-1] == bq.shape[-2],
        f"contraction mismatch: aq {aq.shape[-1]}, bq {bq.shape[-2]}",
    )
    if aq.ndim == 3:
        _require(aq.shape[0] == offs.numel(), "aq group count must match offs")
    if bq.ndim == 3:
        _require(bq.shape[0] == offs.numel(), "bq group count must match offs")
    if bias is not None:
        _require(aq.ndim != 3, "bias not supported for ragged-N")
        _require(
            bias.shape == (offs.numel(), bq.shape[-1]),
            f"bias must be ({offs.numel()}, {bq.shape[-1]})",
        )
        _require(bias.dtype in _FLOAT_DTYPES, f"unsupported bias dtype {bias.dtype}")


def _grouped_scales_are_valid(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    block_size: int,
) -> bool:
    """The three-way scale-layout check shared by `epilogue_scaled_grouped_gemm` and
    `mxfp8_scaled_grouped_gemm` -- identical for both, since it depends only on which
    dim `offs` partitions, not on the operand format.
    """
    if aq.ndim == 2 and bq.ndim == 3:
        blocks = triton.cdiv(aq.shape[1], block_size) if block_size else 1
        return sa.shape == (aq.shape[0], blocks) and sb.shape == (
            bq.shape[0],
            blocks,
            bq.shape[2],
        )
    elif aq.ndim == 2 and bq.ndim == 2:
        return (
            sa.ndim == 2
            and sb.ndim == 2
            and sa.shape[0] == aq.shape[0]
            and sa.shape[1] == sb.shape[0]
            and sb.shape[1] == bq.shape[1]
            and (block_size != 0 or sa.shape[1] == offs.numel())
        )
    else:
        blocks = triton.cdiv(aq.shape[2], block_size) if block_size else 1
        return sa.shape == (aq.shape[0], aq.shape[1], blocks) and sb.shape == (
            blocks,
            bq.shape[1],
        )


def epilogue_scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, out_dtype, block_size, bias, _QUANTIZED_DTYPES
    )
    _require(
        _grouped_scales_are_valid(aq, bq, sa, sb, offs, block_size),
        "invalid scale layout",
    )
    capability = torch.cuda.get_device_capability(aq.device)
    minimum = (8, 9) if aq.dtype.is_floating_point else (8, 0)
    _require(
        capability >= minimum,
        f"scaled grouped GEMM requires SM{minimum[0]}{minimum[1]} or newer, got SM{capability[0]}{capability[1]}",
    )
    return _epilogue_scaled_grouped_mm(
        aq, bq, sa, sb, offs, out_dtype, block_size, bias
    )


def mxfp8_scaled_grouped_gemm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    _check_scaled_grouped_gemm(
        aq, bq, sa, sb, offs, out_dtype, block_size, bias, _MXFP8_FORMAT
    )
    # The ragged-K layout re-tiles its scale blocks *inside* each group at runtime --
    # group g's blocks start where group g-1's ended, sized off `offs` via
    # cdiv(k_size_g, 32) -- so a flat repeat_interleave over the whole scale-block axis
    # would smear a replicated scale across a group boundary. `_mxfp8_scaled_grouped_mm`
    # therefore only supports the kernel's native 32-wide block for ragged-K; the other
    # two layouts (ragged M, ragged N) have a single non-ragged K shared by every group,
    # where one repeat_interleave over the whole axis is exact, so they take the
    # generalized multiple-of-32 condition. Do not simplify this to one shared
    # condition. block_size 0 is the "one block spans the whole segment" sentinel, not
    # a real width -- 0 % 32 == 0 in Python, so it must be excluded explicitly or it
    # would collapse rep_k to 0 in `_mxfp8_scaled_grouped_mm` and hand the kernel a
    # zero-width scale tensor.
    is_ragged_k = aq.ndim == 2 and bq.ndim == 2
    if is_ragged_k:
        _require(
            block_size == _MXFP8_BLOCK_SIZE,
            f"mxfp8 ragged-K grouped GEMM requires block_size == {_MXFP8_BLOCK_SIZE}, got {block_size}",
        )
    else:
        _require(
            block_size != 0 and block_size % _MXFP8_BLOCK_SIZE == 0,
            f"mxfp8 grouped GEMM requires block_size a nonzero multiple of {_MXFP8_BLOCK_SIZE}, got {block_size}",
        )
    _require(
        _grouped_scales_are_valid(aq, bq, sa, sb, offs, block_size),
        "invalid scale layout",
    )
    capability = torch.cuda.get_device_capability(aq.device)
    _require(
        capability >= (10, 0),
        f"mxfp8 scaled grouped GEMM requires SM100 or newer, got SM{capability[0]}{capability[1]}",
    )
    return _mxfp8_scaled_grouped_mm(aq, bq, sa, sb, offs, out_dtype, block_size, bias)


# Dispatch reaches this entry directly, gated only by device/SM capability; see the
# entry's own docstring for what it still checks and why.
@register_kernel(
    op="gemm.mxfp8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(10, 0))}),
)
@triton_op("jit_kernel::mxfp8_scaled_grouped_mm", mutates_args={})
def _mxfp8_scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged scaled grouped GEMM, layout picked from the operand ranks, mxfp8 only.

    At `block_size` 32 always and any multiple of 32 for the ragged-M/ragged-N layouts
    (see `mxfp8_scaled_grouped_gemm`'s ragged-K note for why ragged-K is locked to 32).

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand's axis order with the contraction axis replaced by
    the scale-block axis, so transposing an operand transposes its scale too.

    `bias` is an optional (E,N) tensor broadcast over the output's row dim, added to
    the fp32 accumulator in the epilogue so it never passes through the scales.

    Dispatch reaches this entry directly, gated only by device/SM capability; the
    `mxfp8_scaled_grouped_gemm` function above is a separate, unused-by-dispatch
    reference kept for direct callers and tests.
    """
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    E = offs.shape[0]
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (E,K,N) -> (M,N), ragged M
        M, N, K = 0, bq.shape[2], aq.shape[1]
        output_size = (aq.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (E,M,N), ragged K
        M, N, K = aq.shape[0], bq.shape[1], 0
        output_size = (E, M, N)
    else:  # (E,M,K) x (K,N) -> (M,N), ragged N
        M, N, K = aq.shape[1], 0, aq.shape[2]
        output_size = (M, bq.shape[1])

    c = torch.empty(output_size, device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)

    is_ragged_k = a_is_2d and b_is_2d
    if is_ragged_k:
        sa8 = sa.to(torch.float8_e8m0fnu).view(torch.uint8)
        sb8 = sb.to(torch.float8_e8m0fnu).view(torch.uint8)
    else:
        # Same replication trick as `_mxfp8_scaled_mm`: K is shared by every group here,
        # so one repeat_interleave over the whole scale-block axis is exact. The
        # axis is always last for sa and second-to-last for sb, in both the 2-D
        # and 3-D (E, ...) shapes, so this covers ragged M and ragged N alike.
        rep_k = block_size // _MXFP8_BLOCK_SIZE
        # ceil, not floor: the quantizer pads a trailing partial block whenever
        # block_size does not divide K, and sa/sb legitimately carry that column.
        n_mx = triton.cdiv(K, _MXFP8_BLOCK_SIZE)
        sa_mx = sa.repeat_interleave(rep_k, dim=-1).narrow(-1, 0, n_mx)
        sb_mx = sb.repeat_interleave(rep_k, dim=-2).narrow(-2, 0, n_mx)
        sa8 = sa_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
        sb8 = sb_mx.contiguous().to(torch.float8_e8m0fnu).view(torch.uint8)
    wrap_triton(_scaled_grouped_gemm_mxfp8_kernel)[(num_sms,)](
        aq,
        bq,
        c,
        sa8,
        sb8,
        bias,
        offs,
        E,
        M,
        N,
        K,
        0 if a_is_2d else aq.stride(0),
        aq.stride(-2),
        aq.stride(-1),
        0 if b_is_2d else bq.stride(0),
        bq.stride(-2),
        bq.stride(-1),
        c.stride(0) if c.ndim == 3 else 0,
        c.stride(-2),
        c.stride(-1),
        0 if a_is_2d else sa8.stride(0),
        sa8.stride(-2),
        sa8.stride(-1),
        0 if b_is_2d else sb8.stride(0),
        sb8.stride(-2),
        sb8.stride(-1),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        A_FORMAT=_MXFP8_FORMAT[aq.dtype],
        B_FORMAT=_MXFP8_FORMAT[bq.dtype],
        HAS_BIAS=bias is not None,
    )
    return c


@register_kernel(
    op="gemm.int8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 0))}),
)
@register_kernel(
    op="gemm.fp8_scaled_grouped_mm",
    backend="triton",
    build="jit",
    autograd=False,
    capabilities=frozenset({cuda(min_arch=(8, 9))}),
)
@triton_op("jit_kernel::epilogue_scaled_grouped_mm", mutates_args={})
def _epilogue_scaled_grouped_mm(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    offs: torch.Tensor,
    out_dtype: torch.dtype,
    block_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Ragged scaled grouped GEMM, layout picked from the operand ranks.

    Layouts, by which dim `offs` (1-D int32 cumulative end-offsets) partitions:
        (M,K) x (E,K,N) -> (M,N)    ragged M
        (M,K) x (K,N)   -> (E,M,N)  ragged K
        (E,M,K) x (K,N) -> (M,N)    ragged N

    `sa`/`sb` mirror their operand's axis order with the contraction axis replaced by
    the scale-block axis, so transposing an operand transposes its scale too.
    `block_size` 0 means one scale block spanning the whole contraction segment.

    `bias` is an optional (E,N) tensor broadcast over the output's row dim, added to
    the fp32 accumulator in the epilogue so it never passes through the scales.

    Dispatch reaches this entry directly, gated only by device/SM capability; the
    `epilogue_scaled_grouped_gemm` function above is a separate, unused-by-dispatch
    reference kept for direct callers and tests. Unlike the dense `_epilogue_scaled_mm`, this
    kernel has no numerically-fragile constraint to re-check here: it masks the
    reduction to each scale block's end rather than clamping BLOCK_K to it, so a
    non-power-of-two `block_size` (e.g. 48) is exercised and correct, not just
    tolerated (see `test_scaled_grouped_layouts_match_oracle`'s `bs=48` case).
    """
    a_is_2d, b_is_2d = aq.ndim == 2, bq.ndim == 2
    E = offs.shape[0]
    # the ragged dim's extent is passed as 0: unused, and keeps the autotune key stable
    if a_is_2d and not b_is_2d:  # (M,K) x (E,K,N) -> (M,N), ragged M
        M, N, K = 0, bq.shape[2], aq.shape[1]
        output_size = (aq.shape[0], N)
    elif a_is_2d and b_is_2d:  # (M,K) x (K,N) -> (E,M,N), ragged K
        M, N, K = aq.shape[0], bq.shape[1], 0
        output_size = (E, M, N)
    else:  # (E,M,K) x (K,N) -> (M,N), ragged N
        M, N, K = aq.shape[1], 0, aq.shape[2]
        output_size = (M, bq.shape[1])

    c = torch.empty(output_size, device=aq.device, dtype=out_dtype)
    num_sms = _num_sms(aq.device)

    wrap_triton(_scaled_grouped_gemm_kernel)[(num_sms,)](
        aq,
        bq,
        c,
        sa,
        sb,
        bias,
        offs,
        E,
        M,
        N,
        K,
        0 if a_is_2d else aq.stride(0),
        aq.stride(-2),
        aq.stride(-1),
        0 if b_is_2d else bq.stride(0),
        bq.stride(-2),
        bq.stride(-1),
        c.stride(0) if c.ndim == 3 else 0,
        c.stride(-2),
        c.stride(-1),
        0 if a_is_2d else sa.stride(0),
        sa.stride(-2),
        sa.stride(-1),
        0 if b_is_2d else sb.stride(0),
        sb.stride(-2),
        sb.stride(-1),
        0 if bias is None else bias.stride(0),
        0 if bias is None else bias.stride(1),
        NUM_SMS=num_sms,
        A_IS_2D=a_is_2d,
        B_IS_2D=b_is_2d,
        SCALE_BLOCK_SIZE=block_size,
        HAS_BIAS=bias is not None,
    )
    return c
