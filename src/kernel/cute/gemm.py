"""CuTe DSL MXFP8 dense GEMM for SM120.

The kernel keeps both operands K-contiguous, stages XOR-swizzled 128^3 tiles with
`cp.async`, and applies E8M0 scales in the tensor core. This avoids host relayouts
and shared-memory bank conflicts at the cost of a fixed tile shape and pipeline.
M/N edges are predicated; K must align to the 32-element MXFP8 scale block.
"""

from __future__ import annotations

import cutlass
import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as blackwell_helpers
import cutlass.utils.blockscaled_layout as blockscaled_layout
import torch
from cutlass.cute.atom import make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu import warp as cute_warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import Float8E8M0FNU, Float32

TILE_M = 128
TILE_N = 128
TILE_K = 128
assert TILE_M == TILE_N == 128 and TILE_K == 128, (
    "the SF write view, the SFB mode regroup and copy_a's layout are derived "
    "for a 128^3 tile"
)
SCALE_VEC = 32
TILE_SF = TILE_K // SCALE_VEC
THREADS = 256
SF_TILE_BYTES = TILE_M * TILE_SF
# Swizzle 16-byte copy granules by row to spread MMA reads across all 32 banks.
# The parameters must permute exactly one TILE_K-byte row: a smaller shift loses
# bank spreading, while a larger shift reaches outside the row and breaks results.
SWIZZLE_BITS, SWIZZLE_BASE, SWIZZLE_SHIFT = 3, 4, 3
assert (
    TILE_K == (1 << SWIZZLE_BASE) << SWIZZLE_BITS and SWIZZLE_SHIFT == SWIZZLE_BITS
), (
    f"Swizzle<{SWIZZLE_BITS},{SWIZZLE_BASE},{SWIZZLE_SHIFT}> is not a permutation of "
    f"one {TILE_K}-byte K-major row: need BASE + BITS == BASE + SHIFT == "
    f"log2(TILE_K) == {TILE_K.bit_length() - 1}, so that the swizzle XORs the row "
    f"index into the granule index and touches nothing else"
)
# Three stages maximize overlap while fitting SM120's 99 KiB shared-memory limit.
STAGES = 3
assert STAGES >= 2, "the mainloop needs at least one tile in flight behind the MMA"
SMEM_BYTES = STAGES * (TILE_M * TILE_K + TILE_N * TILE_K + 2 * SF_TILE_BYTES)
assert SMEM_BYTES <= 99 * 1024, (
    f"staged shared memory is {SMEM_BYTES} bytes, over sm120's 99 KiB limit -- "
    f"lower STAGES or shrink the tile (TILE_K halves it fastest)"
)

# M stays dynamic; schedule, dimensions, and dtypes baked into the artifact form the
# cache key.
_COMPILED: dict[tuple, object] = {}
_SCHEDULES = frozenset({"cpasync"})

FP8_FORMATS = frozenset({torch.float8_e4m3fn, torch.float8_e5m2})
OUT_DTYPES = frozenset({torch.float32, torch.float16, torch.bfloat16})


class _MmaMXF8Op(cute_warp.MmaMXF8Op):
    """`MmaMXF8Op` with the A and B formats chosen independently.

    The stock wrapper shares one dtype, although the atom accepts separate E4M3/E5M2
    qualifiers with identical fragment layouts. `b_dtype` is outside the base frozen
    dataclass, so the compile cache must continue to key on both operand dtypes.
    """

    def __init__(self, a_dtype, b_dtype):
        super().__init__(a_dtype, Float32, Float8E8M0FNU)
        object.__setattr__(self, "b_dtype", b_dtype)  # the base is a frozen dataclass

    def _make_trait(self, *, loc=None, ip=None, **kwargs) -> cute_warp.MmaMXF8Trait:
        ty = cute_nvgpu_ir.MmaAtomSM120BlockScaledType.get(
            _pack_shape(self.shape_mnk, loc=loc, ip=ip).type.attribute,
            self.sf_vec_size,
            self.use_sf_layout_TV,
            self.ab_dtype.mlir_type,
            self.b_dtype.mlir_type,
            self.acc_dtype.mlir_type,
            self.sf_type.mlir_type,
        )
        return cute_warp.MmaMXF8Trait(make_atom(ty, loc=loc, ip=ip))


def _make_mxfp8_tiled_mma(a_dtype, b_dtype) -> cute.TiledMma:
    """Build the eight-warp MMA with the N permutation required by scale fragments."""
    op = _MmaMXF8Op(a_dtype, b_dtype)
    permutation = blackwell_helpers.get_permutation_mnk(
        (TILE_M, TILE_N, TILE_K), SCALE_VEC, True
    )
    return cute.make_tiled_mma(op, (4, 2, 1), permutation_mnk=permutation)


def _operand_smem_layout(tile_mn: int) -> cute.ComposedLayout:
    """Use K-major tiles for vector loads and XOR rows to avoid bank conflicts."""
    return cute.make_composed_layout(
        cute.make_swizzle(SWIZZLE_BITS, SWIZZLE_BASE, SWIZZLE_SHIFT),
        0,
        cute.make_ordered_layout((tile_mn, TILE_K, STAGES), order=(1, 0, 2)),
    )


def _partition_mxfp8_sf(sf: cute.Tensor, tiled_mma: cute.TiledMma, tidx, is_a: bool):
    """Return this thread's scale view; stock helpers discard the source view."""
    thrfrg = blackwell_helpers.thrfrg_SFA if is_a else blackwell_helpers.thrfrg_SFB
    thr = cute.make_tensor(sf.iterator, thrfrg(sf.layout, tiled_mma))
    vmnk = tiled_mma.thr_layout_vmnk.get_flat_coord(tidx)
    coord = (vmnk[0], (vmnk[1], vmnk[3])) if is_a else (vmnk[0], (vmnk[2], vmnk[3]))
    part = cute.group_modes(cute.flatten(thr[coord, (None, None)]), 0, 2)
    # B spans four 32-wide MMA N tiles, so its flattened view has one extra mode.
    return part if is_a else cute.group_modes(part, 1, 3)


def _tiled_copy(dtype, thr_shape, thr_stride, val_shape, is_async) -> cute.TiledCopy:
    """Copy one aligned vector per predicate, optionally with `cp.async`."""
    op = cute.nvgpu.cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_tiled_copy_tv(
        cute.make_copy_atom(
            op,
            dtype,
            num_bits_per_copy=dtype.width * val_shape[0] * val_shape[1],
        ),
        cute.make_layout(thr_shape, stride=thr_stride),
        cute.make_layout(val_shape),
    )


def _sf_stage(buf: cute.Tensor, stage, layout: cute.Layout) -> cute.Tensor:
    """Restore known alignment lost through byte-wise stage pointer arithmetic."""
    return cute.make_tensor((buf.iterator + stage * SF_TILE_BYTES).align(16), layout)


def _group_rest(t: cute.Tensor) -> cute.Tensor:
    """Collapse a partitioned tensor's trailing modes, leaving (V, rest)."""
    return cute.group_modes(t, 1, cute.rank(t))


def _tile_coords(shape, tiler, tile):
    """The global coordinates of one tile, shaped exactly like the tile itself."""
    return cute.zipped_divide(cute.make_identity_tensor(shape), tiler)[
        (None, None), tile
    ]


def _copy_tile(tiled_copy, tidx, gS, cS, sD, bound, zero_k_tail: bool):
    """Stage in-bounds vectors and zero dropped K vectors in the same stage.

    Stale M/N-edge values are never stored, but stale K-tail values multiply into
    every output -- and an uninitialised e8m0 byte can be the 0xFF NaN encoding, which
    `NaN * 0` spreads across an accumulator row. Zeroing during staging targets the
    buffer the MMA will later consume, which is what staging requires: the tail tile
    is copied several iterations before it is read. Scale tensors are padded instead
    so their vector copies remain in bounds.
    """
    copy = tiled_copy.get_slice(tidx)
    src = _group_rest(copy.partition_S(gS))
    crd = _group_rest(copy.partition_S(cS))
    dst = _group_rest(copy.partition_D(sD))
    n_vec = cute.size(src, mode=[1])
    pred = cute.make_rmem_tensor((1, n_vec), cutlass.Boolean)
    for i in range(n_vec):
        pred[0, i] = cute.elem_less(crd[0, i], bound)
    cute.copy(tiled_copy, src, dst, pred=pred)
    if zero_k_tail:
        zeros = cute.make_fragment_like(dst)
        zeros.fill(0)
        zpred = cute.make_rmem_tensor((1, n_vec), cutlass.Boolean)
        for i in range(n_vec):
            zpred[0, i] = crd[0, i][1] >= bound[1]
        cute.copy(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                sD.element_type,
                num_bits_per_copy=sD.element_type.width * cute.size(dst, mode=[0]),
            ),
            zeros,
            dst,
            pred=zpred,
        )


@cute.jit
def _stage_k_tile(k_tile, stage, tidx, k_tail, a, b, sfa, sfb):
    """Stage one K tile; split the CTA because each scale operand needs half.

    `@cute.jit` is required to stage the `tidx`-dependent control flow.
    """
    k = (None, None, k_tile)
    _copy_tile(a[0], tidx, a[1][k], a[2][k], a[3][None, None, stage], a[4], k_tail)
    _copy_tile(b[0], tidx, b[1][k], b[2][k], b[3][None, None, stage], b[4], k_tail)
    half = THREADS // 2
    if tidx < half:
        dst = _sf_stage(sfa[3], stage, sfa[4])
        _copy_tile(sfa[0], tidx, sfa[1][k], sfa[2][k], dst, sfa[5], False)
    else:
        dst = _sf_stage(sfb[3], stage, sfb[4])
        _copy_tile(sfb[0], tidx - half, sfb[1][k], sfb[2][k], dst, sfb[5], False)


@cute.kernel
def _scaled_gemm_mxfp8_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    tile_m, tile_n, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    sA = smem.allocate_tensor(mA.element_type, _operand_smem_layout(TILE_M), 16)
    sB = smem.allocate_tensor(mB.element_type, _operand_smem_layout(TILE_N), 16)
    sSFA = smem.allocate_tensor(
        Float8E8M0FNU, cute.make_layout(SF_TILE_BYTES * STAGES), 16
    )
    sSFB = smem.allocate_tensor(
        Float8E8M0FNU, cute.make_layout(SF_TILE_BYTES * STAGES), 16
    )
    # A compact write view over the scale layouts' shared 32x4x4 swizzle:
    # offset = 16 * (row % 32) + 4 * (row // 32) + scale. One view serves both
    # operands, since either one's k-tile of scales is the same 128x4 block.
    sf_write = cute.make_layout(((32, 4), TILE_SF), stride=((16, 4), 1))

    gA = cute.zipped_divide(mA, (TILE_M, TILE_K))[(None, None), (tile_m, None)]
    gB = cute.zipped_divide(mB, (TILE_N, TILE_K))[(None, None), (tile_n, None)]
    gSFA = cute.zipped_divide(mSFA, (TILE_M, TILE_SF))[(None, None), (tile_m, None)]
    gSFB = cute.zipped_divide(mSFB, (TILE_N, TILE_SF))[(None, None), (tile_n, None)]
    gD = cute.zipped_divide(mD, (TILE_M, TILE_N))[(None, None), (tile_m, tile_n)]
    cA = _tile_coords(mA.shape, (TILE_M, TILE_K), (tile_m, None))
    cB = _tile_coords(mB.shape, (TILE_N, TILE_K), (tile_n, None))
    cSFA = _tile_coords(mSFA.shape, (TILE_M, TILE_SF), (tile_m, None))
    cSFB = _tile_coords(mSFB.shape, (TILE_N, TILE_SF), (tile_n, None))
    cD = _tile_coords(mD.shape, (TILE_M, TILE_N), (tile_m, tile_n))

    # Match copy widths to one swizzle granule and one K tile's scale group.
    copy_a = _tiled_copy(mA.element_type, (32, 8), (8, 1), (1, 16), True)
    copy_b = _tiled_copy(mB.element_type, (32, 8), (8, 1), (1, 16), True)
    copy_sf = _tiled_copy(Float8E8M0FNU, (TILE_M, 1), (1, 1), (1, TILE_SF), True)

    thr_mma = tiled_mma.get_slice(tidx)
    tCgD = thr_mma.partition_C(gD)
    tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA[None, None, 0]))
    tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB[None, None, 0]))
    tCrSFA = cute.make_fragment_like(
        _partition_mxfp8_sf(_sf_stage(sSFA, 0, sfa_smem_layout), tiled_mma, tidx, True)
    )
    tCrSFB = cute.make_fragment_like(
        _partition_mxfp8_sf(_sf_stage(sSFB, 0, sfb_smem_layout), tiled_mma, tidx, False)
    )

    acc = cute.make_rmem_tensor(tCgD.shape, Float32)
    acc.fill(0.0)

    k_tail = mA.shape[1] % TILE_K != 0
    n_k = cute.size(gA, mode=[2])

    job_a = (copy_a, gA, cA, sA, mA.shape)
    job_b = (copy_b, gB, cB, sB, mB.shape)
    job_sfa = (copy_sf, gSFA, cSFA, sSFA, sf_write, mSFA.shape)
    job_sfb = (copy_sf, gSFB, cSFB, sSFB, sf_write, mSFB.shape)

    # Empty groups keep wait counts aligned when K has fewer than STAGES tiles.
    for s in range(STAGES - 1):
        if s < n_k:
            _stage_k_tile(s, s, tidx, k_tail, job_a, job_b, job_sfa, job_sfb)
        cute.arch.cp_async_commit_group()

    # Unroll four iterations so the compiler can overlap MMA with shared-to-register
    # loads; deeper unrolling adds pressure without enough additional overlap.
    for k_tile in cutlass.range(n_k, unroll=4):
        stage = k_tile % STAGES
        cute.arch.cp_async_wait_group(STAGES - 2)
        # wait_group is per-thread; the barrier makes the stage visible CTA-wide.
        cute.arch.barrier()

        # Refill first to overlap the copy and shorten address-register live ranges.
        nxt = k_tile + STAGES - 1
        if nxt < n_k:
            _stage_k_tile(
                nxt, nxt % STAGES, tidx, k_tail, job_a, job_b, job_sfa, job_sfb
            )
        cute.arch.cp_async_commit_group()

        cute.autovec_copy(thr_mma.partition_A(sA[None, None, stage]), tCrA)
        cute.autovec_copy(thr_mma.partition_B(sB[None, None, stage]), tCrB)
        cute.autovec_copy(
            _partition_mxfp8_sf(
                _sf_stage(sSFA, stage, sfa_smem_layout), tiled_mma, tidx, True
            ),
            tCrSFA,
        )
        cute.autovec_copy(
            _partition_mxfp8_sf(
                _sf_stage(sSFB, stage, sfb_smem_layout), tiled_mma, tidx, False
            ),
            tCrSFB,
        )
        cute.gemm(tiled_mma, acc, [tCrA, tCrSFA], [tCrB, tCrSFB], acc)

    tCrD = cute.make_fragment_like(tCgD)
    tCrD.store(acc.load().to(mD.element_type))
    # Interior tiles use faster vector stores; edge tiles require element-wise
    # predication. Both paths convert first and write the same bytes.
    if (tile_m + 1) * TILE_M <= mD.shape[0] and (tile_n + 1) * TILE_N <= mD.shape[1]:
        cute.autovec_copy(tCrD, tCgD)
    else:
        tCcD = thr_mma.partition_C(cD)
        for i in cutlass.range(cute.size(tCgD), unroll_full=True):
            if cute.elem_less(tCcD[i], mD.shape):
                tCgD[i] = tCrD[i]


@cute.jit
def _scaled_gemm_mxfp8_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa: cute.Tensor,
    mSFB: cute.Tensor,
    mD: cute.Tensor,
):
    tiled_mma = _make_mxfp8_tiled_mma(mA.element_type, mB.element_type)
    M, K = mA.shape
    N = mB.shape[0]
    # Restate SFA so its dynamic M retains the padded scale-block row pitch.
    padded_blocks = sfa.shape[1]
    mSFA = cute.make_tensor(
        sfa.iterator, cute.make_layout((M, padded_blocks), stride=(padded_blocks, 1))
    )
    tile = (TILE_M, TILE_N, TILE_K)
    # Keep each helper authoritative even though both scale layouts currently match.
    sfa_smem_layout = cute.select(
        blockscaled_layout.sm120_make_smem_layout_sfa(tiled_mma, tile, SCALE_VEC, 1),
        mode=[0, 1],
    )
    sfb_smem_layout = cute.select(
        blockscaled_layout.sm120_make_smem_layout_sfb(tiled_mma, tile, SCALE_VEC, 1),
        mode=[0, 1],
    )
    _scaled_gemm_mxfp8_kernel(
        mA, mB, mSFA, mSFB, mD, tiled_mma, sfa_smem_layout, sfb_smem_layout
    ).launch(
        grid=[cute.ceil_div(M, TILE_M), cute.ceil_div(N, TILE_N), 1],
        block=[THREADS, 1, 1],
    )


def scaled_gemm_mxfp8(
    aq: torch.Tensor,
    bq: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out_dtype: torch.dtype,
    schedule: str = "cpasync",
) -> torch.Tensor:
    """Compute `(M,K) x (N,K)^T` with E4M3/E5M2 data and E8M0 block scales.

    Operands and scales must be K-contiguous. M/N may be arbitrary; K must be a
    multiple of the 32-element MXFP8 scale block.
    """
    M, K = aq.shape
    N = bq.shape[0]
    assert aq.dtype in FP8_FORMATS and bq.dtype in FP8_FORMATS, (
        f"the CuTe mxfp8 GEMM takes e4m3 or e5m2 operands, got {aq.dtype} x {bq.dtype}"
    )
    assert bq.shape[1] == K, f"contraction mismatch: aq {K}, bq {bq.shape[1]}"
    # Operand strides are baked into the compiled kernel.
    assert aq.is_contiguous() and bq.is_contiguous(), "aq and bq must be contiguous"
    assert K % SCALE_VEC == 0, f"K must be a multiple of {SCALE_VEC}, got {K}"
    assert sa.shape == (M, K // SCALE_VEC) and sb.shape == (N, K // SCALE_VEC), (
        f"scale shapes {tuple(sa.shape)} / {tuple(sb.shape)} do not match "
        f"({M}, {K // SCALE_VEC}) / ({N}, {K // SCALE_VEC})"
    )
    assert out_dtype in OUT_DTYPES, f"unsupported out_dtype {out_dtype}"
    assert schedule in _SCHEDULES, f"unsupported schedule {schedule}"

    out = torch.empty(M, N, device=aq.device, dtype=out_dtype)
    sa8 = sa.to(torch.float8_e8m0fnu).contiguous()
    sb8 = sb.to(torch.float8_e8m0fnu).contiguous()
    # Scale copies are four bytes wide, so pad short groups with zero. E8M0's 0xFF
    # is NaN and would poison the accumulator even when the paired operand is zero.
    n_scale_blocks = K // SCALE_VEC
    padded_blocks = -(-n_scale_blocks // TILE_SF) * TILE_SF
    if padded_blocks != n_scale_blocks:
        sa_padded = torch.zeros(M, padded_blocks, device=sa8.device, dtype=sa8.dtype)
        sa_padded[:, :n_scale_blocks] = sa8
        sa8 = sa_padded
        sb_padded = torch.zeros(N, padded_blocks, device=sb8.device, dtype=sb8.dtype)
        sb_padded[:, :n_scale_blocks] = sb8
        sb8 = sb_padded
    args = (
        from_dlpack(aq, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=aq.dim_order(), divisibility=1
        ),
        from_dlpack(bq, assumed_align=16),
        from_dlpack(sa8, assumed_align=4).mark_compact_shape_dynamic(
            mode=0, stride_order=sa8.dim_order(), divisibility=1
        ),
        from_dlpack(sb8, assumed_align=4),
        from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=out.dim_order(), divisibility=1
        ),
    )
    key = (schedule, K, N, aq.dtype, bq.dtype, out_dtype)
    compiled = _COMPILED.get(key)
    if compiled is None:
        compiled = cute.compile(_scaled_gemm_mxfp8_host, *args)
        _COMPILED[key] = compiled
    compiled(*args)
    return out
