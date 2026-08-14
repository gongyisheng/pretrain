"""SM120 CuTe MXFP8 GEMM with cp.async and cooperative TMA schedules."""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass._mlir.dialects.cute_nvgpu as cute_nvgpu_ir
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell_helpers
import cutlass.utils.blockscaled_layout as blockscaled_layout
import cutlass.utils.hopper_helpers as hopper_helpers
import torch
from cutlass.cute.atom import make_atom
from cutlass.cute.core import _pack_shape
from cutlass.cute.nvgpu import cpasync
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
COOP_MMA_WARPS = 8
COOP_TMA_WARPS = 1
COOP_THREADS = (COOP_MMA_WARPS + COOP_TMA_WARPS) * 32
COOP_LOAD_REGISTERS = 40
COOP_MMA_REGISTERS = 232
COOP_BUFFER_ALIGNMENT = 1024
COOP_EPILOGUE_TILE = (64, 32)
COOP_CLUSTER_SHAPE = (1, 1, 1)
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
_SCHEDULES = frozenset({"cpasync", "cooperative"})

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


def _cooperative_operand_smem_layout(tile_mn: int, stages: int):
    """Build the same 128-byte swizzle with an output-dtype-dependent stage count."""
    return cute.make_composed_layout(
        cute.make_swizzle(SWIZZLE_BITS, SWIZZLE_BASE, SWIZZLE_SHIFT),
        0,
        cute.make_ordered_layout((tile_mn, TILE_K, stages), order=(1, 0, 2)),
    )


def _cooperative_epilogue_smem_layout(out_dtype):
    """Use one compact row-major epilogue stage so every output dtype keeps two AB stages."""
    atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        hopper_helpers.get_smem_layout_atom(
            utils.LayoutEnum.ROW_MAJOR, out_dtype, COOP_EPILOGUE_TILE[1]
        ),
        out_dtype,
    )
    return cute.tile_to_shape(
        atom,
        cute.append(COOP_EPILOGUE_TILE, 1),
        order=(0, 1, 2),
    )


def _cooperative_stage_count(a_dtype, b_dtype, out_dtype, tiled_mma):
    """Reserve one epilogue tile, then spend the remaining SM120 capacity on AB/SF."""
    tile = (TILE_M, TILE_N, TILE_K)
    a_stage = cute.slice_(_cooperative_operand_smem_layout(TILE_M, 1), (None, None, 0))
    b_stage = cute.slice_(_cooperative_operand_smem_layout(TILE_N, 1), (None, None, 0))
    sfa_stage = cute.slice_(
        blockscaled_layout.sm120_make_smem_layout_sfa(tiled_mma, tile, SCALE_VEC, 1),
        (None, None, 0),
    )
    sfb_stage = cute.slice_(
        blockscaled_layout.sm120_make_smem_layout_sfb(tiled_mma, tile, SCALE_VEC, 1),
        (None, None, 0),
    )
    epi_stage = cute.slice_(
        _cooperative_epilogue_smem_layout(out_dtype), (None, None, 0)
    )
    mainloop_bytes = (
        cute.size_in_bytes(a_dtype, a_stage)
        + cute.size_in_bytes(b_dtype, b_stage)
        + cute.size_in_bytes(Float8E8M0FNU, cute.filter_zeros(sfa_stage))
        + cute.size_in_bytes(Float8E8M0FNU, cute.filter_zeros(sfb_stage))
    )
    epilogue_bytes = cute.size_in_bytes(out_dtype, epi_stage)
    smem_capacity = utils.get_smem_capacity_in_bytes("sm_120")
    stages = (smem_capacity - COOP_BUFFER_ALIGNMENT - epilogue_bytes) // mainloop_bytes
    assert stages >= 2, (
        f"cooperative schedule needs two AB stages, got {stages} from "
        f"{smem_capacity} B SMEM ({mainloop_bytes} B/stage, "
        f"{epilogue_bytes} B epilogue)"
    )
    used = stages * mainloop_bytes + epilogue_bytes + COOP_BUFFER_ALIGNMENT
    assert used <= smem_capacity, (
        f"cooperative layouts require {used} B, over SM120's {smem_capacity} B"
    )
    return stages


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
def _scaled_gemm_mxfp8_cpasync_kernel(
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
    _scaled_gemm_mxfp8_cpasync_kernel(
        mA, mB, mSFA, mSFB, mD, tiled_mma, sfa_smem_layout, sfb_smem_layout
    ).launch(
        grid=[cute.ceil_div(M, TILE_M), cute.ceil_div(N, TILE_N), 1],
        block=[THREADS, 1, 1],
    )


# Cooperative schedule portions are adapted from NVIDIA CUTLASS commit
# 564d267e4c992c456d12ad02665f9acedf7708f1 under the following license:
#
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
@cute.kernel
def _scaled_gemm_mxfp8_cooperative_kernel(
    tma_atom_a: cute.CopyAtom,
    mA: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB: cute.Tensor,
    mSFA: cute.Tensor,
    mSFB: cute.Tensor,
    tma_atom_d: cute.CopyAtom,
    mD_tma: cute.Tensor,
    mD: cute.Tensor,
    tiled_mma: cute.TiledMma,
    a_dtype: cutlass.Constexpr,
    b_dtype: cutlass.Constexpr,
    out_dtype: cutlass.Constexpr,
    a_smem_layout_staged,
    b_smem_layout_staged,
    sfa_smem_layout_staged: cute.Layout,
    sfb_smem_layout_staged: cute.Layout,
    epilogue_smem_layout_staged,
    ab_stages: cutlass.Constexpr,
    tma_copy_bytes: cutlass.Constexpr,
    tile_sched_params: utils.PersistentTileSchedulerParams,
    SharedStorage: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)
        cpasync.prefetch_descriptor(tma_atom_d)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    mainloop_pipeline = pipeline.PipelineTmaAsync.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, COOP_MMA_WARPS),
        tx_count=tma_copy_bytes,
        barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
        cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        defer_sync=True,
    )
    scale_pipeline = pipeline.PipelineAsync.create(
        num_stages=ab_stages,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp, COOP_MMA_WARPS),
        barrier_storage=storage.scale_pipeline_array_ptr.data_ptr(),
        defer_sync=True,
    )
    cute.arch.mbarrier_init_fence()
    pipeline.agent_sync(pipeline.Agent.ThreadBlock)

    sA = storage.sA.get_tensor(
        a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
    )
    sB = storage.sB.get_tensor(
        b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
    )
    sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
    sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
    sD = storage.sD.get_tensor(
        epilogue_smem_layout_staged.outer,
        swizzle=epilogue_smem_layout_staged.inner,
    )

    gA = cute.local_tile(mA, (TILE_M, TILE_K), (None, None))
    gB = cute.local_tile(mB, (TILE_N, TILE_K), (None, None))
    gD = cute.local_tile(mD, (TILE_M, TILE_N), (None, None))
    gD_tma = cute.local_tile(mD_tma, (TILE_M, TILE_N), (None, None))
    scale_tile = (TILE_M, TILE_SF)
    gSFA = cute.zipped_divide(mSFA, scale_tile)
    gSFB = cute.zipped_divide(mSFB, scale_tile)
    cSFA = cute.zipped_divide(cute.make_identity_tensor(mSFA.shape), scale_tile)
    cSFB = cute.zipped_divide(cute.make_identity_tensor(mSFB.shape), scale_tile)
    sf_write = cute.make_layout(((32, 4), TILE_SF), stride=((16, 4), 1))
    scale_store = _tiled_copy(Float8E8M0FNU, (32, 1), (1, 1), (1, TILE_SF), True)

    tAsA, tAgA = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sA, 0, 2),
        cute.group_modes(gA, 0, 2),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sB, 0, 2),
        cute.group_modes(gB, 0, 2),
    )
    thr_mma = tiled_mma.get_slice(tidx)
    tCsA = thr_mma.partition_A(sA)
    tCsB = thr_mma.partition_B(sB)
    tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
    tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
    tCrSFA = blackwell_helpers.partition_fragment_SFA(
        sSFA[None, None, 0], thr_mma, tidx
    )
    tCrSFB = blackwell_helpers.partition_fragment_SFB(
        sSFB[None, None, 0], thr_mma, tidx
    )
    tCrSFA = cute.group_modes(tCrSFA, 2, cute.rank(tCrSFA))
    tCrSFB = cute.group_modes(tCrSFB, 2, cute.rank(tCrSFB))

    tCgD = thr_mma.partition_C(gD)
    accumulators = cute.make_rmem_tensor(tCgD.shape[:3], Float32)
    k_tile_count = cute.size(gA, mode=[3])
    tile_sched = utils.StaticPersistentTileScheduler.create(
        tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
    )
    work_tile = tile_sched.initial_work_tile_info()
    producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, ab_stages
    )
    consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, ab_stages
    )
    scale_producer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, ab_stages
    )
    scale_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, ab_stages
    )
    epilogue_barrier = pipeline.NamedBarrier(
        barrier_id=2, num_threads=COOP_MMA_WARPS * 32
    )

    if warp_idx < COOP_MMA_WARPS:
        cute.arch.setmaxregister_increase(COOP_MMA_REGISTERS)
        atom_ldmatrix_a = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), a_dtype
        )
        atom_ldmatrix_b = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), b_dtype
        )
        smem_copy_a = cute.make_tiled_copy_A(atom_ldmatrix_a, tiled_mma)
        smem_copy_b = cute.make_tiled_copy_B(atom_ldmatrix_b, tiled_mma)
        thr_copy_a = smem_copy_a.get_slice(tidx)
        thr_copy_b = smem_copy_b.get_slice(tidx)
        tCsA_copy = thr_copy_a.partition_S(sA)
        tCsB_copy = thr_copy_b.partition_S(sB)
        tCrA_copy = thr_copy_a.retile(tCrA)
        tCrB_copy = thr_copy_b.retile(tCrB)

        scale_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float8E8M0FNU)
        scale_copy_a = cute.make_tiled_copy(
            scale_atom,
            blackwell_helpers.get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        scale_copy_b = cute.make_tiled_copy(
            scale_atom,
            blackwell_helpers.get_layoutSFB_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        thr_scale_a = scale_copy_a.get_slice(tidx)
        thr_scale_b = scale_copy_b.get_slice(tidx)
        tCsSFA_copy = thr_scale_a.partition_S(sSFA)
        tCsSFB_copy = thr_scale_b.partition_S(sSFB)
        tCrSFA_copy = thr_scale_a.retile(tCrSFA)
        tCrSFB_copy = thr_scale_b.retile(tCrSFB)
        num_k_blocks = cute.size(tCrA, mode=[2])
        r2s_atom = blackwell_helpers.sm120_get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR,
            elem_ty_d=out_dtype,
            elem_ty_acc=Float32,
        )
        copy_atom_d = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(False, 2),
            # This atom supplies TV geometry; the R2S atom carries the payload type.
            cutlass.Float16,
        )
        tiled_copy_d = cute.make_tiled_copy_C_atom(copy_atom_d, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(r2s_atom, tiled_copy_d)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_copy_r2s.partition_D(sD)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)
        rD_shape = cute.shape(thr_copy_r2s.partition_S(sD))
        rD_layout = cute.make_layout(rD_shape[:3])
        tRS_rD = cute.make_rmem_tensor(rD_layout.shape, Float32)
        tma_store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, COOP_MMA_WARPS * 32
            ),
        )
        epi_buffer = cutlass.Int32(0)

        while work_tile.is_valid_tile:
            tile_coord = work_tile.tile_idx
            accumulators.fill(0.0)
            consumer_state.reset_count()
            scale_consumer_state.reset_count()

            ab_ready = mainloop_pipeline.consumer_try_wait(consumer_state)
            sf_ready = scale_pipeline.consumer_try_wait(scale_consumer_state)
            mainloop_pipeline.consumer_wait(consumer_state, ab_ready)
            scale_pipeline.consumer_wait(scale_consumer_state, sf_ready)
            stage = consumer_state.index
            tCsA_stage = tCsA_copy[None, None, None, stage]
            tCsB_stage = tCsB_copy[None, None, None, stage]
            tCsSFA_stage = cute.filter_zeros(tCsSFA_copy[None, None, None, stage])
            tCsSFB_stage = cute.filter_zeros(tCsSFB_copy[None, None, None, stage])
            tCrSFA_filtered = cute.filter_zeros(tCrSFA_copy)
            tCrSFB_filtered = cute.filter_zeros(tCrSFB_copy)
            cute.copy(smem_copy_a, tCsA_stage[None, None, 0], tCrA_copy[None, None, 0])
            cute.copy(smem_copy_b, tCsB_stage[None, None, 0], tCrB_copy[None, None, 0])
            cute.copy(
                scale_copy_a,
                tCsSFA_stage[None, None, 0],
                tCrSFA_filtered[None, None, 0],
            )
            cute.copy(
                scale_copy_b,
                tCsSFB_stage[None, None, 0],
                tCrSFB_filtered[None, None, 0],
            )

            for _ in range(0, k_tile_count - 1, 1, unroll=1):
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                    if k_block == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(consumer_state)
                        scale_pipeline.consumer_release(scale_consumer_state)
                        consumer_state.advance()
                        scale_consumer_state.advance()
                        ab_ready = mainloop_pipeline.consumer_try_wait(consumer_state)
                        sf_ready = scale_pipeline.consumer_try_wait(
                            scale_consumer_state
                        )
                        stage = consumer_state.index
                        tCsA_stage = tCsA_copy[None, None, None, stage]
                        tCsB_stage = tCsB_copy[None, None, None, stage]
                        tCsSFA_stage = cute.filter_zeros(
                            tCsSFA_copy[None, None, None, stage]
                        )
                        tCsSFB_stage = cute.filter_zeros(
                            tCsSFB_copy[None, None, None, stage]
                        )
                        mainloop_pipeline.consumer_wait(consumer_state, ab_ready)
                        scale_pipeline.consumer_wait(scale_consumer_state, sf_ready)

                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [tCrA[None, None, k_block], tCrSFA[None, None, k_block]],
                        [tCrB[None, None, k_block], tCrSFB[None, None, k_block]],
                        accumulators,
                    )
                    cute.copy(
                        smem_copy_a,
                        tCsA_stage[None, None, k_block_next],
                        tCrA_copy[None, None, k_block_next],
                    )
                    cute.copy(
                        smem_copy_b,
                        tCsB_stage[None, None, k_block_next],
                        tCrB_copy[None, None, k_block_next],
                    )
                    cute.copy(
                        scale_copy_a,
                        tCsSFA_stage[None, None, k_block_next],
                        tCrSFA_filtered[None, None, k_block_next],
                    )
                    cute.copy(
                        scale_copy_b,
                        tCsSFB_stage[None, None, k_block_next],
                        tCrSFB_filtered[None, None, k_block_next],
                    )

            for k_block in cutlass.range_constexpr(num_k_blocks):
                k_block_next = 0 if k_block + 1 == num_k_blocks else k_block + 1
                if k_block == num_k_blocks - 1:
                    cute.arch.fence_proxy("async.shared", space="cta")
                    mainloop_pipeline.consumer_release(consumer_state)
                    scale_pipeline.consumer_release(scale_consumer_state)
                    consumer_state.advance()
                    scale_consumer_state.advance()
                if k_block_next > 0:
                    cute.copy(
                        smem_copy_a,
                        tCsA_stage[None, None, k_block_next],
                        tCrA_copy[None, None, k_block_next],
                    )
                    cute.copy(
                        smem_copy_b,
                        tCsB_stage[None, None, k_block_next],
                        tCrB_copy[None, None, k_block_next],
                    )
                    cute.copy(
                        scale_copy_a,
                        tCsSFA_stage[None, None, k_block_next],
                        tCrSFA_filtered[None, None, k_block_next],
                    )
                    cute.copy(
                        scale_copy_b,
                        tCsSFB_stage[None, None, k_block_next],
                        tCrSFB_filtered[None, None, k_block_next],
                    )
                cute.gemm(
                    tiled_mma,
                    accumulators,
                    [tCrA[None, None, k_block], tCrSFA[None, None, k_block]],
                    [tCrB[None, None, k_block], tCrSFB[None, None, k_block]],
                    accumulators,
                )

            gD_tile = gD[(None, None, tile_coord[0], tile_coord[1])]
            tCgD_tile = thr_mma.partition_C(gD_tile)
            # TMA output descriptors require a 16-byte-aligned global row pitch.
            if (
                (tile_coord[0] + 1) * TILE_M <= mD.shape[0]
                and (tile_coord[1] + 1) * TILE_N <= mD.shape[1]
                and (mD.shape[1] * out_dtype.width) % 128 == 0
            ):
                gD_tma_tile = gD_tma[(None, None, tile_coord[0], tile_coord[1])]
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_d,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sD, 0, 2),
                    cute.zipped_divide(gD_tma_tile, COOP_EPILOGUE_TILE),
                )
                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                mma_tile_m = TILE_M // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = TILE_N // cute.size(tRS_rAcc, mode=[2])
                mma_per_epi_m = COOP_EPILOGUE_TILE[0] // mma_tile_m
                mma_per_epi_n = COOP_EPILOGUE_TILE[1] // mma_tile_n

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        for mma_n_in_epi in cutlass.range_constexpr(mma_per_epi_n):
                            for mma_m_in_epi in cutlass.range_constexpr(mma_per_epi_m):
                                mma_m = epi_m * mma_per_epi_m + mma_m_in_epi
                                mma_n = epi_n * mma_per_epi_n + mma_n_in_epi
                                rD_slice = tRS_rD[(None, mma_m_in_epi, mma_n_in_epi)]
                                acc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                for elem in cutlass.range_constexpr(
                                    cute.size(rD_slice)
                                ):
                                    rD_slice[elem] = acc_slice[elem]

                        rD_out = cute.make_rmem_tensor(rD_layout.shape, out_dtype)
                        rD_out.store(tRS_rD.load().to(out_dtype))
                        epi_buffer = (epi_buffer + 1) % cute.size(tRS_sD, mode=[3])
                        epilogue_barrier.arrive_and_wait()
                        cute.copy(
                            tiled_copy_r2s,
                            rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        epilogue_barrier.arrive_and_wait()
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_d,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, (epi_m, epi_n))],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()
            else:
                tCrD = cute.make_fragment_like(tCgD_tile)
                tCrD.store(accumulators.load().to(out_dtype))
                cD_tile = _tile_coords(
                    mD.shape, (TILE_M, TILE_N), (tile_coord[0], tile_coord[1])
                )
                tCcD = thr_mma.partition_C(cD_tile)
                for i in cutlass.range(cute.size(tCgD_tile), unroll_full=True):
                    if cute.elem_less(tCcD[i], mD.shape):
                        tCgD_tile[i] = tCrD[i]

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

    elif warp_idx == COOP_MMA_WARPS:
        cute.arch.setmaxregister_decrease(COOP_LOAD_REGISTERS)
        while work_tile.is_valid_tile:
            tile_coord = work_tile.tile_idx
            tAgA_tile = tAgA[(None, tile_coord[0], None)]
            tBgB_tile = tBgB[(None, tile_coord[1], None)]
            producer_state.reset_count()
            scale_producer_state.reset_count()

            for _ in range(0, k_tile_count, 1, unroll=1):
                mainloop_pipeline.producer_acquire(producer_state)
                scale_pipeline.producer_acquire(scale_producer_state)
                barrier = mainloop_pipeline.producer_get_barrier(producer_state)
                k_tile = producer_state.count
                stage = producer_state.index
                cute.copy(
                    tma_atom_a,
                    tAgA_tile[(None, k_tile)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_tile[(None, k_tile)],
                    tBsB[(None, stage)],
                    tma_bar_ptr=barrier,
                )
                sfa_tile = (None, None), (tile_coord[0], k_tile)
                sfb_tile = (None, None), (tile_coord[1], k_tile)
                _copy_tile(
                    scale_store,
                    tidx % 32,
                    gSFA[sfa_tile],
                    cSFA[sfa_tile],
                    _sf_stage(sSFA, stage, sf_write),
                    mSFA.shape,
                    False,
                )
                _copy_tile(
                    scale_store,
                    tidx % 32,
                    gSFB[sfb_tile],
                    cSFB[sfb_tile],
                    _sf_stage(sSFB, stage, sf_write),
                    mSFB.shape,
                    False,
                )
                # The generic scale barrier is signaled only after cp.async completion.
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                mainloop_pipeline.producer_commit(producer_state)
                scale_pipeline.producer_commit(scale_producer_state)
                producer_state.advance()
                scale_producer_state.advance()

            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()

        mainloop_pipeline.producer_tail(producer_state)
        scale_pipeline.producer_tail(scale_producer_state)


@cute.jit
def _scaled_gemm_mxfp8_cooperative_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    sfa: cute.Tensor,
    sfb: cute.Tensor,
    mD: cute.Tensor,
    max_active_clusters: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    tiled_mma = _make_mxfp8_tiled_mma(mA.element_type, mB.element_type)
    ab_stages = _cooperative_stage_count(
        mA.element_type, mB.element_type, mD.element_type, tiled_mma
    )
    tile = (TILE_M, TILE_N, TILE_K)
    a_smem_layout_staged = _cooperative_operand_smem_layout(TILE_M, ab_stages)
    b_smem_layout_staged = _cooperative_operand_smem_layout(TILE_N, ab_stages)
    sfa_smem_layout_staged = blockscaled_layout.sm120_make_smem_layout_sfa(
        tiled_mma, tile, SCALE_VEC, ab_stages
    )
    sfb_smem_layout_staged = blockscaled_layout.sm120_make_smem_layout_sfb(
        tiled_mma, tile, SCALE_VEC, ab_stages
    )
    epilogue_smem_layout_staged = _cooperative_epilogue_smem_layout(mD.element_type)
    tma_copy_bytes = cute.size_in_bytes(
        mA.element_type,
        cute.slice_(a_smem_layout_staged, (None, None, 0)),
    ) + cute.size_in_bytes(
        mB.element_type,
        cute.slice_(b_smem_layout_staged, (None, None, 0)),
    )

    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mA,
        cute.slice_(a_smem_layout_staged, (None, None, 0)),
        (TILE_M, TILE_K),
    )
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mB,
        cute.slice_(b_smem_layout_staged, (None, None, 0)),
        (TILE_N, TILE_K),
    )
    tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        mD,
        cute.slice_(epilogue_smem_layout_staged, (None, None, 0)),
        COOP_EPILOGUE_TILE,
    )

    gD = cute.zipped_divide(mD, (TILE_M, TILE_N))
    num_ctas_mn = gD[(0, (None, None))].shape
    tile_sched_params = utils.PersistentTileSchedulerParams(
        (*num_ctas_mn, 1), COOP_CLUSTER_SHAPE
    )
    grid = utils.StaticPersistentTileScheduler.get_grid_shape(
        tile_sched_params, max_active_clusters
    )

    @cute.struct
    class SharedStorage:
        mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
        scale_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, ab_stages * 2]
        sA: cute.struct.Align[
            cute.struct.MemRange[mA.element_type, cute.cosize(a_smem_layout_staged)],
            COOP_BUFFER_ALIGNMENT,
        ]
        sB: cute.struct.Align[
            cute.struct.MemRange[mB.element_type, cute.cosize(b_smem_layout_staged)],
            COOP_BUFFER_ALIGNMENT,
        ]
        sSFA: cute.struct.Align[
            cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfa_smem_layout_staged)],
            COOP_BUFFER_ALIGNMENT,
        ]
        sSFB: cute.struct.Align[
            cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfb_smem_layout_staged)],
            COOP_BUFFER_ALIGNMENT,
        ]
        sD: cute.struct.Align[
            cute.struct.MemRange[
                mD.element_type, cute.cosize(epilogue_smem_layout_staged)
            ],
            COOP_BUFFER_ALIGNMENT,
        ]

    _scaled_gemm_mxfp8_cooperative_kernel(
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        sfa,
        sfb,
        tma_atom_d,
        tma_tensor_d,
        mD,
        tiled_mma,
        mA.element_type,
        mB.element_type,
        mD.element_type,
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        epilogue_smem_layout_staged,
        ab_stages,
        tma_copy_bytes,
        tile_sched_params,
        SharedStorage,
    ).launch(
        grid=grid,
        block=[COOP_THREADS, 1, 1],
        cluster=COOP_CLUSTER_SHAPE,
        stream=stream,
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

    Operands and scales must be K-contiguous. Positive M/N may be arbitrary; K
    must be a positive multiple of the 32-element MXFP8 scale block.
    """
    M, K = aq.shape
    N = bq.shape[0]
    assert M > 0 and N > 0 and K > 0, f"M, N, and K must be positive, got {M}, {N}, {K}"
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
        if schedule == "cooperative":
            max_active_clusters = utils.HardwareInfo().get_max_active_clusters(1)
            stream = cuda.CUstream(torch.cuda.current_stream(aq.device).cuda_stream)
            compiled = cute.compile(
                _scaled_gemm_mxfp8_cooperative_host,
                *args,
                max_active_clusters,
                stream,
            )
        else:
            compiled = cute.compile(_scaled_gemm_mxfp8_host, *args)
        _COMPILED[key] = compiled
    if schedule == "cooperative":
        stream = cuda.CUstream(torch.cuda.current_stream(aq.device).cuda_stream)
        compiled(*args, stream)
    else:
        compiled(*args)
    return out
