# Qwen3-51M CuTe MXFP8 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the CuTe SM120 MXFP8 dense GEMM to at least 85% of cuBLASLt throughput on the model-weighted Qwen3-51M training GEMMs, with no dominant MLP GEMM below 75%.

**Architecture:** Consolidate all scaled-GEMM measurements into one benchmark, preserve the current correct `cp.async` kernel as a baseline schedule, and add a CUTLASS-style persistent cooperative schedule with one TMA producer warp and eight MMA consumer warps. Keep the public native `(rows, K//32)` scale layout; establish the TMA schedule ceiling with blocked scales first, then adapt the scale feed without folding layout conversion into kernel-only timing.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu130, CUDA 13.0/13.3 driver stack, `nvidia-cutlass-dsl[cu13]` 4.7.0, CuTe DSL, CUTLASS pipelines/TMA, Nsight Compute, pytest-xdist, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-13-qwen3-51m-cute-mxfp8-optimization-design.md`

## Global Constraints

- At the start of each execution session, run `nvidia-smi`, choose an idle numeric GPU ID, and set `task_gpu=1` (replace `1` once if another device is the idle device). Prefix every GPU command with `CUDA_VISIBLE_DEVICES="$task_gpu"`.
- Use `uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6` for the focused CUDA suite and `uv run pytest tests/fast/kernel -n 12 --dist load` for the full kernel tree. Never run pytest without `-n`.
- Accumulate in FP32; preserve caller-selected FP32, FP16, or BF16 output dtype.
- Keep the tensor-core configuration at SM120 MXFP8 `m16n8k32`, CTA tile `128x128x128`, TNN orientation, and align-16 A/B inputs until profiler evidence justifies a shape-specialized alternative.
- Preserve mixed E4M3/E5M2 operand support, ragged M/N predication, K-tail zeroing, partial scale groups, and the dynamic-M/static-KN compile cache.
- Change one performance variable at a time and record the Qwen3 benchmark after each accepted change.
- Do not dispatch to `torch._scaled_mm` as a substitute for the custom kernel.
- Use the BSD-3-Clause CUTLASS reference at commit `564d267e4c992c456d12ad02665f9acedf7708f1`; retain required attribution for any copied portions and otherwise reimplement only the MXFP8 subset.

## File Structure

| Path | Responsibility |
|---|---|
| `benchmarks/gemm/bench_scaled_gemm.py` | The single scaled-GEMM benchmark, including Qwen3-51M forward/dgrad/wgrad shapes and CuTe/native/Triton MXFP8 arms. |
| `benchmarks/gemm/bench_cute_gemm.py` | Delete after its useful logic is consolidated. |
| `src/kernel/cute/gemm.py` | Baseline and cooperative CuTe schedules, shared input validation, compilation cache, and public wrapper. |
| `tests/fast/kernel/cute/test_gemm.py` | Numerical, schedule-selection, cache, benchmark-import, and API regression tests. |
| `docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md` | Exact baseline kernel names, Nsight metrics, accepted/rejected hypotheses, and before/after performance tables. |

The upstream cooperative reference is:

```text
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
  dense_blockscaled_gemm_persistent_cooperative.py
  blockscaled_gemm_dispatch.py
```

The upstream ping-pong reference, used only if the cooperative schedule misses the target, is:

```text
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/
  dense_blockscaled_gemm_persistent_pingpong.py
```

---

### Task 1: Consolidate the Qwen3-51M benchmark

**Files:**
- Modify: `benchmarks/gemm/bench_scaled_gemm.py`
- Delete: `benchmarks/gemm/bench_cute_gemm.py`
- Modify: `tests/fast/kernel/cute/test_gemm.py`

**Interfaces:**
- Consumes: `scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype)` from `src.kernel.cute.gemm`.
- Produces: `QWEN3_51M_LINEARS`, `_qwen3_training_shapes(tokens)`, `_aggregate_qwen3(rows)`, CuTe fields in `_bench_mxfp8`, and CLI flags `--qwen3-training`, `--json-out`, and `--min-cute-native-ratio`.

- [ ] **Step 1: Write CPU-safe metadata and consolidation tests**

Replace `test_bench_cute_gemm_importable` with:

```python
def test_bench_scaled_gemm_contains_qwen3_training_shapes():
    import importlib

    bench = importlib.import_module("benchmarks.gemm.bench_scaled_gemm")
    rows = bench._qwen3_training_shapes(16384)
    got = {(r["projection"], r["role"], r["M"], r["K"], r["N"], r["count"]) for r in rows}
    assert ("mlp_gate_up", "forward", 16384, 512, 3072, 1) in got
    assert ("mlp_gate_up", "dgrad", 16384, 3072, 512, 1) in got
    assert ("mlp_gate_up", "wgrad", 3072, 16384, 512, 1) in got
    assert ("attn_kv", "forward", 16384, 512, 256, 2) in got
    assert len(rows) == 12


def test_cute_benchmark_was_consolidated():
    from pathlib import Path

    assert not Path("benchmarks/gemm/bench_cute_gemm.py").exists()
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 -k 'qwen3_training_shapes or consolidated'
```

Expected: the first test fails because `_qwen3_training_shapes` is absent; the second fails because `bench_cute_gemm.py` still exists.

- [ ] **Step 3: Add exact model metadata and shape expansion**

Add near the benchmark constants:

```python
QWEN3_51M_TOKENS = 16 * 1024
QWEN3_51M_LINEARS = [
    ("attn_qo", 2, 512, 512),
    ("attn_kv", 2, 512, 256),
    ("mlp_gate_up", 1, 512, 3072),
    ("mlp_down", 1, 1536, 512),
]


def _qwen3_training_shapes(tokens):
    rows = []
    for projection, count, k_in, n_out in QWEN3_51M_LINEARS:
        rows.extend(
            [
                dict(projection=projection, role="forward", M=tokens, K=k_in, N=n_out, count=count),
                dict(projection=projection, role="dgrad", M=tokens, K=n_out, N=k_in, count=count),
                dict(projection=projection, role="wgrad", M=n_out, K=tokens, N=k_in, count=count),
            ]
        )
    return rows
```

Update the legacy `SHAPES` entry for the fused gate/up projection from `(512, 1536)` to `(512, 3072)`.

- [ ] **Step 4: Use identical quantized operands in all MXFP8 arms**

In `_bench_mxfp8`, prepare the operands once:

```python
aq, sa = quantize_operand(a, -1, _FMT[E4M3], scaling)
bq, sb = quantize_operand(b, -2, _FMT[E4M3], scaling)
sa_e8 = sa.to(torch.float8_e8m0fnu).contiguous()
sb_e8 = sb.t().contiguous().to(torch.float8_e8m0fnu)
bq_cute = bq.t().contiguous()

a_blk = _to_blocked_native(sa_e8)
b_blk = _to_blocked_native(sb_e8)
b_arg = bq_cute.t()
```

Time only the GEMM in the CuTe arm:

```python
def cute_fn():
    return scaled_gemm_mxfp8(
        aq.contiguous(), bq_cute, sa_e8, sb_e8, torch.bfloat16
    )
```

Keep `quantize_operand`, E8M0 conversion, transpose/contiguous operations, and `_to_blocked_native` outside all timed closures. Add `cute_ms` and `cute_relerr` to the returned dictionary; set them to `None` when compute capability is below `(12, 0)`.

- [ ] **Step 5: Add weighted aggregation and threshold enforcement**

Add:

```python
def _aggregate_qwen3(rows):
    total_flops = sum(2 * r["M"] * r["K"] * r["N"] * r["count"] for r in rows)
    out = {"flops": total_flops}
    for impl in ("cute", "triton", "native"):
        missing = [r for r in rows if r.get(f"{impl}_ms") is None]
        if missing:
            raise RuntimeError(f"{impl} is unavailable for {len(missing)} Qwen3 rows")
        out[f"{impl}_ms"] = sum(r[f"{impl}_ms"] * r["count"] for r in rows)
        out[f"{impl}_tflops"] = total_flops / out[f"{impl}_ms"] / 1e9
    out["cute_native_ratio"] = out["cute_tflops"] / out["native_tflops"]
    return out
```

When `--min-cute-native-ratio X` is supplied, exit nonzero if the weighted ratio is below `X` or any dominant MLP row is below `0.75`. Print each failing row before exiting.

- [ ] **Step 6: Merge output/plot logic and delete the old script**

Extend `_SCHEME_COLOR` and plotting loops with `"cute"`. For schemes other than MXFP8, render CuTe as `n/a`. Add `--qwen3-training` to run the 12 rows from `_qwen3_training_shapes(QWEN3_51M_TOKENS)` and `--json-out PATH` to serialize rows plus aggregate using `json.dump({"rows": rows, "aggregate": aggregate}, output_file, indent=2)`.

Delete `benchmarks/gemm/bench_cute_gemm.py` with `apply_patch` after all unique logic has moved.

- [ ] **Step 7: Run tests and benchmark smoke check**

Run:

```bash
uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 -k 'qwen3_training_shapes or consolidated or bench'
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py --qwen3-training --no-plot --json-out /tmp/qwen3-before.json
```

Expected: tests pass; the benchmark emits 12 rows, a weighted aggregate, and valid JSON.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff check benchmarks/gemm/bench_scaled_gemm.py tests/fast/kernel/cute/test_gemm.py
uv run ruff format --check benchmarks/gemm/bench_scaled_gemm.py tests/fast/kernel/cute/test_gemm.py
git add -A benchmarks/gemm tests/fast/kernel/cute/test_gemm.py
git commit -m "bench: consolidate Qwen3 MXFP8 GEMMs"
```

---

### Task 2: Freeze profiler evidence and the performance acceptance test

**Files:**
- Create: `docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md`
- Modify: `benchmarks/gemm/bench_scaled_gemm.py` only if profiling exposes a timing-scope defect.

**Interfaces:**
- Consumes: `--qwen3-training`, `--json-out`, and `--min-cute-native-ratio` from Task 1.
- Produces: a recorded baseline, exact cuBLASLt/CuTe kernel identities, Nsight metrics, and a reproducible failing performance gate.

- [ ] **Step 1: Check hardware and freeze the baseline**

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --json-out /tmp/qwen3-before.json
```

Copy the six dominant MLP rows and weighted aggregate into the profile note with GPU name, driver, CUDA, torch, and CUTLASS DSL versions.

- [ ] **Step 2: Demonstrate that the target gate currently fails**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --min-cute-native-ratio 0.85
```

Expected: nonzero exit because the baseline weighted ratio is below 0.85 and several MLP rows are below 0.75.

- [ ] **Step 3: Capture exact kernel names with PyTorch profiler**

Profile `(M,K,N)=(16384,1536,512)` after five warmups. Record these two kernel names in the note:

```text
kernel_cutlass__scaled_gemm_mxfp8_kernel_*
cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_*_128x128x128_*_tnn_align16_*
```

Also record that the CuTe public wrapper launches no FP32-to-E8M0 conversion kernels when `sa_e8` and `sb_e8` are passed.

- [ ] **Step 4: Capture Nsight Compute metrics for each kernel**

Run one representative launch per kernel with:

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" ncu --target-processes all --set full --launch-count 1 \
  --kernel-name 'regex:kernel_cutlass__scaled_gemm_mxfp8_kernel.*' \
  uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot
```

Repeat with:

```text
regex:cutlass3x_sm120_bstensorop_s16832gemm_block_scaled.*
```

Record registers/thread, shared memory/CTA, block size, achieved occupancy, tensor-pipe utilization, DRAM throughput, and the top three warp stall reasons. If performance counters are permission-blocked, record the exact NCU error and retain kernel names plus launch statistics from PyTorch profiler.

- [ ] **Step 5: Record the confirmed hypothesis and commit**

The note must state: both kernels use SM120 block-scaled `s16x8x32`, `128x128x128`, TNN, align-16; the current difference is the nonpersistent all-warp `cp.async` schedule versus CUTLASS's persistent TMA warp-specialized schedule.

```bash
git add -f docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md
git commit -m "docs: record Qwen3 MXFP8 kernel baseline"
```

---

### Task 3: Add a schedule seam without changing behavior

**Files:**
- Modify: `src/kernel/cute/gemm.py`
- Modify: `tests/fast/kernel/cute/test_gemm.py`

**Interfaces:**
- Consumes: the existing `_scaled_gemm_mxfp8_host` compiled callable.
- Produces: `scaled_gemm_mxfp8(aq, bq, sa, sb, out_dtype, schedule: str = "cpasync")`, `_SCHEDULES`, a `--cute-schedule` benchmark flag, and a compile-cache key containing `schedule`.

- [ ] **Step 1: Write schedule-selection tests**

Add:

```python
def test_scaled_gemm_mxfp8_rejects_unknown_schedule():
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    aq, bq, sa, sb = _operands(128, 128, 128)
    with pytest.raises(AssertionError, match="unsupported schedule"):
        scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32, schedule="unknown")


def test_compile_cache_separates_schedules():
    from src.kernel.cute import gemm as cute_gemm

    aq, bq, sa, sb = _operands(128, 128, 128)
    cute_gemm.scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32, schedule="cpasync")
    assert any(key[0] == "cpasync" for key in cute_gemm._COMPILED)
```

- [ ] **Step 2: Run and verify failure**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 \
  -k 'unknown_schedule or separates_schedules'
```

Expected: `TypeError` because the public function does not accept `schedule`.

- [ ] **Step 3: Implement the minimal seam**

Add:

```python
_SCHEDULES = frozenset({"cpasync"})
```

Extend the public signature with `schedule: str = "cpasync"`, assert membership, and change the cache key to:

```python
key = (schedule, K, N, aq.dtype, bq.dtype, out_dtype)
```

Keep compilation routed to `_scaled_gemm_mxfp8_host`; no kernel behavior changes in this task.

Add `--cute-schedule` with default `"cpasync"` to `bench_scaled_gemm.py`, and pass its value as the `schedule` keyword in the CuTe timed closure.

- [ ] **Step 4: Verify exact baseline behavior**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --json-out /tmp/qwen3-seam.json
```

Expected: all tests pass and the weighted result stays within 5% of `/tmp/qwen3-before.json`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py
uv run ruff format --check src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py
git add src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py
git commit -m "refactor: separate CuTe GEMM schedules"
```

---

### Task 4: Implement the persistent cooperative TMA schedule

**Files:**
- Modify: `src/kernel/cute/gemm.py`
- Modify: `tests/fast/kernel/cute/test_gemm.py`

**Interfaces:**
- Consumes: native `(M,K)`, `(N,K)`, `(M,K//32)`, `(N,K//32)` tensors and the existing `_MmaMXF8Op` mixed-format atom.
- Produces: `_scaled_gemm_mxfp8_cooperative_host`, `_scaled_gemm_mxfp8_cooperative_kernel`, and `schedule="cooperative"`.

- [ ] **Step 1: Read the complete upstream cooperative reference**

Read every line of the two CUTLASS files named in File Structure at commit `564d267e4c992c456d12ad02665f9acedf7708f1`. Record in the profile note the exact producer/consumer groups, `PipelineTmaAsync` transaction count, register budgets, persistent scheduler, ldmatrix copies, and TMA epilogue sequence. Do not copy FP4, mixed FP4/FP8, batching, CLI, or reference-test code.

- [ ] **Step 2: Add a failing cooperative numerical test**

Parameterize the strongest existing scale-placement test:

```python
@pytest.mark.parametrize("schedule", ["cpasync", "cooperative"])
@pytest.mark.parametrize("shape", [(256, 512, 128), (192, 160, 128), (129, 128, 130)])
def test_scaled_gemm_mxfp8_schedule_adds_no_error(shape, schedule):
    from src.kernel.cute.gemm import scaled_gemm_mxfp8

    M, K, N = shape
    aq, bq, sa, sb = _operands_spread_scales(M, K, N)
    got = scaled_gemm_mxfp8(aq, bq, sa, sb, torch.float32, schedule=schedule)
    want = scaled_gemm_ref(aq, bq.t(), sa, sb.t(), 32)
    rel = (got - want).norm() / want.norm()
    assert rel < 1e-6, rel
```

- [ ] **Step 3: Run and verify the new case fails**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 \
  -k schedule_adds_no_error
```

Expected: cooperative cases fail with `unsupported schedule` while cpasync cases pass.

- [ ] **Step 4: Add the cooperative schedule's static configuration**

Import `cuda.bindings.driver as cuda`, `cutlass.pipeline as pipeline`, `cutlass.utils as utils`, and `cutlass.cute.nvgpu.cpasync as cpasync`.

Use:

```python
COOP_MMA_WARPS = 8
COOP_TMA_WARPS = 1
COOP_THREADS = (COOP_MMA_WARPS + COOP_TMA_WARPS) * 32
COOP_LOAD_REGISTERS = 40
COOP_MMA_REGISTERS = 232
COOP_BUFFER_ALIGNMENT = 1024
```

Build the same `(4,2,1)` tiled MMA with `_MmaMXF8Op`, preserving independent E4M3/E5M2 operand types.

- [ ] **Step 5: Build TMA descriptors and staged shared storage**

Use `cpasync.make_tiled_tma_atom` with `cpasync.CopyBulkTensorTileG2SOp()` for A and B, passing each global tensor, its shared layout, the CTA tile, and the MMA tiler. Use the same builder with `cpasync.CopyBulkTensorTileS2GOp()` for the BF16/FP16/FP32 output, passing the global output tensor and epilogue layout. Define `@cute.struct SharedStorage` with:

```python
mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
sA: cute.struct.Align[cute.struct.MemRange[a_dtype, cute.cosize(a_layout)], 1024]
sB: cute.struct.Align[cute.struct.MemRange[b_dtype, cute.cosize(b_layout)], 1024]
sSFA: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfa_layout)], 1024]
sSFB: cute.struct.Align[cute.struct.MemRange[Float8E8M0FNU, cute.cosize(sfb_layout)], 1024]
sD: cute.struct.Align[cute.struct.MemRange[out_dtype, cute.cosize(epi_layout)], 1024]
```

Compute `AB_STAGES` from the actual SM120 shared-memory capacity after reserving one epilogue stage; assert that the chosen layouts fit before compilation.

- [ ] **Step 6: Implement the producer/consumer pipeline**

Create:

```python
mainloop_pipeline = pipeline.PipelineTmaAsync.create(
    num_stages=AB_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, COOP_MMA_WARPS),
    tx_count=tma_copy_bytes,
    barrier_storage=mainloop_pipeline_array_ptr,
    cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
)
```

Warp `COOP_MMA_WARPS` is the sole producer. It calls `setmaxregister_decrease(40)`, acquires each producer stage, issues TMA A/B copies, issues the selected native-scale staging operation from Step 7, advances the producer state, and calls `producer_tail` after its persistent work ends. Set `PipelineTmaAsync.tx_count` to the exact bytes transferred by the TMA operations attached to its barrier; do not include bytes copied by a separate scale pipeline.

Warps `< COOP_MMA_WARPS` call `setmaxregister_increase(232)`, wait on consumer stages, use `LdMatrix8x8x16bOp` tiled copies for A/B, use the current SFA/SFB fragment mapping, issue `_MmaMXF8Op`, release stages, and advance consumer state.

- [ ] **Step 7: Preserve native scale inputs with a measured two-path experiment**

Implement and measure these paths separately; do not combine them in one benchmark run:

1. A private benchmark-only blocked-scale ceiling helper: use the official TMA scale layout and preblocked scale tensors to prove the schedule ceiling. Do not add this helper to `_SCHEDULES` because its input contract differs from the public native scale layout.
2. `cooperative`: retain native `(rows,K//32)` scales. First attempt a TMA tensor view whose logical K mode is `(K//32,32)` with the 32-element mode stride zero and use `cute.filter_zeros` so only unique scale bytes transfer. If `make_tiled_tma_atom` rejects the zero-stride descriptor, keep TMA for A/B and use a separate `pipeline.PipelineAsync` stage for producer-warp scale copies into the current `sm120_make_smem_layout_sfa/sfb` layouts. Consumers must wait on both the A/B TMA stage and its paired scale stage before reading shared memory, then release and advance both states together.

The accepted `cooperative` path must not launch a global scale-swizzle kernel. Record the rejected path and compiler/runtime diagnostic in the profile note.

- [ ] **Step 8: Add persistent tile scheduling and TMA epilogue**

Use `utils.StaticPersistentTileScheduler` with cluster shape `(1,1,1)` and `HardwareInfo.get_max_active_clusters(1)`. Both consumer warp groups cooperate on one output tile. Convert accumulators to the requested output dtype, stage them with the SM120 shared-store atom, and use `PipelineTmaStore` plus `CopyBulkTensorTileS2GOp` for aligned interior output tiles. Retain the current predicated register-to-global epilogue for ragged edges if TMA store cannot encode the boundary safely.

- [ ] **Step 9: Compile and route the schedule**

Add only `"cooperative"` to `_SCHEDULES`. Compile `_scaled_gemm_mxfp8_cooperative_host` for that schedule and keep `"cpasync"` routed to the existing host function. Compile the blocked-scale ceiling through a private benchmark helper. Include schedule, K, N, operand dtypes, and output dtype in the public cache key.

- [ ] **Step 10: Run correctness after each structural milestone**

After descriptor creation, pipeline creation, mainloop, and epilogue are each added, run:

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 \
  -k 'schedule_adds_no_error or agrees_with_scaled_mm or ragged or partial_scale'
```

Do not benchmark a milestone until these tests pass.

- [ ] **Step 11: Benchmark the schedule ceiling and native-scale path**

Run the consolidated Qwen benchmark with `--cute-schedule cpasync` and `--cute-schedule cooperative`, writing `/tmp/qwen3-cpasync.json` and `/tmp/qwen3-cooperative.json`. Run the private blocked-scale helper into `/tmp/qwen3-blocked-scale-ceiling.json`. Accept the native-scale cooperative schedule only if it materially improves the weighted ratio and preserves all numerical gates. Compare its result against the blocked-scale ceiling to quantify the cost of retaining native scale layout.

- [ ] **Step 12: Commit the correct cooperative schedule**

```bash
uv run ruff check src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py
uv run ruff format --check src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py
git add src/kernel/cute/gemm.py tests/fast/kernel/cute/test_gemm.py \
  docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md
git commit -m "perf: add persistent TMA MXFP8 schedule"
```

---

### Task 5: Select the production schedule and close remaining profiler gaps

**Files:**
- Modify: `src/kernel/cute/gemm.py`
- Modify: `benchmarks/gemm/bench_scaled_gemm.py`
- Modify: `tests/fast/kernel/cute/test_gemm.py`
- Modify: `docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md`

**Interfaces:**
- Consumes: public `cpasync` and `cooperative` schedules plus the private blocked-scale ceiling measurement.
- Produces: `DEFAULT_SCHEDULE`, optional `pingpong`, final cache dispatch, and passing 0.85/0.75 acceptance gates.

- [ ] **Step 1: Set the default only after evidence**

If `cooperative` reaches the 0.85 weighted target and every dominant MLP row reaches 0.75, set:

```python
DEFAULT_SCHEDULE = "cooperative"
```

and use it as the public argument default. Keep the blocked-scale ceiling helper private to benchmarks.

- [ ] **Step 2: If cooperative misses, profile one representative shape again**

Repeat Task 2's Nsight capture for `(16384,1536,512)`. Compare the top stalls and utilization to cuBLASLt. Choose exactly one next hypothesis:

- high memory-dependency stalls: add the ping-pong producer/consumer state machine from the upstream ping-pong reference;
- low occupancy from registers/shared memory: reduce stages or epilogue staging without changing the MMA tile;
- poor wgrad utilization with otherwise healthy large-M shapes: add a wgrad-specific tile/schedule keyed by static K/N.

Record the selected hypothesis before editing.

- [ ] **Step 3: Implement at most one measured follow-up at a time**

For ping-pong, add `schedule="pingpong"`, preserve one TMA producer warp, split the eight MMA warps into two four-warp consumer groups, and use independent consumer pipeline phases exactly as the upstream SM120 ping-pong reference. For an occupancy fix, change only stage counts. For wgrad specialization, add only the profiler-selected tile and key it explicitly in the compile cache.

- [ ] **Step 4: Re-run correctness and the acceptance benchmark**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --min-cute-native-ratio 0.85 \
  --json-out /tmp/qwen3-final.json
```

Expected: all numerical tests pass; the benchmark exits zero; every dominant MLP row is at least 0.75 of native.

- [ ] **Step 5: Remove rejected experimental schedules**

Delete code for schedules that lost or failed correctness. Keep `cpasync` as an explicit diagnostic fallback and the winning schedule as default. Ensure `_SCHEDULES` contains only maintained paths.

- [ ] **Step 6: Commit final selection**

```bash
git add src/kernel/cute/gemm.py benchmarks/gemm/bench_scaled_gemm.py \
  tests/fast/kernel/cute/test_gemm.py \
  docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md
git commit -m "perf: select the Qwen3 MXFP8 schedule"
```

If three isolated schedule hypotheses fail to meet the gate, stop instead of attempting a fourth. Report the best measured result and reopen the architecture choice between native-scale CuTe DSL and a separate CUTLASS C++ design.

---

### Task 6: Full regression and end-to-end training validation

**Files:**
- Modify: `docs/superpowers/notes/2026-08-13-qwen3-51m-cute-mxfp8-profile.md`
- Modify other files only for failures directly caused by this work.

**Interfaces:**
- Consumes: final benchmark and schedule from Tasks 1–5.
- Produces: verified kernel suite, final performance record, and Qwen3-51M end-to-end result.

- [ ] **Step 1: Recheck hardware and run the focused suite**

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6
```

- [ ] **Step 2: Run the complete fast kernel tree**

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run pytest tests/fast/kernel -n 12 --dist load
```

- [ ] **Step 3: Run lint and formatting**

```bash
uv run ruff check src/ tests/ benchmarks/
uv run ruff format --check src/ tests/ benchmarks/
git diff --check
```

- [ ] **Step 4: Run the final standalone benchmark twice**

Run the Qwen3 acceptance benchmark twice on an otherwise idle GPU. Record both runs and their variation in the profile note; do not claim a marginal win from one run.

- [ ] **Step 5: Run end-to-end Qwen3-51M training benchmark**

First ensure the CuTe path is actually selected by the MXFP8 training config. If backend dispatch is still intentionally unwired, benchmark a temporary explicit selection without changing the repository default. Run:

```bash
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/bench_train.py \
  --config experiments/quant_dev/qwen3_51m_mxfp8.yaml
```

Record step time, tokens/s, peak memory, and whether the expected CuTe kernels appear in the profiler. Compare with the same config using the native cuBLASLt backend and with the pre-change CuTe schedule.

- [ ] **Step 6: Final verification commit**

```bash
git add -A
git commit -m "test: validate Qwen3 MXFP8 optimization"
```

Do not create an empty commit if the validation step changed only ignored `/tmp` outputs. In that case, amend the profile note in the previous commit or make one documentation-only verification commit.

- [ ] **Step 7: Push all task commits**

Confirm the branch and upstream, then push the current branch:

```bash
git status --short
git branch --show-current
git push -u origin "$(git branch --show-current)"
```

Expected: the working tree contains no uncommitted task changes and the remote branch advances through the final verification commit.
