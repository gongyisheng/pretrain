# Qwen3-51M CuTe MXFP8 profile baseline

This note freezes the pre-optimization evidence from commit `823b246` (`bench:
consolidate Qwen3 MXFP8 GEMMs`). All GPU work used physical GPU 1 through
`CUDA_VISIBLE_DEVICES=1`; inside CUDA and profiler output this device is numbered 0.
The benchmark retains its kernel-only timing scope: quantization, E8M0 preparation,
B layout conversion, and native scale blocking happen before the timed closures.
No benchmark or kernel source was changed while collecting this evidence.

## Environment

Before every benchmark or profiler session, `nvidia-smi` showed physical GPU 1 idle
at 15 MiB and 0% utilization. The first preflight reported:

```text
Thu Aug 13 23:55:48 2026
NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3
1  NVIDIA GeForce RTX 5060 Ti     15MiB / 16311MiB          0%      Default
```

Software versions were captured in the same pinned environment:

```bash
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python - <<'PY'
import importlib.metadata
import torch
print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"nvidia-cutlass-dsl={importlib.metadata.version('nvidia-cutlass-dsl')}")
PY
```

```text
torch=2.11.0+cu130
torch.version.cuda=13.0
nvidia-cutlass-dsl=4.7.0
```

Nsight Compute was `2025.3.1.0` (build `36398880`). Thus the frozen environment is
an NVIDIA GeForce RTX 5060 Ti (SM120), driver/KMD 610.43.02, CUDA UMD 13.3, PyTorch
2.11.0+cu130 with CUDA runtime 13.0, and NVIDIA CUTLASS DSL 4.7.0.

## Frozen Qwen3-51M baseline

Command:

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --json-out /tmp/qwen3-before.json
```

The six dominant MLP rows below are the exact millisecond values in
`/tmp/qwen3-before.json` from that run. `count` is the per-layer multiplicity used by
the weighted aggregate.

| projection | role | count | M | K | N | CuTe ms | Triton ms | native ms | native/CuTe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp_gate_up | forward | 1 | 16384 | 512 | 3072 | 0.5131420982070267 | 0.5379868391901255 | 0.3731424594298005 | 0.727172 |
| mlp_gate_up | dgrad | 1 | 16384 | 3072 | 512 | 0.6360487197525799 | 0.6577553786337376 | 0.31359003856778145 | 0.493028 |
| mlp_gate_up | wgrad | 1 | 3072 | 16384 | 512 | 0.5582630797289312 | 0.7256124005652964 | 0.3148727398365736 | 0.564022 |
| mlp_down | forward | 1 | 16384 | 1536 | 512 | 0.2514711208641529 | 0.3751935809850693 | 0.1688111387193203 | 0.671294 |
| mlp_down | dgrad | 1 | 16384 | 512 | 1536 | 0.26288063963875175 | 0.3811892797239125 | 0.19108350155875087 | 0.726883 |
| mlp_down | wgrad | 1 | 512 | 16384 | 1536 | 0.33717280020937324 | 0.40741775883361697 | 0.2090153214521706 | 0.619906 |

Exact weighted aggregate JSON:

```json
{
  "flops": 309237645312,
  "cute_ms": 5.585358135867864,
  "cute_tflops": 55.36576845916972,
  "triton_ms": 7.651272600051016,
  "triton_tflops": 40.41649820579365,
  "native_ms": 2.5210023252293468,
  "native_tflops": 122.66456171708104,
  "cute_native_ratio": 0.4513591185926036
}
```

The console summary, rounded only by the benchmark's display formatting, was:

```text
weighted aggregate cute=55.4 TFLOP/s triton=40.4 TFLOP/s native=122.7 TFLOP/s cute/native=0.451
wrote /tmp/qwen3-before.json
```

## Reproducible failing acceptance gate

Command:

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot --min-cute-native-ratio 0.85
```

The command exited 1. Its failure diagnostics were:

```text
weighted aggregate cute=55.2 TFLOP/s triton=40.9 TFLOP/s native=121.8 TFLOP/s cute/native=0.453
weighted cute/native ratio 0.453 is below 0.850
dominant MLP row below 0.75: mlp_gate_up forward 0.729
dominant MLP row below 0.75: mlp_gate_up dgrad 0.493
dominant MLP row below 0.75: mlp_gate_up wgrad 0.571
dominant MLP row below 0.75: mlp_down forward 0.685
dominant MLP row below 0.75: mlp_down dgrad 0.730
dominant MLP row below 0.75: mlp_down wgrad 0.620
```

This is the performance acceptance test for the schedule work: the weighted ratio
must reach at least 0.85, while the existing per-dominant-MLP-row floor remains 0.75.

## PyTorch profiler: exact kernels and launch statistics

The representative `(M,K,N)=(16384,1536,512)` inputs, quantized operands, E8M0
scales, and native blocked-scale layouts were prepared before profiling. A temporary
untracked helper ran five CuTe/native warmup pairs, synchronized, then recorded one
public `scaled_gemm_mxfp8(aq, bq, sa_e8, sb_e8, torch.bfloat16)` call and one native
`torch.nn.functional.scaled_mm` call:

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" uv run python /tmp/profile_qwen3_mxfp8.py
```

The temporary helper can be reproduced from the repository root with this body
(the imports reuse benchmark setup helpers but never enter `_time`):

```python
import json
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from benchmarks.gemm.bench_scaled_gemm import (
    MXFP8_BLOCK, _FMT, _make, _scaling, _to_blocked_native,
)
from src.kernel.cute.gemm import scaled_gemm_mxfp8
from src.quant.quantize import quantize_operand

M, K, N = 16384, 1536, 512
a, b = _make(M, K, N)
scaling = _scaling("blockwise", MXFP8_BLOCK, "fp8_e8m0")
aq, sa = quantize_operand(a, -1, _FMT[torch.float8_e4m3fn], scaling)
bq, sb = quantize_operand(b, -2, _FMT[torch.float8_e4m3fn], scaling)
sa_e8 = sa.to(torch.float8_e8m0fnu).contiguous()
sb_e8 = sb.t().contiguous().to(torch.float8_e8m0fnu)
bq_cute = bq.t().contiguous()
a_blk = _to_blocked_native(sa_e8)
b_blk = _to_blocked_native(sb_e8)
b_arg = bq_cute.t()

def cute_fn():
    return scaled_gemm_mxfp8(aq.contiguous(), bq_cute, sa_e8, sb_e8, torch.bfloat16)

def native_fn():
    st = F.ScalingType.BlockWise1x32
    sw = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        aq.contiguous(), b_arg, a_blk, st, b_blk, st,
        swizzle_a=sw, swizzle_b=sw, output_dtype=torch.bfloat16,
    )

for _ in range(5):
    cute_fn()
    native_fn()
torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
) as prof:
    with torch.profiler.record_function("representative_cute_public_wrapper"):
        cute_fn()
    with torch.profiler.record_function("representative_native_scaled_mm"):
        native_fn()
    torch.cuda.synchronize()
prof.export_chrome_trace("/tmp/qwen3-mxfp8-profile.json")

with open("/tmp/qwen3-mxfp8-profile.json") as trace_file:
    trace = json.load(trace_file)
for event in trace["traceEvents"]:
    if event.get("cat") == "kernel":
        print(event["name"], event.get("dur"), event.get("args", {}))
```

The exported trace was `/tmp/qwen3-mxfp8-profile.json`. It contained exactly two
CUDA kernel events:

```text
kernel_cutlass__scaled_gemm_mxfp8_kernel_tensorptrf8E4M3FNgmemalign16o153615361_tensorptrf8E4M3FNgmemalign16o512153615361_tensorptrf8E8M0FNUgmemalign4o48481_tensorptrf8E8M0FNUgmemalign4o5_0
cutlass3x_sm120_bstensorop_s16832gemm_block_scaled_ue8m0xe4m3_ue8m0xe4m3_f32_bf16_bf16_128x128x128_1x1x1_0_tnn_align16_q_bias_bf16_relu
```

Launch statistics reported by the PyTorch Chrome trace:

| implementation | duration µs | grid | block | registers/thread | static+dynamic shared memory/CTA | trace blocks/SM | trace warps/SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| CuTe | 269.633 | `[128, 4, 1]` | `[256, 1, 1]` | 166 | 101376 B | 14.222222 | 113.777779 |
| native CUTLASS | 161.985 | `[4, 128, 1]` | `[384, 1, 1]` | 168 | 81920 B | 14.222222 | 170.666672 |

The durations are one profiler sample, not replacements for the benchmark timing.
The trace fields named `blocks per SM` and `warps per SM` describe whole-grid launch
distribution; they are not Nsight's achieved-occupancy measurement.

Because the trace recorded exactly the CuTe GEMM and native GEMM kernel events, the
CuTe public wrapper launched no FP32-to-E8M0 conversion kernels when `sa_e8` and
`sb_e8` were already E8M0 and contiguous. The wrapper's `.to(...).contiguous()`
operations were therefore idempotent and did not expand the timed GPU-kernel scope.

## Nsight Compute

CuTe command:

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" ncu --target-processes all --set full --launch-count 1 \
  --kernel-name 'regex:kernel_cutlass__scaled_gemm_mxfp8_kernel.*' \
  uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot
```

Native command:

```bash
nvidia-smi
task_gpu=1
CUDA_VISIBLE_DEVICES="$task_gpu" ncu --target-processes all --set full --launch-count 1 \
  --kernel-name 'regex:cutlass3x_sm120_bstensorop_s16832gemm_block_scaled.*' \
  uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --qwen3-training --no-plot
```

Both commands connected to the Python process, matched their target kernel, printed
the following exact diagnostic, disconnected, and exited 1:

```text
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM
```

Consequently, NCU registers/thread, shared memory/CTA, achieved occupancy,
tensor-pipe utilization, DRAM throughput, and the top three warp-stall reasons are
not available in this environment. Registers/thread, shared memory/CTA, and block
size remain recorded above from the PyTorch profiler; no values are guessed for the
permission-blocked counters. The roughly 100 ms timings printed by the benchmark
under the failed NCU attempts are profiler-interception artifacts and are not
baseline measurements.

## Confirmed schedule hypothesis

Both exact kernel identities are SM120 block-scaled `s16x8x32`, `128x128x128`, TNN,
align-16 implementations. The current difference is the CuTe kernel's nonpersistent,
all-warp `cp.async` schedule versus CUTLASS's persistent, TMA, warp-specialized
schedule. The next optimization should change that schedule directly; it must not
dispatch the CuTe path to `torch._scaled_mm`, move quantization/layout work into the
timed region, or weaken the acceptance gate.

## Task 4 cooperative schedule

The implementation was ported from NVIDIA CUTLASS commit
`564d267e4c992c456d12ad02665f9acedf7708f1` (BSD-3-Clause), specifically the
complete `dense_blockscaled_gemm_persistent_cooperative.py` and
`blockscaled_gemm_dispatch.py` references. The MXFP8 subset uses one producer warp
(warp 8), eight consumer warps, a `(4,2,1)` MMA atom layout, producer/consumer
register budgets of 40/232, and a static persistent scheduler with cluster shape
`(1,1,1)`. The upstream main pipeline uses `Agent.Thread` producer and eight-thread
consumer cooperative groups. Its transaction count covers A, B, SFA, and SFB TMA
bytes: 33,792 B/stage (16,384 A + 16,384 B + 512 SFA + 512 SFB). Consumers use
`LdMatrix8x8x16bOp(False, 4)` for both A and B; the epilogue converts registers,
uses the SM120 shared-store atom, synchronizes eight MMA warps with a named barrier,
and issues `PipelineTmaStore` S2G copies.

The public native-scale path keeps TMA A/B and uses a paired `PipelineAsync` for
producer-warp scale copies. Its TMA transaction count is exactly 32,768 bytes per
stage (A+B only); scale bytes are deliberately excluded. Consumers wait, release,
and advance both stage states together. Two A/B stages plus one 64x32 epilogue
stage fit the actual SM120 shared-memory capacity. Aligned interior outputs use the
TMA epilogue; ragged edges and row pitches not divisible by 16 bytes retain the
predicated register store.

### Native-scale TMA experiment

A standalone descriptor probe accepted a native scale view with logical K shape
`(K//32,32)` and zero stride on the 32-element mode. The real data path rejected or
misexecuted it:

- K=512 failed IR verification with
  `cute.copy expects same size in restAtomVRank`, reporting source shape
  `((1,(1,4,32,16,1)),(1))` and destination shape `((1,(512,1)),(1))`.
- K=128 lowered, but launch failed with `cudaErrorIllegalInstruction`.

The maintained schedule therefore uses the paired native-scale pipeline and does
not launch a global scale-swizzle kernel. A cp.async-aware scale-barrier variant was
also numerically correct but slower in the private weighted run, so it was not
retained.

### Numerical milestones

The initial red run was the requested six-case schedule test: three cpasync cases
passed and all three cooperative cases failed with `unsupported schedule`. After
descriptor, paired-pipeline, and direct-epilogue milestones, the focused structural
suite passed all 16 cases. The first TMA-epilogue version exposed an FP32 store
geometry error and a non-16-byte ragged-N row-pitch error; the corrected geometry
and safe fallback restored `16 passed`. Mixed E4M3/E5M2 operands, all requested
output dtypes, and schedule cache isolation then passed 15 additional cases.

### Final public matrix

Both consolidated runs used physical GPU 1 and wrote
`/tmp/scaled-gemm-cpasync.json` and `/tmp/scaled-gemm-cooperative.json`. Positive
speedup means cooperative is faster.

| shape | M | cpasync ms | cooperative ms | speedup | cooperative relerr |
|---|---:|---:|---:|---:|---:|
| attn_proj | 4096 | 0.264200 | 0.276784 | -4.55% | 0.037558 |
| attn_proj | 16384 | 0.248326 | 0.266486 | -6.81% | 0.037483 |
| mlp_up | 4096 | 0.244316 | 0.273599 | -10.70% | 0.037500 |
| mlp_up | 16384 | 0.524389 | 0.389701 | +34.56% | 0.037505 |
| mlp_down | 4096 | 0.253381 | 0.266990 | -5.10% | 0.037576 |
| mlp_down | 16384 | 0.250951 | 0.273458 | -8.23% | 0.037501 |

### Private Qwen training matrix and blocked-scale ceiling

The private helper remained outside the public benchmark and wrote
`/tmp/qwen3-cpasync.json`, `/tmp/qwen3-cooperative.json`, and
`/tmp/qwen3-blocked-scale-ceiling.json`.

The two temporary helpers and their exact final invocations were:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_qwen3_mxfp8.py \
  --schedule cpasync --json-out /tmp/qwen3-cpasync.json
nvidia-smi
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_qwen3_mxfp8.py \
  --schedule cooperative --json-out /tmp/qwen3-cooperative.json
nvidia-smi
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_blocked_scale_ceiling.py \
  --json-out /tmp/qwen3-blocked-scale-ceiling.json
```

`/tmp/bench_qwen3_mxfp8.py`:

```python
import argparse
import json
import sys

sys.path.insert(0, ".")

from benchmarks.gemm.bench_scaled_gemm import _bench_mxfp8, _make

TOKENS = 16 * 1024
LINEARS = [
    ("attn_qo", 2, 512, 512),
    ("attn_kv", 2, 512, 256),
    ("mlp_gate_up", 1, 512, 3072),
    ("mlp_down", 1, 1536, 512),
]


def shapes():
    rows = []
    for projection, count, k_in, n_out in LINEARS:
        rows.extend(
            [
                dict(projection=projection, role="forward", M=TOKENS, K=k_in, N=n_out, count=count),
                dict(projection=projection, role="dgrad", M=TOKENS, K=n_out, N=k_in, count=count),
                dict(projection=projection, role="wgrad", M=n_out, K=TOKENS, N=k_in, count=count),
            ]
        )
    return rows


def aggregate(rows):
    total_flops = sum(2 * row["M"] * row["K"] * row["N"] * row["count"] for row in rows)
    result = {"flops": total_flops}
    for impl in ("cute", "triton", "native"):
        result[f"{impl}_ms"] = sum(row[f"{impl}_ms"] * row["count"] for row in rows)
        result[f"{impl}_tflops"] = total_flops / result[f"{impl}_ms"] / 1e9
    result["cute_native_ratio"] = result["cute_tflops"] / result["native_tflops"]
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--schedule", required=True, choices=("cpasync", "cooperative"))
parser.add_argument("--json-out", required=True)
parser.add_argument("--enforce-gates", action="store_true")
args = parser.parse_args()

rows = []
for shape in shapes():
    a, b = _make(shape["M"], shape["K"], shape["N"])
    reference = a.float() @ b.float()
    row = {**shape, **_bench_mxfp8(a, b, reference, args.schedule)}
    rows.append(row)
    print(
        row["projection"], row["role"],
        f"cute={row['cute_ms']:.6f}",
        f"native={row['native_ms']:.6f}",
        f"native/cute={row['native_ms'] / row['cute_ms']:.6f}",
    )

summary = aggregate(rows)
mlp_ratios = {
    f"{row['projection']}:{row['role']}": row["native_ms"] / row["cute_ms"]
    for row in rows
    if row["projection"].startswith("mlp_")
}
gates = {
    "weighted_target": 0.85,
    "weighted_actual": summary["cute_native_ratio"],
    "weighted_pass": summary["cute_native_ratio"] >= 0.85,
    "mlp_row_target": 0.75,
    "mlp_row_actual": mlp_ratios,
    "mlp_rows_pass": all(ratio >= 0.75 for ratio in mlp_ratios.values()),
}
print("aggregate", json.dumps(summary, sort_keys=True))
with open(args.json_out, "w") as output_file:
    json.dump(
        {"schedule": args.schedule, "rows": rows, "aggregate": summary, "gates": gates},
        output_file,
        indent=2,
    )

if args.enforce_gates and not (gates["weighted_pass"] and gates["mlp_rows_pass"]):
    raise SystemExit(f"acceptance gates failed: {json.dumps(gates, sort_keys=True)}")
```

`/tmp/bench_blocked_scale_ceiling.py`:

```python
import argparse
import json
import re
import subprocess
import sys

TOKENS = 16 * 1024
LINEARS = [
    ("attn_qo", 2, 512, 512),
    ("attn_kv", 2, 512, 256),
    ("mlp_gate_up", 1, 512, 3072),
    ("mlp_down", 1, 1536, 512),
]
UPSTREAM = "/tmp/cutlass-ref.bKwpnV/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py"


def shapes():
    rows = []
    for projection, count, k_in, n_out in LINEARS:
        rows.extend(
            [
                dict(projection=projection, role="forward", M=TOKENS, K=k_in, N=n_out, count=count),
                dict(projection=projection, role="dgrad", M=TOKENS, K=n_out, N=k_in, count=count),
                dict(projection=projection, role="wgrad", M=n_out, K=TOKENS, N=k_in, count=count),
            ]
        )
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--json-out", required=True)
args = parser.parse_args()

rows = []
for shape in shapes():
    cmd = [
        sys.executable,
        UPSTREAM,
        "--mnkl", f"{shape['M']},{shape['N']},{shape['K']},1",
        "--tile_shape_mnk", "128,128,128",
        "--epi_tile", "64,32",
        "--a_dtype", "Float8E4M3FN",
        "--b_dtype", "Float8E4M3FN",
        "--sf_dtype", "Float8E8M0FNU",
        "--sf_vec_size", "32",
        "--c_dtype", "BFloat16",
        "--acc_dtype", "Float32",
        "--warmup_iterations", "10",
        "--iterations", "50",
        "--skip_ref_check",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    match = re.search(r"Execution time: ([0-9.]+) microseconds", completed.stdout)
    if match is None:
        raise RuntimeError(completed.stdout + completed.stderr)
    row = {**shape, "blocked_scale_ms": float(match.group(1)) / 1000.0}
    rows.append(row)
    print(shape["projection"], shape["role"], f"blocked_scale={row['blocked_scale_ms']:.6f}")

total_flops = sum(2 * row["M"] * row["K"] * row["N"] * row["count"] for row in rows)
total_ms = sum(row["blocked_scale_ms"] * row["count"] for row in rows)
aggregate = {
    "flops": total_flops,
    "blocked_scale_ms": total_ms,
    "blocked_scale_tflops": total_flops / total_ms / 1e9,
}
print("aggregate", json.dumps(aggregate, sort_keys=True))
with open(args.json_out, "w") as output_file:
    json.dump({"rows": rows, "aggregate": aggregate}, output_file, indent=2)
```

| projection | role | cpasync ms | cooperative ms | speedup |
|---|---|---:|---:|---:|
| mlp_gate_up | forward | 0.517089 | 0.383271 | +34.91% |
| mlp_gate_up | dgrad | 0.637038 | 0.567360 | +12.28% |
| mlp_gate_up | wgrad | 0.556171 | 0.422504 | +31.64% |
| mlp_down | forward | 0.259648 | 0.280822 | -7.54% |
| mlp_down | dgrad | 0.263042 | 0.277938 | -5.36% |
| mlp_down | wgrad | 0.338244 | 0.272332 | +24.20% |

The full weighted cpasync aggregate was 5.654098 ms, 54.69 TFLOP/s, and a
CuTe/native throughput ratio of 0.45127. Cooperative was 5.530528 ms,
55.91 TFLOP/s, and 0.46370: a 2.23% weighted latency improvement. The official
preblocked-scale schedule ceiling was 2.597623 ms and 119.05 TFLOP/s, showing that
native strided scale staging remains the dominant gap.

### Final launch profile

The retained cooperative kernel for `(M,K,N)=(16384,1536,512)` launched one
persistent CTA per SM: grid `[1,1,36]`, block `[288,1,1]`, 126 registers/thread,
and 72,704 bytes shared memory/CTA. Its single profiler sample was 239.327 us,
versus the frozen cpasync sample of 269.633 us. NCU hardware counters remain
unavailable because this host still returns `ERR_NVGPUCTRPERM`; no occupancy,
throughput, or stall counters are inferred.

The cooperative schedule materially improves the compute-heavy MLP rows, but small
attention and MLP-down rows regress and the weighted ratio remains below the frozen
0.85 target. The cpasync schedule therefore remains the default fallback while the
cooperative path is explicit and cache-isolated.

## Task 5 compact native-scale staging

The Task 4 native-scale cooperative path reached only 0.46370 of native, while the
private preblocked-scale ceiling reached 119.05 TFLOP/s. The single follow-up
hypothesis is therefore to replace the producer warp's scattered scale copies with
ordinary TMA G2S loads from the unchanged native `(rows, K//32)` scale tensors into
compact shared `(128, 4, stages)` tiles. SFA and SFB's exact 512 bytes per stage
would join A and B in the main TMA barrier transaction count, and consumers would
gather the compact logical coordinates into the existing `_MmaMXF8Op` scale
fragments. The A/B tile, MMA, A/B TMA, scheduler, epilogue, and public scale
contract remain fixed.

The experiment started behind the temporary internal schedule
`cooperative_compact_sf`. Its focused spread-scale test failed before production
changes with `AssertionError: unsupported schedule cooperative_compact_sf`, which
confirmed that the test exercised the missing path rather than an existing
schedule.

The attempted ordinary scale descriptor used `CopyBulkTensorTileG2SOp` with
`internal_type=cutlass.Int16`, matching the pinned CUTLASS helper, and an internal
16-byte global row pitch for short scale rows. Each `(128, 4)` shared tile used
`cute.make_layout(((32, 4), 4), stride=((16, 4), 1))`; equivalently, scale
`(row, group)` maps to byte offset
`16 * (row % 32) + 4 * (row // 32) + group`. This is the same physical swizzle as
the existing blockscaled SFA/SFB layouts, so the consumer could alias the compact
allocation with those layouts and retain `partition_fragment_SFA/SFB`,
`get_layoutSFA_TV/SFB`, and the exact `_MmaMXF8Op` register fragments. The main
TMA barrier expected 33,792 bytes per stage: 16,384 A + 16,384 B + 512 SFA +
512 SFB.

After marking the experiment switch `cutlass.Constexpr[bool]`, lowering succeeded
and the launch was accepted, but synchronization returned
`cudaErrorIllegalInstruction`.
The same failure reproduced on the fully aligned `(M,N,K)=(128,128,128)` case.
Compute Sanitizer reported a warp illegal instruction at kernel PC `+0x2570` for
threads 256--287, the producer warp, and an error summary of 34 errors. The scale
descriptors already used `cutlass.Int16`; rechecking them against the official
helper therefore did not provide a distinct safe variant. The compact TMA path was
not timed, and the temporary schedule, focused test, descriptors, compact storage,
and consumer aliases were removed rather than weakening correctness.

After removal, the complete focused file remained green:

```text
CUDA_VISIBLE_DEVICES=1 uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6
48 passed, 1056 warnings in 37.95s
```

Fresh public MXFP8 rows (`/tmp/scaled-gemm-cpasync-final.json` and
`/tmp/scaled-gemm-final.json`) were:

| shape | M | cpasync ms | cooperative ms | native ms | cooperative speedup | cooperative/native |
|---|---:|---:|---:|---:|---:|---:|
| attn_proj | 4096 | 0.267154 | 0.269394 | 0.072542 | -0.83% | 0.26928 |
| attn_proj | 16384 | 0.256743 | 0.270921 | 0.073645 | -5.23% | 0.27183 |
| mlp_up | 4096 | 0.265400 | 0.267309 | 0.096446 | -0.71% | 0.36080 |
| mlp_up | 16384 | 0.523598 | 0.389498 | 0.382305 | +34.43% | 0.98153 |
| mlp_down | 4096 | 0.257943 | 0.271833 | 0.071562 | -5.11% | 0.26326 |
| mlp_down | 16384 | 0.269593 | 0.278084 | 0.169887 | -3.05% | 0.61092 |

The private 16K-token forward/dgrad/wgrad comparison used the helper above. The
final cooperative invocation deliberately enforced both gates after writing its
JSON:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_qwen3_mxfp8.py \
  --schedule cpasync --json-out /tmp/qwen3-cpasync-final.json
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_qwen3_mxfp8.py \
  --schedule cooperative --json-out /tmp/qwen3-final.json --enforce-gates
```

The results were:

| projection | role | cpasync ms | cooperative ms | native ms | cooperative speedup | cooperative/native |
|---|---|---:|---:|---:|---:|---:|
| attn_qo | forward | 0.251591 | 0.266384 | 0.069360 | -5.55% | 0.26038 |
| attn_qo | dgrad | 0.254850 | 0.268013 | 0.070060 | -4.91% | 0.26141 |
| attn_qo | wgrad | 0.259296 | 0.264124 | 0.097889 | -1.83% | 0.37062 |
| attn_kv | forward | 0.257661 | 0.275113 | 0.068569 | -6.34% | 0.24924 |
| attn_kv | dgrad | 0.295965 | 0.281772 | 0.072069 | +5.04% | 0.25577 |
| attn_kv | wgrad | 0.257325 | 0.263638 | 0.097903 | -2.39% | 0.37135 |
| mlp_gate_up | forward | 0.513821 | 0.384546 | 0.376015 | +33.62% | 0.97781 |
| mlp_gate_up | dgrad | 0.635840 | 0.566100 | 0.313841 | +12.32% | 0.55439 |
| mlp_gate_up | wgrad | 0.555026 | 0.422401 | 0.317782 | +31.40% | 0.75232 |
| mlp_down | forward | 0.261135 | 0.270261 | 0.169251 | -3.38% | 0.62625 |
| mlp_down | dgrad | 0.265979 | 0.263088 | 0.191964 | +1.10% | 0.72966 |
| mlp_down | wgrad | 0.338205 | 0.266968 | 0.209018 | +26.68% | 0.78293 |

Cpasync totaled 5.723383 ms and 54.03 TFLOP/s, or 0.44801 of its paired
native measurement. Cooperative totaled 5.411453 ms and 57.15 TFLOP/s, 5.45%
less latency than cpasync but only 0.46745 of its paired 122.25-TFLOP/s native
measurement. The gated cooperative helper wrote `/tmp/qwen3-final.json` and
exited nonzero: the weighted ratio missed 0.85, while three of six MLP rows missed
0.75 (the ratios were 0.97781, 0.55439, 0.75232, 0.62625, 0.72966, and 0.78293).
The earlier private preblocked-scale ceiling remains 119.05 TFLOP/s.

Consequently, the literal `cpasync` argument default remains the production
schedule and diagnostic fallback, while Task 4's correct cooperative schedule
remains an explicit, cache-isolated option for favorable shapes. `_SCHEDULES`
contains only those two maintained paths. No `DEFAULT_SCHEDULE` constant or
`pingpong` path is introduced because the acceptance gates did not pass. The
benchmark module docstring and shape comment are now generic; its public text no
longer labels this matrix as Qwen-specific.

## Preblocked-scale benchmark and promotion gate

Commit `77b5458` added the explicit cooperative
`scale_layout="swizzle_32_4_4"` kernel path. The benchmark follow-up quantizes each
operand once and prepares both E8M0 `SWIZZLE_32_4_4` scale tensors before any timed
closure is defined. `cute_ms` now measures that preblocked cooperative path,
`cute_native_scale_ms` measures the maintained native-scale path selected by
`--cute-schedule`, and the existing native arm receives the same preblocked scale
tensors as the primary CuTe arm. The generic public matrix remains 120 rows and no
public Qwen mode was added.

Every GPU command below was preceded by `nvidia-smi` and pinned to physical GPU 1.
The preflights at 08:34, 08:36, 08:37, and 08:38 on August 14 showed GPU 1 at 15 MiB
and 0% utilization before the public benchmark, private gate, profiler, and full
kernel suite respectively.

### RED and GREEN contract evidence

The benchmark contract test keeps the real Triton, preblocked CuTe, native-scale
CuTe, and native calls for numerical output, while replacing `_time` with the
deterministic sequence `1.0`, `2.0`, `4.0`, and `8.0` ms. Its traced scale converter
must run twice before the first `_time` call and never while a timed closure is
executing. It also checks the exact CuTe schedule/layout pairs and requires all four
CuTe timing/ratio fields to be explicit `None` on a non-MXFP8 row.

RED command and result:

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6 \
  -k "scheme_matrix or benchmark or mxfp8"
```

```text
2 failed, 57 passed, 1461 warnings in 44.70s
```

Both failures were the intended missing contract: the MXFP8 row had no
`cute_native_scale_ms`, and non-MXFP8 rows lacked the new explicit `None` fields.
After the minimal benchmark wiring, the complete focused file was GREEN:

```bash
CUDA_VISIBLE_DEVICES=1 uv run pytest tests/fast/kernel/cute/test_gemm.py -n 6
```

```text
63 passed, 1704 warnings in 48.79s
```

### Generic public matrix

Command and retained JSON artifact:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmarks/gemm/bench_scaled_gemm.py \
  --no-plot --cute-schedule cpasync \
  --json-out /tmp/scaled-gemm-preblocked.json
```

An independent JSON validation confirmed exactly 120 rows and exactly six MXFP8
rows. Every present `*_ms` value was finite and nonnegative. All five required
MXFP8 values (`cute_ms`, `cute_native_scale_ms`, `native_ms`,
`cute_native_ratio`, and `cute_native_scale_ratio`) were finite and nonnegative;
the 114 other rows had both CuTe measurements and both CuTe/native ratios set to
`None`. The traced contract test above confirms that neither scale conversion is
inside a timed closure.

| shape | M | K | N | preblocked ms | native-scale CuTe ms | native ms | native/preblocked latency | native/native-scale CuTe latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_proj | 4096 | 512 | 512 | 0.254955059 | 0.263033819 | 0.074765140 | 0.293248 | 0.284242 |
| attn_proj | 16384 | 512 | 512 | 0.246966919 | 0.261229619 | 0.073518639 | 0.297686 | 0.281433 |
| mlp_up | 4096 | 512 | 3072 | 0.250092102 | 0.253760682 | 0.099659222 | 0.398490 | 0.392729 |
| mlp_up | 16384 | 512 | 3072 | 0.375914602 | 0.523860960 | 0.382602580 | 1.017791 | 0.730351 |
| mlp_down | 4096 | 1536 | 512 | 0.277278021 | 0.260269400 | 0.074529918 | 0.268791 | 0.286357 |
| mlp_down | 16384 | 1536 | 512 | 0.267776982 | 0.270876221 | 0.169715858 | 0.633796 | 0.626544 |

Across the six rows, preblocked CuTe totaled `1.672983684` ms, native-scale CuTe
totaled `1.833030700` ms, and native totaled `0.874791357` ms. Thus the aggregate
sum-of-latencies native/preblocked latency ratio was `0.522892940`, versus
`0.477237701` for native/native-scale CuTe latency. Preblocking improved this public
aggregate, and the 16K MLP-up row slightly exceeded native, but the aggregate
remained materially below the native kernel.

### Private Qwen3-51M gate

The existing private `/tmp/bench_qwen3_mxfp8.py` harness remained outside the
public benchmark. It covers 12 projection/role rows; its six MLP rows are the
per-row gate set. Through the updated `_bench_mxfp8`, `cute_ms` uses the preblocked
scale tensors and `scale_layout="swizzle_32_4_4"`, while
`cute_native_scale_ms` retains native scales. The helper writes JSON before
enforcing either gate.

```bash
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/bench_qwen3_mxfp8.py \
  --schedule cpasync --json-out /tmp/qwen3-preblocked.json --enforce-gates
```

The JSON was written successfully and the command then exited 1 because the
acceptance gates missed:

| projection | role | count | M | K | N | preblocked ms | native-scale CuTe ms | native ms | native/preblocked latency | native/native-scale CuTe latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| attn_qo | forward | 2 | 16384 | 512 | 512 | 0.244665840 | 0.261496119 | 0.072921580 | 0.298046 | 0.278863 |
| attn_qo | dgrad | 2 | 16384 | 512 | 512 | 0.249974539 | 0.250992901 | 0.073241221 | 0.292995 | 0.291806 |
| attn_qo | wgrad | 2 | 512 | 16384 | 512 | 0.243713439 | 0.253129159 | 0.098109900 | 0.402563 | 0.387588 |
| attn_kv | forward | 2 | 16384 | 512 | 256 | 0.248262761 | 0.247516821 | 0.074712939 | 0.300943 | 0.301850 |
| attn_kv | dgrad | 2 | 16384 | 256 | 512 | 0.252918920 | 0.252678541 | 0.075740220 | 0.299464 | 0.299749 |
| attn_kv | wgrad | 2 | 256 | 16384 | 512 | 0.243969241 | 0.255581960 | 0.097934720 | 0.401422 | 0.383183 |
| mlp_gate_up | forward | 1 | 16384 | 512 | 3072 | 0.371174740 | 0.519899679 | 0.374524500 | 1.009025 | 0.720378 |
| mlp_gate_up | dgrad | 1 | 16384 | 3072 | 512 | 0.565531682 | 0.638883261 | 0.313394901 | 0.554160 | 0.490535 |
| mlp_gate_up | wgrad | 1 | 3072 | 16384 | 512 | 0.422383200 | 0.559281479 | 0.318708720 | 0.754549 | 0.569854 |
| mlp_down | forward | 1 | 16384 | 1536 | 512 | 0.263081740 | 0.259053840 | 0.173456639 | 0.659326 | 0.669578 |
| mlp_down | dgrad | 1 | 16384 | 512 | 1536 | 0.249042201 | 0.268369659 | 0.197175820 | 0.791737 | 0.734717 |
| mlp_down | wgrad | 1 | 512 | 16384 | 1536 | 0.248802879 | 0.339113621 | 0.209509260 | 0.842069 | 0.617814 |

The multiplicity-weighted totals were `5.087025922` ms for preblocked CuTe,
`5.627392540` ms for native-scale CuTe, and `2.572091001` ms for native. The
weighted native/preblocked latency ratio (equivalently, preblocked/native
throughput ratio) was `0.505617829`, below the `0.85` target. The six MLP
native/preblocked latency ratios were `1.009025`, `0.554160`,
`0.754549`, `0.659326`, `0.791737`, and `0.842069`; gate-up dgrad and MLP-down
forward missed the `0.75` row target. Both the weighted gate and all-MLP-rows gate
therefore failed.

The preblocked path remains explicit kernel-only experimental support. The public
default and training call sites remain on `scale_layout="native"`, so
`benchmarks/bench_train.py` is not applicable to this isolated kernel option.

### Representative launch profile

The worst MLP gate row, gate-up dgrad `(M,K,N)=(16384,3072,512)`, was profiled
with preblocked scale tensors. The temporary helper changed only the CuTe scale
arguments and layout, and wrote `/tmp/qwen3-mxfp8-preblocked-profile.json`:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python /tmp/profile_qwen3_mxfp8.py
```

| implementation | profiler duration us | grid | block | registers/thread | shared memory/CTA |
|---|---:|---:|---:|---:|---:|
| preblocked CuTe cooperative | 547.200 | `[1,1,36]` | `[288,1,1]` | 128 | 72,704 B |
| native CUTLASS | 336.256 | `[4,128,1]` | `[384,1,1]` | 168 | 81,920 B |

The preblocked kernel has one persistent CTA per SM and nine warps per CTA, while
the native launch has 512 total CTAs and twelve warps per CTA. CuTe uses fewer
registers per thread and less shared memory per CTA, so the remaining gap is not
explained by a simple register or shared-memory ceiling. The evidence instead
points to launch geometry and schedule efficiency after scale conversion has been
removed. The trace's `blocks per SM` and `warps per SM` values describe whole-grid
distribution, not achieved occupancy; hardware counters remain unavailable, so
the profile does not claim a measured occupancy or warp-stall cause.

### Final validation

```text
CUDA_VISIBLE_DEVICES=1 uv run pytest tests/fast/kernel -n 12 --dist load
227 passed, 1962 warnings in 55.00s

uv run ruff check src/ tests/
All checks passed!

uv run ruff format --check src/ tests/
80 files already formatted

git diff --check
exit 0
```
