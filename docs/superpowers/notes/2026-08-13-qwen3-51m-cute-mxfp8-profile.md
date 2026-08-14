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
print("aggregate", json.dumps(summary, sort_keys=True))
with open(args.json_out, "w") as output_file:
    json.dump({"schedule": args.schedule, "rows": rows, "aggregate": summary}, output_file, indent=2)
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
