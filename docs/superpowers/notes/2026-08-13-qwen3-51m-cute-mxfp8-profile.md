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
