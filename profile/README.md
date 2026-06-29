# profile/

Standalone forward/backward kernel profiler for **nsys** / **ncu**. Drives any
`configs/*.yaml` with a fixed synthetic batch (no `Trainer`, no dataloader), so
shapes stay constant and kernels are stable enough for per-kernel inspection.

NVTX ranges label `forward` / `backward` / `iter/N`; `cudaProfilerApi` gates the
capture to steady-state steps (warmup runs outside the window, excluding
`torch.compile` tracing + autotuning).

## What gets profiled

By default the **compiled bf16** path — what training actually runs. Kernels show
up as fused Inductor/Triton kernels (`triton_*`) plus cuBLAS/cutlass GEMMs.
`unique_kernel_names` is on so names are readable for attribution.

`--no-compile` profiles **eager**, giving clean aten attribution — you see
`aten::_grouped_mm`, sdpa/flash, and the MoE scatter/gather index kernels
directly. Useful to understand what the compiled path fuses.

## Flags

| flag | default | meaning |
|---|---|---|
| `--config` | (required) | any `configs/*.yaml` |
| `--warmup` | 10 | steps before capture (compile + autotune settle here) |
| `--steps` | 5 | profiled steps inside the capture window |
| `--batch-size` | from YAML | override batch size |
| `--seq-len` | from YAML | override sequence length |
| `--no-compile` | off | profile eager instead of compiled |
| `--forward-only` | off | skip backward |

## nsys

```bash
mkdir -p logs/profiles
nsys profile --capture-range=cudaProfilerApi -o logs/profiles/qwen3_moe \
    python profile/profile_model.py --config configs/qwen3_moe_133m.yaml

# kernel time ranking
nsys stats --report cuda_gpu_kern_sum logs/profiles/qwen3_moe.nsys-rep
```

## ncu

ncu replays each kernel (~100× slowdown) — use `--steps 1` and scope with NVTX
(and optionally a kernel-name regex):

```bash
ncu --profile-from-start off --nvtx --nvtx-include "iter/" \
    -o logs/profiles/qwen3_moe_ncu \
    python profile/profile_model.py --config configs/qwen3_moe_133m.yaml --steps 1

# scope to specific kernels, full metric set
ncu --profile-from-start off --nvtx --nvtx-include "iter/" --set full \
    -k regex:"grouped|scatter|gather" \
    -o logs/profiles/qwen3_moe_grouped \
    python profile/profile_model.py --config configs/qwen3_moe_133m.yaml --steps 1
```

`--profile-from-start off` makes ncu honor the `cudaProfilerStart/Stop` gate.

## Tips

- Compare compiled vs `--no-compile` to see what Inductor fused.
- Keep `--steps` small for ncu, larger for nsys timeline stability.
- bf16 is the target (dropless MoE requires it); fp16/fp32 configs aren't the
  intended use.
