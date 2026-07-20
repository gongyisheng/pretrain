#!/usr/bin/env bash
# Profile training kernels with Nsight Systems (nsys), for both the compiled and
# eager (--disable-torch-compile) paths, then print a per-kernel time summary for
# each so the fusion effect is easy to compare.
#
# Usage:
#   benchmarks/profile_train.sh [CONFIG] [STEPS] [WARMUP] [GPU]
#
# Examples:
#   benchmarks/profile_train.sh
#   benchmarks/profile_train.sh configs/gpt2_124m.yaml
#   benchmarks/profile_train.sh configs/qwen3_183m_a51m.yaml 5 3 0
#
# Outputs (per mode): logs/profiles/<name>.nsys-rep + <name>.kernels.txt
#
# nsys records only the measured steps (cudaProfilerStart/Stop capture range), so
# the summary excludes compile/autotune warmup. WARMUP must be large enough for
# torch.compile to finish before it — bump it if the compiled trace still shows
# one-off Triton/autotune kernels.
set -euo pipefail

CONFIG="${1:-configs/qwen3_51m.yaml}"
STEPS="${2:-5}"
WARMUP="${3:-3}"
GPU="${4:-0}"

OUT_DIR="logs/profiles"
mkdir -p "$OUT_DIR"

# Trace name derives from the config's basename (e.g. gpt2_124m).
BASE="$(basename "$CONFIG" .yaml)"

profile_one() {
    local mode="$1"          # "compiled" or "eager"
    local extra_flag="$2"    # "" or "--disable-torch-compile"
    local rep="$OUT_DIR/${BASE}_${mode}"

    echo "=== profiling: $BASE ($mode) ==="
    # --cuda-profiler makes the bench call cudaProfilerStart/Stop around the
    # measured steps; --capture-range=cudaProfilerApi limits nsys to that window,
    # excluding compile/autotune warmup.
    CUDA_VISIBLE_DEVICES="$GPU" nsys profile \
        --output="$rep" \
        --trace=cuda,nvtx,cudnn,cublas,osrt \
        --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        uv run python benchmarks/bench_train.py \
            --config "$CONFIG" --steps "$STEPS" --warmup "$WARMUP" \
            --cuda-profiler --emit-nvtx $extra_flag

    # Per-kernel summary: call count + total/avg/min/max time, sorted by time%.
    # --force-export re-exports the .sqlite from the current .nsys-rep (else a
    # stale export from a prior run makes nsys stats error out).
    nsys stats --force-export=true --report cuda_gpu_kern_sum "${rep}.nsys-rep" \
        | tee "${rep}.kernels.txt"
    echo "saved: ${rep}.nsys-rep and ${rep}.kernels.txt"
    echo
}

profile_one compiled ""
profile_one eager "--disable-torch-compile"

echo "Done. Compare kernel breakdowns:"
echo "  $OUT_DIR/${BASE}_compiled.kernels.txt   (fewer fused Triton kernels)"
echo "  $OUT_DIR/${BASE}_eager.kernels.txt      (more small unfused kernels)"
