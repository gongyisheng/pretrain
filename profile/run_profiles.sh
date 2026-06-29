#!/usr/bin/env bash
# Profile several models with nsys and print a per-model kernel-time ranking.
#
# Same batch/seq for every model so kernel costs are comparable. Edit the
# `configs` array (or pass configs as args) and tune B/S/STEPS via env vars.
#
# Usage:
#   bash profile/run_profiles.sh                       # default config list
#   bash profile/run_profiles.sh configs/qwen3_51m.yaml configs/gpt2_124m.yaml
#   GPU=1 B=8 S=1024 STEPS=5 bash profile/run_profiles.sh
#   COMPILE=0 bash profile/run_profiles.sh             # eager (aten attribution)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
B="${B:-8}"
S="${S:-1024}"
WARMUP="${WARMUP:-10}"
STEPS="${STEPS:-5}"
COMPILE="${COMPILE:-1}"     # 1 = compiled bf16 (default), 0 = eager
OUTDIR="${OUTDIR:-logs/profiles}"

configs=("$@")
if [ ${#configs[@]} -eq 0 ]; then
  configs=(
    configs/gpt2_124m.yaml
    configs/qwen3_51m.yaml
    configs/qwen3_183m_a51m.yaml
  )
fi

compile_flag=""
suffix="compiled"
if [ "$COMPILE" = "0" ]; then
  compile_flag="--no-compile"
  suffix="eager"
fi

mkdir -p "$OUTDIR"
echo "GPU=$GPU B=$B S=$S warmup=$WARMUP steps=$STEPS mode=$suffix"

for config in "${configs[@]}"; do
  name="$(basename "$config" .yaml)_${suffix}"
  rep="$OUTDIR/$name"
  echo "=== profiling $config -> $rep.nsys-rep ==="
  CUDA_VISIBLE_DEVICES="$GPU" nsys profile \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    --force-overwrite=true -o "$rep" \
    uv run python profile/profile_model.py \
      --config "$config" --batch-size "$B" --seq-len "$S" \
      --warmup "$WARMUP" --steps "$STEPS" $compile_flag
  echo "--- top kernels: $config ---"
  nsys stats --report cuda_gpu_kern_sum "$rep.nsys-rep" | tee "$rep.kern_sum.txt"
done

echo "done. reports + kernel summaries in $OUTDIR/"
