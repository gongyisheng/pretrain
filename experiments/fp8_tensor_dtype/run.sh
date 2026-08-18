#!/bin/bash
# Run per-tensor FP8 sensitivity sweep (4 tensors x 2 formats) vs bf16 at 51M on Qwen3.
# Stage 1 (e4m3) ranks the four tensors; stage 2 (e5m2) repeats the ladder in the
# other format and is only worth running where stage 1 found a gap.
# Usage: nohup bash experiments/fp8_tensor_dtype/run.sh [1|2] > logs/fp8_tensor_dtype.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

stage=${1:-1}
layouts=(w8a16 w16a8 w16a16dx8 w16a16dw8)
# bf16 anchor: drop it if fp8_dtype's seed-42 bf16 has already run (same base config).
stage1=(qwen3_51m_bf16)
stage2=()
for fmt in e4m3 e5m2; do
    for layout in "${layouts[@]}"; do
        config="qwen3_51m_fp8_${fmt}_${layout}"
        if [ "${fmt}" = e4m3 ]; then
            stage1+=("${config}")
        else
            stage2+=("${config}")
        fi
    done
done
# forward pair, for the weight x act interaction; also fp8_dtype's backward-2x2 corner.
stage1+=(qwen3_51m_fp8_e4m3_w8a8)

if [ "${stage}" = 1 ]; then
    configs=("${stage1[@]}")
elif [ "${stage}" = 2 ]; then
    configs=("${stage2[@]}")
else
    echo "unknown stage: ${stage} (expected 1 or 2)" >&2
    exit 1
fi

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_tensor_dtype/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_tensor_dtype stage ${stage} complete ==="
