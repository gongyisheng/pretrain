#!/bin/bash
# Qwen3-51M bf16 baseline plus int4 W4A16 with 1D and 2D blockwise scales, with and without global scale.
# Usage: nohup bash experiments/int4_weight_global_scale/run.sh > logs/int4_weight_global_scale_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

layouts=(blockwise1d blockwise2d)
global_scales=(off on)

configs=(qwen3_51m_bf16)
for layout in "${layouts[@]}"; do
    for global_scale in "${global_scales[@]}"; do
        configs+=("qwen3_51m_int4_w4a16_${layout}_16_gs_${global_scale}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_global_scale/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_weight_global_scale runs complete ==="
