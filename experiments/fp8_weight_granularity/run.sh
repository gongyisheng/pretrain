#!/bin/bash
# Run the FP8 E4M3 weight-scale granularity sweep (10 W8A16 granularities) plus E5M2 blockwise2d-32 vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/fp8_weight_granularity/run.sh > logs/fp8_weight_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for granularity in tensorwise rowwise; do
    configs+=("qwen3_51m_fp8_w8a16_${granularity}")
done
for layout in blockwise1d blockwise2d; do
    for extent in 16 32 64 128; do
        configs+=("qwen3_51m_fp8_w8a16_${layout}_${extent}")
    done
done
configs+=(qwen3_51m_fp8_w8a16_blockwise2d_32_e5m2)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_weight_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_weight_granularity runs complete ==="
