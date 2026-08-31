#!/bin/bash
# Run the int8 scale-granularity sweep (10 granularities x W8A16/W8A8) vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/int8_granularity/run.sh > logs/int8_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for arm in w8a16 w8a8; do
    for granularity in tensorwise rowwise; do
        configs+=("qwen3_51m_int8_${arm}_${granularity}")
    done
    for layout in blockwise1d blockwise2d; do
        for extent in 16 32 64 128; do
            configs+=("qwen3_51m_int8_${arm}_${layout}_${extent}")
        done
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int8_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int8_granularity runs complete ==="
