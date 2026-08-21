#!/bin/bash
# Run weight-only int8 scale-granularity sweep (10 recipes) vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/int8_granularity/run.sh > logs/int8_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16 qwen3_51m_int8_tensorwise qwen3_51m_int8_rowwise)
for layout in blockwise1d blockwise2d; do
    for extent in 16 32 64 128; do
        configs+=("qwen3_51m_int8_${layout}_${extent}")
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
