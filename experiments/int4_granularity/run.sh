#!/bin/bash
# Run the int4 scale-granularity sweep (6 blockwise granularities x W8A16/W8A8) vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/int4_granularity/run.sh > logs/int4_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for arm in w8a16 w8a8; do
    for layout in blockwise1d blockwise2d; do
        for extent in 16 32 64; do
            configs+=("qwen3_51m_int4_${arm}_${layout}_${extent}")
        done
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_granularity runs complete ==="
