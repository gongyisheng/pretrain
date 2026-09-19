#!/bin/bash
# Run the int4 weight-granularity sweep (8 blockwise granularities) vs bf16 at 51M on Qwen3.
# Usage: nohup bash experiments/int4_weight_granularity/run.sh > logs/int4_weight_granularity.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_bf16)
for layout in blockwise1d blockwise2d; do
    for extent in 16 32 64 128; do
        configs+=("qwen3_51m_int4_w4a16_${layout}_${extent}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M int4_weight_granularity runs complete ==="
