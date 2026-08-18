#!/bin/bash
# Run FP8 scale-granularity sweep at 51M on Qwen3.
# Usage: nohup bash experiments/fp8_granularity/run.sh > logs/fp8_granularity.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

primary_configs=(qwen3_51m_bf16 qwen3_51m_fp8_tensorwise qwen3_51m_fp8_rowwise)
for layout in blockwise1d blockwise2d; do
    for extent in 16 32 64 128; do
        primary_configs+=("qwen3_51m_fp8_${layout}_${extent}")
    done
done

for config in "${primary_configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_granularity/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Weight-transpose consistency sweep ==="
for grad in g16 g8; do
    for layout in blockwise1d blockwise2d; do
        config="qwen3_51m_fp8_${layout}_32_w8a16${grad}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/fp8_granularity/${config}.yaml"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== 51M FP8 runs complete ==="
