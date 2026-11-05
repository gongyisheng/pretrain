#!/bin/bash
# Run Qwen3 51M bf16 versus fp32 experiment.
# Usage: nohup bash experiments/fp32/run.sh > logs/fp32.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 fp32)
configs=()
for variant in "${variants[@]}"; do
    configs+=("qwen3_51m_${variant}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp32/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All precision runs complete ==="
