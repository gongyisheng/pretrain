#!/bin/bash
# Run Qwen3 51M bf16 versus fp16 experiment.
# Usage: nohup bash experiments/fp16/run.sh > logs/fp16.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(bf16 fp16)
configs=()
for variant in "${variants[@]}"; do
    configs+=("qwen3_51m_${variant}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp16/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All precision runs complete ==="
