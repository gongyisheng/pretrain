#!/bin/bash
# Run the d_model versus depth sweep.
# Usage: nohup bash experiments/d_model_x_depth/run.sh > logs/d_model_x_depth.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

model_sizes=(114m_a101m_d256 126m_a101m_d512 152m_a101m_d1024)

for model_size in "${model_sizes[@]}"; do
    config="qwen3_${model_size}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/d_model_x_depth/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== d_model versus depth sweep complete ==="
