#!/bin/bash
# Run the AdamW beta1/beta2 sweep.
# Usage: nohup bash experiments/adamw_b1_b2/run.sh > logs/adamw_b1_b2.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

beta1s=(0.8 0.9 0.95 0.99)
beta2s=(0.9 0.95 0.99 0.999)

for beta1 in "${beta1s[@]}"; do
    for beta2 in "${beta2s[@]}"; do
        config="qwen3_51m_adamw_b1_${beta1}_b2_${beta2}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/adamw_b1_b2/${config}.yaml" "$@"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== AdamW beta1/beta2 sweep complete ==="
