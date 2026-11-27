#!/bin/bash
# Run the AdamH learning-rate sweep with AdamW and AdamC baselines.
# Usage: nohup bash experiments/adamh/run.sh > logs/adamh.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_adamw qwen3_51m_adamc)
adamh_lrs=(1e-3 2e-3 5e-3 1e-2 2e-2 5e-2)
for lr in "${adamh_lrs[@]}"; do
    configs+=("qwen3_51m_adamh_lr${lr}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/adamh/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done
