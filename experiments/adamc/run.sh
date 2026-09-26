#!/bin/bash
# Run the Qwen3-51M AdamW and AdamC comparison.
# Usage: nohup bash experiments/adamc/run.sh > logs/adamc_51m.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

optimizers=(adamw adamc)
for optimizer in "${optimizers[@]}"; do
    config="qwen3_51m_${optimizer}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/adamc/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== AdamW/AdamC comparison complete ==="
