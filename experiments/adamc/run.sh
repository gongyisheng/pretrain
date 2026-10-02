#!/bin/bash
# Run Qwen3-51M AdamW and AdamC experiment variants.
# Usage: nohup bash experiments/adamc/run.sh > logs/adamc_51m.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

optimizers=(adamw adamc)
variants=(50k 200k 800k 200k_bs_256)
for variant in "${variants[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        config="qwen3_51m_${optimizer}_steps_${variant}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/adamc/${config}.yaml" "$@"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== AdamW/AdamC comparison complete ==="
