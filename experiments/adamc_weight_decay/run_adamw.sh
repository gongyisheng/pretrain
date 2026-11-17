#!/bin/bash
# Run the AdamW weight-decay sweep.
# Usage: nohup bash experiments/adamc_weight_decay/run_adamw.sh > logs/adamc_weight_decay_baseline.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

weight_decays=(0.1 0.2 0.5)

for weight_decay in "${weight_decays[@]}"; do
    config="qwen3_51m_adamw_wd_${weight_decay}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/adamc_weight_decay/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== AdamW weight-decay sweep complete ==="
