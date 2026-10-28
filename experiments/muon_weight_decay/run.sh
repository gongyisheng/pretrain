#!/bin/bash
# Run the Muon weight-decay sweep.
# Usage: nohup bash experiments/muon_weight_decay/run.sh > logs/muon_weight_decay.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

weight_decays=(0 0.01 0.02 0.05 0.1 0.2 0.5 1)

for weight_decay in "${weight_decays[@]}"; do
    config="qwen3_51m_muon_wd_${weight_decay}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muon_weight_decay/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Muon weight-decay sweep complete ==="
