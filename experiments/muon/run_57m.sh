#!/bin/bash
# Run the qwen3 57M muon experiments: AdamW baseline then Muon.
# Usage: nohup bash experiments/muon/run_57m.sh > logs/muon_57m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

optimizers=("adamw" "muon")
configs=()
for opt in "${optimizers[@]}"; do
    configs+=("qwen3_57m_${opt}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muon/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All muon 57m runs complete ==="
