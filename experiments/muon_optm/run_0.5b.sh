#!/bin/bash
# Run the qwen3 0.5B muon experiments: AdamW baseline then Muon.
# Usage: nohup bash experiments/muon_optm/run_0.5b.sh > logs/muon_0.5b.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

optimizers=("adamw" "muon")
configs=()
for opt in "${optimizers[@]}"; do
    configs+=("qwen3_0.5b_${opt}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muon_optm/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All muon qwen3 0.5b runs complete ==="
