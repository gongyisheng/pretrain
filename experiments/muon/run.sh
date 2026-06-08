#!/bin/bash
# Run all muon experiments sequentially: per model size, AdamW baseline then Muon.
# Usage: nohup bash experiments/muon/run.sh > logs/muon.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

sizes=("57m" "0.5b")
optimizers=("adamw" "muon")
configs=()
for size in "${sizes[@]}"; do
    for opt in "${optimizers[@]}"; do
        configs+=("qwen3_${size}_${opt}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muon/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All muon runs complete ==="
