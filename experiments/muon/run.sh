#!/bin/bash
# Run AdamW vs Muon comparison across 16M / 30M / 55M sequentially (smallest first).
# Usage: nohup bash experiments/muon/run.sh > muon.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for size in 16m 30m 55m; do
    for opt in adamw muon; do
        config="gpt2_${size}_${opt}"
        echo "=== ${config} === $(date)"
        python scripts/train.py --config "experiments/muon/${config}.yaml"
        echo ""
    done
done

echo "=== All Muon comparison runs complete ==="
