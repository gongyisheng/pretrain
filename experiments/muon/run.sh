#!/bin/bash
# Run AdamW vs Muon comparison experiments sequentially
# Usage: nohup bash experiments/muon/run.sh > muon.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in gpt2_adamw gpt2_muon; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/muon/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All Muon comparison runs complete ==="
