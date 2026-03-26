#!/bin/bash
# Run all scaling law experiments sequentially (smallest first)
# Usage: nohup bash experiments/scaling_law/gpt2/run.sh > scaling_law_gpt2.log 2>&1 &

set -e
cd "$(dirname "$0")/../../.."

for config in gpt2_16m gpt2_30m gpt2_55m gpt2_124m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/scaling_law/gpt2/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All scaling law runs complete ==="
