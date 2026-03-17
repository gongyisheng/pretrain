#!/bin/bash
# Run all AttnRes scaling law experiments sequentially (smallest first)
# Usage: nohup bash experiments/attn_res/run.sh > attn_res.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in gpt2_16m gpt2_30m gpt2_55m gpt2_124m; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/attn_res/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All AttnRes runs complete ==="
