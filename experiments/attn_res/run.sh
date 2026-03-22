#!/bin/bash
# Run all AttnRes scaling law experiments sequentially (smallest first)
# Usage: nohup bash experiments/attn_res/run.sh > attn_res.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for config in gpt2_d512_l4 gpt2_d512_l8 gpt2_d512_l12 gpt2_d512_l16 gpt2_d512_l20; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    python scripts/train.py --config "experiments/attn_res/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All AttnRes runs complete ==="
