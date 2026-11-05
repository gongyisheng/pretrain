#!/bin/bash
# Run Qwen3 attention gating and ungated controls.
# Usage: nohup bash experiments/attn_gate/run.sh > logs/attn_gate.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

configs=()
for group in "51m headwise" "53m elementwise"; do
    read -r size gate <<< "$group"
    for variant in ungated "$gate"; do
        configs+=("qwen3_${size}_${variant}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/attn_gate/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All attention gate runs complete ==="
