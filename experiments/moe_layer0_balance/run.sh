#!/bin/bash
# moe_layer0_balance: remedies for layer-0 routing imbalance.
# Usage: nohup bash experiments/moe_layer0_balance/run.sh > logs/moe_layer0_balance.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."

arms=(baseline dense_first shallow_bias_boost)

for arm in "${arms[@]}"; do
    echo "=== ${arm} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_layer0_balance/${arm}.yaml"
    echo "Finished at: $(date)"
    echo ""
done
echo "=== All moe_layer0_balance runs complete ==="
