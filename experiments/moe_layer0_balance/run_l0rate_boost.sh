#!/bin/bash
# moe_layer0_balance / shallow bias boost: sweep layer-0 expert_bias_update_rate
# (layers 1-7 fixed at 1e-3).
# Usage: nohup bash experiments/moe_layer0_balance/run_l0rate_boost.sh > logs/moe_layer0_balance_l0rate_boost.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."

configs=()
l0_rates=(2e-3 5e-3 1e-2)
for r in "${l0_rates[@]}"; do
    configs+=("qwen3_188m_a51m_l0rate${r}")
done

for cfg in "${configs[@]}"; do
    echo "=== ${cfg} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_layer0_balance/${cfg}.yaml"
    echo "Finished at: $(date)"
    echo ""
done
echo "=== All l0rate_boost runs complete ==="
