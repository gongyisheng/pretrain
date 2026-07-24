#!/bin/bash
# moe_layer0_balance / l0 aux-loss: switch layer 0 to Switch-style aux-loss
# balancing (layers 1-7 stay on loss-free expert_bias at 1e-3), sweeping the
# layer-0 aux_loss_coef.
# Usage: nohup bash experiments/moe_layer0_balance/run_l0auxloss.sh > logs/moe_layer0_balance_l0auxloss.log 2>&1 &
set -e
cd "$(dirname "$0")/../.."

configs=()
l0_coefs=(1e-2 1e-1)
for c in "${l0_coefs[@]}"; do
    configs+=("qwen3_188m_a51m_l0auxcoef${c}")
done

for cfg in "${configs[@]}"; do
    echo "=== ${cfg} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_layer0_balance/${cfg}.yaml"
    echo "Finished at: $(date)"
    echo ""
done
echo "=== All l0auxcoef runs complete ==="
