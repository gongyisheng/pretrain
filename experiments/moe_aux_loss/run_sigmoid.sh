#!/bin/bash
# MoE Switch aux-loss coefficient sweep, sigmoid router (183m testbed: 64 routed, top-8, no shared).
# Usage: nohup bash experiments/moe_aux_loss/run_sigmoid.sh > logs/moe_aux_loss_sigmoid.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

coefs=(0 1e-3 1e-2 1e-1 1e-0)

configs=()
for c in "${coefs[@]}"; do
    configs+=("qwen3_183m_a51m_sigmoid_aux_coef${c}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_aux_loss/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_aux_loss sigmoid runs complete ==="
