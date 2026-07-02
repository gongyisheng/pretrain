#!/bin/bash
# moe_muon_optm: AdamW vs Muon on the 183M/a51M MoE, optimizer swap only.
# Usage: nohup bash experiments/moe_muon_optm/run.sh > logs/moe_muon_optm.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

optimizers=("adamw" "muon")
configs=()
for opt in "${optimizers[@]}"; do
    configs+=("qwen3_183m_a51m_${opt}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    uv run python scripts/train.py --config "experiments/moe_muon_optm/${config}.yaml"
done

echo "=== All moe_muon_optm runs complete ==="
