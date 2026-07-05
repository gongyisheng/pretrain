#!/bin/bash
# Run the qwen3 51M fp8 (tensorwise) optimizer comparison: AdamW then Muon.
# Usage: nohup bash experiments/fp8_muon/run.sh > logs/fp8_muon_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

optimizers=("adamw" "muon")
configs=()
for opt in "${optimizers[@]}"; do
    configs+=("qwen3_51m_fp8_${opt}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_muon/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_muon runs complete ==="
