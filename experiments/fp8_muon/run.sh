#!/bin/bash
# Run the qwen3 51M optimizer/precision sweep: bf16 baselines + fp8 tensorwise
# {std, alle4m3} x {adamw, muon}.
# Usage: nohup bash experiments/fp8_muon/run.sh > logs/fp8_muon_51m.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

precisions=("bf16" "fp8_std" "fp8_alle4m3")
optimizers=("adamw" "muon")
configs=()
for precision in "${precisions[@]}"; do
    for opt in "${optimizers[@]}"; do
        configs+=("qwen3_51m_${precision}_${opt}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/fp8_muon/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M fp8_muon runs complete ==="
