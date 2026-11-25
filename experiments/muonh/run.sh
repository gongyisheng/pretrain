#!/bin/bash
# Run the MuonH learning-rate sweep with MuonW and MuonC baselines.
# Usage: nohup bash experiments/muonh/run.sh > logs/muonh_experiment.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

configs=(qwen3_51m_muonw qwen3_51m_muonc)
muonh_lrs=(1e-3 2e-3 5e-3 1e-2 2e-2 5e-2)
for lr in "${muonh_lrs[@]}"; do
    configs+=("qwen3_51m_muonh_lr${lr}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muonh/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done
