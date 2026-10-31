#!/bin/bash
# Run the Muon momentum sweep.
# Usage: nohup bash experiments/muon_momentum/run.sh > logs/muon_momentum.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

muon_momenta=(0.8 0.9 0.95 0.99)

for muon_momentum in "${muon_momenta[@]}"; do
    config="qwen3_51m_muon_momentum_${muon_momentum}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/muon_momentum/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== Muon momentum sweep complete ==="
