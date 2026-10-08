#!/bin/bash
# Run Qwen3-51M Muon and MuonC experiment variants.
# Usage: nohup bash experiments/muonc/run.sh > logs/muonc_51m.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

variants=(50k 200k 800k 200k_bs_256)
optimizers=(muon muonc)
for variant in "${variants[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        config="qwen3_51m_${optimizer}_steps_${variant}"
        echo "=== ${config} ==="
        echo "Started at: $(date)"
        uv run python scripts/train.py --config "experiments/muonc/${config}.yaml" "$@"
        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=== Muon/MuonC comparison complete ==="
