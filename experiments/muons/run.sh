#!/bin/bash
# Run the Qwen3-51M Muon/MuonS comparison.
# Usage: nohup bash experiments/muons/run.sh > logs/muons.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

for optimizer in muon muons; do
    config="qwen3_51m_${optimizer}"
    uv run python scripts/train.py --config "experiments/muons/${config}.yaml" "$@"
done
