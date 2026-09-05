#!/bin/bash
# Run the SwiGLU up_shift sweep (0.0 vs 1.0) at 51M on Qwen3.
# Usage: nohup bash experiments/activation_up_shift/run.sh > logs/activation_up_shift.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for shift in 0 1; do
    config="qwen3_51m_up_shift${shift}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/activation_up_shift/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== 51M activation_up_shift runs complete ==="
