#!/bin/bash
# Run QK norm ablation experiments (on vs off)
# Usage: nohup bash experiments/qk_norm/run.sh > logs/qk_norm.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for qk in on off; do
    config="qwen3_57m_qk_norm_${qk}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/qk_norm/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All QK norm runs complete ==="
