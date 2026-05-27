#!/bin/bash
# Run activation comparison experiments at Qwen3-57M.
# Usage: nohup bash experiments/activation/run.sh > logs/activation.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

UNGATED=(relu gelu silu leaky_relu relu2 gelu2 silu2 leaky_relu2)
GATED=(reglu geglu swiglu leaky_reglu reglu2 geglu2 swiglu2 leaky_reglu2 bilinear bilinear2 powlu)

for variant in "${UNGATED[@]}" "${GATED[@]}"; do
    config="qwen3_57m_${variant}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/activation/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All activation runs complete ==="
