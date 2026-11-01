#!/bin/bash
# Run the GQA query-head-count sweep.

set -euo pipefail
cd "$(dirname "$0")/../.."

n_heads_values=(2 4 8 16 32 64)

for n_heads in "${n_heads_values[@]}"; do
    config="qwen3_51m_h${n_heads}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/gqa_n_heads/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== GQA query-head-count sweep complete ==="
