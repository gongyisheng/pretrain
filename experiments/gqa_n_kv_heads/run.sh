#!/bin/bash
# Run the GQA KV-head-count sweep.
# Usage: nohup bash experiments/gqa_n_kv_heads/run.sh > logs/gqa_n_kv_heads.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

n_kv_heads_values=(1 2 4 8)

for n_kv_heads in "${n_kv_heads_values[@]}"; do
    config="qwen3_51m_kv${n_kv_heads}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/gqa_n_kv_heads/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== GQA KV-head-count sweep complete ==="
