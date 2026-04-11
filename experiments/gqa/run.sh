#!/bin/bash
# Run all GQA KV head sweep experiments sequentially (MQA to MHA)
# Usage: nohup bash experiments/gqa/run.sh > logs/gqa.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for kv in 1 2 4 8; do
    config="qwen3_57m_kv${kv}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/gqa/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All GQA runs complete ==="
