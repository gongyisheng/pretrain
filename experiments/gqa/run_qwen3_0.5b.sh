#!/bin/bash
# Run all GQA KV head sweep experiments for Qwen3 0.5B (MQA to MHA)
# Architecture: d=1024, L=28, h=16 (matches Qwen3 0.6B)
# Usage: nohup bash experiments/gqa/run_0.5b.sh > logs/gqa_0.5b.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

for kv in 1 2 4 8 16; do
    config="qwen3_0.5b_kv${kv}"
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/gqa/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All GQA 0.5B runs complete ==="
