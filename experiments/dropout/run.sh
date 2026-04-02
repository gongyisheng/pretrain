#!/bin/bash
# Run all dropout experiments sequentially (smallest to largest)
# Usage: nohup bash experiments/dropout/run.sh > logs/dropout.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

RATES="0.0 0.01 0.02 0.05 0.1 0.2 0.5 0.9 0.95 0.99"

echo "=== All dropout (embd+attn+ffn) ==="
for rate in $RATES; do
    config="qwen3_57m_drop${rate}"
    echo "--- ${config} --- Started at: $(date)"
    uv run python scripts/train.py --config "experiments/dropout/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== Embedding dropout only ==="
for rate in $RATES; do
    config="qwen3_57m_drop_embd${rate}"
    echo "--- ${config} --- Started at: $(date)"
    uv run python scripts/train.py --config "experiments/dropout/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== Attention dropout only ==="
for rate in $RATES; do
    config="qwen3_57m_drop_attn${rate}"
    echo "--- ${config} --- Started at: $(date)"
    uv run python scripts/train.py --config "experiments/dropout/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== FFN dropout only ==="
for rate in $RATES; do
    config="qwen3_57m_drop_ffn${rate}"
    echo "--- ${config} --- Started at: $(date)"
    uv run python scripts/train.py --config "experiments/dropout/${config}.yaml"
    echo "Finished at: $(date)"
done

echo "=== All dropout runs complete ==="
