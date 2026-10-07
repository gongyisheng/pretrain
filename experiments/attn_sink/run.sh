#!/bin/bash
# Run Qwen3 51M attention-sink ablation.
# Usage: nohup bash experiments/attn_sink/run.sh > logs/attn_sink.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

variants=(attn_sink_off attn_sink_on)
configs=()
for variant in "${variants[@]}"; do
    configs+=("qwen3_51m_${variant}")
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/attn_sink/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All attention-sink runs complete ==="
