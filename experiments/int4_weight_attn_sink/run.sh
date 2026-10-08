#!/bin/bash
# Compare attention sinks with bf16 and W4A16 pretraining.
# Usage: nohup bash experiments/int4_weight_attn_sink/run.sh > logs/int4_weight_attn_sink.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

precisions=(bf16 w4a16)
sinks=(on off)

configs=()
for precision in "${precisions[@]}"; do
    for sink in "${sinks[@]}"; do
        configs+=("qwen3_51m_${precision}_attn_sink_${sink}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/int4_weight_attn_sink/${config}.yaml" "$@"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All int4 weight attention-sink runs complete ==="
