#!/bin/bash
# Run all lion_optm experiments sequentially (AdamW baseline first, then Lion grid)
# Usage: nohup bash experiments/lion_optm/run.sh > logs/lion_optm.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=("qwen3_57m_adamw_lr5e-4_wd0.1")
lrs=("5e-5" "1e-4" "2e-4")
wds=("0.1" "0.2" "0.5" "1.0")
for lr in "${lrs[@]}"; do
    for wd in "${wds[@]}"; do
        configs+=("qwen3_57m_lion_lr${lr}_wd${wd}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lion_optm/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lion_optm runs complete ==="
