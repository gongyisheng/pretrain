#!/bin/bash
# Run all lion_optm experiments sequentially (AdamW baseline first, then Lion grid)
# Usage: nohup bash experiments/lion_optm/run.sh > logs/lion_optm.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

configs=(
    "qwen3_57m_adamw_lr5e-4_wd0.1"
    "qwen3_57m_lion_lr5e-5_wd0.1"
    "qwen3_57m_lion_lr5e-5_wd0.2"
    "qwen3_57m_lion_lr5e-5_wd0.5"
    "qwen3_57m_lion_lr5e-5_wd1.0"
    "qwen3_57m_lion_lr1e-4_wd0.1"
    "qwen3_57m_lion_lr1e-4_wd0.2"
    "qwen3_57m_lion_lr1e-4_wd0.5"
    "qwen3_57m_lion_lr1e-4_wd1.0"
    "qwen3_57m_lion_lr2e-4_wd0.1"
    "qwen3_57m_lion_lr2e-4_wd0.2"
    "qwen3_57m_lion_lr2e-4_wd0.5"
    "qwen3_57m_lion_lr2e-4_wd1.0"
)

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/lion_optm/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All lion_optm runs complete ==="
