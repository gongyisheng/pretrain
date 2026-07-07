#!/bin/bash
# MoE router score-fn comparison: softmax vs sigmoid gating, swept over aux_loss_coef.
# The two score fns feed different-magnitude scores into the Switch aux-loss formula,
# so the balancing coef must be swept per score fn rather than matched. coef=0 keeps
# aux_loss on (MaxVio still logged) but applies no balancing pressure.
# Usage: nohup bash experiments/moe_router_score_fn/run.sh > logs/moe_router_score_fn.log 2>&1 &

set -e
cd "$(dirname "$0")/../.."

score_fns=(softmax sigmoid)
coefs=(1e-2 3e-3 1e-3 1e-4 3e-5 0)

configs=()
for fn in "${score_fns[@]}"; do
    for coef in "${coefs[@]}"; do
        configs+=("qwen3_183m_a51m_${fn}_aux${coef}")
    done
done

for config in "${configs[@]}"; do
    echo "=== ${config} ==="
    echo "Started at: $(date)"
    uv run python scripts/train.py --config "experiments/moe_router_score_fn/${config}.yaml"
    echo "Finished at: $(date)"
    echo ""
done

echo "=== All moe_router_score_fn runs complete ==="
