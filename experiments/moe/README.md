# MoE Experiments

Grid search over `n_experts_per_token` (top-k) for Qwen3 MoE architecture.

## Base Architecture

| Param | Value |
|-------|-------|
| `d_model` | 384 |
| `n_layers` | 8 |
| `n_heads` | 6 |
| `n_kv_heads` | 3 |
| `intermediate_size` (per expert) | 512 |
| `n_experts` | 8 |
| `vocab_size` | 50257 |

Total params: ~61M. Active params scale with `n_experts_per_token`.

## Grid Search Configs

| Config | k | Active params |
|--------|---|---------------|
| `qwen3_moe_61m_a28m` | 1 | ~28M |
| `qwen3_moe_61m_a32m` | 2 | ~32M |
| `qwen3_moe_61m_a42m` | 4 | ~42M |
| `qwen3_moe_61m_a61m` | 8 | ~61M (dense-equiv) |

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe/qwen3_moe/run.sh > logs/moe_grid.log 2>&1 &

# Single config:
python scripts/train.py --config experiments/moe/qwen3_moe/qwen3_moe_61m_a32m.yaml
```

## Aux Loss

MoE training uses Switch Transformer load-balancing loss:

```
L_total = L_CE + α * n_experts * Σ_i(f_i * P_i)
```

where `f_i` = fraction of tokens routed to expert `i` (hard), `P_i` = mean router softmax
probability for expert `i` (soft, carries gradient). `α` = `moe_aux_loss_coef` (default 0.01).
