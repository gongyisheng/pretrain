# MoE Experiments

## Scale-up to 0.5B

Single config scaling the MoE architecture to ~500M total params.

### Architecture

| Param | Value |
|-------|-------|
| `d_model` | 1024 |
| `n_layers` | 16 |
| `n_heads` | 16 |
| `n_kv_heads` | 8 |
| `moe_intermediate_size` (per expert) | 128 |
| `moe_n_experts` | 64 |
| `moe_n_experts_per_token` | 4 |
| `vocab_size` | 50257 |

Total params: ~506M (weight-tied). Active per token: ~128M.

Sparsity: 4/64 = 6.25% of experts active per token (matches Qwen3-0.6B-MoE ratio).

### Grid Search Configs

| Config | k | Active params | Sparsity |
|--------|---|---------------|----------|
| `qwen3_moe_506m_a109m` | 1 | ~109M | 1.6% |
| `qwen3_moe_506m_a115m` | 2 | ~115M | 3.1% |
| `qwen3_moe_506m_a128m` | 4 | ~128M | 6.3% |
| `qwen3_moe_506m_a153m` | 8 | ~153M | 12.5% |
| `qwen3_moe_506m_a204m` | 16 | ~204M | 25% |
| `qwen3_moe_506m_a304m` | 32 | ~304M | 50% |
| `qwen3_moe_506m_a506m` | 64 | ~506M | 100% (dense-equiv) |

### Running

```bash
# All configs sequentially:
nohup bash experiments/moe/qwen3_moe/run.sh > logs/moe_500m.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe/qwen3_moe_506m_a128m.yaml
```

> **Memory note:** Model params + optimizer ~6GB (BF16). If activations OOM on your GPU,
> add `--training.activation_checkpointing=true`.

## Aux Loss

MoE training uses Switch Transformer load-balancing loss:

```
L_total = L_CE + α * n_experts * Σ_i(f_i * P_i)
```

where `f_i` = fraction of tokens routed to expert `i` (hard), `P_i` = mean router softmax
probability for expert `i` (soft, carries gradient). `α` = `moe_aux_loss_coef` (default 0.01).
