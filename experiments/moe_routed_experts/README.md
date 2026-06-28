# MoE Routed-Experts Experiments

Top-k routing grid search: fix total params (64 experts, per-expert
`intermediate_size=128`), sweep `n_routed_experts_per_token` (k) from 1 to 64.
Only active params/token change; total stays constant. Run at two model sizes.

### Architecture

| Param | 133M | 506M |
|-------|------|------|
| `d_model` | 512 | 1024 |
| `n_layers` | 8 | 16 |
| `n_heads` | 8 | 16 |
| `n_kv_heads` | 4 | 8 |
| `moe_intermediate_size` (per expert) | 128 | 128 |
| `moe_n_routed_experts` | 64 | 64 |
| `vocab_size` | 50257 | 50257 |

Weight-tied. k=2 (a35m/a115m) ≈ 3.1% sparsity, matching the Qwen3-0.6B-MoE ratio.

### Grid Search Configs

133M total:

| Config | k | Active params | Sparsity |
|--------|---|---------------|----------|
| `qwen3_133m_a34m` | 1 | ~34M | 1.6% |
| `qwen3_133m_a35m` | 2 | ~35M | 3.1% |
| `qwen3_133m_a39m` | 4 | ~39M | 6.3% |
| `qwen3_133m_a45m` | 8 | ~45M | 12.5% |
| `qwen3_133m_a57m` | 16 | ~57M | 25% |
| `qwen3_133m_a83m` | 32 | ~83M | 50% |
| `qwen3_133m_a133m` | 64 | ~133M | 100% (dense-equiv) |

506M total:

| Config | k | Active params | Sparsity |
|--------|---|---------------|----------|
| `qwen3_506m_a109m` | 1 | ~109M | 1.6% |
| `qwen3_506m_a115m` | 2 | ~115M | 3.1% |
| `qwen3_506m_a128m` | 4 | ~128M | 6.3% |
| `qwen3_506m_a153m` | 8 | ~153M | 12.5% |
| `qwen3_506m_a204m` | 16 | ~204M | 25% |
| `qwen3_506m_a304m` | 32 | ~304M | 50% |
| `qwen3_506m_a506m` | 64 | ~506M | 100% (dense-equiv) |

### Running

```bash
# Each grid sequentially (lowest k first):
nohup bash experiments/moe_routed_experts/run_133m.sh > logs/moe_routed_133m.log 2>&1 &
nohup bash experiments/moe_routed_experts/run_506m.sh > logs/moe_routed_506m.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_routed_experts/qwen3_133m_a39m.yaml
```

> **Memory note:** 506M model params + optimizer ~6GB (BF16). If activations OOM on your GPU,
> add `--training.activation_checkpointing=true`.

## Aux Loss

MoE training uses Switch Transformer load-balancing loss:

```
L_total = L_CE + α * n_routed_experts * Σ_i(f_i * P_i)
```

where `f_i` = fraction of tokens routed to expert `i` (hard), `P_i` = mean router softmax
probability for expert `i` (soft, carries gradient). `α` = `moe_aux_loss_coef` (default 0.01).
