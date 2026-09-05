# MoE Ultra Sparsity

## Hypothesis

[`moe_sparsity`](../moe_sparsity/) grew the routed pool to `E=256` (2.3% routed sparsity) at a
fixed active budget. This experiment pushes the same axis into the **ultra-sparse** regime —
`E ∈ {384, 512, 768, 1024}`, down to 0.59% routed sparsity — and asks two questions at once:

1. **Where does pool scaling stop paying?** Active capacity `(s+k)·is = 1536` (~51M active
   params) is constant, so FLOPs/token never change; only total params grow, from 0.94B to
   2.46B. If val loss keeps falling, more total capacity is still exploitable by top-6 routing.
   If it flattens (or regresses), the router — a single `(E, d_model)` linear over up to 1024
   classes trained on `k/E` positive signal per token — has become the bottleneck, not capacity.
2. **Does the optimizer change the answer?** Experts are 96-99% of params here, and each routed
   expert sees only ~`k/E` of the tokens, so its gradient is sparse and noisy — the exact regime
   where [`moe_muon_optm`](../moe_muon_optm/) flagged Newton–Schulz as risky. Muon may extend the
   useful pool size (better-conditioned updates for rarely-hit experts) or break down first
   (orthogonalizing near-noise). Each `E` is therefore run under both AdamW and Muon.

## Setup

Common backbone — identical to `moe_sparsity`, so those runs serve as the low-`E` end of this
curve. Only `n_routed_experts` and `optimizer.optimizer_cls` vary.

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 192 |
| `n_shared_experts` (s) | 2 |
| `n_routed_experts_per_token` (k) | 6 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

Invariant: active experts per token `s + k = 8`, active capacity `(s+k)·is = 1536` (~51M active
params) in every cell. Load balancing is aux-loss-free (`expert_bias: true`,
`expert_bias_update_rate=0.001`, `router_score_fn: sigmoid`); no expert capacity limit, so no
token drops. `is` = per-expert `intermediate_size`.

### Configs

Config filename = ckpt dir = W&B run name. Eight runs = 4 pool sizes × 2 optimizers.

| E (pool) | Pool (E·is) | Total | Active | Sparsity (k/E) | AdamW config | Muon config |
|:--------:|:-----------:|:-----:|:------:|:--------------:|---|---|
|  384 |  73728 | 0.94B | 51M | 1.56%  | `qwen3_944m_a51m_is192_e384_s2r6_adamw`   | `..._muon` |
|  512 |  98304 | 1.25B | 51M | 1.17%  | `qwen3_1247m_a51m_is192_e512_s2r6_adamw`  | `..._muon` |
|  768 | 147456 | 1.85B | 51M | 0.78%  | `qwen3_1852m_a51m_is192_e768_s2r6_adamw`  | `..._muon` |
| 1024 | 196608 | 2.46B | 51M | 0.59%  | `qwen3_2457m_a51m_is192_e1024_s2r6_adamw` | `..._muon` |

Reference points from [`moe_sparsity`](../moe_sparsity/) (Muon only, same backbone):
`E=64` → 188M (the `qwen3_188m_a51m` benchmark), `E=128` → 339M, `E=256` → 642M.

Training (all runs): batch 64 × grad-accum 4 × seq 1024 = 0.26M tokens/step (effective batch
256, matching `moe_sparsity`), `max_steps` 50000 (~13.1B tokens), cosine LR 1e-3 → 1e-4, warmup
1500, `grad_clip` 1.0, bf16, OpenWebText. `eval_every=100`, `eval_steps=100`, `eval_batch_size=64`,
`checkpoint_every=5000`, `log_every=100`. Muon runs use the hybrid `MuonAdamWOptimizer`
(`momentum=0.95`, `nesterov`, `match_rms_adamw`): 2D/3D hidden weights incl. the stacked expert
tensors → Muon; embeddings, `lm_head`, router gate, and 1D norm scales → AdamW.

## Running

Each arm is its own script (cheapest pool first), so the two optimizers can run on separate GPUs
or be scheduled independently:

```bash
nohup bash experiments/moe_ultra_sparse/run_muon.sh  > logs/moe_ultra_sparse_muon.log 2>&1 &
nohup bash experiments/moe_ultra_sparse/run_adamw.sh > logs/moe_ultra_sparse_adamw.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_ultra_sparse/qwen3_944m_a51m_is192_e384_s2r6_muon.yaml
```

Training is single-device; pin each arm with `CUDA_VISIBLE_DEVICES=<idx>` if running both at
once. Note the memory table below — the two arms do **not** fit concurrently on one GPU past
`E=512`.

W&B project: `pretrain-moe-ultra-sparse`.

## Results

| E (pool) | Total | Active | AdamW val loss | Muon val loss | Δ (Muon − AdamW) |
|:--------:|:-----:|:------:|:--------------:|:-------------:|:----------------:|
|  384 | 0.94B | 51M | | | |
|  512 | 1.25B | 51M | | | |
|  768 | 1.85B | 51M | | | |
| 1024 | 2.46B | 51M | | | |

## Notes

- **Memory is the binding constraint.** Weights and gradients are fp32 (bf16 autocast), so
  resident state before activations is `4·N` bytes for AdamW (params + grads + `exp_avg` +
  `exp_avg_sq`) and `3·N` for Muon (params + grads + one momentum buffer):

  | E | Total params | AdamW state | Muon state |
  |:-:|:---:|:---:|:---:|
  |  384 | 944M  | 14.1 GiB | 10.6 GiB |
  |  512 | 1.25B | 18.6 GiB | 13.9 GiB |
  |  768 | 1.85B | 27.6 GiB | 20.7 GiB |
  | 1024 | 2.46B | 36.6 GiB | 27.5 GiB |

  On a 45 GiB L40S the `E=1024` **AdamW** cell leaves only ~8 GiB for activations, compile
  workspace, and MoE dispatch buffers — and at `batch_size 64` activations are 4× what they'd be
  at batch 16, so the larger-`E` AdamW cells are expected to be tight. If a cell OOMs, halve
  `batch_size` and double `gradient_accumulation_steps` (effective batch is preserved), or enable
  activation checkpointing; drop the cell rather than changing the effective batch.
- Active params (~51M) and FLOPs/token are constant across all 8 runs. Step time still grows with
  `E`: the router softmax/sigmoid is over `E` classes and the gather/scatter touches a larger
  expert table, so wall-clock is not flat even though FLOPs/token is.
- **Router saturation is the thing to watch.** With `k/E < 1%`, each expert receives ~0.6% of
  tokens per step — roughly 400 tokens per expert per micro-batch at batch 64 × seq 1024. Track
  per-expert load and the `expert_bias` drift: if a large fraction of experts are effectively
  never selected, the pool is dead weight and the val-loss curve should flatten accordingly.
- The Muon arm is a direct scale-up of [`moe_muon_optm`](../moe_muon_optm/)'s open question. That
  experiment ran `E=64` with a capacity factor and aux loss; here routing is aux-loss-free and the
  pool is 6-16× larger, so per-expert gradients are correspondingly sparser. A Muon advantage that
  shrinks or inverts as `E` grows is the signal that Newton–Schulz is amplifying noise.
- Comparability with `moe_sparsity`: same backbone, same effective batch (256), token budget,
  micro-batch shape (`64 × 4`), and eval protocol (`eval_steps 100`, `eval_batch_size 64`), so val
  loss is directly comparable across the two sweeps.
