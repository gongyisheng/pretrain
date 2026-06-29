# MoE: Muon vs AdamW

Compare Muon ([Jordan et al.](https://kellerjordan.github.io/posts/muon/), [Moonshot scaling](https://arxiv.org/pdf/2502.16982)) against AdamW on the 183M/a51M MoE, holding everything except `optimizer.name` fixed.

Unlike [`muon_optm`](../muon_optm/) (dense, 2D weights only), this experiment exercises Muon on the **MoE expert layers**: the experts are stored as stacked 3D tensors `(E, out, in)`, which `MuonOptimizer` orthogonalizes per-expert via batched Newton–Schulz. Experts are 82% of all params (96% of the Muon-routed params), so this run is mostly a test of whether Muon helps where it previously could not reach.

## Hypothesis

Muon orthogonalizes each weight's momentum so every singular direction gets a comparable step. For MoE this is the regime [Moonlight/Kimi K2](https://arxiv.org/pdf/2502.16982) validated at scale — Muon on the experts beat AdamW at matched compute. The open risk is the expert gradients: each routed expert sees only its share of tokens, so its update matrix is sparser/noisier, and Newton–Schulz renormalizes singular directions that may be dominated by noise. `expert_capacity_factor=1.25` + `aux_loss_coef=0.001` keep per-expert token counts reasonably even, which should bound the worst case. Expectation: Muon matches or beats AdamW val loss at equal `lr`/`wd`/schedule.

## Setup

The `muon` run uses the hybrid `MuonAdamWOptimizer`: matrix hidden weights (attention/MLP projections **and** the 3D experts) → Muon; embeddings, `lm_head`, the **router gate**, and 1D params (RMSNorm scales) → AdamW. The two configs are identical except `optimizer.name`.

| Config | Optimizer | Total params | Active | Muon-routed | of which experts | LR | min_lr | WD | Eff. batch |
|---|---|---|---|---|---|---|---|---|---|
| qwen3_183m_a51m_adamw | AdamW | 183.3M | 51M | — | — | 1e-3 | 1e-4 | 0.1 | 256 |
| qwen3_183m_a51m_muon  | Muon  | 183.3M | 51M | 157.3M (86%) | 151.0M (82% of total) | 1e-3 | 1e-4 | 0.1 | 256 |

- Architecture: `d_model=512`, `n_layers=8`, GQA (8 heads / 4 kv, qk_norm), 64 routed experts, top-8, `intermediate_size=192`, `expert_capacity_factor=1.25`, `aux_loss_coef=0.001`.
- `lr=1e-3`/`min_lr=1e-4` matches the rest of the `moe_*` sweeps; Muon reuses it via `match_rms_adamw`.
- Muon hyperparams at defaults: `momentum=0.95`, `nesterov=true`, `ns_steps=5`, shared `eps=1e-8`.
- All runs: seq_len=1024, batch_size=16, grad_accum=16 (effective batch=256), 50K steps (~13.1B tokens), cosine schedule with 1500 warmup, bf16, OpenWebText.
- `eval_every=100`, `eval_steps=100`, `checkpoint_every=5000`, `log_every=10`.

## Run

```bash
nohup bash experiments/moe_muon_optm/run.sh > logs/moe_muon_optm.log 2>&1 &
```

## Results

| Optimizer | val loss | val PPL | Δ vs AdamW |
|---|---|---|---|
| AdamW | TBD | TBD | — |
| Muon  | TBD | TBD | TBD |

## Notes

- Muon stores one `momentum_buffer` per routed param (including the 3D expert stacks); the AdamW sub-optimizer keeps `exp_avg`+`exp_avg_sq` for the rest. `optim/momentum_norm`/`optim/variance_norm` in W&B reflect only the AdamW-routed params — Muon's buffer is not aggregated by `metric_utils`.
- The router gate is intentionally on AdamW: orthogonalizing a tiny `(n_experts, d_model)` classifier makes little sense.
- Watch `train/aux_loss` and per-expert balance — if Muon destabilizes routing (experts collapsing), the noisy-gradient risk is materializing; compare against the AdamW run's aux loss.
- If turnaround matters, compare the 5K/10K-step intermediate eval losses before committing both to 50K.
