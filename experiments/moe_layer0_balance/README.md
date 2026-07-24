# MoE Layer-0 Routing Balance

## Hypothesis

MoE routing imbalance tends to be worst at the shallowest layer: layer-0
hidden states are the least differentiated (closest to the raw token
embedding), so the router has the least signal to separate tokens into
experts, and a few experts absorb a disproportionate share of load. Once the
imbalance sets in early it can compound in later layers via load-balancing
bias/aux-loss dynamics. Two remedies:

- **l0dense**: replace layer 0's MoE with a dense SwiGLU MLP sized to
  match the MoE layers' active FFN width. No routing happens at layer 0 at
  all, so there's nothing to imbalance — trades this layer's conditional
  compute (a routing decision) for unconditional compute of the same size.
- **l0 aux loss** (`l0auxcoef*`): keep layer 0 as MoE but switch it from
  loss-free `expert_bias` to Switch-style auxiliary load-balancing loss
  (`aux_loss`), while layers 1-7 stay on loss-free `expert_bias` at 1e-3.
  Unlike the bias controller (a slow additive-bias feedback loop), the aux loss
  applies a *gradient* toward balance that backprops into the router and the
  layer-0 inputs (≈ the token embeddings), so it can reshape the embeddings to
  be more separable — targeting the root cause (poor differentiation) rather
  than just the symptom. Swept over layer-0 `aux_loss_coef` **1e-2 / 1e-1**.
  (This replaces the earlier `expert_bias_update_rate` boost, which prior
  experiments showed does not help — a faster bias controller can't fix layer 0
  if the states are too undifferentiated to separate.)

Both are compared against **`qwen3_188m_a51m`** (baseline): uniform MoE with the
same `expert_bias_update_rate` (1e-3) at every layer.

## Setup

Fixed testbed: `qwen3_188m_a51m` architecture (same as
`../moe_expert_bias/qwen3_188m_a51m_sigmoid_expert_bias_rate1e-3.yaml`) — 512
d_model, 8 layers, 64 routed experts + 2 always-on shared experts, top-6
routed, no expert capacity limit, `expert_bias: true` / `aux_loss: false`,
sigmoid router, muon lr 1e-3, warmup 1500. Active MoE FFN width
`(k + s)·intermediate_size = (6+2)·192 = 1536`; `l0dense`'s layer-0 dense
MLP uses `intermediate_size: 1536, activation: silu, gated: true` to match it
(SwiGLU, compute-comparable to a routed+shared MoE layer).

| Param | Value |
|-------|-------|
| d_model / n_layers | 512 / 8 |
| n_routed_experts / top-k | 64 / 6 |
| n_shared_experts | 2 |
| intermediate_size (per expert) | 192 |
| batch × grad_accum × seq | 64 × 4 × 1024 (≈0.26M tok/step) |
| max_steps | 50000 (≈13B tokens) |
| lr / min_lr / warmup | 1e-3 / 1e-4 / 1500 |

| Config | Layer 0 | Layers 1-7 | Layer 0 balancing | Total params | Active params |
|--------|---------|------------|--------------------|--------------|----------------|
| `qwen3_188m_a51m` (baseline) | MoE | MoE | expert_bias 1e-3 | 188.03M | 51.19M |
| `qwen3_171m_a51m_l0dense` | Dense SwiGLU (is=1536) | MoE | n/a (dense) | 170.89M | 51.16M |
| `qwen3_188m_a51m_l0auxcoef1e-2` | MoE | MoE | aux_loss 1e-2 | 188.03M | 51.19M |
| `qwen3_188m_a51m_l0auxcoef1e-1` | MoE | MoE | aux_loss 1e-1 | 188.03M | 51.19M |

Layers 1-7 always use loss-free `expert_bias` at 1e-3; only layer 0's balancing
strategy varies across arms.

Param counts computed via:
```bash
uv run python -c "from src.utils.config import load_config; from src.utils.metric_utils import count_parameters; c=load_config('experiments/moe_layer0_balance/qwen3_188m_a51m.yaml'); print(count_parameters(c))"
```
`{'total': 188032512, 'non_emb': 162276864, 'active_non_emb': 25437696}` for
the baseline and every `l0auxcoef*` arm (identical param shapes — only the
layer-0 balancing strategy/coef differs) and `{'total': 170894848, 'non_emb':
145139200, 'active_non_emb': 25404928}` for `l0dense`. `l0dense`'s total is lower
because a dense SwiGLU layer at `is=1536` has fewer parameters than a
64-routed-expert MoE layer at `is=192` (only the top-6 + 2 shared experts are
ever active per token, but all 64 routed experts' weights are resident).
Active params (embeddings + active_non_emb) are ~51.2M / ~51.16M across all
arms, so they are compute-comparable per forward pass.

## Run

```bash
# l0dense arm (baseline + l0dense):
nohup bash experiments/moe_layer0_balance/run_l0dense.sh > logs/moe_layer0_balance_l0dense.log 2>&1 &
# l0 aux-loss sweep (layer-0 aux_loss_coef 1e-2 / 1e-1):
nohup bash experiments/moe_layer0_balance/run_l0auxloss.sh > logs/moe_layer0_balance_l0auxloss.log 2>&1 &
# single:
uv run python scripts/train.py --config experiments/moe_layer0_balance/qwen3_188m_a51m.yaml
```

## Results

| arm | val_loss | layer-0 balance (MaxVio) | notes |
|-----|----------|--------------------------|-------|
| `qwen3_188m_a51m` (baseline) | | | |
| `qwen3_171m_a51m_l0dense` | | | |
| `qwen3_188m_a51m_l0auxcoef1e-2` | | | |
| `qwen3_188m_a51m_l0auxcoef1e-1` | | | |

## Notes

- Layer-0 balance is read from the existing per-layer MoE MaxVio metrics
  (`src/utils/metric_utils.py::compute_moe_maxvio` /
  `compute_moe_global_maxvio`, logged per layer during training) — compare
  `layer_0` MaxVio across the baseline and the `l0auxcoef*` arms directly, and
  look for a monotonic trend with the layer-0 `aux_loss_coef`.
  `l0dense` has no layer-0 MoE metric (layer 0 is dense), so its "balance"
  is by construction 0 there; compare its layers 1-7 MaxVio against the
  corresponding layers in the baseline to see whether removing layer-0 routing
  changes downstream balance.
- Compare `val_loss` across all arms to see whether either remedy costs (or
  gains) quality relative to the baseline, since both trade off something:
  `l0dense` removes a routing decision (fixed compute, no adaptivity), the
  `l0auxcoef*` arms add a gradient-based balance term at layer 0 that perturbs
  the LM objective there (too high a coef can cost quality; too low won't
  balance) — no compute change.
- Same testbed as `../moe_expert_bias/`, so the baseline here is expected to
  reproduce `qwen3_188m_a51m_sigmoid_expert_bias_rate1e-3` if the two ever
  need to be cross-checked (config-diff, not just result-diff).
