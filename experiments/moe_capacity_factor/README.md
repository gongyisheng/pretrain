# MoE Expert Capacity Factor

## Hypothesis

How much does **fixed expert capacity** cost in quality, and how much does it buy in speed?

The MoE block dispatches routed tokens into a padded `(E, capacity, D)` buffer and runs one
batched GEMM over all experts. Two ways to size `capacity`:

- **Dynamic (no capacity factor):** `capacity = max expert load` each step → no token is
  dropped, but the buffer is sized to the *busiest* expert, so every expert pays for the peak
  (wasted compute on padding when load is imbalanced).
- **Fixed (`expert_capacity_factor = f`):** `capacity = floor(T·k·f/E)`, constant across
  steps. Cheaper and stable, but any expert receiving more than `capacity` tokens **drops**
  the overflow (those tokens skip the MLP, keeping only the residual).

Lower `f` → less compute and memory, but more dropped tokens → potentially worse loss. This
sweep maps that tradeoff on the standard `qwen3_183m_a51m` benchmark.

## Setup

Common backbone (Qwen3-style, identical across all runs except `expert_capacity_factor`):

| Param | Value |
|-------|-------|
| `d_model` | 512 |
| `n_layers` | 8 |
| `n_heads` / `n_kv_heads` | 8 / 4 (`qk_norm: true`) |
| `intermediate_size` (per expert) | 192 |
| `n_routed_experts` (E) | 64 |
| `n_routed_experts_per_token` (k) | 8 |
| `aux_loss_coef` | 0.001 |
| `vocab_size` | 50257 |
| norm / pos_emb | rmsnorm / rope (θ=10000) |

The only knob is `expert_capacity_factor`. Total 183M / active 51M params, sparsity
`k/E = 12.5%` — identical across cells. At training batch (64 × seq 1024 = 65536 tokens),
the balanced per-expert load is `T·k/E = 8192`, so fixed capacity `= floor(8192·f)`.

### Configs

Config filename = ckpt dir = W&B run name.

| Config | `expert_capacity_factor` | Capacity (T=65536) | Behavior |
|--------|:------------------------:|:------------------:|----------|
| `qwen3_183m_a51m_capf_none` | none (dynamic) | = max load | no drops; sized to peak |
| `qwen3_183m_a51m_capf_1.0`  | 1.0  | 8192  | drops above balanced load |
| `qwen3_183m_a51m_capf_1.25` | 1.25 | 10240 | codebase default |
| `qwen3_183m_a51m_capf_1.5`  | 1.5  | 12288 | |
| `qwen3_183m_a51m_capf_2.0`  | 2.0  | 16384 | |

Training (all runs): batch 64 × grad-accum 4 × seq 1024 ≈ 0.26M tokens/step, `max_steps`
50000 (~13B tokens), Muon lr 1e-3 → 1e-4, warmup 1500, bf16.

## Running

```bash
# All configs sequentially:
nohup bash experiments/moe_capacity_factor/run.sh > logs/moe_capacity_factor.log 2>&1 &

# Single config:
uv run python scripts/train.py --config experiments/moe_capacity_factor/qwen3_183m_a51m_capf_1.25.yaml
```

W&B project: `pretrain-moe-capacity-factor`.

## Results

| Config | `f` | tokens/sec | MaxVio | est. drop % | Final val loss | Final PPL |
|--------|:---:|:----------:|:------:|:-----------:|:--------------:|:---------:|
| `qwen3_183m_a51m_capf_none` | none | | | 0 | | |
| `qwen3_183m_a51m_capf_1.0`  | 1.0  | | | | | |
| `qwen3_183m_a51m_capf_1.25` | 1.25 | | | | | |
| `qwen3_183m_a51m_capf_1.5`  | 1.5  | | | | | |
| `qwen3_183m_a51m_capf_2.0`  | 2.0  | | | | | |

- tokens/sec = `perf/tokens_per_sec`; val loss/PPL from the final eval.
- **MaxVio** (`val/moe_maxvio`) = max-violation load imbalance — the routing-balance signal.
- **est. drop %** is *not* logged directly. `expert_load` records pre-drop per-expert counts,
  so the drop fraction = `sum(max(load − capacity, 0)) / (T·k)`, where `capacity = floor(2048·f)`.
  Derive it offline from the logged loads, or add a `drop_rate` metric in `metrics.py`
  (small follow-up). `capf_none` drops nothing by construction.

## Notes

- Preliminary throughput microbenchmark (fwd+bwd, batch 4, untrained, RTX 5060 Ti):
  `none` ≈ 208 ms/step, `1.25` ≈ 142 ms/step (1.47×), `1.0` ≈ 133 ms/step; dense `qwen3_51m`
  ≈ 63 ms/step. So fixed capacity recovers a large slice of the gap, but ~2× over dense is
  intrinsic to fine-grained MoE (many small expert GEMMs + routing) and is *not* tunable here.
- Drop rate depends on routing balance, which the `aux_loss` term improves over training:
  imbalance (`val/moe_maxvio`, and thus drops / dynamic capacity) is worst early and falls as
  load equalizes. Untrained worst-case drops at this E=64 setting were high (~38% at `f=1.25`),
  so the trained-equilibrium drop rate is the number that matters — derive it from the logged
  `expert_load`, not the init.
- Expect the curve: as `f` rises, tokens/sec falls and val loss improves, converging toward
  `capf_none` (no drops). The question is where the knee is — i.e. the smallest `f` whose
  loss matches `none` while still being meaningfully faster.
