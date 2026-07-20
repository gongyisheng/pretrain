# LatentMoE

LatentMoE (arXiv:2601.18089): routed experts run in a compressed latent space ℓ. A
shared down-projection `d→ℓ` runs once per token before dispatch, experts compute in
ℓ (intermediate width `m` unchanged), and a shared up-projection `ℓ→d` runs after
aggregation. The router scores the original token at `d`, and the shared experts stay
at `d`. Compression factor α = d/ℓ. See `src/layers/mlp.py` (`SparseMoEBlock`,
`latent_moe`/`latent_dim`).

## Hypothesis

1. **Compression alone hurts** (`run_base`). Shrinking the routed-expert I/O rank
   `d→ℓ` at a fixed pool (E=64, k=6) loses capacity, so val loss rises as ℓ falls.
2. **Reinvesting the savings recovers, then beats the benchmark** (`run_eff`,
   `run_acc`). Spending the α = d/ℓ savings on a larger routed pool along the
   α-matched diagonal (N′ = αN from the E=64 base) restores combinatorial capacity.
   The eff variant keeps top-k fixed; the acc variant additionally scales active
   experts (K′ = αK), the paper's recommended setting, and should match or beat the
   standard-MoE benchmark on accuracy per active FLOP/param.

All runs share the `qwen3_188m_a51m` recipe (d=512, n_layers=8, GQA 8/4 + qk_norm,
`intermediate_size=192`, 2 shared experts, sigmoid router, aux-loss-free `expert_bias`
balancing) and identical training (Muon lr 1e-3, min_lr 1e-4, warmup 1500, 50k steps,
bf16, effective batch 256). Only `latent_moe`/`latent_dim`, `n_routed_experts`, and
(acc only) `n_routed_experts_per_token` differ, so differences are attributable to
architecture at a fixed token budget.

## Setup

`total` / `active` are analytic parameter counts (active includes the always-on token
embedding, matching the `aXXm` naming). Config name:
`qwen3_<total>m_a<active>m[_l<ℓ>]_e<E>[_k<k>]` (baseline omits `l`; eff omits `k`
since top-k = 6 = benchmark).

### run_base — benchmark + compression only (E=64, k=6)

| config | ℓ | α | E | top-k | total | active |
|---|---|---|---|---|---|---|
| `qwen3_188m_a51m_e64` (benchmark) | — | 1 | 64 | 6 | 188M | 51M |
| `qwen3_115m_a46m_l256_e64` | 256 | 2 | 64 | 6 | 115M | 46M |
| `qwen3_76m_a42m_l128_e64` | 128 | 4 | 64 | 6 | 76M | 42M |
| `qwen3_56m_a39m_l64_e64` | 64 | 8 | 64 | 6 | 56M | 39M |

### run_eff — ℓ-MoE_eff, N′ = αN, top-k fixed

| config | ℓ | α | E | top-k | total | active |
|---|---|---|---|---|---|---|
| `qwen3_190m_a46m_l256_e128` | 256 | 2 | 128 | 6 | 190M | 46M |
| `qwen3_190m_a42m_l128_e256` | 128 | 4 | 256 | 6 | 190M | 42M |
| `qwen3_190m_a41m_l64_e512` | 64 | 8 | 512 | 6 | 190M | 41M |

### run_acc — ℓ-MoE_acc, N′ = αN and K′ = αK

| config | ℓ | α | E | top-k | total | active |
|---|---|---|---|---|---|---|
| `qwen3_190m_a54m_l256_e128_k12` | 256 | 2 | 128 | 12 | 190M | 54M |
| `qwen3_190m_a53m_l128_e256_k24` | 128 | 4 | 256 | 24 | 190M | 53M |
| `qwen3_190m_a54m_l64_e512_k48` | 64 | 8 | 512 | 48 | 190M | 54M |

The three reinvest points share ~190M total (the pool grows as ℓ shrinks, keeping
stored expert params roughly constant); eff holds active near the compressed base,
acc lifts active back to ≈ the 51M benchmark by scaling top-k.

## Run

```bash
# all three groups
nohup bash experiments/latent_moe/run.sh all > logs/latent_moe.log 2>&1 &

# one group at a time
bash experiments/latent_moe/run.sh base   # or: eff | acc

# a single config
uv run python scripts/train.py --config experiments/latent_moe/qwen3_190m_a54m_l256_e128_k12.yaml
```

## Results

Final val loss / perplexity after 50k steps (fill in after running).

| group | config | ℓ | E | top-k | active | val loss | val ppl |
|---|---|---|---|---|---|---|---|
| base | `qwen3_188m_a51m_e64` (benchmark) | — | 64 | 6 | 51M | | |
| base | `qwen3_115m_a46m_l256_e64` | 256 | 64 | 6 | 46M | | |
| base | `qwen3_76m_a42m_l128_e64` | 128 | 64 | 6 | 42M | | |
| base | `qwen3_56m_a39m_l64_e64` | 64 | 64 | 6 | 39M | | |
| eff | `qwen3_190m_a46m_l256_e128` | 256 | 128 | 6 | 46M | | |
| eff | `qwen3_190m_a42m_l128_e256` | 128 | 256 | 6 | 42M | | |
| eff | `qwen3_190m_a41m_l64_e512` | 64 | 512 | 6 | 41M | | |
| acc | `qwen3_190m_a54m_l256_e128_k12` | 256 | 128 | 12 | 54M | | |
| acc | `qwen3_190m_a53m_l128_e256_k24` | 128 | 256 | 24 | 53M | | |
| acc | `qwen3_190m_a54m_l64_e512_k48` | 64 | 512 | 48 | 54M | | |

## Notes

- **Fixed-compute framing.** All runs use the same 50k-step token budget, so this
  compares architectures at equal training compute, not compute-optimal per size.
- **eff vs acc.** eff (top-k fixed) reinvests only into the pool, so active cost tracks
  the compressed base; acc (K′ = αK) reinvests into active experts too, restoring
  active params to ≈ the benchmark — the fairer accuracy-per-active-FLOP comparison.
- **Memory.** The acc runs with large top-k (k=48) dispatch many more tokens through
  the grouped GEMM. If a run OOMs, set `training.activation_checkpointing: true`
  and/or lower `batch_size` (raise `gradient_accumulation_steps` to keep effective
  batch 256).
