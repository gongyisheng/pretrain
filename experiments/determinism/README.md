# Determinism

Test whether enabling deterministic CUDA algorithms affects training loss or convergence, and whether two identical runs produce bit-identical results with determinism on vs. divergent results with it off.

## Hypothesis

With `use_deterministic_algo=true`, two runs with the same seed should produce identical loss curves. With it off, cuBLAS non-determinism introduces per-step floating-point differences that compound through backprop, causing loss curves to diverge over thousands of steps — even with identical seeds, data order, and hyperparameters.

## Setup

Fixed: Qwen3 57M architecture, `seq_len=1024`, `lr=6e-4`, 5K steps, `seed=42`, `batch_size=8`, `grad_accum=32`.

| Config | use_deterministic_algo |
|---|---|
| qwen3_57m_deterministic | true |
| qwen3_57m_nondeterministic | false |

To test reproducibility, run each config twice and compare whether the loss curves are identical across runs.

## Run

```bash
nohup bash experiments/determinism/run.sh > logs/determinism.log 2>&1 &
```

Or a single run:

```bash
uv run python scripts/train.py --config experiments/determinism/qwen3_57m_deterministic.yaml
```

## W&B

Project: `pretrain-determinism`. Compare runs by `val/loss` vs `train/step`.

## Results

TODO

## Notes

### cuBLAS workspace configuration

cuBLAS GEMMs use a scratch **workspace buffer** for parallel partial-sum reductions. By default, cuBLAS dynamically allocates a few MB, allowing many parallel threads to write partial sums — but the accumulation order depends on thread scheduling, which varies run-to-run. Because float addition is non-associative (`(a+b)+c ≠ a+(b+c)`), different orders produce different last-bit results.

`CUBLAS_WORKSPACE_CONFIG=:4096:8` constrains the workspace to 8 x 4096-byte buffers (32KB), forcing a single algorithm with a fixed reduction tree — guaranteeing bit-identical output. These per-GEMM differences are tiny (~1 ULP), but compound through hundreds of layers and thousands of backprop steps, causing "identical" runs to diverge. Combined with `torch.use_deterministic_algorithms(True)`, this covers cuDNN and other CUDA ops as well.

The env var is per-process — concurrent experiments each get their own 32KB workspace with no interference. Expect ~1-5% throughput regression from constrained algorithm selection. `:4096:8` is preferred over `:16:8` as the larger buffer allows better-performing deterministic kernels.

### Sources of non-determinism in PyTorch

`torch.use_deterministic_algorithms(True)` addresses two root causes:

1. **cuBLAS parallel reductions** — variable thread ordering in GEMM ops (every linear layer). Fixed by the workspace config above.
2. **atomicAdd races** — backward passes of scatter/index ops where multiple gradients accumulate to the same index. Fixed by serialized fallback kernels. Affected ops: `cross_entropy`/`nll_loss` backward, `Embedding` backward, `scatter_add`, `index_add`, `index_put(accumulate=True)`, `index_select` backward, `repeat_interleave` backward, `Conv1d/2d/3d` (cuDNN), `ctc_loss`, `interpolate`, `grid_sample`.

For transformers, the ops that matter: all linear layers (cuBLAS), embedding backward (scatter_add), and cross_entropy backward (atomicAdd). Convolution/pooling non-determinism (cuDNN auto-tuning) is irrelevant here.
