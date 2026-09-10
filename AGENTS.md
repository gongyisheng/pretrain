# AGENTS.md

## Project

Single-GPU LLM pretraining in PyTorch, configured with YAML and logged to W&B. See [README.md](README.md) for setup, data preparation, and training commands.

`build_model` in `src/model/__init__.py` constructs `TransformerLM` from registered components in `src/layers/`. The trainer compiles the whole model when enabled; keep layer operations as plain functions, without individual compile decorators.

## Development

- Code will be read and reviewed by humans, so prioritize readability with clear names and straightforward logic.
- Run relevant tests before and after layer/model changes. Benchmark performance-sensitive changes with `benchmarks/bench_train.py` before and after.
- Before GPU tests, training, or benchmarks, run `nvidia-smi` and pin a free device with `CUDA_VISIBLE_DEVICES=<idx>`. Do not assume GPU count or VRAM; training uses one device.
- Keep `docs/superpowers/` ignored and uncommitted. If files there are tracked, untrack them while preserving local copies.

```bash
uv sync
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Tests

Run fast and e2e trees separately. Always specify `-n`, including for subsets and single files: use `-n 6` by default, `-n 12 --dist load` for `kernel`, `quant`, and `metrics`, and `-n 0` only for e2e or debugging.

```bash
uv run pytest tests/fast -n 6
uv run pytest tests/fast/kernel tests/fast/quant tests/fast/metrics -n 12 --dist load
uv run pytest tests/e2e -n 0
```

Keep test files API-focused and test only public APIs. Merge cases for the same API into one parameterized test and fold related properties into it rather than adding near-duplicates. Name general and error tests `test_<api>` and `test_<api>_raise_error`; name class/property variants `test_<class>_<api>_<property>` and `test_<class>_<api>_<property>_raise_error`; use `test_<api>_precision` for oracle checks.

For kernel tests, put numerical precision and oracle tests in each backend's test module. Put validation and shape tests in the matching ops test module, and exercise them only through the ops API.

Do not add tests for specific kernels' backend choices. Keep backend-selection tests focused on the generic selector API.

Error tests should assert the exception type only; do not match or assert error messages.

For readability, define each quantization dtype constant as a self-contained dictionary literal with explicit `weight`, `act`, and `grad_out` values, in that order. Do not derive these constants from recipes, other constants, dictionary unpacking, or helper calls.

Build reusable, independent parameter axes with one argument per `pytest.mark.parametrize` decorator and explicit module-level case lists from product constants. Case lists must contain plain data, never `pytest.param` entries (including marks or IDs); apply marks and skips at the parametrization or test level; never specify explicit pytest IDs (`id=` or `ids=`). Use tuples only for bound expectations or illegal Cartesian combinations; skip invalid cells with the reason. Name quantization cases by per-tensor storage width, such as `w8a8`, `w16a8`, and `g8` (int4–int8 weights are `w8`).

Derive numeric tolerances from the worst case across the full grid, with a non-round 3–10× margin. Prefer `atol` with `rtol=0` for magnitude-independent error, match the oracle to the stored dtype, and use exact comparison when both paths run the same operations. Assert effects directly, including what changes and what remains bit-identical.

## Configuration and components

`src/utils/config.py` owns config defaults and validation: `ModelConfig.__post_init__` fills resolved kwargs and rejects invalid combinations. Components receive explicit resolved values. Norm, positional embedding, and residual use `*_cls` plus `*_kwargs`; attention and MLP are per-layer `{*_cls, *_kwargs, layer_idx?}` lists. One unscoped entry applies to every unclaimed layer; scoped entries override named layers, and at most one unscoped fallback is allowed. `resolve_attn(i)` and `resolve_mlp(i)` are the source of per-layer resolution.

Keep canonical `model:` key order: `d_model`, `n_layers`, `vocab_size`, `attn`, `mlp`, then remaining keys. All rope-bearing layers must share a rope head dimension, and all layers must use the same `attn_implementation`. Add a component in its owning layer file and register it in the corresponding registry.

Fused operations must support float32, float16, and bfloat16. Preserve the caller's dtype throughout; only use an explicitly documented accumulation dtype (such as float32 for reductions).

## Experiments

Each `experiments/` folder is self-contained and must include a `README.md` covering the hypothesis, setup table (configs, key parameters, approximate parameter count), run command, results table, and notes. Experiment YAML must explicitly set `batch_size: 16`, `gradient_accumulation_steps: 16`, `checkpoint_every: 5000`, `eval_every: 100`, and `eval_steps: 100` unless intentionally changed. Build sweep config lists in `run.sh` with nested loops over swept axes.
