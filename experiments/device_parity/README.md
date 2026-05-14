# Device Parity

Train a fixed Qwen3 57M config from scratch on different GPUs and compare final loss to check that results are reproducible across devices.

## Hypothesis

With the same seed, data, and hyperparameters, training on different GPUs should converge to a similar final val loss. `use_deterministic_algo` is left off, so within-device runs are not bit-identical — the goal is to confirm that cross-device deltas stay within the same band as run-to-run noise on a single device, not to assert equality.

## Setup

Fixed: Qwen3 57M, `seq_len=1024`, `lr=6e-4`, `seed=42`, `batch_size=16`, `grad_accum=16`, 50K steps, `bf16`, `backend=torch`.

| Config | use_deterministic_algo | max_steps |
|---|---|---|
| qwen3_57m | false | 50000 |

## Run

```bash
nohup bash experiments/device_parity/run.sh > logs/device_parity.log 2>&1 &
```

`run.sh` reads the GPU model via `nvidia-smi`, trains, and writes `results/<gpu_slug>.json` with the final `val_loss`, `val_ppl`, wallclock, hostname, and git commit. Run it on each device you want to compare; one JSON per device accumulates in `results/`.

## W&B

Project: `pretrain-device-parity`. Compare runs by `val/loss` vs `train/step`.

## Results

| GPU | val_loss | val_ppl | wallclock (s) | commit |
|---|---|---|---|---|
| TODO | | | | |

## Notes

- The trainer logs `[eval] val_loss=X | val_ppl=Y` to stdout at every eval; `run.sh` parses the last such line. If the training run is interrupted, the JSON will reflect the last completed eval rather than step 50000.
- Cross-device parity is not bit-level: cuBLAS/cuDNN ship different kernels per architecture, and with `use_deterministic_algo=false` even same-device reruns drift. Treat per-device numbers as samples, not ground truth.
- Sibling experiment [`../determinism`](../determinism) covers same-device bit-identical reproducibility.
