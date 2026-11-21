# AdamW Initialization Sweep

## Hypothesis

The initialization method and standard-deviation gain affect early training stability and validation loss for Qwen3 51M on OpenWebText. Compare each setting with the fixed, gain 1 control at matched optimizer steps.

## Setup

The sweep runs 20 configurations. Substitute each gain in the table into its config filename.

| Initialization method | Config filename | Gains | Approx. parameters |
|---|---|---|---:|
| Fixed | `qwen3_51m_adamw_fixed_gain_<gain>.yaml` | 0.25, 0.5, 1, 2, 4 | 51M |
| LeCun | `qwen3_51m_adamw_lecun_gain_<gain>.yaml` | 0.25, 0.5, 1, 2, 4 | 51M |
| Xavier | `qwen3_51m_adamw_xavier_gain_<gain>.yaml` | 0.25, 0.5, 1, 2, 4 | 51M |
| Kaiming | `qwen3_51m_adamw_kaiming_gain_<gain>.yaml` | 0.25, 0.5, 1, 2, 4 | 51M |

All runs use normal initialization. Gain multiplies the method's standard deviation for attention and MLP projections. Fixed starts at 0.02, and embeddings and the LM head use 0.02 at every gain.

All runs use seed 42, sequence length 1024, batch size 16 with 16 accumulation steps, bf16, and gradient clipping at 1.0. AdamW uses betas `(0.9, 0.95)` and weight decay 0.1. The 50,000-step cosine schedule runs from `5e-4` to `5e-5` with 1,500 warmup steps. Evaluation runs every 100 steps for 100 batches, and checkpoints are saved every 5,000 steps. [Weight-decay baseline](../adamw_weight_decay/qwen3_51m_adamw_wd_0.1.yaml)

## Run

From the repository root, check `nvidia-smi` and pin a free device with `CUDA_VISIBLE_DEVICES`. Run the sweep for 50,000 steps per configuration or screen each configuration for 10,000 steps.

```bash
bash experiments/adamw_init_method/run.sh
bash experiments/adamw_init_method/run.sh --training.early_stop=10000
```

Screening retains the 50,000-step cosine schedule. Start full runs from scratch. To keep both sets of outputs, use the YAML defaults for screening and give each full run its own checkpoint directory and W&B name, as in this example for an individual finalist.

```bash
stem=qwen3_51m_adamw_fixed_gain_1
uv run python scripts/train.py --config "experiments/adamw_init_method/${stem}.yaml" \
  --training.checkpoint_dir="checkpoints/adamw_init_method/${stem}_full/" \
  --logging.wandb_run_name="${stem}_full"
```

The launcher passes the same overrides to every configuration, so a shared checkpoint directory override would collide. The W&B project is `pretrain-adamw-init-method`.

## Results

| Initialization method | Gain | Validation loss at 10k | Validation loss at 50k | Status |
|---|---:|---:|---:|---|
| Fixed | 0.25 | — | — | Pending |
| Fixed | 0.5 | — | — | Pending |
| Fixed (control) | 1 | — | — | Pending |
| Fixed | 2 | — | — | Pending |
| Fixed | 4 | — | — | Pending |
| LeCun | 0.25 | — | — | Pending |
| LeCun | 0.5 | — | — | Pending |
| LeCun | 1 | — | — | Pending |
| LeCun | 2 | — | — | Pending |
| LeCun | 4 | — | — | Pending |
| Xavier | 0.25 | — | — | Pending |
| Xavier | 0.5 | — | — | Pending |
| Xavier | 1 | — | — | Pending |
| Xavier | 2 | — | — | Pending |
| Xavier | 4 | — | — | Pending |
| Kaiming | 0.25 | — | — | Pending |
| Kaiming | 0.5 | — | — | Pending |
| Kaiming | 1 | — | — | Pending |
| Kaiming | 2 | — | — | Pending |
| Kaiming | 4 | — | — | Pending |

## Notes

Use the training loss and gradient norm curves to assess early stability. Compare validation loss at the same optimizer step.
