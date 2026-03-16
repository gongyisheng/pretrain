# Experiments

Each experiment is a self-contained folder with:
- `README.md` — hypothesis, setup, results
- `config.yaml` — experiment config (can use `base_config` to inherit)
- Custom Python files — override specific components

## Creating an experiment

1. Create a folder: `experiments/<name>/`
2. Write a `config.yaml` that inherits from a base:
   ```yaml
   base_config: "configs/gpt2_124m.yaml"
   overrides:
     optimizer_module: "experiments/<name>/optimizer.py"
   optimizer:
     name: "custom"
   ```
3. Run: `python scripts/train.py --config experiments/<name>/config.yaml`

## Graduating an experiment

If results are good, move the custom code into `src/` and update the base config.
