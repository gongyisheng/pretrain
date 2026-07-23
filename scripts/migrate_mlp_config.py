"""One-off: migrate model.mlp_cls/mlp_kwargs -> model.mlp list schema.

uv run python scripts/migrate_mlp_config.py           # dry run, lists files
uv run python scripts/migrate_mlp_config.py --apply    # rewrite in place
"""

import sys
from pathlib import Path
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True


def migrate_file(path: Path, apply: bool) -> bool:
    data = yaml.load(path.read_text())
    if not isinstance(data, dict):
        return False
    model = data.get("model")
    if not isinstance(model, dict):
        return False
    if "mlp_cls" not in model and "mlp_kwargs" not in model:
        return False
    mlp_cls = model.pop("mlp_cls", "dense")
    mlp_kwargs = model.pop("mlp_kwargs", {})
    item = {"mlp_cls": mlp_cls, "mlp_kwargs": mlp_kwargs}
    model["mlp"] = [item]
    if apply:
        with path.open("w") as f:
            yaml.dump(data, f)
    return True


def main():
    apply = "--apply" in sys.argv
    root = Path(__file__).resolve().parent.parent
    changed = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            if migrate_file(path, apply):
                changed.append(path.relative_to(root))
        except Exception as e:
            print(f"SKIP {path}: {e}")
    verb = "migrated" if apply else "would migrate"
    for p in changed:
        print(f"{verb}: {p}")
    print(f"\n{len(changed)} files {verb}")


if __name__ == "__main__":
    main()
