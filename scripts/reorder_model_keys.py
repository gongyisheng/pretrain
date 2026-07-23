"""One-off: reorder model.* keys to d_model, n_layers, vocab_size, attn, mlp,
then the rest (existing relative order).

  uv run python scripts/reorder_model_keys.py            # dry run
  uv run python scripts/reorder_model_keys.py --apply     # rewrite in place
"""

import sys
from pathlib import Path

from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True


def reorder_file(path: Path, apply: bool) -> bool:
    data = yaml.load(path.read_text())
    if not isinstance(data, dict):
        return False
    model = data.get("model")
    if not isinstance(model, dict):
        return False
    before = list(model.keys())
    for key in ("mlp", "attn", "vocab_size", "n_layers", "d_model"):
        if key in model:
            model.move_to_end(key, last=False)
    after = list(model.keys())
    if before == after:
        return False
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
            if reorder_file(path, apply):
                changed.append(path.relative_to(root))
        except Exception as e:
            print(f"SKIP {path}: {e}")
    verb = "reordered" if apply else "would reorder"
    for p in changed:
        print(f"{verb}: {p}")
    print(f"\n{len(changed)} files {verb}")


if __name__ == "__main__":
    main()
