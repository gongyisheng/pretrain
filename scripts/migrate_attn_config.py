"""One-off: wrap model.attn_cls/attn_kwargs -> model.attn list, and reorder
model keys to attn, mlp, then the rest.

  uv run python scripts/migrate_attn_config.py            # dry run
  uv run python scripts/migrate_attn_config.py --apply     # rewrite in place
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
    if "attn_cls" not in model and "attn_kwargs" not in model:
        return False
    attn_cls = model.pop("attn_cls", "gqa")
    attn_kwargs = model.pop("attn_kwargs", {})
    model["attn"] = [{"attn_cls": attn_cls, "attn_kwargs": attn_kwargs}]
    # canonical order: attn, mlp, then the rest (existing relative order)
    if "mlp" in model:
        model.move_to_end("mlp", last=False)
    model.move_to_end("attn", last=False)
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
