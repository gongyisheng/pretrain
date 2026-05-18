"""Emit the SuperBPE grid sweep configs (V x t x m) into per-V folders.

Run from the repo root:

    uv run python experiments/superbpe/generate_configs.py

The script is idempotent — re-running overwrites with identical content.
After writing, it validates every emitted YAML against the design
invariants and exits non-zero if any check fails.

Spec: docs/superpowers/specs/2026-05-18-superbpe-grid-design.md
"""

from __future__ import annotations

import sys
from pathlib import Path

# 256 byte tokens + 1 special token (<|endoftext|>).
ALPHABET = 257

V_VALUES = [50, 100, 150, 200]      # vocab targets, in thousands of merges
M_VALUES = [2, 3, 4, 5, 6]           # max_superword_words
T_STEP = 20                          # transition step, in thousands of merges

EXPERIMENT_ROOT = Path("experiments/superbpe")
TOKENIZER_ROOT = "tokenizers/experiments"  # value lives inside the YAML
WANDB_PROJECT = "tokenizer-superbpe"
DATASET = "openwebtext"
TRAIN_NUM_SAMPLES = 100000
EVAL_EVERY = 5000
EVAL_NUM_DOCS = 1000

BPE_TEMPLATE = """\
model:
  vocab_size: {vocab_size}
data:
  dataset: {dataset}
  tokenizer_path: {tokenizer_root}/{name}
  tokenizer_train_method: bpe
  tokenizer_train_num_samples: {train_num_samples}
  tokenizer_train_eval_every: {eval_every}
logging:
  wandb_project: {wandb_project}
  wandb_run_name: {name}
"""

SUPERBPE_TEMPLATE = """\
model:
  vocab_size: {vocab_size}
data:
  dataset: {dataset}
  tokenizer_path: {tokenizer_root}/{name}
  tokenizer_train_method: superbpe
  tokenizer_train_method_kwargs:
    transition_size: {transition_size}
    max_superword_words: {max_superword_words}
    eval_num_docs: {eval_num_docs}
  tokenizer_train_num_samples: {train_num_samples}
  tokenizer_train_eval_every: {eval_every}
logging:
  wandb_project: {wandb_project}
  wandb_run_name: {name}
"""


def bpe_yaml(V: int) -> tuple[str, str]:
    """Return (filename, yaml_text) for a BPE baseline at vocab V (in thousands)."""
    name = f"bpe_v{V}k"
    text = BPE_TEMPLATE.format(
        vocab_size=V * 1000 + ALPHABET,
        dataset=DATASET,
        tokenizer_root=TOKENIZER_ROOT,
        name=name,
        train_num_samples=TRAIN_NUM_SAMPLES,
        eval_every=EVAL_EVERY,
        wandb_project=WANDB_PROJECT,
    )
    return f"{name}.yaml", text


def superbpe_yaml(V: int, t: int, m: int) -> tuple[str, str]:
    """Return (filename, yaml_text) for a SuperBPE config (V, t, m all in thousands for V/t)."""
    name = f"superbpe_v{V}k_t{t}k_m{m}"
    text = SUPERBPE_TEMPLATE.format(
        vocab_size=V * 1000 + ALPHABET,
        dataset=DATASET,
        tokenizer_root=TOKENIZER_ROOT,
        name=name,
        transition_size=t * 1000 + ALPHABET,
        max_superword_words=m,
        eval_num_docs=EVAL_NUM_DOCS,
        train_num_samples=TRAIN_NUM_SAMPLES,
        eval_every=EVAL_EVERY,
        wandb_project=WANDB_PROJECT,
    )
    return f"{name}.yaml", text


def generate() -> list[Path]:
    """Write all 114 YAMLs and return their paths."""
    written: list[Path] = []
    for V in V_VALUES:
        folder = EXPERIMENT_ROOT / f"v{V}k"
        folder.mkdir(parents=True, exist_ok=True)

        # BPE baseline
        fname, text = bpe_yaml(V)
        path = folder / fname
        path.write_text(text)
        written.append(path)

        # SuperBPE sweep
        for t in range(T_STEP, V, T_STEP):
            for m in M_VALUES:
                fname, text = superbpe_yaml(V, t, m)
                path = folder / fname
                path.write_text(text)
                written.append(path)
    return written


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

EXPECTED_COUNTS = {50: 11, 100: 21, 150: 36, 200: 46}  # per-V folder counts


def _parse_name(stem: str) -> dict:
    """Parse 'bpe_v200k' / 'superbpe_v200k_t80k_m4' into a fields dict."""
    parts = stem.split("_")
    if parts[0] == "bpe":
        assert len(parts) == 2 and parts[1].startswith("v") and parts[1].endswith("k")
        return {"method": "bpe", "V": int(parts[1][1:-1])}
    if parts[0] == "superbpe":
        assert len(parts) == 4
        assert parts[1].startswith("v") and parts[1].endswith("k")
        assert parts[2].startswith("t") and parts[2].endswith("k")
        assert parts[3].startswith("m")
        return {
            "method": "superbpe",
            "V": int(parts[1][1:-1]),
            "t": int(parts[2][1:-1]),
            "m": int(parts[3][1:]),
        }
    raise AssertionError(f"unrecognized name: {stem}")


def validate(paths: list[Path]) -> None:
    """Check the invariants listed in the design spec.

    Raises AssertionError on any failure.
    """
    # Total file count
    assert len(paths) == 114, f"expected 114 files, got {len(paths)}"

    # Per-V folder counts
    for V, expected in EXPECTED_COUNTS.items():
        actual = sum(1 for p in paths if p.parent.name == f"v{V}k")
        assert actual == expected, f"v{V}k: expected {expected} files, got {actual}"

    for path in paths:
        text = path.read_text()
        fields = _parse_name(path.stem)

        # vocab_size = V*1000 + 257
        expected_vocab = fields["V"] * 1000 + ALPHABET
        assert f"vocab_size: {expected_vocab}" in text, (
            f"{path}: missing vocab_size: {expected_vocab}"
        )

        # tokenizer_path tail and wandb_run_name match filename stem
        assert f"tokenizer_path: {TOKENIZER_ROOT}/{path.stem}" in text, (
            f"{path}: tokenizer_path mismatch"
        )
        assert f"wandb_run_name: {path.stem}" in text, (
            f"{path}: wandb_run_name mismatch"
        )

        if fields["method"] == "bpe":
            assert "tokenizer_train_method: bpe" in text
            assert "tokenizer_train_method_kwargs" not in text, (
                f"{path}: bpe baseline must not have tokenizer_train_method_kwargs"
            )
        else:
            assert "tokenizer_train_method: superbpe" in text
            expected_t = fields["t"] * 1000 + ALPHABET
            assert f"transition_size: {expected_t}" in text, (
                f"{path}: missing transition_size: {expected_t}"
            )
            assert f"max_superword_words: {fields['m']}" in text
            # t < V
            assert fields["t"] < fields["V"], (
                f"{path}: t ({fields['t']}) must be < V ({fields['V']})"
            )


def main() -> int:
    if not EXPERIMENT_ROOT.exists():
        print(
            f"error: run from the repo root; {EXPERIMENT_ROOT} not found",
            file=sys.stderr,
        )
        return 1
    paths = generate()
    validate(paths)
    print(f"Generated and validated {len(paths)} configs in {EXPERIMENT_ROOT}/v*k/")
    for V in V_VALUES:
        count = sum(1 for p in paths if p.parent.name == f"v{V}k")
        print(f"  v{V}k/: {count} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
