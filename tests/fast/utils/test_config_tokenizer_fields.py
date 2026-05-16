"""DataConfig must expose tokenizer-method fields with backward-compatible defaults."""

import tempfile
from pathlib import Path

from src.utils.config import DataConfig, load_config


def test_dataconfig_defaults_preserve_bpe():
    dc = DataConfig()
    assert dc.tokenizer_method == "bpe"
    assert dc.tokenizer_transition_size is None
    assert dc.tokenizer_max_superword_words == 4


def test_dataconfig_loads_superbpe_yaml():
    yaml_text = """
model:
  vocab_size: 200000
data:
  dataset: openwebtext
  tokenizer_path: tokenizers/superbpe_200k_t80k
  tokenizer_method: superbpe
  tokenizer_transition_size: 80000
  tokenizer_max_superword_words: 4
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    try:
        cfg = load_config(path)
        assert cfg.data.tokenizer_method == "superbpe"
        assert cfg.data.tokenizer_transition_size == 80000
        assert cfg.data.tokenizer_max_superword_words == 4
    finally:
        Path(path).unlink()
