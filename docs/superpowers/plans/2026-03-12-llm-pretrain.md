# LLM Pretraining Codebase Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure PyTorch LLM pretraining codebase and run GPT-2 small pretraining on OpenWebText.

**Architecture:** Modular package with config-driven training. Separate modules for model, data, training, evaluation. YAML configs define experiments. Custom BPE tokenizer. Memory-mapped binary data files. W&B logging.

**Tech Stack:** Python 3.10+, PyTorch, HuggingFace `tokenizers` + `datasets`, W&B, PyYAML, NumPy

**Spec:** `docs/superpowers/specs/2026-03-12-llm-pretrain-design.md`

---

## File Structure

```
pretrain/
├── configs/
│   └── gpt2_small.yaml              # GPT-2 small (124M) experiment config
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── components.py             # MultiHeadAttention, LayerNorm, GeLU, TransformerBlock
│   │   ├── gpt2.py                   # GPT2Model
│   │   └── registry.py               # MODEL_REGISTRY + build_model()
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py              # train_tokenizer() function
│   │   └── dataset.py                # PretrainDataset (memory-mapped .bin reader)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizer.py              # build_optimizer(), build_scheduler()
│   │   ├── logger.py                 # WandbLogger wrapper
│   │   └── trainer.py                # Trainer class (core loop)
│   └── utils/
│       ├── __init__.py
│       └── config.py                 # TrainConfig dataclass + load_config()
├── scripts/
│   ├── train.py                      # Training entry point
│   ├── train_tokenizer.py            # Tokenizer training entry point
│   ├── preprocess_data.py            # Data preprocessing entry point
│   └── eval.py                       # Evaluation entry point
├── tests/
│   ├── test_config.py
│   ├── test_components.py
│   ├── test_gpt2.py
│   ├── test_dataset.py
│   └── test_trainer.py
├── experiments/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Chunk 1: Foundation (Config + Dependencies)

### Task 1: Project setup and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/training/__init__.py`
- Create: `src/utils/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0
tokenizers>=0.15
datasets>=2.14
wandb>=0.15
pyyaml>=6.0
numpy>=1.24
pytest>=7.0
```

- [ ] **Step 2: Create all `__init__.py` files**

Empty files for: `src/`, `src/model/`, `src/data/`, `src/training/`, `src/utils/`

- [ ] **Step 3: Create .gitignore**

```
data/
checkpoints/
tokenizers/
wandb/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Install dependencies**

Run: `pip install -r requirements.txt`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/ .gitignore
git commit -m "feat: add project skeleton and dependencies"
```

---

### Task 2: Config system

**Files:**
- Create: `src/utils/config.py`
- Create: `tests/test_config.py`
- Create: `configs/gpt2_small.yaml`

- [ ] **Step 1: Write config test**

```python
# tests/test_config.py
import pytest
import tempfile
import os
import yaml
from src.utils.config import TrainConfig, load_config


def _write_yaml(tmp_dir, data):
    path = os.path.join(tmp_dir, "test.yaml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


MINIMAL_CONFIG = {
    "max_seq_len": 128,
    "model": {"arch": "gpt2", "n_layers": 2, "n_heads": 2, "d_model": 64, "vocab_size": 256, "dropout": 0.0},
    "data": {"dataset": "test", "tokenizer_path": "tok", "data_dir": "data/", "val_split": 0.01, "num_workers": 0},
    "training": {
        "batch_size": 2, "gradient_accumulation_steps": 1, "max_steps": 10,
        "mixed_precision": "no", "activation_checkpointing": False,
        "grad_clip": 1.0, "checkpoint_dir": "ckpt/", "checkpoint_every": 5,
        "eval_every": 5, "eval_steps": 2,
    },
    "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1, "betas": [0.9, 0.95]},
    "scheduler": {"name": "cosine", "warmup_steps": 2, "min_lr": 1e-4},
    "logging": {"wandb_project": "test", "wandb_run_name": "test", "log_every": 1},
}


def test_load_config_from_yaml():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.max_seq_len == 128
        assert config.model.arch == "gpt2"
        assert config.model.n_layers == 2
        assert config.optimizer.lr == 1e-3


def test_config_d_ff_default():
    """d_ff defaults to 4 * d_model when not specified."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        assert config.model.d_ff == 4 * 64  # 4 * d_model


def test_config_d_ff_explicit():
    """d_ff can be explicitly set."""
    data = {**MINIMAL_CONFIG, "model": {**MINIMAL_CONFIG["model"], "d_ff": 128}}
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, data)
        config = load_config(path)
        assert config.model.d_ff == 128


def test_config_cli_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path, overrides=["optimizer.lr=3e-4", "training.batch_size=8"])
        assert config.optimizer.lr == 3e-4
        assert config.training.batch_size == 8


def test_config_to_dict_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_yaml(tmp, MINIMAL_CONFIG)
        config = load_config(path)
        d = config.to_dict()
        assert d["max_seq_len"] == 128
        assert d["model"]["arch"] == "gpt2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement config system**

```python
# src/utils/config.py
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    arch: str = "gpt2"
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 0  # 0 means 4 * d_model, set in post_init
    vocab_size: int = 50257
    dropout: float = 0.1

    def __post_init__(self):
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model


@dataclass
class DataConfig:
    dataset: str = "openwebtext"
    tokenizer_path: str = "tokenizers/custom_bpe"
    data_dir: str = "data/"
    val_split: float = 0.01
    num_workers: int = 4


@dataclass
class TrainingConfig:
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_steps: int = 100000
    mixed_precision: str = "bf16"
    activation_checkpointing: bool = False
    grad_clip: float = 1.0
    checkpoint_dir: str = "checkpoints/"
    checkpoint_every: int = 5000
    eval_every: int = 1000
    eval_steps: int = 200


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: List[float] = field(default_factory=lambda: [0.9, 0.95])


@dataclass
class SchedulerConfig:
    name: str = "cosine"
    warmup_steps: int = 2000
    min_lr: float = 6e-5


@dataclass
class LoggingConfig:
    wandb_project: str = "pretrain"
    wandb_run_name: str = ""
    log_every: int = 10


@dataclass
class TrainConfig:
    max_seq_len: int = 1024
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self):
        return asdict(self)


def _apply_overrides(config: TrainConfig, overrides: List[str]):
    """Apply CLI overrides like 'optimizer.lr=3e-4'."""
    for override in overrides:
        key, value = override.split("=", 1)
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        field_name = parts[-1]
        # Infer type from existing field
        current = getattr(obj, field_name)
        if isinstance(current, bool):
            value = value.lower() in ("true", "1", "yes")
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(obj, field_name, value)


def load_config(path: str, overrides: Optional[List[str]] = None) -> TrainConfig:
    """Load config from YAML, optionally applying CLI overrides."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    config = TrainConfig(
        max_seq_len=raw.get("max_seq_len", 1024),
        model=ModelConfig(**{k: v for k, v in raw.get("model", {}).items()}),
        data=DataConfig(**{k: v for k, v in raw.get("data", {}).items()}),
        training=TrainingConfig(**{k: v for k, v in raw.get("training", {}).items()}),
        optimizer=OptimizerConfig(**{k: v for k, v in raw.get("optimizer", {}).items()}),
        scheduler=SchedulerConfig(**{k: v for k, v in raw.get("scheduler", {}).items()}),
        logging=LoggingConfig(**{k: v for k, v in raw.get("logging", {}).items()}),
    )

    if overrides:
        _apply_overrides(config, overrides)

    return config
```

- [ ] **Step 4: Run tests**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 5: Create GPT-2 small config YAML**

```yaml
# configs/gpt2_small.yaml
max_seq_len: 1024

model:
  arch: "gpt2"
  n_layers: 12
  n_heads: 12
  d_model: 768
  vocab_size: 50257
  dropout: 0.1

data:
  dataset: "openwebtext"
  tokenizer_path: "tokenizers/custom_bpe_50k"
  data_dir: "data/"
  val_split: 0.01
  num_workers: 4

training:
  batch_size: 12
  gradient_accumulation_steps: 5
  max_steps: 100000
  mixed_precision: "bf16"
  activation_checkpointing: false
  grad_clip: 1.0
  checkpoint_dir: "checkpoints/"
  checkpoint_every: 5000
  eval_every: 1000
  eval_steps: 200

optimizer:
  name: "adamw"
  lr: 6e-4
  weight_decay: 0.1
  betas: [0.9, 0.95]

scheduler:
  name: "cosine"
  warmup_steps: 2000
  min_lr: 6e-5

logging:
  wandb_project: "pretrain"
  wandb_run_name: "gpt2-small-openwebtext"
  log_every: 10
```

- [ ] **Step 6: Commit**

```bash
git add src/utils/config.py tests/test_config.py configs/gpt2_small.yaml
git commit -m "feat: add config system with YAML loading and CLI overrides"
```

---

## Chunk 2: Model

### Task 3: Model components

**Files:**
- Create: `src/model/components.py`
- Create: `tests/test_components.py`

- [ ] **Step 1: Write component tests**

```python
# tests/test_components.py
import pytest
import torch
from src.model.components import MultiHeadAttention, TransformerBlock


def test_multihead_attention_output_shape():
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)  # (batch, seq, d_model)
    out = mha(x)
    assert out.shape == (2, 16, 64)


def test_multihead_attention_is_causal():
    """Future tokens should not affect past token outputs."""
    mha = MultiHeadAttention(d_model=64, n_heads=4, dropout=0.0)
    mha.eval()
    x = torch.randn(1, 8, 64)
    out_full = mha(x)
    # Changing token 7 should not change output at token 0
    x2 = x.clone()
    x2[0, 7, :] = torch.randn(64)
    out_modified = mha(x2)
    assert torch.allclose(out_full[0, :7], out_modified[0, :7], atol=1e-6)


def test_transformer_block_output_shape():
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert out.shape == (2, 16, 64)


def test_transformer_block_residual():
    """Output should differ from input (not identity) but be same shape."""
    block = TransformerBlock(d_model=64, n_heads=4, d_ff=256, dropout=0.0)
    x = torch.randn(2, 16, 64)
    out = block(x)
    assert not torch.allclose(out, x)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_components.py -v`
Expected: FAIL

- [ ] **Step 3: Implement components**

```python
# src/model/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, d_head)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal_mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```

- [ ] **Step 4: Run tests**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_components.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/components.py tests/test_components.py
git commit -m "feat: add transformer components (attention, block)"
```

---

### Task 4: GPT-2 model and registry

**Files:**
- Create: `src/model/gpt2.py`
- Create: `src/model/registry.py`
- Create: `tests/test_gpt2.py`

- [ ] **Step 1: Write GPT-2 model tests**

```python
# tests/test_gpt2.py
import pytest
import torch
from src.model.gpt2 import GPT2Model
from src.model.registry import build_model
from src.utils.config import ModelConfig


def _small_config():
    return ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=256, dropout=0.0)


def test_gpt2_forward_shape():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))  # (batch, seq)
    logits = model(x)
    assert logits.shape == (2, 32, 256)  # (batch, seq, vocab)


def test_gpt2_loss():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    # Shift: predict next token
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)
    )
    assert loss.item() > 0
    assert loss.requires_grad


def test_gpt2_param_count():
    config = _small_config()
    model = GPT2Model(config, max_seq_len=128)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0
    # Rough check: should be small for this tiny config
    assert n_params < 1_000_000


def test_registry_build_model():
    config = _small_config()

    class FakeTrainConfig:
        model = config
        max_seq_len = 128

    model = build_model(FakeTrainConfig())
    assert isinstance(model, GPT2Model)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_gpt2.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GPT-2 model**

```python
# src/model/gpt2.py
import torch
import torch.nn as nn
from src.model.components import TransformerBlock
from src.utils.config import ModelConfig


class GPT2Model(nn.Module):
    def __init__(self, config: ModelConfig, max_seq_len: int = 1024):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)

        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
```

- [ ] **Step 4: Implement registry**

```python
# src/model/registry.py
from src.model.gpt2 import GPT2Model

MODEL_REGISTRY = {
    "gpt2": GPT2Model,
}


def build_model(config):
    """Build model from config. config must have .model.arch and .max_seq_len."""
    arch = config.model.arch
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[arch](config.model, max_seq_len=config.max_seq_len)
```

- [ ] **Step 5: Run tests**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_gpt2.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add src/model/gpt2.py src/model/registry.py tests/test_gpt2.py
git commit -m "feat: add GPT-2 model and model registry"
```

---

## Chunk 3: Data Pipeline

### Task 5: Tokenizer training

**Files:**
- Create: `src/data/tokenizer.py`
- Create: `scripts/train_tokenizer.py`

- [ ] **Step 1: Implement tokenizer training module**

```python
# src/data/tokenizer.py
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


def train_tokenizer(
    dataset_iter,
    vocab_size: int = 50257,
    save_path: str = "tokenizers/custom_bpe",
):
    """Train a BPE tokenizer on an iterator of text strings.

    Args:
        dataset_iter: Iterator yielding text strings.
        vocab_size: Target vocabulary size.
        save_path: Directory to save the tokenizer.
    """
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(dataset_iter, trainer=trainer)

    os.makedirs(save_path, exist_ok=True)
    tokenizer.save(os.path.join(save_path, "tokenizer.json"))
    print(f"Tokenizer saved to {save_path}/ (vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a trained tokenizer from disk."""
    return Tokenizer.from_file(os.path.join(path, "tokenizer.json"))
```

- [ ] **Step 2: Implement train_tokenizer script**

```python
# scripts/train_tokenizer.py
"""Train a custom BPE tokenizer on a HuggingFace dataset."""
import argparse
import sys
sys.path.insert(0, ".")

from datasets import load_dataset
from src.utils.config import load_config
from src.data.tokenizer import train_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--num_samples", type=int, default=1_000_000,
                        help="Number of text samples to train on")
    args = parser.parse_args()

    config = load_config(args.config)

    print(f"Loading dataset: {config.data.dataset}")
    ds = load_dataset(config.data.dataset, split="train", streaming=True)

    def text_iter():
        for i, sample in enumerate(ds):
            if i >= args.num_samples:
                break
            yield sample["text"]

    train_tokenizer(
        dataset_iter=text_iter(),
        vocab_size=config.model.vocab_size,
        save_path=config.data.tokenizer_path,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/data/tokenizer.py scripts/train_tokenizer.py
git commit -m "feat: add BPE tokenizer training"
```

---

### Task 6: Data preprocessing

**Files:**
- Create: `scripts/preprocess_data.py`

- [ ] **Step 1: Implement preprocessing script**

```python
# scripts/preprocess_data.py
"""Tokenize a HuggingFace dataset and save as memory-mapped .bin files.

Uses a two-pass approach to avoid loading all tokens into RAM:
  Pass 1: Stream dataset, tokenize, write to a single temporary .bin file
  Pass 2: Split the temp file into train.bin and val.bin
"""
import argparse
import os
import sys
import numpy as np
sys.path.insert(0, ".")

from datasets import load_dataset
from src.utils.config import load_config
from src.data.tokenizer import load_tokenizer

CHUNK_SIZE = 1024 * 1024  # write in chunks of ~1M tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to process (None = all)")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = load_tokenizer(config.data.tokenizer_path)
    eot_token = tokenizer.token_to_id("<|endoftext|>")

    os.makedirs(config.data.data_dir, exist_ok=True)
    tmp_path = os.path.join(config.data.data_dir, "all_tokens.bin")
    train_path = os.path.join(config.data.data_dir, "train.bin")
    val_path = os.path.join(config.data.data_dir, "val.bin")

    # Pass 1: Stream, tokenize, and write chunks to temp file
    print(f"Loading dataset: {config.data.dataset} (streaming)")
    ds = load_dataset(config.data.dataset, split="train", streaming=True)

    total_tokens = 0
    buffer = []

    with open(tmp_path, "wb") as f:
        for i, sample in enumerate(ds):
            if args.max_samples and i >= args.max_samples:
                break
            ids = tokenizer.encode(sample["text"]).ids
            buffer.extend(ids)
            buffer.append(eot_token)

            # Flush buffer when large enough
            if len(buffer) >= CHUNK_SIZE:
                chunk = np.array(buffer, dtype=np.uint16)
                f.write(chunk.tobytes())
                total_tokens += len(buffer)
                buffer = []

            if (i + 1) % 10000 == 0:
                print(f"  Tokenized {i+1} documents ({total_tokens + len(buffer):,} tokens)")

        # Flush remaining
        if buffer:
            chunk = np.array(buffer, dtype=np.uint16)
            f.write(chunk.tobytes())
            total_tokens += len(buffer)

    print(f"Total tokens: {total_tokens:,}")

    # Pass 2: Split into train/val via memmap
    all_data = np.memmap(tmp_path, dtype=np.uint16, mode="r")
    n_val = int(len(all_data) * config.data.val_split)
    n_train = len(all_data) - n_val

    train_data = np.memmap(train_path, dtype=np.uint16, mode="w+", shape=(n_train,))
    train_data[:] = all_data[:n_train]
    train_data.flush()

    val_data = np.memmap(val_path, dtype=np.uint16, mode="w+", shape=(n_val,))
    val_data[:] = all_data[n_train:]
    val_data.flush()

    # Cleanup temp file
    del all_data
    os.remove(tmp_path)

    print(f"Train: {n_train:,} tokens -> {train_path}")
    print(f"Val:   {n_val:,} tokens -> {val_path}")

    # Verification: decode a sample
    verify = np.memmap(train_path, dtype=np.uint16, mode="r")
    sample_ids = verify[:100].tolist()
    decoded = tokenizer.decode(sample_ids)
    print(f"\nSample (first 100 tokens decoded):\n{decoded[:500]}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/preprocess_data.py
git commit -m "feat: add data preprocessing script"
```

---

### Task 7: Dataset (memory-mapped reader)

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write dataset test**

```python
# tests/test_dataset.py
import pytest
import numpy as np
import os
import tempfile
import torch
from src.data.dataset import PretrainDataset


@pytest.fixture
def tmp_bin_file():
    """Create a temp .bin file with sequential uint16 tokens."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.bin")
        tokens = np.arange(1024, dtype=np.uint16)
        tokens.tofile(path)
        yield path


def test_dataset_length(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    # 1024 tokens / 128 seq_len = 8 sequences (need seq_len+1 for target, but we have enough for 7)
    assert len(ds) == 1024 // 128 - 1  # 7


def test_dataset_getitem_shape(tmp_bin_file):
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    x, y = ds[0]
    assert x.shape == (128,)
    assert y.shape == (128,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_dataset_getitem_shift(tmp_bin_file):
    """y should be x shifted by 1 (next token prediction)."""
    ds = PretrainDataset(tmp_bin_file, seq_len=128)
    x, y = ds[0]
    # x = tokens[0:128], y = tokens[1:129]
    assert torch.equal(y[:-1], x[1:])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement dataset**

```python
# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    """Memory-mapped dataset that reads fixed-length chunks from a .bin file.

    Each sample returns (x, y) where y = x shifted by 1 token (next-token prediction).
    """

    def __init__(self, bin_path: str, seq_len: int):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        # Number of complete sequences we can form (need seq_len + 1 tokens each)
        self.n_sequences = len(self.data) // seq_len - 1

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
```

- [ ] **Step 4: Run tests**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_dataset.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: add memory-mapped pretrain dataset"
```

---

## Chunk 4: Training Infrastructure

### Task 8: Optimizer and scheduler

**Files:**
- Create: `src/training/optimizer.py`

- [ ] **Step 1: Implement optimizer and scheduler builders**

```python
# src/training/optimizer.py
import math
import torch
from src.utils.config import TrainConfig


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Build optimizer with weight decay applied only to non-bias, non-layernorm params."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=config.optimizer.lr,
        betas=tuple(config.optimizer.betas),
    )
    return optimizer


class CosineWarmupScheduler:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int, min_lr: float, max_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / self.warmup_steps
        if self.current_step >= self.max_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


def build_scheduler(optimizer, config: TrainConfig):
    """Build LR scheduler from config."""
    return CosineWarmupScheduler(
        optimizer=optimizer,
        warmup_steps=config.scheduler.warmup_steps,
        max_steps=config.training.max_steps,
        min_lr=config.scheduler.min_lr,
        max_lr=config.optimizer.lr,
    )
```

- [ ] **Step 2: Commit**

```bash
git add src/training/optimizer.py
git commit -m "feat: add optimizer and cosine warmup scheduler"
```

---

### Task 9: W&B logger

**Files:**
- Create: `src/training/logger.py`

- [ ] **Step 1: Implement W&B logger**

```python
# src/training/logger.py
import wandb
from src.utils.config import TrainConfig


class WandbLogger:
    """Thin wrapper around W&B for training metrics."""

    def __init__(self, config: TrainConfig, enabled: bool = True):
        self.enabled = enabled
        if self.enabled:
            wandb.init(
                project=config.logging.wandb_project,
                name=config.logging.wandb_run_name or None,
                config=config.to_dict(),
            )

    def log(self, metrics: dict, step: int):
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: int):
        if self.enabled:
            wandb.log({key: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self):
        if self.enabled:
            wandb.finish()
```

- [ ] **Step 2: Commit**

```bash
git add src/training/logger.py
git commit -m "feat: add W&B logger wrapper"
```

---

### Task 10: Trainer (core training loop)

**Files:**
- Create: `src/training/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write trainer smoke test**

```python
# tests/test_trainer.py
import pytest
import os
import tempfile
import numpy as np
import torch
from src.utils.config import TrainConfig, ModelConfig, DataConfig, TrainingConfig, OptimizerConfig, SchedulerConfig, LoggingConfig
from src.training.trainer import Trainer


def _tiny_config(tmp_dir):
    """Config for a tiny model that trains in seconds."""
    # Create tiny .bin files
    tokens = np.arange(4096, dtype=np.uint16)
    train_path = os.path.join(tmp_dir, "train.bin")
    val_path = os.path.join(tmp_dir, "val.bin")
    tokens.tofile(train_path)
    tokens[:512].tofile(val_path)

    return TrainConfig(
        max_seq_len=64,
        model=ModelConfig(arch="gpt2", n_layers=2, n_heads=2, d_model=64, vocab_size=4096, dropout=0.0),
        data=DataConfig(dataset="test", tokenizer_path="", data_dir=tmp_dir, val_split=0.01, num_workers=0),
        training=TrainingConfig(
            batch_size=4, gradient_accumulation_steps=1, max_steps=5,
            mixed_precision="no", activation_checkpointing=False,
            grad_clip=1.0, checkpoint_dir=os.path.join(tmp_dir, "ckpt"),
            checkpoint_every=3, eval_every=3, eval_steps=2,
        ),
        optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=0.0, betas=[0.9, 0.95]),
        scheduler=SchedulerConfig(name="cosine", warmup_steps=1, min_lr=1e-4),
        logging=LoggingConfig(wandb_project="test", wandb_run_name="test", log_every=1),
    )


def test_trainer_runs_without_error():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        assert trainer.step == 5


def test_trainer_saves_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        ckpt_dir = os.path.join(tmp, "ckpt")
        assert os.path.exists(os.path.join(ckpt_dir, "step_3.pt"))


def test_trainer_loss_decreases():
    with tempfile.TemporaryDirectory() as tmp:
        config = _tiny_config(tmp)
        config.training.max_steps = 20
        config.training.eval_every = 100  # don't eval, just train
        config.training.checkpoint_every = 100
        trainer = Trainer(config, wandb_enabled=False)
        trainer.train()
        # First loss should be higher than last
        assert trainer.loss_history[0] > trainer.loss_history[-1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_trainer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement trainer**

```python
# src/training/trainer.py
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from src.model.registry import build_model
from src.data.dataset import PretrainDataset
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.logger import WandbLogger
from src.utils.config import TrainConfig


class Trainer:
    def __init__(self, config: TrainConfig, wandb_enabled: bool = True, resume_from: str = None):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.step = 0
        self.loss_history = []

        # Seed for reproducibility
        self._seed(42)

        # Model
        self.model = build_model(config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {config.model.arch} | {n_params / 1e6:.1f}M parameters | device={self.device}")

        # Data
        train_path = os.path.join(config.data.data_dir, "train.bin")
        val_path = os.path.join(config.data.data_dir, "val.bin")
        self.train_dataset = PretrainDataset(train_path, config.max_seq_len)
        self.val_dataset = PretrainDataset(val_path, config.max_seq_len)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.training.batch_size,
            shuffle=True, num_workers=config.data.num_workers, pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.training.batch_size,
            shuffle=False, num_workers=config.data.num_workers, pin_memory=True,
        )

        # Optimizer & scheduler
        self.optimizer = build_optimizer(self.model, config)
        self.scheduler = build_scheduler(self.optimizer, config)

        # Mixed precision
        self.use_amp = config.training.mixed_precision != "no" and self.device == "cuda"
        self.amp_dtype = torch.bfloat16 if config.training.mixed_precision == "bf16" else torch.float16
        self.scaler = torch.amp.GradScaler(enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Activation checkpointing
        if config.training.activation_checkpointing:
            for block in self.model.blocks:
                block._original_forward = block.forward
                def make_ckpt_forward(b):
                    def ckpt_forward(x):
                        return torch.utils.checkpoint.checkpoint(b._original_forward, x, use_reentrant=False)
                    return ckpt_forward
                block.forward = make_ckpt_forward(block)

        # Logger
        self.logger = WandbLogger(config, enabled=wandb_enabled)

        # Checkpoint dir
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)

        # Resume
        if resume_from:
            self._load_checkpoint(resume_from)

    def _seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train(self):
        cfg = self.config.training
        self.model.train()

        train_iter = iter(self.train_loader)
        accum_loss = 0.0
        t_last_log = time.time()
        tokens_since_log = 0

        while self.step < cfg.max_steps:
            self.optimizer.zero_grad()

            for micro_step in range(cfg.gradient_accumulation_steps):
                try:
                    x, y = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    x, y = next(train_iter)

                x, y = x.to(self.device), y.to(self.device)

                with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                    logits = self.model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / cfg.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                accum_loss += loss.item()
                tokens_since_log += x.numel()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self.step += 1
            self.loss_history.append(accum_loss)

            # Logging
            if self.step % self.config.logging.log_every == 0:
                elapsed = time.time() - t_last_log
                tokens_per_sec = tokens_since_log / elapsed if elapsed > 0 else 0
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log({
                    "train/loss": accum_loss,
                    "train/perplexity": min(float(torch.exp(torch.tensor(accum_loss))), 1e6),
                    "lr": lr,
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "tokens_per_sec": tokens_per_sec,
                }, step=self.step)
                print(f"step {self.step}/{cfg.max_steps} | loss={accum_loss:.4f} | lr={lr:.2e} | tok/s={tokens_per_sec:.0f}")
                t_last_log = time.time()
                tokens_since_log = 0

            accum_loss = 0.0

            # Evaluation
            if self.step % cfg.eval_every == 0:
                self._evaluate()

            # Checkpoint
            if self.step % cfg.checkpoint_every == 0:
                self._save_checkpoint()

        self.logger.finish()
        print(f"Training complete. Final step: {self.step}")

    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.training.eval_steps:
                break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(self.device, dtype=self.amp_dtype, enabled=self.use_amp):
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        ppl = min(float(torch.exp(torch.tensor(avg_loss))), 1e6)
        self.logger.log({"val/loss": avg_loss, "val/perplexity": ppl}, step=self.step)
        print(f"  [eval] val_loss={avg_loss:.4f} | val_ppl={ppl:.2f}")

        # Generate a text sample for qualitative check
        self._generate_sample()

        self.model.train()

    @torch.no_grad()
    def _generate_sample(self, max_new_tokens: int = 50):
        """Generate a short text sample for qualitative monitoring."""
        self.model.eval()
        # Start with a single EOT token (id=0 typically)
        idx = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits = self.model(idx_cond)
            logits = logits[:, -1, :]  # last token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        # Log raw token IDs (decoding requires tokenizer, which trainer doesn't hold)
        token_ids = idx[0].tolist()
        self.logger.log_text("generated_tokens", str(token_ids[:60]), step=self.step)

    def _save_checkpoint(self):
        path = os.path.join(self.config.training.checkpoint_dir, f"step_{self.step}.pt")
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "grad_scaler": self.scaler.state_dict(),
            "step": self.step,
            "config": self.config.to_dict(),
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        torch.save(checkpoint, path)
        print(f"  [ckpt] saved to {path}")

    def _load_checkpoint(self, path: str):
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.scaler.load_state_dict(checkpoint["grad_scaler"])
        self.step = checkpoint["step"]

        rng = checkpoint.get("rng_states", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.random.set_rng_state(rng["torch"])
        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng["cuda"])

        print(f"Resumed at step {self.step}")
```

- [ ] **Step 4: Run tests**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/test_trainer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/training/trainer.py tests/test_trainer.py
git commit -m "feat: add training loop with mixed precision, checkpointing, evaluation"
```

---

## Chunk 5: Scripts and End-to-End

### Task 11: Entry point scripts

**Files:**
- Create: `scripts/train.py`
- Create: `scripts/eval.py`
- Create: `experiments/README.md`

- [ ] **Step 1: Implement train.py**

```python
# scripts/train.py
"""Pretrain an LLM."""
import argparse
import sys
sys.path.insert(0, ".")

from src.utils.config import load_config
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Pretrain an LLM")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args, remaining = parser.parse_known_args()

    # Parse CLI overrides (e.g. --optimizer.lr=3e-4)
    overrides = []
    for arg in remaining:
        if arg.startswith("--") and "=" in arg:
            overrides.append(arg.removeprefix("--"))

    config = load_config(args.config, overrides=overrides if overrides else None)
    trainer = Trainer(config, wandb_enabled=not args.no_wandb, resume_from=args.resume)
    trainer.train()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement eval.py**

```python
# scripts/eval.py
"""Evaluate a pretrained model checkpoint."""
import argparse
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")

from src.utils.config import load_config
from src.model.registry import build_model
from src.data.dataset import PretrainDataset
from torch.utils.data import DataLoader
import os


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_steps", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load val data
    val_path = os.path.join(config.data.data_dir, "val.bin")
    val_dataset = PretrainDataset(val_path, config.max_seq_len)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)

    eval_steps = args.eval_steps or config.training.eval_steps
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= eval_steps:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = float(torch.exp(torch.tensor(avg_loss)))
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Step: {ckpt.get('step', 'unknown')}")
    print(f"Val loss: {avg_loss:.4f}")
    print(f"Val perplexity: {ppl:.2f}")
    print(f"Evaluated on {n_batches} batches")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create experiments README**

```markdown
# Experiments

Each experiment is a self-contained folder with:
- `README.md` — hypothesis, setup, results
- `config.yaml` — experiment config (can use `base_config` to inherit)
- Custom Python files — override specific components

## Creating an experiment

1. Create a folder: `experiments/<name>/`
2. Write a `config.yaml` that inherits from a base:
   ```yaml
   base_config: "configs/gpt2_small.yaml"
   overrides:
     optimizer_module: "experiments/<name>/optimizer.py"
   optimizer:
     name: "custom"
   ```
3. Run: `python scripts/train.py --config experiments/<name>/config.yaml`

## Graduating an experiment

If results are good, move the custom code into `src/` and update the base config.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py scripts/eval.py experiments/README.md
git commit -m "feat: add train/eval scripts and experiments README"
```

---

### Task 12: Run the full GPT-2 pretraining pipeline

This is the end-to-end execution. Run each step sequentially.

- [ ] **Step 1: Run all tests to verify everything works**

Run: `cd /media/hdddisk/yisheng/pretrain && python -m pytest tests/ -v`
Expected: All tests pass. Fix any failures before proceeding to the data pipeline.

- [ ] **Step 2: Train the tokenizer**

Run: `cd /media/hdddisk/yisheng/pretrain && python scripts/train_tokenizer.py --config configs/gpt2_small.yaml --num_samples 500000`

Expected: Tokenizer saved to `tokenizers/custom_bpe_50k/tokenizer.json`

- [ ] **Step 3: Preprocess the data**

Run: `cd /media/hdddisk/yisheng/pretrain && python scripts/preprocess_data.py --config configs/gpt2_small.yaml`

Expected: `data/train.bin` and `data/val.bin` created with token counts printed.

Note: OpenWebText is large (~38GB). This uses streaming so it won't OOM, but it will take a while. Consider `--max_samples 500000` for a smaller initial run to validate the pipeline first.

- [ ] **Step 4: Start pretraining**

Run: `cd /media/hdddisk/yisheng/pretrain && python scripts/train.py --config configs/gpt2_small.yaml`

Expected: Training starts, W&B run is created, loss decreases over time, checkpoints are saved every 5000 steps.
