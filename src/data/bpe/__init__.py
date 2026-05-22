"""BPE trainer package — pure-Python control flow + C++ hot-loop extension.

Public surface:
  - `BpeTrainer`: Python-side config + train() entry point (see `trainer.py`).
  - `BpeEngine`: native C++ class owning chunks + heap + merge loop
    (see `bpe_engine.{h,cpp}` and the `_bpe_engine` extension).
"""

from src.data.bpe._bpe_engine import BpeEngine, hello  # noqa: F401
from src.data.bpe.trainer import (  # noqa: F401
    BpeTrainer,
    _BYTE_TO_UNICODE,
    _UNICODE_TO_BYTE,
    _build_chunks,
    _byte_encode,
    _pretokenize,
)
