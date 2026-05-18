"""Native C++ hot loops for BPE training.

The compiled `_bpe_native` extension is the only public surface. See
`src/data/bpe.py` for how it's used.
"""

from src.data.bpe_native._bpe_native import hello  # noqa: F401

# `BpeState` is added in Task 2 once the C++ class exists.
