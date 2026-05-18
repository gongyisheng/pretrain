"""Native C++ hot loops for BPE training.

The compiled `_bpe_native` extension is the only public surface. See
`src/data/bpe.py` for how it's used.
"""

from src.data.bpe_native._bpe_native import BpeState, hello  # noqa: F401
