"""Build configuration for the C++ _bpe_engine extension under src/data/bpe/.

Pure-Python packages are still declared in pyproject.toml; this file exists
only because pybind11's Pybind11Extension needs a setup.py hook for the C++
extension build. `uv sync` invokes this automatically.
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "src.data.bpe._bpe_engine",
        sources=[
            "src/data/bpe/bindings.cpp",
            "src/data/bpe/bpe_engine.cpp",
        ],
        include_dirs=["src/data/bpe"],
        cxx_std=17,
        extra_compile_args=["-O3", "-fopenmp", "-march=native"],
        extra_link_args=["-fopenmp"],
    )
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
