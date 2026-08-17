"""Build configuration for the project's native extensions.

Pure-Python packages are still declared in pyproject.toml; this file exists
because the extension builders need a setup.py hook. `uv sync` invokes this
automatically.
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

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

if CUDA_HOME is not None:
    ext_modules.append(
        CUDAExtension(
            "src.kernel.backends.cublaslt._C",
            [
                "src/kernel/backends/cublaslt/csrc/gemm.cpp",
                "src/kernel/backends/cublaslt/csrc/cuda_build_sentinel.cu",
                "src/kernel/backends/cublaslt/csrc/mxfp8_grouped.cu",
            ],
            libraries=["cublasLt"],
            extra_compile_args={"cxx": ["-O3"]},
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
