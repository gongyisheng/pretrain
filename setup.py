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
            "src.kernel.backends.cuda._C",
            [
                "src/kernel/backends/cuda/csrc/gemm.cpp",
                "src/kernel/backends/cuda/csrc/matmul_scales.cu",
                "src/kernel/backends/cuda/csrc/fp8.cu",
                "src/kernel/backends/cuda/csrc/int8.cu",
                "src/kernel/backends/cuda/csrc/nvfp4.cu",
                "src/kernel/backends/cuda/csrc/mxfp8.cu",
            ],
            depends=[
                "src/kernel/backends/cuda/csrc/quantize.cuh",
                "src/kernel/backends/cuda/csrc/reduce.cuh",
            ],
            libraries=["cublasLt"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    )

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)},
)
