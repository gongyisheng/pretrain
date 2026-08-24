# syntax=docker/dockerfile:1
#
# Single-stage dev/interactive image for the pretrain codebase.
#   Python 3.12  •  CUDA 13.2  •  torch 2.12 (cu132 wheels)
#
# The image is a pure environment: source is cloned and deps are synced by
# entrypoint.sh at container START (not build), so the image needs no local
# build context and never goes stale. Build with a tiny context:
#   docker build -f docker/Dockerfile -t pretrain docker/
#
# Pick the branch/tag to run at `docker run` time (default: main):
#   docker run -e GIT_REF=v0.1.0 pretrain
#
# The cu132 CUDA-13.2 devel base gives us nvcc + cublasLt, required to compile:
#   - src/data/bpe/_bpe_engine            (pybind11, C++17, OpenMP, -march=native)
#   - src/kernel/backends/cublaslt/_C     (torch CUDAExtension, links cublasLt)
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# python3-dev supplies Python.h — pybind11 needs it whenever uv builds the
# native extensions against the system interpreter instead of a managed one.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv drives all dependency resolution/installs from uv.lock (reproducible).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/venv/bin:$PATH \
    # Target only the RTX 5090 (Blackwell, sm_120) — same-host build, no fat binary.
    TORCH_CUDA_ARCH_LIST=12.0 \
    # Keep the venv OUT of /workspace so a runtime bind-mount can't shadow it.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Repo + ref are ENV (not ARG) so they can be overridden at `docker run` time
# with `-e GIT_REF=...`; entrypoint.sh clones/updates /workspace/pretrain on start.
ENV GIT_REPO=https://github.com/gongyisheng/pretrain.git \
    GIT_REF=main

WORKDIR /workspace

# entrypoint.sh clones the repo, runs `uv sync --frozen` (setup.py compiles both
# native extensions — the CUDA devel base supplies nvcc + cublasLt), downloads
# the dataset, then runs the given command — all at container start.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
