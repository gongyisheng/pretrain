# syntax=docker/dockerfile:1
#
# Single-stage dev/interactive image for the pretrain codebase.
#   Python 3.12  •  CUDA 13.2  •  torch 2.12 (cu132 wheels)
#
# Source is cloned from GitHub at build time, so no local build context is
# needed — build with a tiny context (the docker/ dir works fine):
#   docker build -f docker/Dockerfile -t pretrain docker/
#
# Pin a branch/tag to build from (default: main):
#   docker build -f docker/Dockerfile --build-arg GIT_REF=v0.1.0 -t <tag> docker/
#
# The cu132 CUDA-13.2 devel base gives us nvcc + cublasLt, required to compile:
#   - src/data/bpe/_bpe_engine            (pybind11, C++17, OpenMP, -march=native)
#   - src/kernel/backends/cublaslt/_C     (torch CUDAExtension, links cublasLt)
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
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

ARG GIT_REPO=https://github.com/gongyisheng/pretrain.git
ARG GIT_REF=main

# Bust the clone layer's cache whenever the ref moves: the commits API response
# changes on every push, so a rebuild re-clones (harmless for pinned tags).
ADD https://api.github.com/repos/gongyisheng/pretrain/commits/${GIT_REF} /tmp/commit.json
RUN git clone --depth 1 --branch ${GIT_REF} ${GIT_REPO} /workspace

WORKDIR /workspace

# setup.py compiles both native extensions during `uv sync`; the CUDA devel
# base supplies nvcc + cublasLt. Editable install points at /workspace/src, so
# in-container edits take effect immediately.
RUN uv sync --frozen

# Downloads the dataset on container start, then runs the given command.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
