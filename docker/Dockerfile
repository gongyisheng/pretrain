# syntax=docker/dockerfile:1
#
# Dev image: Python 3.12 / CUDA 13.2 / torch 2.12. Source and deps are handled by
# entrypoint.sh at container start, so the build context is just docker/:
#   docker build -f docker/Dockerfile -t pretrain docker/
#   docker run -e GIT_REF=v0.1.0 pretrain
#
# devel base for nvcc + cublasLt, needed by _bpe_engine and cublaslt/_C.
FROM nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# python3-dev supplies Python.h for the pybind11 build.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:/opt/venv/bin:$PATH \
    # RTX 5090 (Blackwell, sm_120) only — no fat binary.
    TORCH_CUDA_ARCH_LIST=12.0 \
    # Venv outside /workspace so a bind-mount can't shadow it.
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON=3.12 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# ENV (not ARG) so they can be overridden with `docker run -e GIT_REF=...`.
ENV GIT_REPO=https://github.com/gongyisheng/pretrain.git \
    GIT_REF=main

WORKDIR /workspace

# Clones the repo, runs `uv sync --frozen`, downloads the dataset, then runs CMD.
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
