# Docker

Dev image: Python 3.12 · CUDA 13.2 · torch 2.12. Source is cloned from GitHub at
build time, so no local build context is needed.

**Prerequisites:** CUDA 13-capable NVIDIA driver +
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(for `--gpus all`).

## Build

```bash
docker build -f docker/Dockerfile -t pretrain_sm120 docker/

# from a specific branch/tag (default: main)
docker build -f docker/Dockerfile --build-arg GIT_REF=v0.1.0 -t pretrain_sm120 docker/
```

## Run

```bash
docker run --rm -it --gpus all --env-file .env \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  pretrain_sm120
```

Drops into a shell with the venv on `PATH`:

```bash
uv run pytest tests/fast -n 6
uv run python scripts/train.py --config configs/gpt2_124m.yaml --no-wandb
```

## Notes

- Image reflects the **pushed** repo, not local uncommitted changes.
- `-march=native` pins the BPE extension to this host's CPU — build and run on the same machine.
- Rebuild re-clones when the branch head moves (`ADD ...commits/${GIT_REF}` cache-bust).
