# Docker

Dev image: Python 3.12 · CUDA 13.2 · torch 2.12. Source is cloned from GitHub at
build time, so no local build context is needed.

**Prerequisites:** CUDA 13-capable NVIDIA driver +
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(for `--gpus all`).

## Build

```bash
docker build -f docker/Dockerfile -t pretrain docker/

# from a specific branch/tag (default: main)
docker build -f docker/Dockerfile --build-arg GIT_REF=v0.1.0 -t pretrain docker/
```

## Run

```bash
docker run --rm -it --gpus all --env-file .env \
  -v "$PWD/data:/workspace/data" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  pretrain
```

On start it downloads the dataset (`gongyisheng/openwebtext-exp` into
`/workspace`, idempotent) then drops into a shell with the venv on `PATH`:

```bash
uv run pytest tests/fast -n 6
uv run python scripts/train.py --config configs/gpt2_124m.yaml --no-wandb
```

Mount `-v "$PWD/data:/workspace/data"` so the download persists across
containers. For a private dataset, pass `-e HF_TOKEN=...`.

## Publish

```bash
docker login                                                  # use an access token
docker tag pretrain <dockerhub-user>/pretrain:latest
docker push <dockerhub-user>/pretrain:latest
```
