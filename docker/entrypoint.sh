#!/usr/bin/env bash
set -euo pipefail

cd /workspace

if [ -d pretrain/.git ]; then
    git -C pretrain fetch --depth 1 origin "$GIT_REF"
    git -C pretrain reset --hard FETCH_HEAD
else
    git clone --depth 1 --branch "$GIT_REF" "$GIT_REPO" pretrain
fi

cd /workspace/pretrain
uv sync --frozen
uv run hf download gongyisheng/openwebtext-exp --repo-type dataset --local-dir .

exec "$@"
