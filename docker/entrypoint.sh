#!/usr/bin/env bash
set -euo pipefail

uv run hf download gongyisheng/openwebtext-exp --repo-type dataset --local-dir .

exec "$@"
