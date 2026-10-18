#!/usr/bin/env bash
# Start the cached demo; no dependency installation or downloads.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
exec .venv/bin/python tools/manage_demo.py start --device cuda:0 "$@"
