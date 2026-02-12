#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
PROFILE=${PROFILE:-4090}
MODE=${MODE:-full}
GPU=${GPU:-0}
OMNIGRAPH_HF_CACHE=${OMNIGRAPH_HF_CACHE:-/media/disk/02drive/13hias/.cache}
OMNIGRAPH_HF_ENDPOINT=${OMNIGRAPH_HF_ENDPOINT:-https://hf-mirror.com}
export OMNIGRAPH_HF_CACHE OMNIGRAPH_HF_ENDPOINT

exec "$PYTHON_BIN" "$WORKDIR/scripts/train/infra_launcher.py" \
  --profile "$PROFILE" \
  --mode "$MODE" \
  --gpu "$GPU" \
  "$@"
