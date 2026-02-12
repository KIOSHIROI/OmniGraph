#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
PROFILE=${PROFILE:-4090}
MODE=${MODE:-full}
GPU=${GPU:-0}

exec "$PYTHON_BIN" "$WORKDIR/scripts/train/infra_launcher.py" \
  --profile "$PROFILE" \
  --mode "$MODE" \
  --gpu "$GPU" \
  "$@"
