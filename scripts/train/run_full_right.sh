#!/usr/bin/env bash
set -euo pipefail

# Lightweight end-to-end run:
# graph_bootstrap -> graph_refine -> multimodal_tune -> GQA eval
# Intended for fast verification with lower GPU budget.

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
PROFILE=${PROFILE:-pro6000}
GPU=${GPU:-0}
DATA_CONFIG=${DATA_CONFIG:-configs/train/data_paths.server.json}

if [ ! -f "$DATA_CONFIG" ]; then
  echo "[LightRun] data config not found: $DATA_CONFIG"
  exit 1
fi

# Keep runtime/GPU cost low while preserving full pipeline behavior.
S2A_MAX_STEPS=${S2A_MAX_STEPS:-1500}
S2B_MAX_STEPS=${S2B_MAX_STEPS:-1200}
S3_MAX_STEPS=${S3_MAX_STEPS:-900}
S2A_VAL_CHECK_INTERVAL=${S2A_VAL_CHECK_INTERVAL:-300}
S2B_VAL_CHECK_INTERVAL=${S2B_VAL_CHECK_INTERVAL:-300}
S3_VAL_CHECK_INTERVAL=${S3_VAL_CHECK_INTERVAL:-300}
S2A_CHECKPOINT_EVERY_N_STEPS=${S2A_CHECKPOINT_EVERY_N_STEPS:-100}
S2B_CHECKPOINT_EVERY_N_STEPS=${S2B_CHECKPOINT_EVERY_N_STEPS:-100}
S3_CHECKPOINT_EVERY_N_STEPS=${S3_CHECKPOINT_EVERY_N_STEPS:-100}

exec "$PYTHON_BIN" scripts/train/infra_launcher.py \
  --profile "$PROFILE" \
  --mode full \
  --gpu "$GPU" \
  --data-config "$DATA_CONFIG" \
  --preflight \
  --set AUTO_RESUME_ON_OOM=1 \
  --set AUTO_BATCH_RETRY_ON_OOM=1 \
  --set RUN_ROUND2_ON_FAIL=0 \
  --set S2A_MAX_STEPS="$S2A_MAX_STEPS" \
  --set S2B_MAX_STEPS="$S2B_MAX_STEPS" \
  --set S3_MAX_STEPS="$S3_MAX_STEPS" \
  --set S2A_VAL_CHECK_INTERVAL="$S2A_VAL_CHECK_INTERVAL" \
  --set S2B_VAL_CHECK_INTERVAL="$S2B_VAL_CHECK_INTERVAL" \
  --set S3_VAL_CHECK_INTERVAL="$S3_VAL_CHECK_INTERVAL" \
  --set S2A_CHECKPOINT_EVERY_N_STEPS="$S2A_CHECKPOINT_EVERY_N_STEPS" \
  --set S2B_CHECKPOINT_EVERY_N_STEPS="$S2B_CHECKPOINT_EVERY_N_STEPS" \
  --set S3_CHECKPOINT_EVERY_N_STEPS="$S3_CHECKPOINT_EVERY_N_STEPS" \
  "$@"

