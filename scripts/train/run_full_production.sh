#!/usr/bin/env bash
set -euo pipefail

# Full production run:
# graph_bootstrap -> graph_refine -> multimodal_tune -> GQA eval
# Uses profile defaults (long schedule, optional round-2 on fail).

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
PROFILE=${PROFILE:-pro6000}
GPU=${GPU:-0}
DATA_CONFIG=${DATA_CONFIG:-configs/train/data_paths.server.json}

if [ ! -f "$DATA_CONFIG" ]; then
  echo "[FullRun] data config not found: $DATA_CONFIG"
  exit 1
fi

S2A_CHECKPOINT_EVERY_N_STEPS=${S2A_CHECKPOINT_EVERY_N_STEPS:-500}
S2B_CHECKPOINT_EVERY_N_STEPS=${S2B_CHECKPOINT_EVERY_N_STEPS:-500}
S3_CHECKPOINT_EVERY_N_STEPS=${S3_CHECKPOINT_EVERY_N_STEPS:-500}
RUN_ROUND2_ON_FAIL=${RUN_ROUND2_ON_FAIL:-1}

exec "$PYTHON_BIN" scripts/train/infra_launcher.py \
  --profile "$PROFILE" \
  --mode full \
  --gpu "$GPU" \
  --data-config "$DATA_CONFIG" \
  --preflight \
  --set AUTO_RESUME_ON_OOM=1 \
  --set AUTO_BATCH_RETRY_ON_OOM=1 \
  --set RUN_ROUND2_ON_FAIL="$RUN_ROUND2_ON_FAIL" \
  --set S2A_CHECKPOINT_EVERY_N_STEPS="$S2A_CHECKPOINT_EVERY_N_STEPS" \
  --set S2B_CHECKPOINT_EVERY_N_STEPS="$S2B_CHECKPOINT_EVERY_N_STEPS" \
  --set S3_CHECKPOINT_EVERY_N_STEPS="$S3_CHECKPOINT_EVERY_N_STEPS" \
  "$@"

