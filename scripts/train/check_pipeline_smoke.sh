#!/usr/bin/env bash
set -euo pipefail

# Lightweight smoke checks for strict pipeline CLI contracts.
# This script does not launch training.

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}

echo "[Smoke] CLI --help checks"
set +e
"$PYTHON_BIN" omnigraph/train/train_projector.py --help >/tmp/omnigraph_help_stage2a.txt 2>/tmp/omnigraph_help_stage2a.err
S2A_RC=$?
"$PYTHON_BIN" omnigraph/train/train_stage2B.py --help >/tmp/omnigraph_help_stage2b.txt 2>/tmp/omnigraph_help_stage2b.err
S2B_RC=$?
"$PYTHON_BIN" omnigraph/train/train_stage3.py --help >/tmp/omnigraph_help_stage3.txt 2>/tmp/omnigraph_help_stage3.err
S3_RC=$?
"$PYTHON_BIN" scripts/eval/infer_gqa.py --help >/tmp/omnigraph_help_infergqa.txt 2>/tmp/omnigraph_help_infergqa.err
INF_RC=$?
set -e

echo "  train_projector.py --help rc=$S2A_RC"
echo "  train_stage2B.py --help rc=$S2B_RC"
echo "  train_stage3.py --help rc=$S3_RC"
echo "  infer_gqa.py --help rc=$INF_RC"

if [ "$S2A_RC" -ne 0 ] || [ "$S2B_RC" -ne 0 ] || [ "$S3_RC" -ne 0 ]; then
  echo "[Smoke] training CLI help failed. Likely missing runtime deps (e.g. pytorch_lightning)."
fi

echo "[Smoke] strict-stage CLI contract checks"
set +e
"$PYTHON_BIN" omnigraph/train/train_projector.py --scene_graphs a --regions b >/tmp/omnigraph_smoke_stage2a_missing.txt 2>/tmp/omnigraph_smoke_stage2a_missing.err
MISS_STAGE1_RC=$?
"$PYTHON_BIN" omnigraph/train/train_stage3.py --init_ckpt foo >/tmp/omnigraph_smoke_stage3_legacy.txt 2>/tmp/omnigraph_smoke_stage3_legacy.err
LEGACY_STAGE3_RC=$?
set -e

echo "  Stage2A missing --stage1_qformer_ckpt rc=$MISS_STAGE1_RC (expect non-zero)"
echo "  Stage3 legacy --init_ckpt rc=$LEGACY_STAGE3_RC (expect non-zero)"
echo "[Smoke] logs:"
echo "  /tmp/omnigraph_smoke_stage2a_missing.err"
echo "  /tmp/omnigraph_smoke_stage3_legacy.err"
