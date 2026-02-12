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

TRAIN_DEPS_OK=1
if [ "$S2A_RC" -ne 0 ] || [ "$S2B_RC" -ne 0 ] || [ "$S3_RC" -ne 0 ]; then
  echo "[Smoke] training CLI help failed. Likely missing runtime deps (e.g. pytorch_lightning)."
  set +e
  if command -v rg >/dev/null 2>&1; then
    rg -qi "No module named 'pytorch_lightning'" /tmp/omnigraph_help_stage2a.err /tmp/omnigraph_help_stage2b.err /tmp/omnigraph_help_stage3.err
    HAS_PL_MISSING=$?
  else
    grep -Eqi "No module named 'pytorch_lightning'" /tmp/omnigraph_help_stage2a.err /tmp/omnigraph_help_stage2b.err /tmp/omnigraph_help_stage3.err
    HAS_PL_MISSING=$?
  fi
  set -e
  if [ "$HAS_PL_MISSING" -eq 0 ]; then
    TRAIN_DEPS_OK=0
  fi
fi

echo "[Smoke] strict-stage CLI contract checks"
set +e
"$PYTHON_BIN" omnigraph/train/train_projector.py \
  --scene_graphs a --regions b \
  --graph_tokenizer_type perceiver \
  --stage2A_bootstrap_mode no_stage1 \
  >/tmp/omnigraph_smoke_stage2a_no_stage1.txt 2>/tmp/omnigraph_smoke_stage2a_no_stage1.err
NO_STAGE1_RC=$?
"$PYTHON_BIN" omnigraph/train/train_projector.py \
  --scene_graphs a --regions b \
  --graph_tokenizer_type qformer \
  --stage2A_bootstrap_mode legacy_stage1 \
  >/tmp/omnigraph_smoke_stage2a_legacy_missing.txt 2>/tmp/omnigraph_smoke_stage2a_legacy_missing.err
LEGACY_MISS_STAGE1_RC=$?
"$PYTHON_BIN" omnigraph/train/train_stage3.py --init_ckpt foo >/tmp/omnigraph_smoke_stage3_legacy.txt 2>/tmp/omnigraph_smoke_stage3_legacy.err
LEGACY_STAGE3_RC=$?
set -e

if [ "$TRAIN_DEPS_OK" -eq 1 ]; then
  set +e
  if command -v rg >/dev/null 2>&1; then
    rg -qi "stage1_qformer_ckpt" /tmp/omnigraph_smoke_stage2a_no_stage1.err
    NO_STAGE1_HAS_STAGE1_MSG=$?
    rg -qi "stage1_qformer_ckpt" /tmp/omnigraph_smoke_stage2a_legacy_missing.err
    LEGACY_HAS_STAGE1_MSG=$?
  else
    grep -Eqi "stage1_qformer_ckpt" /tmp/omnigraph_smoke_stage2a_no_stage1.err
    NO_STAGE1_HAS_STAGE1_MSG=$?
    grep -Eqi "stage1_qformer_ckpt" /tmp/omnigraph_smoke_stage2a_legacy_missing.err
    LEGACY_HAS_STAGE1_MSG=$?
  fi
  set -e
else
  NO_STAGE1_HAS_STAGE1_MSG=-1
  LEGACY_HAS_STAGE1_MSG=-1
fi

echo "  Stage2A no_stage1+perceiver rc=$NO_STAGE1_RC (expect non-zero due fake data, but no stage1-ckpt error)"
echo "  Stage2A legacy_stage1+qformer (no ckpt) rc=$LEGACY_MISS_STAGE1_RC (expect non-zero with stage1-ckpt error)"
echo "  Stage3 legacy --init_ckpt rc=$LEGACY_STAGE3_RC (expect non-zero)"
if [ "$TRAIN_DEPS_OK" -eq 1 ]; then
  echo "  no_stage1 stderr contains stage1_qformer_ckpt? rc=$NO_STAGE1_HAS_STAGE1_MSG (expect 1)"
  echo "  legacy stderr contains stage1_qformer_ckpt? rc=$LEGACY_HAS_STAGE1_MSG (expect 0)"
else
  echo "  stage1-ckpt stderr assertions skipped (missing pytorch_lightning in env)."
fi
echo "[Smoke] logs:"
echo "  /tmp/omnigraph_smoke_stage2a_no_stage1.err"
echo "  /tmp/omnigraph_smoke_stage2a_legacy_missing.err"
echo "  /tmp/omnigraph_smoke_stage3_legacy.err"
