#!/usr/bin/env bash
set -euo pipefail

echo "[Deprecated] stage2.sh is no longer used in strict pipeline mode."
echo "Primary entry:"
echo "  scripts/train/run_4090_gqa_sprint.sh"
echo "  ./train_infra.sh  (standardized profile launcher)"
echo "Single-stage wrappers:"
echo "  ./train_stage2A.sh   (PIPELINE_MODE=stage2a)"
echo "  ./train_stage2B.sh   (PIPELINE_MODE=stage2b)"
echo "  ./train_stage3.sh    (PIPELINE_MODE=stage3)"
exit 1
