#!/usr/bin/env bash
set -euo pipefail

echo "[Deprecated] stage2.sh is no longer used in strict pipeline mode."
echo "Primary entry:"
echo "  scripts/train/run_4090_gqa_sprint.sh"
echo "  ./train_infra.sh  (standardized profile launcher)"
echo "Recommended semantic wrappers:"
echo "  ./train_graph_bootstrap.sh      (PIPELINE_MODE=graph_bootstrap)"
echo "  ./train_graph_refine.sh         (PIPELINE_MODE=graph_refine)"
echo "  ./train_multimodal_tune.sh      (PIPELINE_MODE=multimodal_tune)"
echo "Legacy wrappers (still supported):"
echo "  ./train_stage2A.sh   (compat)"
echo "  ./train_stage2B.sh   (compat)"
echo "  ./train_stage3.sh    (compat)"
exit 1
