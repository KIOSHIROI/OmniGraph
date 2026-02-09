#!/usr/bin/env bash
set -euo pipefail

echo "[Deprecated] stage2.sh is no longer used in strict pipeline mode."
echo "Use the new training chain instead:"
echo "  1) ./train_stage1.sh"
echo "  2) ./train_stage2A.sh"
echo "  3) ./train_stage2B.sh"
echo "  4) ./train_stage3.sh"
exit 1
