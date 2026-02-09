#!/usr/bin/env bash
set -euo pipefail

# Stage3 paper-sprint profile (VG-only, stronger GQA-like transfer)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-0}

SCENE_GRAPHS=${SCENE_GRAPHS:-data/vg/contents/sceneGraphs/scene_graphs.json}
REGIONS=${REGIONS:-data/vg/contents/regionDescriptions/region_descriptions.json}
IMAGE_ROOT=${IMAGE_ROOT:-data/vg}

STAGE2B_DIR=${STAGE2B_DIR:-checkpoints_projector_vg/stage2B_paper}
STAGE2B_META=${STAGE2B_META:-$STAGE2B_DIR/stage2B_meta.json}
STAGE2B_FALLBACK=${STAGE2B_FALLBACK:-$STAGE2B_DIR/last.ckpt}
STAGE2B_CKPT=${STAGE2B_CKPT:-$("$PYTHON_BIN" scripts/train/select_best_ckpt.py --meta "$STAGE2B_META" --fallback "$STAGE2B_FALLBACK")}

SAVE_DIR=${SAVE_DIR:-checkpoints_stage3_paper}

echo "[Stage3] using Stage2B ckpt: $STAGE2B_CKPT"

"$PYTHON_BIN" omnigraph/train/train_stage3.py \
  --scene_graphs "$SCENE_GRAPHS" \
  --regions "$REGIONS" \
  --image_root "$IMAGE_ROOT" \
  --stage2B_ckpt "$STAGE2B_CKPT" \
  --graph_qa_max_per_image 4 \
  --graph_qa_repeat 2 \
  --gpu "$GPU" \
  --batch_size 2 \
  --precision 16 \
  --max_length 256 \
  --lr 2e-5 \
  --max_steps 50000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 14 \
  --min_delta 0.0005 \
  --save_dir "$SAVE_DIR"
