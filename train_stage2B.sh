#!/usr/bin/env bash
set -euo pipefail

# Stage2B paper-sprint profile (VG-only, stronger GQA-like transfer)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-0}

SCENE_GRAPHS=${SCENE_GRAPHS:-data/vg/contents/sceneGraphs/scene_graphs.json}
REGIONS=${REGIONS:-data/vg/contents/regionDescriptions/region_descriptions.json}

STAGE2A_DIR=${STAGE2A_DIR:-checkpoints_projector_vg/stage2A_paper}
STAGE2A_META=${STAGE2A_META:-$STAGE2A_DIR/stage2A_meta.json}
STAGE2A_FALLBACK=${STAGE2A_FALLBACK:-$STAGE2A_DIR/last.ckpt}
STAGE2A_CKPT=${STAGE2A_CKPT:-$("$PYTHON_BIN" scripts/train/select_best_ckpt.py --meta "$STAGE2A_META" --fallback "$STAGE2A_FALLBACK")}

SAVE_DIR=${SAVE_DIR:-checkpoints_projector_vg/stage2B_paper}

echo "[Stage2B] using Stage2A ckpt: $STAGE2A_CKPT"

"$PYTHON_BIN" omnigraph/train/train_stage2B.py \
  --scene_graphs "$SCENE_GRAPHS" \
  --regions "$REGIONS" \
  --stage2A_ckpt "$STAGE2A_CKPT" \
  --graph_qa_max_per_image 6 \
  --graph_qa_repeat 4 \
  --gpu "$GPU" \
  --batch_size 3 \
  --precision 16 \
  --max_length 256 \
  --lr 1.2e-5 \
  --max_steps 60000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 16 \
  --min_delta 0.0005 \
  --save_dir "$SAVE_DIR"
