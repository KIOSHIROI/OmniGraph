#!/usr/bin/env bash
set -euo pipefail

# Stage2A paper-sprint profile (VG-only, stronger GQA-like transfer)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-0}

SCENE_GRAPHS=${SCENE_GRAPHS:-data/vg/contents/sceneGraphs/scene_graphs.json}
REGIONS=${REGIONS:-data/vg/contents/regionDescriptions/region_descriptions.json}
STAGE1_CKPT=${STAGE1_CKPT:-graph_qformer_stage1.pt}
SAVE_DIR=${SAVE_DIR:-checkpoints_projector_vg/stage2A_paper}

"$PYTHON_BIN" omnigraph/train/train_projector.py \
  --scene_graphs "$SCENE_GRAPHS" \
  --regions "$REGIONS" \
  --stage1_qformer_ckpt "$STAGE1_CKPT" \
  --graph_qa_max_per_image 5 \
  --graph_qa_repeat 3 \
  --gpu "$GPU" \
  --batch_size 1 \
  --precision 16 \
  --max_length 256 \
  --num_workers 0 \
  --val_ratio 0.02 \
  --patience 16 \
  --min_delta 0.0005 \
  --lr 3e-5 \
  --max_steps 120000 \
  --val_check_interval 1000 \
  --save_dir "$SAVE_DIR"
