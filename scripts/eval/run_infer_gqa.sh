#!/usr/bin/env sh
set -eu

# Example usage for GQA inference
# Adjust paths as needed

PYTHON_BIN=${PYTHON_BIN:-python}

QUESTIONS_JSON=data/gqa/contents/questions/val_balanced_questions.json
QUESTIONS_JSONL=data/gqa/val_balanced.jsonl
SCENEGRAPHS_RAW=data/gqa/contents/sceneGraphs/val_sceneGraphs.json
SCENEGRAPHS_VG=data/gqa/scene_graphs_val_vg.json
IMAGE_ROOT=data/gqa/contents/images

if [ ! -f "$SCENEGRAPHS_VG" ]; then
  $PYTHON_BIN scripts/data_prep/convert_gqa_scene_graphs.py \
    --input "$SCENEGRAPHS_RAW" \
    --output "$SCENEGRAPHS_VG"
fi

if [ ! -f "$QUESTIONS_JSONL" ]; then
  $PYTHON_BIN scripts/data_prep/convert_gqa_questions.py \
    --input "$QUESTIONS_JSON" \
    --output "$QUESTIONS_JSONL" \
    --image_root "$IMAGE_ROOT"
fi

$PYTHON_BIN scripts/eval/infer_gqa.py \
  --questions "$QUESTIONS_JSONL" \
  --scene_graphs "$SCENEGRAPHS_VG" \
  --image_root "$IMAGE_ROOT" \
  --ckpt checkpoints_stage3/omnigraph_stage3_state_dict.pt \
  --output data/gqa/pred_val_balanced.jsonl \
  --batch_size 1 \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu 0 \
  --max_samples 200 \
  --log_every 50
