#!/usr/bin/env sh
set -eu

# Example usage for GQA inference
# Adjust paths as needed

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
GPU=${GPU:-0}

QUESTIONS_JSON=${QUESTIONS_JSON:-$REPO/data/gqa/contents/questions/val_balanced_questions.json}
QUESTIONS_JSONL=${QUESTIONS_JSONL:-$REPO/data/gqa/val_balanced.jsonl}
SCENEGRAPHS_RAW=${SCENEGRAPHS_RAW:-$REPO/data/gqa/contents/sceneGraphs/val_sceneGraphs.json}
SCENEGRAPHS_VG=${SCENEGRAPHS_VG:-$REPO/data/gqa/scene_graphs_val_vg.json}
IMAGE_ROOT=${IMAGE_ROOT:-$REPO/data/gqa/contents/images}
CKPT=${CKPT:-$REPO/checkpoints_stage3_paper/omnigraph_stage3_state_dict.pt}
PRED=${PRED:-$REPO/data/gqa/pred_val_balanced_paper.jsonl}
EVAL_TXT=${EVAL_TXT:-$REPO/data/gqa/eval_val_balanced_paper.txt}

if [ ! -f "$SCENEGRAPHS_VG" ]; then
  $PYTHON_BIN "$REPO/scripts/data_prep/convert_gqa_scene_graphs.py" \
    --input "$SCENEGRAPHS_RAW" \
    --output "$SCENEGRAPHS_VG"
fi

if [ ! -f "$QUESTIONS_JSONL" ]; then
  $PYTHON_BIN "$REPO/scripts/data_prep/convert_gqa_questions.py" \
    --input "$QUESTIONS_JSON" \
    --output "$QUESTIONS_JSONL" \
    --image_root "$IMAGE_ROOT"
fi

$PYTHON_BIN "$REPO/scripts/eval/infer_gqa.py" \
  --questions "$QUESTIONS_JSONL" \
  --scene_graphs "$SCENEGRAPHS_VG" \
  --image_root "$IMAGE_ROOT" \
  --ckpt "$CKPT" \
  --output "$PRED" \
  --batch_size 1 \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu "$GPU" \
  --max_samples 0 \
  --log_every 500

$PYTHON_BIN "$REPO/scripts/eval/eval_gqa_accuracy.py" \
  --gt "$QUESTIONS_JSONL" \
  --pred "$PRED" | tee "$EVAL_TXT"
