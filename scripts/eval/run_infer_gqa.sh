#!/usr/bin/env sh
set -eu

# Example usage for GQA inference
# Adjust paths as needed

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
GPU=${GPU:-0}
ISOLATE_GPU=${ISOLATE_GPU:-1}
ORIG_GPU=${GPU}
if [ "$ISOLATE_GPU" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
  GPU=0
fi
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

DEFAULT_LLM_3B="Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL=${LLM_MODEL:-$DEFAULT_LLM_3B}
VISION_MODEL=${VISION_MODEL:-Salesforce/blip2-flan-t5-xl}
NODE_ENCODER_TYPE=${NODE_ENCODER_TYPE:-hybrid}
NODE_ENCODER_ALPHA_INIT=${NODE_ENCODER_ALPHA_INIT:-0.3}
NODE_ENCODER_OUT_DIM=${NODE_ENCODER_OUT_DIM:-128}

BATCH_SIZE=${BATCH_SIZE:-1}
ALLOW_MISSING_MODALITIES=${ALLOW_MISSING_MODALITIES:-1}
LOW_RAM_EVAL=${LOW_RAM_EVAL:-1}
STREAM_QUESTIONS=${STREAM_QUESTIONS:-1}
if [ "$LOW_RAM_EVAL" = "1" ]; then
  : "${NUM_WORKERS:=0}"
  : "${PREFETCH_FACTOR:=2}"
  : "${PERSISTENT_WORKERS:=0}"
else
  : "${NUM_WORKERS:=4}"
  : "${PREFETCH_FACTOR:=4}"
  : "${PERSISTENT_WORKERS:=1}"
fi
MAX_SAMPLES=${MAX_SAMPLES:-0}
LOG_EVERY=${LOG_EVERY:-500}

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
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --node_encoder_type "$NODE_ENCODER_TYPE" \
  --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
  --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
  --output "$PRED" \
  --batch_size "$BATCH_SIZE" \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu "$GPU" \
  $( [ "$ALLOW_MISSING_MODALITIES" = "1" ] && echo "--allow_missing_modalities" ) \
  $( [ "$STREAM_QUESTIONS" = "1" ] && echo "--stream_questions" ) \
  --num_workers "$NUM_WORKERS" \
  --prefetch_factor "$PREFETCH_FACTOR" \
  $( [ "$PERSISTENT_WORKERS" = "1" ] && echo "--persistent_workers" ) \
  --max_samples "$MAX_SAMPLES" \
  --log_every "$LOG_EVERY"

$PYTHON_BIN "$REPO/scripts/eval/eval_gqa_accuracy.py" \
  --gt "$QUESTIONS_JSONL" \
  --pred "$PRED" | tee "$EVAL_TXT"
