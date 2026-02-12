#!/usr/bin/env bash
set -euo pipefail

# Stage-1 Graph Q-Former training launcher (retrain-friendly).
# Optional env overrides:
#   GPU=0
#   STAGE1_BATCH_SIZE=16
#   STAGE1_LR=1e-4
#   STAGE1_EPOCHS=20
#   STAGE1_GTM_MAX_LEN=512
#   STAGE1_TEXT_MAX_LEN=512
#   STAGE1_GTC_LOSS_WEIGHT=1.0
#   STAGE1_GTM_LOSS_WEIGHT=1.2
#   STAGE1_GTM_WARMUP_STEPS=1000
#   STAGE1_GTM_NEG_PER_POS=2
#   STAGE1_GTM_TEXT_DUP_THRESH=0.98
#   STAGE1_GTM_USE_HARD_NEG=1
#   STAGE1_GTM_LABEL_SMOOTHING=0.0
#   STAGE1_TEXT_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
#   STAGE1_AUX_NODE_TEXT_WEIGHT=0.0
#   STAGE1_AUX_NEIGH_TEXT_WEIGHT=0.0
#   STAGE1_DATASET_PATHS="data/train_instruct_graphmatch.json data/arxiv_pub_node_st_cot_link_mix.json"
#   STAGE1_NODE_TEXT_JSON_PATH=/path/to/node_text.json
#   STAGE1_NODE_FEAT_NPY_PATH=/path/to/node_feat.npy
#   OMNIGRAPH_HF_CACHE=/path/to/cache
#   OMNIGRAPH_HF_ENDPOINT=https://hf-mirror.com

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
GPU=${GPU:-0}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-16}
STAGE1_LR=${STAGE1_LR:-1e-4}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-20}
STAGE1_MAX_STEPS=${STAGE1_MAX_STEPS:--1}
STAGE1_GTM_MAX_LEN=${STAGE1_GTM_MAX_LEN:-512}
STAGE1_TEXT_MAX_LEN=${STAGE1_TEXT_MAX_LEN:-512}
STAGE1_GTM_TEXT_SOURCE=${STAGE1_GTM_TEXT_SOURCE:-qa}
STAGE1_GTC_LOSS_WEIGHT=${STAGE1_GTC_LOSS_WEIGHT:-1.0}
STAGE1_GTM_LOSS_WEIGHT=${STAGE1_GTM_LOSS_WEIGHT:-1.2}
STAGE1_GTM_WARMUP_STEPS=${STAGE1_GTM_WARMUP_STEPS:-1000}
STAGE1_GTM_NEG_PER_POS=${STAGE1_GTM_NEG_PER_POS:-2}
STAGE1_GTM_TEXT_DUP_THRESH=${STAGE1_GTM_TEXT_DUP_THRESH:-0.98}
STAGE1_GTM_USE_HARD_NEG=${STAGE1_GTM_USE_HARD_NEG:-1}
STAGE1_GTM_LABEL_SMOOTHING=${STAGE1_GTM_LABEL_SMOOTHING:-0.0}
STAGE1_ENABLE_EARLY_STOP=${STAGE1_ENABLE_EARLY_STOP:-1}
STAGE1_EARLY_STOP_PATIENCE=${STAGE1_EARLY_STOP_PATIENCE:-2}
STAGE1_EARLY_STOP_MIN_DELTA=${STAGE1_EARLY_STOP_MIN_DELTA:-0.001}
STAGE1_EARLY_STOP_MODE=${STAGE1_EARLY_STOP_MODE:-min}
STAGE1_TEXT_MODEL_NAME=${STAGE1_TEXT_MODEL_NAME:-sentence-transformers/all-mpnet-base-v2}
STAGE1_AUX_NODE_TEXT_WEIGHT=${STAGE1_AUX_NODE_TEXT_WEIGHT:-0.0}
STAGE1_AUX_NEIGH_TEXT_WEIGHT=${STAGE1_AUX_NEIGH_TEXT_WEIGHT:-0.0}
STAGE1_NODE_TEXT_JSON_PATH=${STAGE1_NODE_TEXT_JSON_PATH:-}
STAGE1_NODE_FEAT_NPY_PATH=${STAGE1_NODE_FEAT_NPY_PATH:-}
STAGE1_DATASET_PATHS=${STAGE1_DATASET_PATHS:-"data/train_instruct_graphmatch.json data/arxiv_pub_node_st_cot_link_mix.json"}
STAGE1_EXPORT_PATH=${STAGE1_EXPORT_PATH:-$WORKDIR/graph_qformer_stage1.pt}

read -r -a _DATASET_PATHS_ARR <<< "$STAGE1_DATASET_PATHS"
if [ "${#_DATASET_PATHS_ARR[@]}" -eq 0 ]; then
  echo "[Stage1] empty STAGE1_DATASET_PATHS"
  exit 1
fi

CMD=(
  "$PYTHON_BIN" omnigraph/train/train_graph_qfromer.py
  --batch_size "$STAGE1_BATCH_SIZE"
  --lr "$STAGE1_LR"
  --epochs "$STAGE1_EPOCHS"
  --max_steps "$STAGE1_MAX_STEPS"
  --gtm_max_len "$STAGE1_GTM_MAX_LEN"
  --gtm_text_source "$STAGE1_GTM_TEXT_SOURCE"
  --gtc_loss_weight "$STAGE1_GTC_LOSS_WEIGHT"
  --gtm_loss_weight "$STAGE1_GTM_LOSS_WEIGHT"
  --gtm_warmup_steps "$STAGE1_GTM_WARMUP_STEPS"
  --gtm_negatives_per_pos "$STAGE1_GTM_NEG_PER_POS"
  --gtm_text_dup_thresh "$STAGE1_GTM_TEXT_DUP_THRESH"
  --gtm_use_hard_negative "$STAGE1_GTM_USE_HARD_NEG"
  --gtm_label_smoothing "$STAGE1_GTM_LABEL_SMOOTHING"
  --text_max_len "$STAGE1_TEXT_MAX_LEN"
  --enable_early_stop "$STAGE1_ENABLE_EARLY_STOP"
  --early_stop_patience "$STAGE1_EARLY_STOP_PATIENCE"
  --early_stop_min_delta "$STAGE1_EARLY_STOP_MIN_DELTA"
  --early_stop_mode "$STAGE1_EARLY_STOP_MODE"
  --dataset_paths "${_DATASET_PATHS_ARR[@]}"
  --text_model_name "$STAGE1_TEXT_MODEL_NAME"
  --aux_node_text_weight "$STAGE1_AUX_NODE_TEXT_WEIGHT"
  --aux_neigh_text_weight "$STAGE1_AUX_NEIGH_TEXT_WEIGHT"
)

if [ -n "$STAGE1_NODE_TEXT_JSON_PATH" ]; then
  CMD+=(--node_text_json_path "$STAGE1_NODE_TEXT_JSON_PATH")
fi
if [ -n "$STAGE1_NODE_FEAT_NPY_PATH" ]; then
  CMD+=(--node_feat_npy_path "$STAGE1_NODE_FEAT_NPY_PATH")
fi
if [ -n "${OMNIGRAPH_HF_ENDPOINT:-}" ]; then
  CMD+=(--hf_endpoint "$OMNIGRAPH_HF_ENDPOINT")
fi
if [ -n "${OMNIGRAPH_HF_CACHE:-}" ]; then
  CMD+=(--hf_cache_dir "$OMNIGRAPH_HF_CACHE")
fi

if [ "$GPU" -ge 0 ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
fi

echo "[Stage1] start GPU=${GPU} bs=${STAGE1_BATCH_SIZE} lr=${STAGE1_LR} epochs=${STAGE1_EPOCHS} max_steps=${STAGE1_MAX_STEPS} early_stop=${STAGE1_ENABLE_EARLY_STOP} patience=${STAGE1_EARLY_STOP_PATIENCE} gtm_max_len=${STAGE1_GTM_MAX_LEN} gtm_text_source=${STAGE1_GTM_TEXT_SOURCE} gtm_neg_per_pos=${STAGE1_GTM_NEG_PER_POS} gtm_warmup=${STAGE1_GTM_WARMUP_STEPS}"
"${CMD[@]}"

if [ ! -f "$WORKDIR/graph_qformer_stage1.pt" ]; then
  echo "[Stage1] training finished but graph_qformer_stage1.pt not found."
  exit 2
fi

if [ "$STAGE1_EXPORT_PATH" != "$WORKDIR/graph_qformer_stage1.pt" ]; then
  mkdir -p "$(dirname "$STAGE1_EXPORT_PATH")"
  cp -f "$WORKDIR/graph_qformer_stage1.pt" "$STAGE1_EXPORT_PATH"
  echo "[Stage1] copied: $STAGE1_EXPORT_PATH"
fi
echo "[Stage1] produced: $WORKDIR/graph_qformer_stage1.pt"
