#!/usr/bin/env bash
set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-"$WORKDIR"}
GPU=${GPU:-0}
AUTO_BUILD_MANIFESTS=${AUTO_BUILD_MANIFESTS:-1}
MANIFEST_BUILDER=${MANIFEST_BUILDER:-"$REPO/scripts/data_prep/build_bootstrap_manifests.py"}

# Data manifests
V2G_TRAIN_MANIFEST=${V2G_TRAIN_MANIFEST:-"$REPO/data/vg/v2g_train_manifest.jsonl"}
V2G_VAL_MANIFEST=${V2G_VAL_MANIFEST:-""}
VT_MANIFEST_R1=${VT_MANIFEST_R1:-"$REPO/data/vt/vt_manifest_round1.jsonl"}
VT_MANIFEST_R2=${VT_MANIFEST_R2:-"$VT_MANIFEST_R1"}
VG_SCENE_GRAPHS=${VG_SCENE_GRAPHS:-"$REPO/data/vg/contents/sceneGraphs/scene_graphs.json"}
VG_REGIONS=${VG_REGIONS:-"$REPO/data/vg/contents/regionDescriptions/region_descriptions.json"}
V2G_IMAGE_ROOT=${V2G_IMAGE_ROOT:-"$REPO/data/vg"}
COCO_IMAGE_ROOT=${COCO_IMAGE_ROOT:-}
VT_IMAGE_ROOT=${VT_IMAGE_ROOT:-}
if [ -z "$VT_IMAGE_ROOT" ]; then
  if [ -n "$COCO_IMAGE_ROOT" ]; then
    VT_IMAGE_ROOT="$COCO_IMAGE_ROOT"
  else
    VT_IMAGE_ROOT="$REPO/data"
  fi
fi
if [ -z "$COCO_IMAGE_ROOT" ]; then
  COCO_IMAGE_ROOT="$VT_IMAGE_ROOT"
fi

# Output dirs
V2G_DIR=${V2G_DIR:-"$REPO/checkpoints_v2g/v2g_round1"}
PSEUDO_DIR=${PSEUDO_DIR:-"$REPO/data/pseudo_graphs"}
ROUND1_TAG=${ROUND1_TAG:-"r1"}
ROUND2_TAG=${ROUND2_TAG:-"r2"}

ROUND1_RAW=${ROUND1_RAW:-"$PSEUDO_DIR/pseudo_raw_${ROUND1_TAG}.jsonl"}
ROUND1_FILTERED=${ROUND1_FILTERED:-"$PSEUDO_DIR/pseudo_scene_graphs_round1.json"}
ROUND1_REPORT=${ROUND1_REPORT:-"$PSEUDO_DIR/pseudo_filter_report_round1.json"}

ROUND2_RAW=${ROUND2_RAW:-"$PSEUDO_DIR/pseudo_raw_${ROUND2_TAG}.jsonl"}
ROUND2_FILTERED=${ROUND2_FILTERED:-"$PSEUDO_DIR/pseudo_scene_graphs_round2.json"}
ROUND2_REPORT=${ROUND2_REPORT:-"$PSEUDO_DIR/pseudo_filter_report_round2.json"}

# Training dirs for OmniGraph
ROUND1_STAGE2A_DIR=${ROUND1_STAGE2A_DIR:-"$REPO/checkpoints_projector_vg/graph_bootstrap_v2g_round1"}
ROUND1_STAGE2B_DIR=${ROUND1_STAGE2B_DIR:-"$REPO/checkpoints_projector_vg/graph_refine_v2g_round1"}
ROUND1_STAGE3_DIR=${ROUND1_STAGE3_DIR:-"$REPO/checkpoints_multimodal_tune_v2g_round1"}
ROUND2_STAGE2B_DIR=${ROUND2_STAGE2B_DIR:-"$REPO/checkpoints_projector_vg/graph_refine_v2g_round2"}
ROUND2_STAGE3_DIR=${ROUND2_STAGE3_DIR:-"$REPO/checkpoints_multimodal_tune_v2g_round2"}

# V2G defaults (4090-friendly)
V2G_MODEL_NAME=${V2G_MODEL_NAME:-Salesforce/blip2-flan-t5-xl}
V2G_TORCH_DTYPE=${V2G_TORCH_DTYPE:-bfloat16}
V2G_PRECISION=${V2G_PRECISION:-16-mixed}
V2G_BATCH_SIZE=${V2G_BATCH_SIZE:-1}
V2G_NUM_WORKERS=${V2G_NUM_WORKERS:-4}
V2G_ACCUM_GRAD_BATCHES=${V2G_ACCUM_GRAD_BATCHES:-8}
V2G_MAX_LENGTH=${V2G_MAX_LENGTH:-384}
V2G_MAX_NEW_TOKENS=${V2G_MAX_NEW_TOKENS:-256}
V2G_MAX_STEPS=${V2G_MAX_STEPS:-30000}
V2G_VAL_CHECK_INTERVAL=${V2G_VAL_CHECK_INTERVAL:-1000}
V2G_PATIENCE=${V2G_PATIENCE:-12}
V2G_MIN_DELTA=${V2G_MIN_DELTA:-0.0005}
V2G_LR=${V2G_LR:-1e-4}
V2G_WEIGHT_DECAY=${V2G_WEIGHT_DECAY:-0.01}
V2G_SAVE_TOP_K=${V2G_SAVE_TOP_K:-0}
V2G_SAVE_LAST_CKPT=${V2G_SAVE_LAST_CKPT:-0}
V2G_CKPT_SAVE_WEIGHTS_ONLY=${V2G_CKPT_SAVE_WEIGHTS_ONLY:-1}
V2G_EXPORT_TRAINABLE_ONLY=${V2G_EXPORT_TRAINABLE_ONLY:-1}

NUM_CANDIDATES=${NUM_CANDIDATES:-3}
SYNTH_BATCH_SIZE=${SYNTH_BATCH_SIZE:-1}
SYNTH_MAX_NEW_TOKENS=${SYNTH_MAX_NEW_TOKENS:-256}

FILTER_MIN_NODES=${FILTER_MIN_NODES:-2}
FILTER_MAX_NODES=${FILTER_MAX_NODES:-36}
FILTER_MIN_RELS=${FILTER_MIN_RELS:-1}
FILTER_MAX_RELS=${FILTER_MAX_RELS:-72}
FILTER_LOGPROB_THRESH=${FILTER_LOGPROB_THRESH:--1.10}
FILTER_TEXT_SIM_THRESH=${FILTER_TEXT_SIM_THRESH:-0.55}
FILTER_VISION_SIM_THRESH=${FILTER_VISION_SIM_THRESH:-0.25}
FILTER_SCORE_THRESH=${FILTER_SCORE_THRESH:-0.72}

mkdir -p "$V2G_DIR" "$PSEUDO_DIR"

normalize_precision_for_dtype() {
  local precision="${1:-16-mixed}"
  local dtype="${2:-bfloat16}"
  local p
  local d
  p="$(printf "%s" "$precision" | tr '[:upper:]' '[:lower:]')"
  d="$(printf "%s" "$dtype" | tr '[:upper:]' '[:lower:]')"
  if { [ "$d" = "bfloat16" ] || [ "$d" = "bf16" ]; } && { [ "$p" = "16" ] || [ "$p" = "16-mixed" ]; }; then
    echo "bf16-mixed"
    return
  fi
  if { [ "$d" = "float16" ] || [ "$d" = "fp16" ] || [ "$d" = "half" ]; } && { [ "$p" = "bf16" ] || [ "$p" = "bf16-mixed" ]; }; then
    echo "16-mixed"
    return
  fi
  echo "$precision"
}

V2G_PRECISION_ORIG="$V2G_PRECISION"
V2G_PRECISION="$(normalize_precision_for_dtype "$V2G_PRECISION" "$V2G_TORCH_DTYPE")"
if [ "$V2G_PRECISION" != "$V2G_PRECISION_ORIG" ]; then
  echo "[ConfigFix] V2G precision adjusted: $V2G_PRECISION_ORIG + $V2G_TORCH_DTYPE -> $V2G_PRECISION"
fi

if [ "$AUTO_BUILD_MANIFESTS" = "1" ]; then
  if [ ! -f "$V2G_TRAIN_MANIFEST" ] || [ ! -f "$VT_MANIFEST_R1" ] || [ ! -f "$VT_MANIFEST_R2" ]; then
    echo "[Manifest] missing files detected, auto-building manifests..."
    "$PYTHON_BIN" "$MANIFEST_BUILDER" \
      --coco_root "$COCO_IMAGE_ROOT" \
      --vg_scene_graphs "$VG_SCENE_GRAPHS" \
      --vg_regions "$VG_REGIONS" \
      --vg_image_root "$V2G_IMAGE_ROOT" \
      --out_vt_round1 "$VT_MANIFEST_R1" \
      --out_vt_round2 "$VT_MANIFEST_R2" \
      --out_v2g_train "$V2G_TRAIN_MANIFEST"
  fi
fi

if [ ! -f "$V2G_TRAIN_MANIFEST" ]; then
  echo "[V2G] missing train manifest: $V2G_TRAIN_MANIFEST"
  exit 1
fi
if [ ! -f "$VT_MANIFEST_R1" ]; then
  echo "[V2G] missing VT manifest round1: $VT_MANIFEST_R1"
  exit 1
fi
if [ ! -f "$VT_MANIFEST_R2" ]; then
  echo "[V2G] missing VT manifest round2: $VT_MANIFEST_R2"
  exit 1
fi

echo "[Config] GPU=$GPU"
echo "[Config] V2G train_manifest=$V2G_TRAIN_MANIFEST image_root=$V2G_IMAGE_ROOT"
echo "[Config] VT round1_manifest=$VT_MANIFEST_R1 round2_manifest=$VT_MANIFEST_R2 image_root=$VT_IMAGE_ROOT"
echo "[Config] outputs: round1_filtered=$ROUND1_FILTERED round2_filtered=$ROUND2_FILTERED"

SELECT_CKPT="$REPO/scripts/train/select_best_ckpt.py"
RUN_4090="$REPO/scripts/train/run_4090_gqa_sprint.sh"

# -------------------------
# Round1: V2G train + pseudo + full pipeline
# -------------------------
echo "[Round1] train V2G synthesizer"
V2G_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_v2g_synthesizer.py" \
  --train_manifest "$V2G_TRAIN_MANIFEST" \
  --image_root "$V2G_IMAGE_ROOT" \
  --model_name "$V2G_MODEL_NAME" \
  --torch_dtype "$V2G_TORCH_DTYPE" \
  --precision "$V2G_PRECISION" \
  --batch_size "$V2G_BATCH_SIZE" \
  --num_workers "$V2G_NUM_WORKERS" \
  --accumulate_grad_batches "$V2G_ACCUM_GRAD_BATCHES" \
  --max_length "$V2G_MAX_LENGTH" \
  --max_new_tokens "$V2G_MAX_NEW_TOKENS" \
  --lr "$V2G_LR" \
  --weight_decay "$V2G_WEIGHT_DECAY" \
  --max_steps "$V2G_MAX_STEPS" \
  --val_check_interval "$V2G_VAL_CHECK_INTERVAL" \
  --patience "$V2G_PATIENCE" \
  --min_delta "$V2G_MIN_DELTA" \
  --save_top_k "$V2G_SAVE_TOP_K" \
  --save_last_ckpt "$V2G_SAVE_LAST_CKPT" \
  --checkpoint_save_weights_only "$V2G_CKPT_SAVE_WEIGHTS_ONLY" \
  --export_trainable_only "$V2G_EXPORT_TRAINABLE_ONLY" \
  --gpu "$GPU" \
  --save_dir "$V2G_DIR")
if [ -n "${V2G_VAL_MANIFEST}" ]; then
  V2G_CMD+=(--val_manifest "$V2G_VAL_MANIFEST")
fi
"${V2G_CMD[@]}"

V2G_CKPT="$V2G_DIR/v2g_state_dict.pt"
if [ ! -f "$V2G_CKPT" ]; then
  echo "[Round1] v2g export missing: $V2G_CKPT"
  exit 1
fi

echo "[Round1] synthesize pseudo graphs"
"$PYTHON_BIN" "$REPO/scripts/data_prep/synthesize_pseudo_scene_graphs.py" \
  --vt_manifest "$VT_MANIFEST_R1" \
  --image_root "$VT_IMAGE_ROOT" \
  --ckpt "$V2G_CKPT" \
  --model_name "$V2G_MODEL_NAME" \
  --batch_size "$SYNTH_BATCH_SIZE" \
  --num_candidates "$NUM_CANDIDATES" \
  --max_new_tokens "$SYNTH_MAX_NEW_TOKENS" \
  --gpu "$GPU" \
  --output_raw "$ROUND1_RAW"

echo "[Round1] filter pseudo graphs"
"$PYTHON_BIN" "$REPO/scripts/data_prep/filter_pseudo_scene_graphs.py" \
  --input_raw "$ROUND1_RAW" \
  --output_filtered "$ROUND1_FILTERED" \
  --output_report "$ROUND1_REPORT" \
  --min_nodes "$FILTER_MIN_NODES" \
  --max_nodes "$FILTER_MAX_NODES" \
  --min_rels "$FILTER_MIN_RELS" \
  --max_rels "$FILTER_MAX_RELS" \
  --avg_logprob_thresh "$FILTER_LOGPROB_THRESH" \
  --text_sim_thresh "$FILTER_TEXT_SIM_THRESH" \
  --vision_sim_thresh "$FILTER_VISION_SIM_THRESH" \
  --score_thresh "$FILTER_SCORE_THRESH" \
  --gpu "$GPU"

echo "[Round1] run strict pipeline with pseudo graphs"
EXTRA_SCENE_GRAPHS="$ROUND1_FILTERED" \
STAGE2A_DIR="$ROUND1_STAGE2A_DIR" \
STAGE2B_DIR="$ROUND1_STAGE2B_DIR" \
STAGE3_DIR="$ROUND1_STAGE3_DIR" \
RUN_ROUND2_ON_FAIL=0 \
PIPELINE_MODE=full \
GPU="$GPU" \
bash "$RUN_4090"

ROUND1_STAGE2A_META="$ROUND1_STAGE2A_DIR/graph_bootstrap_meta.json"
if [ ! -f "$ROUND1_STAGE2A_META" ]; then
  ROUND1_STAGE2A_META="$ROUND1_STAGE2A_DIR/stage2A_meta.json"
fi
ROUND1_STAGE2A_CKPT="$($PYTHON_BIN "$SELECT_CKPT" --meta "$ROUND1_STAGE2A_META" --fallback "$ROUND1_STAGE2A_DIR/last.ckpt")"
if [ ! -f "$ROUND1_STAGE2A_CKPT" ]; then
  echo "[Round1] stage2A ckpt missing: $ROUND1_STAGE2A_CKPT"
  exit 1
fi

# -------------------------
# Round2: refresh pseudo + Stage2B/Stage3/eval only
# -------------------------
echo "[Round2] refresh pseudo graphs"
"$PYTHON_BIN" "$REPO/scripts/data_prep/synthesize_pseudo_scene_graphs.py" \
  --vt_manifest "$VT_MANIFEST_R2" \
  --image_root "$VT_IMAGE_ROOT" \
  --ckpt "$V2G_CKPT" \
  --model_name "$V2G_MODEL_NAME" \
  --batch_size "$SYNTH_BATCH_SIZE" \
  --num_candidates "$NUM_CANDIDATES" \
  --max_new_tokens "$SYNTH_MAX_NEW_TOKENS" \
  --gpu "$GPU" \
  --output_raw "$ROUND2_RAW"

"$PYTHON_BIN" "$REPO/scripts/data_prep/filter_pseudo_scene_graphs.py" \
  --input_raw "$ROUND2_RAW" \
  --output_filtered "$ROUND2_FILTERED" \
  --output_report "$ROUND2_REPORT" \
  --min_nodes "$FILTER_MIN_NODES" \
  --max_nodes "$FILTER_MAX_NODES" \
  --min_rels "$FILTER_MIN_RELS" \
  --max_rels "$FILTER_MAX_RELS" \
  --avg_logprob_thresh "$FILTER_LOGPROB_THRESH" \
  --text_sim_thresh "$FILTER_TEXT_SIM_THRESH" \
  --vision_sim_thresh "$FILTER_VISION_SIM_THRESH" \
  --score_thresh "$FILTER_SCORE_THRESH" \
  --gpu "$GPU"

echo "[Round2] GraphRefine"
EXTRA_SCENE_GRAPHS="$ROUND2_FILTERED" \
STAGE2A_CKPT="$ROUND1_STAGE2A_CKPT" \
STAGE2B_DIR="$ROUND2_STAGE2B_DIR" \
PIPELINE_MODE=graph_refine \
RUN_ROUND2_ON_FAIL=0 \
GPU="$GPU" \
bash "$RUN_4090"

ROUND2_STAGE2B_META="$ROUND2_STAGE2B_DIR/graph_refine_meta.json"
if [ ! -f "$ROUND2_STAGE2B_META" ]; then
  ROUND2_STAGE2B_META="$ROUND2_STAGE2B_DIR/stage2B_meta.json"
fi
ROUND2_STAGE2B_CKPT="$($PYTHON_BIN "$SELECT_CKPT" --meta "$ROUND2_STAGE2B_META" --fallback "$ROUND2_STAGE2B_DIR/last.ckpt")"
if [ ! -f "$ROUND2_STAGE2B_CKPT" ]; then
  echo "[Round2] graph_refine ckpt missing: $ROUND2_STAGE2B_CKPT"
  exit 1
fi

echo "[Round2] MultiModalTune"
EXTRA_SCENE_GRAPHS="$ROUND2_FILTERED" \
STAGE2B_CKPT="$ROUND2_STAGE2B_CKPT" \
STAGE3_DIR="$ROUND2_STAGE3_DIR" \
PIPELINE_MODE=multimodal_tune \
RUN_ROUND2_ON_FAIL=0 \
GPU="$GPU" \
bash "$RUN_4090"

echo "[Round2] Eval"
EXTRA_SCENE_GRAPHS="$ROUND2_FILTERED" \
STAGE3_STATE_DICT="$ROUND2_STAGE3_DIR/omnigraph_multimodal_tune_state_dict.pt" \
PIPELINE_MODE=eval \
RUN_ROUND2_ON_FAIL=0 \
GPU="$GPU" \
bash "$RUN_4090"

echo "[Done] v2g bootstrap rounds completed"
echo "[Artifacts]"
echo "  round1_filtered=$ROUND1_FILTERED"
echo "  round2_filtered=$ROUND2_FILTERED"
echo "  round1_stage3=$ROUND1_STAGE3_DIR"
echo "  round2_stage3=$ROUND2_STAGE3_DIR"
