#!/usr/bin/env bash
set -euo pipefail

# End-to-end strict pipeline:
# graph_bootstrap -> graph_refine -> multimodal_tune -> GQA infer/eval
# If strict accuracy < threshold, auto-run fixed round-2 recipe.

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-"$WORKDIR"}
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="$REPO:$PYTHONPATH"
else
  export PYTHONPATH="$REPO"
fi
GPU=${GPU:-0}
ISOLATE_GPU=${ISOLATE_GPU:-1}
ORIG_GPU=${GPU}
AUTO_BATCH_BY_VRAM=${AUTO_BATCH_BY_VRAM:-1}
if [ "$ISOLATE_GPU" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
  GPU=0
fi
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
OMNIGRAPH_HF_CACHE=${OMNIGRAPH_HF_CACHE:-/media/disk/02drive/13hias/.cache}
OMNIGRAPH_HF_ENDPOINT=${OMNIGRAPH_HF_ENDPOINT:-https://hf-mirror.com}
export OMNIGRAPH_HF_CACHE OMNIGRAPH_HF_ENDPOINT

CPU_CORES=$(nproc 2>/dev/null || echo 8)
THREAD_DEFAULT=$((CPU_CORES / 2))
if [ "$THREAD_DEFAULT" -lt 1 ]; then THREAD_DEFAULT=1; fi
if [ "$THREAD_DEFAULT" -gt 16 ]; then THREAD_DEFAULT=16; fi

ensure_positive_int_env() {
  local key="$1"
  local fallback="$2"
  local cur="${!key-}"
  if [[ ! "$cur" =~ ^[1-9][0-9]*$ ]]; then
    export "$key=$fallback"
    echo "[Config][Fix] ${key} invalid (${cur:-<unset>}) -> ${fallback}"
  fi
}

ensure_positive_int_env OMP_NUM_THREADS "$THREAD_DEFAULT"
ensure_positive_int_env MKL_NUM_THREADS "$THREAD_DEFAULT"
ensure_positive_int_env OPENBLAS_NUM_THREADS "$OMP_NUM_THREADS"
ensure_positive_int_env NUMEXPR_NUM_THREADS "$OMP_NUM_THREADS"

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

STRICT_TARGET=${STRICT_TARGET:-0.4200}
STRICT_SPRINT_TARGET=${STRICT_SPRINT_TARGET:-0.4400}
STRICT_BASELINE=${STRICT_BASELINE:-0.3912}
COVERAGE_TARGET=${COVERAGE_TARGET:-0.9990}
QUERY_TARGET=${QUERY_TARGET:-0.2600}
QUERY_SPRINT_TARGET=${QUERY_SPRINT_TARGET:-0.2800}
RUN_ROUND2_ON_FAIL=${RUN_ROUND2_ON_FAIL:-1}
INSTALL_DEPS=${INSTALL_DEPS:-0}
LOW_VRAM_4090=${LOW_VRAM_4090:-0}
AUTO_BATCH_RETRY_ON_OOM=${AUTO_BATCH_RETRY_ON_OOM:-1}
AUTO_BATCH_MAX_RETRIES=${AUTO_BATCH_MAX_RETRIES:-8}
AUTO_BATCH_SCALE=${AUTO_BATCH_SCALE:-1.0}
AUTO_RESUME_ON_OOM=${AUTO_RESUME_ON_OOM:-1}
PIPELINE_MODE=${PIPELINE_MODE:-full}
PIPELINE_MODE=$(printf "%s" "$PIPELINE_MODE" | tr '[:upper:]' '[:lower:]')

RUN_STAGE2A=0
RUN_STAGE2B=0
RUN_STAGE3=0
RUN_EVAL=0
case "$PIPELINE_MODE" in
  full)
    RUN_STAGE2A=1
    RUN_STAGE2B=1
    RUN_STAGE3=1
    RUN_EVAL=1
    ;;
  stage2a|graph_bootstrap)
    RUN_STAGE2A=1
    ;;
  stage2b|graph_refine)
    RUN_STAGE2B=1
    ;;
  stage3|multimodal_tune)
    RUN_STAGE3=1
    ;;
  eval|evaluation)
    RUN_EVAL=1
    ;;
  *)
    echo "[Config] invalid PIPELINE_MODE=${PIPELINE_MODE} (expected: full|graph_bootstrap|graph_refine|multimodal_tune|evaluation)"
    exit 1
    ;;
esac
if [ "$PIPELINE_MODE" != "full" ]; then
  RUN_ROUND2_ON_FAIL=0
fi

# Keep track of whether user explicitly set batch size env vars.
USER_SET_S2A_BATCH_SIZE=${S2A_BATCH_SIZE+x}
USER_SET_S2B_BATCH_SIZE=${S2B_BATCH_SIZE+x}
USER_SET_S3_BATCH_SIZE=${S3_BATCH_SIZE+x}
USER_SET_S2B_R2_BATCH_SIZE=${S2B_R2_BATCH_SIZE+x}
USER_SET_S3_R2_BATCH_SIZE=${S3_R2_BATCH_SIZE+x}

DEFAULT_LLM_7B="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LLM_3B="Qwen/Qwen2.5-3B-Instruct"
LLM_MODEL=${LLM_MODEL:-$DEFAULT_LLM_3B}
LLM_DTYPE=${LLM_DTYPE:-bfloat16}
LLM_ATTN_IMPL=${LLM_ATTN_IMPL:-sdpa}
VISION_MODEL=${VISION_MODEL:-Salesforce/blip2-flan-t5-xl}
NODE_ENCODER_TYPE=${NODE_ENCODER_TYPE:-hybrid}
NODE_ENCODER_ALPHA_INIT=${NODE_ENCODER_ALPHA_INIT:-0.3}
NODE_ENCODER_OUT_DIM=${NODE_ENCODER_OUT_DIM:-128}
GRAPH_TOKENIZER_TYPE=${GRAPH_TOKENIZER_TYPE:-perceiver}
STAGE2A_BOOTSTRAP_MODE=${STAGE2A_BOOTSTRAP_MODE:-no_stage1}
PERCEIVER_NUM_LATENTS=${PERCEIVER_NUM_LATENTS:-32}
PERCEIVER_NUM_LAYERS=${PERCEIVER_NUM_LAYERS:-3}
PERCEIVER_NUM_HEADS=${PERCEIVER_NUM_HEADS:-8}
PERCEIVER_FF_MULT=${PERCEIVER_FF_MULT:-4}
PERCEIVER_DROPOUT=${PERCEIVER_DROPOUT:-0.0}
USE_QA_TYPE_TOKEN=${USE_QA_TYPE_TOKEN:-1}
ENABLE_GVL_ADAPTER=${ENABLE_GVL_ADAPTER:-1}
GVL_ADAPTER_GATE_INIT=${GVL_ADAPTER_GATE_INIT:-0.1}
ENABLE_GRAPH_AUX_HEAD=${ENABLE_GRAPH_AUX_HEAD:-1}
S2A_GRAPH_AUX_LOSS_WEIGHT=${S2A_GRAPH_AUX_LOSS_WEIGHT:-0.04}
S2B_GRAPH_AUX_LOSS_WEIGHT=${S2B_GRAPH_AUX_LOSS_WEIGHT:-0.08}
S3_GRAPH_AUX_LOSS_WEIGHT=${S3_GRAPH_AUX_LOSS_WEIGHT:-0.03}
S2B_R2_GRAPH_AUX_LOSS_WEIGHT=${S2B_R2_GRAPH_AUX_LOSS_WEIGHT:-0.10}
S3_R2_GRAPH_AUX_LOSS_WEIGHT=${S3_R2_GRAPH_AUX_LOSS_WEIGHT:-0.04}
ENABLE_XTC=${ENABLE_XTC:-1}
ENABLE_XTM=${ENABLE_XTM:-1}
XTC_LOGIT_SCALE_INIT=${XTC_LOGIT_SCALE_INIT:-2.66}
XTM_DUP_THRESH=${XTM_DUP_THRESH:-0.98}
S2A_XTC_WEIGHT=${S2A_XTC_WEIGHT:-0.08}
S2A_XTM_WEIGHT=${S2A_XTM_WEIGHT:-0.05}
S2B_XTC_WEIGHT=${S2B_XTC_WEIGHT:-0.15}
S2B_XTM_WEIGHT=${S2B_XTM_WEIGHT:-0.10}
S3_XTC_WEIGHT=${S3_XTC_WEIGHT:-0.05}
S3_XTM_WEIGHT=${S3_XTM_WEIGHT:-0.03}
S2B_R2_XTC_WEIGHT=${S2B_R2_XTC_WEIGHT:-$S2B_XTC_WEIGHT}
S2B_R2_XTM_WEIGHT=${S2B_R2_XTM_WEIGHT:-$S2B_XTM_WEIGHT}
S3_R2_XTC_WEIGHT=${S3_R2_XTC_WEIGHT:-$S3_XTC_WEIGHT}
S3_R2_XTM_WEIGHT=${S3_R2_XTM_WEIGHT:-$S3_XTM_WEIGHT}
S3_ENABLE_XGV=${S3_ENABLE_XGV:-1}
S3_XGV_WEIGHT=${S3_XGV_WEIGHT:-0.03}
S3_R2_ENABLE_XGV=${S3_R2_ENABLE_XGV:-$S3_ENABLE_XGV}
S3_R2_XGV_WEIGHT=${S3_R2_XGV_WEIGHT:-$S3_XGV_WEIGHT}
S3_ENABLE_STAGE3_CURRICULUM=${S3_ENABLE_STAGE3_CURRICULUM:-1}
S3_CURRICULUM_PHASE_A_STEPS=${S3_CURRICULUM_PHASE_A_STEPS:-2000}
S3_CURRICULUM_PHASE_B_STEPS=${S3_CURRICULUM_PHASE_B_STEPS:-12000}
S3_PHASE_A_ALIGN_SCALE=${S3_PHASE_A_ALIGN_SCALE:-1.35}
S3_PHASE_C_ALIGN_SCALE=${S3_PHASE_C_ALIGN_SCALE:-0.60}
S3_PHASE_A_AUX_SCALE=${S3_PHASE_A_AUX_SCALE:-0.80}
S3_PHASE_C_AUX_SCALE=${S3_PHASE_C_AUX_SCALE:-1.00}
S3_TRAIN_GRAPH_DROP_PROB=${S3_TRAIN_GRAPH_DROP_PROB:-0.08}
S3_TRAIN_VISION_DROP_PROB=${S3_TRAIN_VISION_DROP_PROB:-0.08}
S3_ENABLE_QA_TYPE_MODALITY_GATE=${S3_ENABLE_QA_TYPE_MODALITY_GATE:-1}
S3_R2_ENABLE_STAGE3_CURRICULUM=${S3_R2_ENABLE_STAGE3_CURRICULUM:-$S3_ENABLE_STAGE3_CURRICULUM}
S3_R2_CURRICULUM_PHASE_A_STEPS=${S3_R2_CURRICULUM_PHASE_A_STEPS:-$S3_CURRICULUM_PHASE_A_STEPS}
S3_R2_CURRICULUM_PHASE_B_STEPS=${S3_R2_CURRICULUM_PHASE_B_STEPS:-$S3_CURRICULUM_PHASE_B_STEPS}
S3_R2_PHASE_A_ALIGN_SCALE=${S3_R2_PHASE_A_ALIGN_SCALE:-$S3_PHASE_A_ALIGN_SCALE}
S3_R2_PHASE_C_ALIGN_SCALE=${S3_R2_PHASE_C_ALIGN_SCALE:-$S3_PHASE_C_ALIGN_SCALE}
S3_R2_PHASE_A_AUX_SCALE=${S3_R2_PHASE_A_AUX_SCALE:-$S3_PHASE_A_AUX_SCALE}
S3_R2_PHASE_C_AUX_SCALE=${S3_R2_PHASE_C_AUX_SCALE:-$S3_PHASE_C_AUX_SCALE}
S3_R2_TRAIN_GRAPH_DROP_PROB=${S3_R2_TRAIN_GRAPH_DROP_PROB:-$S3_TRAIN_GRAPH_DROP_PROB}
S3_R2_TRAIN_VISION_DROP_PROB=${S3_R2_TRAIN_VISION_DROP_PROB:-$S3_TRAIN_VISION_DROP_PROB}
S3_R2_ENABLE_QA_TYPE_MODALITY_GATE=${S3_R2_ENABLE_QA_TYPE_MODALITY_GATE:-$S3_ENABLE_QA_TYPE_MODALITY_GATE}

if [ "$LOW_VRAM_4090" = "1" ]; then
  if [ "$LLM_MODEL" = "$DEFAULT_LLM_7B" ]; then
    LLM_MODEL="$DEFAULT_LLM_3B"
  fi
  # Fallback profile for RTX 4090 24GB.
  : "${S2A_BATCH_SIZE:=1}"
  : "${S2A_MAX_LENGTH:=96}"
  : "${S2A_MAX_GRAPH_TOKENS:=16}"
  : "${S2A_NUM_WORKERS:=4}"
  : "${S2A_PRECISION:=16-mixed}"
  : "${S2A_ACCUM_GRAD_BATCHES:=4}"
  : "${S2A_NUM_SANITY_VAL_STEPS:=0}"
: "${S2A_VAL_CHECK_INTERVAL:=5000}"
: "${S2A_LIMIT_VAL_BATCHES:=0.002}"
: "${S2A_TRAIN_NODE_ENCODER:=1}"
: "${S2A_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2A_ENABLE_LLM_GRADIENT_CHECKPOINTING:=1}"
: "${S2A_PREFETCH_FACTOR:=4}"

  : "${S2B_BATCH_SIZE:=1}"
  : "${S2B_MAX_LENGTH:=96}"
  : "${S2B_MAX_GRAPH_TOKENS:=16}"
  : "${S2B_NUM_WORKERS:=4}"
  : "${S2B_PRECISION:=16-mixed}"
  : "${S2B_ACCUM_GRAD_BATCHES:=4}"
  : "${S2B_NUM_SANITY_VAL_STEPS:=0}"
: "${S2B_VAL_CHECK_INTERVAL:=5000}"
: "${S2B_LIMIT_VAL_BATCHES:=0.002}"
: "${S2B_TRAIN_NODE_ENCODER:=1}"
: "${S2B_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2B_ENABLE_LLM_GRADIENT_CHECKPOINTING:=1}"
: "${S2B_PREFETCH_FACTOR:=4}"

  : "${S3_BATCH_SIZE:=1}"
  : "${S3_MAX_LENGTH:=96}"
  : "${S3_MAX_GRAPH_TOKENS:=16}"
  : "${S3_MAX_VISION_TOKENS:=16}"
  : "${S3_NUM_WORKERS:=2}"
  : "${S3_PRECISION:=16-mixed}"
  : "${S3_ACCUM_GRAD_BATCHES:=4}"
  : "${S3_NUM_SANITY_VAL_STEPS:=0}"
: "${S3_VAL_CHECK_INTERVAL:=5000}"
: "${S3_LIMIT_VAL_BATCHES:=0.002}"
: "${S3_TRAIN_NODE_ENCODER:=0}"
: "${S3_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S3_ENABLE_LLM_GRADIENT_CHECKPOINTING:=1}"
: "${S3_PREFETCH_FACTOR:=4}"

  : "${S2B_R2_BATCH_SIZE:=1}"
  : "${S2B_R2_MAX_LENGTH:=96}"
  : "${S2B_R2_MAX_GRAPH_TOKENS:=16}"
  : "${S2B_R2_NUM_WORKERS:=4}"
  : "${S2B_R2_PRECISION:=16-mixed}"
  : "${S2B_R2_ACCUM_GRAD_BATCHES:=4}"
  : "${S2B_R2_NUM_SANITY_VAL_STEPS:=0}"
: "${S2B_R2_VAL_CHECK_INTERVAL:=5000}"
: "${S2B_R2_LIMIT_VAL_BATCHES:=0.002}"
: "${S2B_R2_TRAIN_NODE_ENCODER:=1}"
: "${S2B_R2_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2B_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING:=1}"
: "${S2B_R2_PREFETCH_FACTOR:=4}"

  : "${S3_R2_BATCH_SIZE:=1}"
  : "${S3_R2_MAX_LENGTH:=96}"
  : "${S3_R2_MAX_GRAPH_TOKENS:=16}"
  : "${S3_R2_MAX_VISION_TOKENS:=16}"
  : "${S3_R2_NUM_WORKERS:=2}"
  : "${S3_R2_PRECISION:=16-mixed}"
  : "${S3_R2_ACCUM_GRAD_BATCHES:=4}"
  : "${S3_R2_NUM_SANITY_VAL_STEPS:=0}"
: "${S3_R2_VAL_CHECK_INTERVAL:=5000}"
: "${S3_R2_LIMIT_VAL_BATCHES:=0.002}"
: "${S3_R2_TRAIN_NODE_ENCODER:=0}"
: "${S3_R2_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S3_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING:=1}"
: "${S3_R2_PREFETCH_FACTOR:=4}"
else
  : "${S2A_BATCH_SIZE:=12}"
  : "${S2A_MAX_LENGTH:=128}"
  : "${S2A_MAX_GRAPH_TOKENS:=24}"
  : "${S2A_NUM_WORKERS:=8}"
  : "${S2A_PRECISION:=16-mixed}"
  : "${S2A_ACCUM_GRAD_BATCHES:=2}"
  : "${S2A_NUM_SANITY_VAL_STEPS:=0}"
: "${S2A_VAL_CHECK_INTERVAL:=5000}"
: "${S2A_LIMIT_VAL_BATCHES:=0.002}"
: "${S2A_TRAIN_NODE_ENCODER:=1}"
: "${S2A_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2A_ENABLE_LLM_GRADIENT_CHECKPOINTING:=0}"
: "${S2A_PREFETCH_FACTOR:=6}"

  : "${S2B_BATCH_SIZE:=12}"
  : "${S2B_MAX_LENGTH:=128}"
  : "${S2B_MAX_GRAPH_TOKENS:=24}"
  : "${S2B_NUM_WORKERS:=8}"
  : "${S2B_PRECISION:=16-mixed}"
  : "${S2B_ACCUM_GRAD_BATCHES:=2}"
  : "${S2B_NUM_SANITY_VAL_STEPS:=0}"
: "${S2B_VAL_CHECK_INTERVAL:=5000}"
: "${S2B_LIMIT_VAL_BATCHES:=0.002}"
: "${S2B_TRAIN_NODE_ENCODER:=1}"
: "${S2B_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2B_ENABLE_LLM_GRADIENT_CHECKPOINTING:=0}"
: "${S2B_PREFETCH_FACTOR:=6}"

  : "${S3_BATCH_SIZE:=6}"
  : "${S3_MAX_LENGTH:=128}"
  : "${S3_MAX_GRAPH_TOKENS:=24}"
  : "${S3_MAX_VISION_TOKENS:=24}"
  : "${S3_NUM_WORKERS:=4}"
  : "${S3_PRECISION:=16-mixed}"
  : "${S3_ACCUM_GRAD_BATCHES:=2}"
  : "${S3_NUM_SANITY_VAL_STEPS:=0}"
: "${S3_VAL_CHECK_INTERVAL:=5000}"
: "${S3_LIMIT_VAL_BATCHES:=0.002}"
: "${S3_TRAIN_NODE_ENCODER:=0}"
: "${S3_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S3_ENABLE_LLM_GRADIENT_CHECKPOINTING:=0}"
: "${S3_PREFETCH_FACTOR:=6}"

  : "${S2B_R2_BATCH_SIZE:=12}"
  : "${S2B_R2_MAX_LENGTH:=128}"
  : "${S2B_R2_MAX_GRAPH_TOKENS:=24}"
  : "${S2B_R2_NUM_WORKERS:=8}"
  : "${S2B_R2_PRECISION:=16-mixed}"
  : "${S2B_R2_ACCUM_GRAD_BATCHES:=2}"
  : "${S2B_R2_NUM_SANITY_VAL_STEPS:=0}"
: "${S2B_R2_VAL_CHECK_INTERVAL:=5000}"
: "${S2B_R2_LIMIT_VAL_BATCHES:=0.002}"
: "${S2B_R2_TRAIN_NODE_ENCODER:=1}"
: "${S2B_R2_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S2B_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING:=0}"
: "${S2B_R2_PREFETCH_FACTOR:=6}"

  : "${S3_R2_BATCH_SIZE:=6}"
  : "${S3_R2_MAX_LENGTH:=128}"
  : "${S3_R2_MAX_GRAPH_TOKENS:=24}"
  : "${S3_R2_MAX_VISION_TOKENS:=24}"
  : "${S3_R2_NUM_WORKERS:=4}"
  : "${S3_R2_PRECISION:=16-mixed}"
  : "${S3_R2_ACCUM_GRAD_BATCHES:=2}"
  : "${S3_R2_NUM_SANITY_VAL_STEPS:=0}"
: "${S3_R2_VAL_CHECK_INTERVAL:=5000}"
: "${S3_R2_LIMIT_VAL_BATCHES:=0.002}"
: "${S3_R2_TRAIN_NODE_ENCODER:=0}"
: "${S3_R2_CHECKPOINT_EVERY_N_STEPS:=1000}"
: "${S3_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING:=0}"
: "${S3_R2_PREFETCH_FACTOR:=6}"
fi

: "${S2A_GRAPH_QA_MAX_PER_IMAGE:=5}"
: "${S2A_GRAPH_QA_REPEAT:=3}"
: "${S2A_PSEUDO_GRAPH_QA_MAX_PER_IMAGE:=2}"
: "${S2A_PSEUDO_GRAPH_QA_REPEAT:=1}"
: "${S2A_VAL_RATIO:=0.02}"
: "${S2A_PATIENCE:=16}"
: "${S2A_MIN_DELTA:=0.0005}"
: "${S2A_LR:=2e-5}"
: "${S2A_MAX_STEPS:=100000}"

: "${S2B_GRAPH_QA_MAX_PER_IMAGE:=6}"
: "${S2B_GRAPH_QA_REPEAT:=6}"
: "${S2B_PSEUDO_GRAPH_QA_MAX_PER_IMAGE:=2}"
: "${S2B_PSEUDO_GRAPH_QA_REPEAT:=1}"
: "${S2B_VAL_RATIO:=0.02}"
: "${S2B_PATIENCE:=18}"
: "${S2B_MIN_DELTA:=0.0003}"
: "${S2B_LR:=9e-6}"
: "${S2B_MAX_STEPS:=100000}"

: "${S3_GRAPH_QA_MAX_PER_IMAGE:=4}"
: "${S3_GRAPH_QA_REPEAT:=2}"
: "${S3_PSEUDO_GRAPH_QA_MAX_PER_IMAGE:=2}"
: "${S3_PSEUDO_GRAPH_QA_REPEAT:=1}"
: "${S3_VAL_RATIO:=0.02}"
: "${S3_PATIENCE:=12}"
: "${S3_MIN_DELTA:=0.0003}"
: "${S3_LR:=1e-5}"
: "${S3_MAX_STEPS:=40000}"

: "${S2B_R2_GRAPH_QA_MAX_PER_IMAGE:=8}"
: "${S2B_R2_GRAPH_QA_REPEAT:=7}"
: "${S2B_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE:=2}"
: "${S2B_R2_PSEUDO_GRAPH_QA_REPEAT:=1}"
: "${S2B_R2_VAL_RATIO:=0.02}"
: "${S2B_R2_PATIENCE:=22}"
: "${S2B_R2_MIN_DELTA:=0.0002}"
: "${S2B_R2_LR:=7e-6}"
: "${S2B_R2_MAX_STEPS:=120000}"

: "${S3_R2_GRAPH_QA_MAX_PER_IMAGE:=5}"
: "${S3_R2_GRAPH_QA_REPEAT:=3}"
: "${S3_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE:=2}"
: "${S3_R2_PSEUDO_GRAPH_QA_REPEAT:=1}"
: "${S3_R2_VAL_RATIO:=0.02}"
: "${S3_R2_PATIENCE:=16}"
: "${S3_R2_MIN_DELTA:=0.0002}"
: "${S3_R2_LR:=8e-6}"
: "${S3_R2_MAX_STEPS:=50000}"

S2A_PRECISION_ORIG="$S2A_PRECISION"
S2B_PRECISION_ORIG="$S2B_PRECISION"
S3_PRECISION_ORIG="$S3_PRECISION"
S2B_R2_PRECISION_ORIG="$S2B_R2_PRECISION"
S3_R2_PRECISION_ORIG="$S3_R2_PRECISION"
S2A_PRECISION="$(normalize_precision_for_dtype "$S2A_PRECISION" "$LLM_DTYPE")"
S2B_PRECISION="$(normalize_precision_for_dtype "$S2B_PRECISION" "$LLM_DTYPE")"
S3_PRECISION="$(normalize_precision_for_dtype "$S3_PRECISION" "$LLM_DTYPE")"
S2B_R2_PRECISION="$(normalize_precision_for_dtype "$S2B_R2_PRECISION" "$LLM_DTYPE")"
S3_R2_PRECISION="$(normalize_precision_for_dtype "$S3_R2_PRECISION" "$LLM_DTYPE")"
if [ "$S2A_PRECISION" != "$S2A_PRECISION_ORIG" ] || [ "$S2B_PRECISION" != "$S2B_PRECISION_ORIG" ] || [ "$S3_PRECISION" != "$S3_PRECISION_ORIG" ] || [ "$S2B_R2_PRECISION" != "$S2B_R2_PRECISION_ORIG" ] || [ "$S3_R2_PRECISION" != "$S3_R2_PRECISION_ORIG" ]; then
  echo "[ConfigFix] precision adjusted for LLM_DTYPE=$LLM_DTYPE -> S2A=$S2A_PRECISION S2B=$S2B_PRECISION S3=$S3_PRECISION S2B_R2=$S2B_R2_PRECISION S3_R2=$S3_R2_PRECISION"
fi

OMNIGRAPH_DATA_ROOT=${OMNIGRAPH_DATA_ROOT:-"$REPO/data"}
VG_SCENE_GRAPHS=${VG_SCENE_GRAPHS:-"$OMNIGRAPH_DATA_ROOT/vg/contents/sceneGraphs/scene_graphs.json"}
VG_REGIONS=${VG_REGIONS:-"$OMNIGRAPH_DATA_ROOT/vg/contents/regionDescriptions/region_descriptions.json"}
VG_IMAGE_ROOT=${VG_IMAGE_ROOT:-"$OMNIGRAPH_DATA_ROOT/vg"}
STAGE1_QFORMER_CKPT=${STAGE1_QFORMER_CKPT:-"$REPO/graph_qformer_stage1.pt"}
EXTRA_SCENE_GRAPHS=${EXTRA_SCENE_GRAPHS:-""}
if [ -n "${GRAPH_TOKENIZER_PRETRAIN_CKPT:-}" ]; then
  STAGE1_QFORMER_CKPT="${GRAPH_TOKENIZER_PRETRAIN_CKPT}"
fi

STAGE2A_BOOTSTRAP_MODE_LC="$(printf "%s" "$STAGE2A_BOOTSTRAP_MODE" | tr '[:upper:]' '[:lower:]')"
GRAPH_TOKENIZER_TYPE_LC="$(printf "%s" "$GRAPH_TOKENIZER_TYPE" | tr '[:upper:]' '[:lower:]')"
REQUIRE_STAGE1_CKPT=0
if [ "$STAGE2A_BOOTSTRAP_MODE_LC" = "legacy_stage1" ]; then
  REQUIRE_STAGE1_CKPT=1
elif [ "$STAGE2A_BOOTSTRAP_MODE_LC" = "auto" ] && [ "$GRAPH_TOKENIZER_TYPE_LC" = "qformer" ]; then
  REQUIRE_STAGE1_CKPT=1
fi

STAGE2A_DIR=${STAGE2A_DIR:-"$REPO/checkpoints_projector_vg/graph_bootstrap_paper"}
STAGE2B_DIR=${STAGE2B_DIR:-"$REPO/checkpoints_projector_vg/graph_refine_paper"}
STAGE3_DIR=${STAGE3_DIR:-"$REPO/checkpoints_multimodal_tune"}
STAGE2B_R2_DIR=${STAGE2B_R2_DIR:-"$REPO/checkpoints_projector_vg/graph_refine_round2"}
STAGE3_R2_DIR=${STAGE3_R2_DIR:-"$REPO/checkpoints_multimodal_tune_round2"}
if [ -n "${GRAPH_BOOTSTRAP_DIR:-}" ]; then STAGE2A_DIR="${GRAPH_BOOTSTRAP_DIR}"; fi
if [ -n "${GRAPH_REFINE_DIR:-}" ]; then STAGE2B_DIR="${GRAPH_REFINE_DIR}"; fi
if [ -n "${MULTIMODAL_TUNE_DIR:-}" ]; then STAGE3_DIR="${MULTIMODAL_TUNE_DIR}"; fi
if [ -n "${GRAPH_REFINE_R2_DIR:-}" ]; then STAGE2B_R2_DIR="${GRAPH_REFINE_R2_DIR}"; fi
if [ -n "${MULTIMODAL_TUNE_R2_DIR:-}" ]; then STAGE3_R2_DIR="${MULTIMODAL_TUNE_R2_DIR}"; fi

S2A_MANUAL_STOP_FILE=${S2A_MANUAL_STOP_FILE:-"$STAGE2A_DIR/STOP_EARLY"}
S2B_MANUAL_STOP_FILE=${S2B_MANUAL_STOP_FILE:-"$STAGE2B_DIR/STOP_EARLY"}
S3_MANUAL_STOP_FILE=${S3_MANUAL_STOP_FILE:-"$STAGE3_DIR/STOP_EARLY"}
S2B_R2_MANUAL_STOP_FILE=${S2B_R2_MANUAL_STOP_FILE:-"$STAGE2B_R2_DIR/STOP_EARLY"}
S3_R2_MANUAL_STOP_FILE=${S3_R2_MANUAL_STOP_FILE:-"$STAGE3_R2_DIR/STOP_EARLY"}

GQA_QUESTIONS_JSON=${GQA_QUESTIONS_JSON:-"$OMNIGRAPH_DATA_ROOT/gqa/contents/questions/val_balanced_questions.json"}
GQA_QUESTIONS_JSONL=${GQA_QUESTIONS_JSONL:-"$OMNIGRAPH_DATA_ROOT/gqa/val_balanced.jsonl"}
GQA_SCENE_RAW=${GQA_SCENE_RAW:-"$OMNIGRAPH_DATA_ROOT/gqa/contents/sceneGraphs/val_sceneGraphs.json"}
GQA_SCENE_VG=${GQA_SCENE_VG:-"$OMNIGRAPH_DATA_ROOT/gqa/scene_graphs_val_vg.json"}
GQA_IMAGE_ROOT=${GQA_IMAGE_ROOT:-"$OMNIGRAPH_DATA_ROOT/gqa/contents/images"}
GQA_PRED_PAPER=${GQA_PRED_PAPER:-"$OMNIGRAPH_DATA_ROOT/gqa/pred_val_balanced_paper.jsonl"}
GQA_EVAL_PAPER=${GQA_EVAL_PAPER:-"$OMNIGRAPH_DATA_ROOT/gqa/eval_val_balanced_paper.txt"}
GQA_PRED_R2=${GQA_PRED_R2:-"$OMNIGRAPH_DATA_ROOT/gqa/pred_val_balanced_round2.jsonl"}
GQA_EVAL_R2=${GQA_EVAL_R2:-"$OMNIGRAPH_DATA_ROOT/gqa/eval_val_balanced_round2.txt"}

SELECT_CKPT="${REPO}/scripts/train/select_best_ckpt.py"
STAGE2A_CKPT=${STAGE2A_CKPT:-}
STAGE2B_CKPT=${STAGE2B_CKPT:-}
if [ -n "${GRAPH_BOOTSTRAP_CKPT:-}" ] && [ -z "${STAGE2A_CKPT}" ]; then STAGE2A_CKPT="${GRAPH_BOOTSTRAP_CKPT}"; fi
if [ -n "${GRAPH_REFINE_CKPT:-}" ] && [ -z "${STAGE2B_CKPT}" ]; then STAGE2B_CKPT="${GRAPH_REFINE_CKPT}"; fi
STAGE3_STATE_DICT=${STAGE3_STATE_DICT:-"$STAGE3_DIR/omnigraph_multimodal_tune_state_dict.pt"}
STAGE3_R2_STATE_DICT=${STAGE3_R2_STATE_DICT:-"$STAGE3_R2_DIR/omnigraph_multimodal_tune_state_dict.pt"}
if [ -n "${MULTIMODAL_TUNE_STATE_DICT:-}" ]; then STAGE3_STATE_DICT="${MULTIMODAL_TUNE_STATE_DICT}"; fi
if [ -n "${MULTIMODAL_TUNE_R2_STATE_DICT:-}" ]; then STAGE3_R2_STATE_DICT="${MULTIMODAL_TUNE_R2_STATE_DICT}"; fi

echo "[Precheck] repo=${REPO}"
echo "[Precheck] GPU request=${ORIG_GPU} isolate=${ISOLATE_GPU} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} internal_gpu=${GPU}"
if ! "$PYTHON_BIN" -c "import torch, pytorch_lightning as pl" >/dev/null 2>&1; then
  if [ "$INSTALL_DEPS" = "1" ]; then
    echo "[Precheck] missing deps -> installing from requirements.txt"
    "$PYTHON_BIN" -m pip install -r "$REPO/requirements.txt"
  else
    echo "[Precheck] missing deps (torch/pytorch_lightning). Set INSTALL_DEPS=1 to auto-install."
    exit 1
  fi
fi
"$PYTHON_BIN" -c "import torch, pytorch_lightning as pl; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(${GPU}) if torch.cuda.is_available() else 'cpu')"

GPU_VRAM_GB="unknown"
if [ "$AUTO_BATCH_BY_VRAM" = "1" ]; then
  read -r GPU_VRAM_GB AUTO_S2A_BATCH AUTO_S2B_BATCH AUTO_S3_BATCH AUTO_S2B_R2_BATCH AUTO_S3_R2_BATCH <<<"$(
    GPU="$GPU" \
    LLM_MODEL="$LLM_MODEL" \
    S2A_MAX_LENGTH="$S2A_MAX_LENGTH" \
    S2B_MAX_LENGTH="$S2B_MAX_LENGTH" \
    S3_MAX_LENGTH="$S3_MAX_LENGTH" \
    AUTO_BATCH_SCALE="$AUTO_BATCH_SCALE" \
    "$PYTHON_BIN" - <<'PY'
import os
import torch

gpu = int(os.environ.get("GPU", "0"))
llm_model = str(os.environ.get("LLM_MODEL", "")).lower()
s2a_max_len = int(os.environ.get("S2A_MAX_LENGTH", "96"))
s2b_max_len = int(os.environ.get("S2B_MAX_LENGTH", "96"))
s3_max_len = int(os.environ.get("S3_MAX_LENGTH", "96"))
scale = float(os.environ.get("AUTO_BATCH_SCALE", "1.0"))

if not torch.cuda.is_available():
    print("0.0 1 1 1 1 1")
    raise SystemExit(0)

if gpu < 0 or gpu >= torch.cuda.device_count():
    gpu = 0

gb = float(torch.cuda.get_device_properties(gpu).total_memory) / (1024 ** 3)
is_3b = ("3b" in llm_model)
is_7b = ("7b" in llm_model)

# Aggressive stage-wise defaults:
# Start high to better fill VRAM; runtime OOM fallback will auto-reduce.
if gb < 16:
    s2a, s2b, s3 = 1, 1, 1
elif gb < 24:
  if is_3b and s2a_max_len <= 96 and s2b_max_len <= 96 and s3_max_len <= 96:
    s2a, s2b, s3 = 16, 16, 8
  elif is_3b and s2a_max_len <= 160 and s2b_max_len <= 160 and s3_max_len <= 160:
    s2a, s2b, s3 = 12, 12, 6
  elif is_3b and s2a_max_len <= 256 and s2b_max_len <= 256 and s3_max_len <= 256:
    s2a, s2b, s3 = 10, 10, 5
  elif is_7b:
    s2a, s2b, s3 = 4, 4, 2
  else:
    s2a, s2b, s3 = 5, 5, 2
elif gb < 30:
  if is_3b and s2a_max_len <= 96 and s2b_max_len <= 96 and s3_max_len <= 96:
    s2a, s2b, s3 = 16, 16, 8
  elif is_3b:
    s2a, s2b, s3 = 10, 10, 5
  elif is_7b:
    s2a, s2b, s3 = 6, 6, 3
  else:
    s2a, s2b, s3 = 8, 8, 4
elif gb < 40:
    if is_3b:
        if s2a_max_len <= 96 and s2b_max_len <= 96 and s3_max_len <= 96:
            s2a, s2b, s3 = 20, 20, 10
        elif s2a_max_len <= 160 and s2b_max_len <= 160 and s3_max_len <= 160:
            s2a, s2b, s3 = 16, 16, 8
        else:
            s2a, s2b, s3 = 14, 14, 7
    elif is_7b:
        s2a, s2b, s3 = 10, 10, 5
    else:
        s2a, s2b, s3 = 12, 12, 6
elif gb < 56:
    if is_3b:
        if s2a_max_len <= 96 and s2b_max_len <= 96 and s3_max_len <= 96:
            s2a, s2b, s3 = 28, 28, 14
        elif s2a_max_len <= 160 and s2b_max_len <= 160 and s3_max_len <= 160:
            s2a, s2b, s3 = 22, 22, 11
        else:
            s2a, s2b, s3 = 18, 18, 9
    elif is_7b:
        s2a, s2b, s3 = 14, 14, 7
    else:
        s2a, s2b, s3 = 16, 16, 8
elif gb < 80:
    if is_3b:
        if s2a_max_len <= 96 and s2b_max_len <= 96 and s3_max_len <= 96:
            s2a, s2b, s3 = 40, 40, 20
        elif s2a_max_len <= 160 and s2b_max_len <= 160 and s3_max_len <= 160:
            s2a, s2b, s3 = 32, 32, 16
        else:
            s2a, s2b, s3 = 24, 24, 12
    elif is_7b:
        s2a, s2b, s3 = 18, 18, 9
    else:
        s2a, s2b, s3 = 20, 20, 10
else:
    if is_3b:
        s2a, s2b, s3 = 48, 48, 24
    elif is_7b:
        s2a, s2b, s3 = 24, 24, 12
    else:
        s2a, s2b, s3 = 28, 28, 14

if scale > 0 and scale != 1.0:
    s2a = max(1, int(round(s2a * scale)))
    s2b = max(1, int(round(s2b * scale)))
    s3 = max(1, int(round(s3 * scale)))

max_bs = int(os.environ.get("MAX_AUTO_BATCH", "64"))
s2a = min(s2a, max_bs)
s2b = min(s2b, max_bs)
s3 = min(s3, max_bs)

print(f"{gb:.1f} {s2a} {s2b} {s3} {s2b} {s3}")
PY
  )"

  if [ -z "${USER_SET_S2A_BATCH_SIZE:-}" ]; then S2A_BATCH_SIZE="$AUTO_S2A_BATCH"; fi
  if [ -z "${USER_SET_S2B_BATCH_SIZE:-}" ]; then S2B_BATCH_SIZE="$AUTO_S2B_BATCH"; fi
  if [ -z "${USER_SET_S3_BATCH_SIZE:-}" ]; then S3_BATCH_SIZE="$AUTO_S3_BATCH"; fi
  if [ -z "${USER_SET_S2B_R2_BATCH_SIZE:-}" ]; then S2B_R2_BATCH_SIZE="$AUTO_S2B_R2_BATCH"; fi
  if [ -z "${USER_SET_S3_R2_BATCH_SIZE:-}" ]; then S3_R2_BATCH_SIZE="$AUTO_S3_R2_BATCH"; fi
fi

is_oom_log_file() {
  local log_file="$1"
  local pattern="out of memory|cuda out of memory|cuda error: out of memory|cublas_status_alloc_failed"
  if command -v rg >/dev/null 2>&1; then
    rg -qi "$pattern" "$log_file"
  else
    grep -Eqi "$pattern" "$log_file"
  fi
}

extract_save_dir_from_stage_cmd() {
  local -a cmd=("$@")
  local i=0
  local n=${#cmd[@]}
  while [ "$i" -lt "$n" ]; do
    if [ "${cmd[$i]}" = "--save_dir" ]; then
      local j=$((i + 1))
      if [ "$j" -lt "$n" ]; then
        echo "${cmd[$j]}"
        return 0
      fi
    fi
    i=$((i + 1))
  done
  return 1
}

find_resume_ckpt_in_dir() {
  local save_dir="$1"
  if [ "$AUTO_RESUME_ON_OOM" != "1" ]; then
    return 1
  fi
  if [ ! -d "$save_dir" ]; then
    return 1
  fi
  if [ -f "$save_dir/last.ckpt" ]; then
    echo "$save_dir/last.ckpt"
    return 0
  fi
  local latest_ckpt=""
  latest_ckpt=$(ls -1t "$save_dir"/*.ckpt 2>/dev/null | head -n 1 || true)
  if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt" ]; then
    echo "$latest_ckpt"
    return 0
  fi
  return 1
}

run_with_auto_batch_retry() {
  local stage_name="$1"
  local batch_var_name="$2"
  shift 2
  local -a stage_cmd=("$@")
  local batch_size="${!batch_var_name}"
  local try_i=1
  local save_dir=""
  local resume_ckpt=""

  save_dir="$(extract_save_dir_from_stage_cmd "${stage_cmd[@]}" || true)"

  while true; do
    local log_file
    local -a cmd_to_run=("${stage_cmd[@]}")
    log_file="$(mktemp "/tmp/omnigraph_${stage_name}_bs${batch_size}_XXXX.log")"
    echo "[${stage_name}] launch batch_size=${batch_size} (try ${try_i}/${AUTO_BATCH_MAX_RETRIES})"
    if [ "$AUTO_RESUME_ON_OOM" = "1" ] && [ -n "$resume_ckpt" ] && [ -f "$resume_ckpt" ]; then
      echo "[${stage_name}] resume_from_checkpoint=${resume_ckpt}"
      cmd_to_run+=(--resume_from_checkpoint "$resume_ckpt")
    fi

    set +e
    "${cmd_to_run[@]}" --batch_size "$batch_size" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    set -e

    if [ "$rc" -eq 0 ]; then
      printf -v "$batch_var_name" "%s" "$batch_size"
      echo "[${stage_name}] success batch_size=${batch_size}"
      return 0
    fi

    if [ "$AUTO_BATCH_RETRY_ON_OOM" != "1" ]; then
      echo "[${stage_name}] failed (rc=${rc}), auto OOM retry disabled. log=${log_file}"
      return "$rc"
    fi
    if ! is_oom_log_file "$log_file"; then
      echo "[${stage_name}] failed (non-OOM, rc=${rc}). log=${log_file}"
      return "$rc"
    fi
    if [ "$batch_size" -le 1 ]; then
      echo "[${stage_name}] OOM even at batch_size=1. log=${log_file}"
      return "$rc"
    fi
    if [ "$try_i" -ge "$AUTO_BATCH_MAX_RETRIES" ]; then
      echo "[${stage_name}] reached retry limit (${AUTO_BATCH_MAX_RETRIES}). last log=${log_file}"
      return "$rc"
    fi

    if [ -n "$save_dir" ]; then
      resume_ckpt="$(find_resume_ckpt_in_dir "$save_dir" || true)"
      if [ -n "$resume_ckpt" ]; then
        echo "[${stage_name}] found resume checkpoint: ${resume_ckpt}"
      else
        echo "[${stage_name}] no checkpoint found under ${save_dir}; next try will restart from stage init."
      fi
    fi

    local next_batch=$(( (batch_size * 2 + 2) / 3 ))
    if [ "$next_batch" -ge "$batch_size" ]; then
      next_batch=$((batch_size - 1))
    fi
    if [ "$next_batch" -lt 1 ]; then
      next_batch=1
    fi
    echo "[${stage_name}] OOM detected, reduce batch_size ${batch_size} -> ${next_batch}"
    batch_size="$next_batch"
    try_i=$((try_i + 1))
  done
}

resolve_stage2a_ckpt() {
  local ckpt="${STAGE2A_CKPT:-}"
  local meta_path="$STAGE2A_DIR/graph_bootstrap_meta.json"
  if [ ! -f "$meta_path" ]; then
    meta_path="$STAGE2A_DIR/stage2A_meta.json"
  fi
  if [ -z "$ckpt" ]; then
    ckpt=$("$PYTHON_BIN" "$SELECT_CKPT" \
      --meta "$meta_path" \
      --fallback "$STAGE2A_DIR/last.ckpt")
  fi
  if [ ! -f "$ckpt" ] && [ -f "$meta_path" ]; then
    local export_ckpt=""
    export_ckpt=$("$PYTHON_BIN" - "$meta_path" <<'PY'
import json
import os
import sys

meta_path = sys.argv[1]
try:
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit(0)

candidate = str(meta.get("export_path", "") or "").strip()
if candidate and os.path.isfile(candidate):
    print(candidate)
else:
    print("")
PY
)
    if [ -n "$export_ckpt" ] && [ -f "$export_ckpt" ]; then
      echo "[GraphBootstrap] last/best ckpt missing, fallback to exported state_dict: $export_ckpt"
      ckpt="$export_ckpt"
    fi
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[GraphBootstrap] checkpoint not found: $ckpt"
    exit 1
  fi
  STAGE2A_CKPT="$ckpt"
  echo "[GraphBootstrap] using ckpt: $STAGE2A_CKPT"
}

resolve_stage2b_ckpt() {
  local ckpt="${STAGE2B_CKPT:-}"
  if [ -z "$ckpt" ]; then
    local meta_path="$STAGE2B_DIR/graph_refine_meta.json"
    if [ ! -f "$meta_path" ]; then
      meta_path="$STAGE2B_DIR/stage2B_meta.json"
    fi
    ckpt=$("$PYTHON_BIN" "$SELECT_CKPT" \
      --meta "$meta_path" \
      --fallback "$STAGE2B_DIR/last.ckpt")
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[GraphRefine] checkpoint not found: $ckpt"
    exit 1
  fi
  STAGE2B_CKPT="$ckpt"
  echo "[GraphRefine] using ckpt: $STAGE2B_CKPT"
}

echo "[Config] LOW_VRAM_4090=${LOW_VRAM_4090} LLM_MODEL=${LLM_MODEL} VISION_MODEL=${VISION_MODEL}"
echo "[Config] PIPELINE_MODE=${PIPELINE_MODE} run_graph_bootstrap=${RUN_STAGE2A} run_graph_refine=${RUN_STAGE2B} run_multimodal_tune=${RUN_STAGE3} run_eval=${RUN_EVAL}"
echo "[Config] AUTO_BATCH_BY_VRAM=${AUTO_BATCH_BY_VRAM} detected_vram_gb=${GPU_VRAM_GB}"
echo "[Config] AUTO_BATCH_RETRY_ON_OOM=${AUTO_BATCH_RETRY_ON_OOM} AUTO_BATCH_MAX_RETRIES=${AUTO_BATCH_MAX_RETRIES}"
echo "[Config] AUTO_BATCH_SCALE=${AUTO_BATCH_SCALE} MAX_AUTO_BATCH=${MAX_AUTO_BATCH:-64}"
echo "[Config] AUTO_RESUME_ON_OOM=${AUTO_RESUME_ON_OOM}"
if [ "$AUTO_BATCH_BY_VRAM" = "1" ]; then
  if [ -n "${USER_SET_S2A_BATCH_SIZE:-}" ]; then echo "[Config] manual S2A_BATCH_SIZE=${S2A_BATCH_SIZE} -> skip auto init for Stage2A"; fi
  if [ -n "${USER_SET_S2B_BATCH_SIZE:-}" ]; then echo "[Config] manual S2B_BATCH_SIZE=${S2B_BATCH_SIZE} -> skip auto init for Stage2B"; fi
  if [ -n "${USER_SET_S3_BATCH_SIZE:-}" ]; then echo "[Config] manual S3_BATCH_SIZE=${S3_BATCH_SIZE} -> skip auto init for Stage3"; fi
  if [ -n "${USER_SET_S2B_R2_BATCH_SIZE:-}" ]; then echo "[Config] manual S2B_R2_BATCH_SIZE=${S2B_R2_BATCH_SIZE} -> skip auto init for GraphRefine-R2"; fi
  if [ -n "${USER_SET_S3_R2_BATCH_SIZE:-}" ]; then echo "[Config] manual S3_R2_BATCH_SIZE=${S3_R2_BATCH_SIZE} -> skip auto init for MultiModalTune-R2"; fi
fi
echo "[Config] LLM_DTYPE=${LLM_DTYPE} LLM_ATTN_IMPL=${LLM_ATTN_IMPL}"
echo "[Config] NODE_ENCODER_TYPE=${NODE_ENCODER_TYPE} ALPHA=${NODE_ENCODER_ALPHA_INIT} OUT_DIM=${NODE_ENCODER_OUT_DIM}"
echo "[Config] GRAPH_TOKENIZER_TYPE=${GRAPH_TOKENIZER_TYPE} STAGE2A_BOOTSTRAP_MODE=${STAGE2A_BOOTSTRAP_MODE} REQUIRE_STAGE1_CKPT=${REQUIRE_STAGE1_CKPT}"
echo "[Config] PERCEIVER latents=${PERCEIVER_NUM_LATENTS} layers=${PERCEIVER_NUM_LAYERS} heads=${PERCEIVER_NUM_HEADS} ff_mult=${PERCEIVER_FF_MULT} dropout=${PERCEIVER_DROPOUT}"
echo "[Config] QA_TYPE_TOKEN=${USE_QA_TYPE_TOKEN} GVL_ADAPTER=${ENABLE_GVL_ADAPTER} GVL_GATE=${GVL_ADAPTER_GATE_INIT} AUX_HEAD=${ENABLE_GRAPH_AUX_HEAD}"
echo "[Config] AUX_W: S2A=${S2A_GRAPH_AUX_LOSS_WEIGHT} S2B=${S2B_GRAPH_AUX_LOSS_WEIGHT} S3=${S3_GRAPH_AUX_LOSS_WEIGHT} S2B_R2=${S2B_R2_GRAPH_AUX_LOSS_WEIGHT} S3_R2=${S3_R2_GRAPH_AUX_LOSS_WEIGHT}"
echo "[Config] ALIGN: ENABLE_XTC=${ENABLE_XTC} ENABLE_XTM=${ENABLE_XTM} SCALE_INIT=${XTC_LOGIT_SCALE_INIT} DUP_THRESH=${XTM_DUP_THRESH}"
echo "[Config] XTC/XTM W: S2A=${S2A_XTC_WEIGHT}/${S2A_XTM_WEIGHT} S2B=${S2B_XTC_WEIGHT}/${S2B_XTM_WEIGHT} S3=${S3_XTC_WEIGHT}/${S3_XTM_WEIGHT} S2B_R2=${S2B_R2_XTC_WEIGHT}/${S2B_R2_XTM_WEIGHT} S3_R2=${S3_R2_XTC_WEIGHT}/${S3_R2_XTM_WEIGHT}"
echo "[Config] XGV: S3(enable=${S3_ENABLE_XGV},weight=${S3_XGV_WEIGHT}) S3_R2(enable=${S3_R2_ENABLE_XGV},weight=${S3_R2_XGV_WEIGHT})"
echo "[Config] Stage3V2: curriculum=${S3_ENABLE_STAGE3_CURRICULUM} A<${S3_CURRICULUM_PHASE_A_STEPS} B<${S3_CURRICULUM_PHASE_B_STEPS} align(A/C)=${S3_PHASE_A_ALIGN_SCALE}/${S3_PHASE_C_ALIGN_SCALE} aux(A/C)=${S3_PHASE_A_AUX_SCALE}/${S3_PHASE_C_AUX_SCALE}"
echo "[Config] Stage3V2: modality_drop(graph/vision)=${S3_TRAIN_GRAPH_DROP_PROB}/${S3_TRAIN_VISION_DROP_PROB} qa_type_gate=${S3_ENABLE_QA_TYPE_MODALITY_GATE}"
echo "[Config] EXTRA_SCENE_GRAPHS=${EXTRA_SCENE_GRAPHS:-<none>}"
echo "[Config] pseudo_qa: S2A(max=${S2A_PSEUDO_GRAPH_QA_MAX_PER_IMAGE},repeat=${S2A_PSEUDO_GRAPH_QA_REPEAT}) S2B(max=${S2B_PSEUDO_GRAPH_QA_MAX_PER_IMAGE},repeat=${S2B_PSEUDO_GRAPH_QA_REPEAT}) S3(max=${S3_PSEUDO_GRAPH_QA_MAX_PER_IMAGE},repeat=${S3_PSEUDO_GRAPH_QA_REPEAT})"
echo "[Config] pseudo_qa_r2: S2B(max=${S2B_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE},repeat=${S2B_R2_PSEUDO_GRAPH_QA_REPEAT}) S3(max=${S3_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE},repeat=${S3_R2_PSEUDO_GRAPH_QA_REPEAT})"
echo "[Config] S2A bs=${S2A_BATCH_SIZE} max_len=${S2A_MAX_LENGTH} workers=${S2A_NUM_WORKERS} prec=${S2A_PRECISION}"
echo "[Config] S2B bs=${S2B_BATCH_SIZE} max_len=${S2B_MAX_LENGTH} workers=${S2B_NUM_WORKERS} prec=${S2B_PRECISION}"
echo "[Config] S3  bs=${S3_BATCH_SIZE} max_len=${S3_MAX_LENGTH} workers=${S3_NUM_WORKERS} prec=${S3_PRECISION}"
echo "[Config] llm_grad_ckpt: S2A=${S2A_ENABLE_LLM_GRADIENT_CHECKPOINTING} S2B=${S2B_ENABLE_LLM_GRADIENT_CHECKPOINTING} S3=${S3_ENABLE_LLM_GRADIENT_CHECKPOINTING} S2B_R2=${S2B_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING} S3_R2=${S3_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING}"
echo "[Config] prefetch: S2A=${S2A_PREFETCH_FACTOR} S2B=${S2B_PREFETCH_FACTOR} S3=${S3_PREFETCH_FACTOR} S2B_R2=${S2B_R2_PREFETCH_FACTOR} S3_R2=${S3_R2_PREFETCH_FACTOR}"
echo "[Config] manual_stop: S2A=${S2A_MANUAL_STOP_FILE} S2B=${S2B_MANUAL_STOP_FILE} S3=${S3_MANUAL_STOP_FILE}"
echo "[Config] step_ckpt: S2A=${S2A_CHECKPOINT_EVERY_N_STEPS} S2B=${S2B_CHECKPOINT_EVERY_N_STEPS} S3=${S3_CHECKPOINT_EVERY_N_STEPS} S2B_R2=${S2B_R2_CHECKPOINT_EVERY_N_STEPS} S3_R2=${S3_R2_CHECKPOINT_EVERY_N_STEPS}"
echo "[Config] val: S2A(interval=${S2A_VAL_CHECK_INTERVAL},limit=${S2A_LIMIT_VAL_BATCHES}) S2B(interval=${S2B_VAL_CHECK_INTERVAL},limit=${S2B_LIMIT_VAL_BATCHES}) S3(interval=${S3_VAL_CHECK_INTERVAL},limit=${S3_LIMIT_VAL_BATCHES})"
echo "[Config] train_node_encoder: S2A=${S2A_TRAIN_NODE_ENCODER} S2B=${S2B_TRAIN_NODE_ENCODER} S3=${S3_TRAIN_NODE_ENCODER} S2B_R2=${S2B_R2_TRAIN_NODE_ENCODER} S3_R2=${S3_R2_TRAIN_NODE_ENCODER}"
if [ "$RUN_STAGE2A" = "1" ] && [ "$REQUIRE_STAGE1_CKPT" = "1" ]; then
  test -f "$STAGE1_QFORMER_CKPT"
fi
if [ "$RUN_STAGE2A" = "1" ] || [ "$RUN_STAGE2B" = "1" ] || [ "$RUN_STAGE3" = "1" ]; then
  test -f "$VG_SCENE_GRAPHS"
  test -f "$VG_REGIONS"
  test -f "$SELECT_CKPT"
  if [ -n "${EXTRA_SCENE_GRAPHS}" ]; then
    IFS=',' read -r -a _extra_scene_paths <<< "$EXTRA_SCENE_GRAPHS"
    for _sg in "${_extra_scene_paths[@]}"; do
      _sg="$(printf "%s" "$_sg" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
      [ -z "$_sg" ] && continue
      test -f "$_sg"
    done
    unset _extra_scene_paths _sg
  fi
fi
if [ "$RUN_STAGE3" = "1" ]; then
  test -d "$VG_IMAGE_ROOT" || true
fi
if [ "$RUN_EVAL" = "1" ]; then
  if [ ! -f "$GQA_QUESTIONS_JSON" ]; then
    echo "[GQA][Error] missing file: GQA_QUESTIONS_JSON=$GQA_QUESTIONS_JSON"
    echo "[GQA][Hint] provide --data-config with GQA_* paths or set RUN_EVAL=0 for training-only run."
    exit 1
  fi
  if [ ! -f "$GQA_SCENE_RAW" ]; then
    echo "[GQA][Error] missing file: GQA_SCENE_RAW=$GQA_SCENE_RAW"
    echo "[GQA][Hint] provide --data-config with GQA_* paths or set RUN_EVAL=0 for training-only run."
    exit 1
  fi
fi

if [ "$RUN_STAGE2A" = "1" ]; then
  echo "[GraphBootstrap] start"
  S2A_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_graph_bootstrap.py" \
    --scene_graphs "$VG_SCENE_GRAPHS" \
    --regions "$VG_REGIONS" \
    --extra_scene_graphs "$EXTRA_SCENE_GRAPHS" \
    --bootstrap_mode "$STAGE2A_BOOTSTRAP_MODE" \
    --llm "$LLM_MODEL" \
    --llm_dtype "$LLM_DTYPE" \
    --llm_attn_implementation "$LLM_ATTN_IMPL" \
    --enable_llm_gradient_checkpointing "$S2A_ENABLE_LLM_GRADIENT_CHECKPOINTING" \
    --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
    --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
    --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
    --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
    --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
    --perceiver_dropout "$PERCEIVER_DROPOUT" \
    --node_encoder_type "$NODE_ENCODER_TYPE" \
    --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
    --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
    --use_qa_type_token "$USE_QA_TYPE_TOKEN" \
    --enable_gvl_adapter "$ENABLE_GVL_ADAPTER" \
    --gvl_adapter_gate_init "$GVL_ADAPTER_GATE_INIT" \
    --enable_graph_aux_head "$ENABLE_GRAPH_AUX_HEAD" \
    --graph_aux_loss_weight "$S2A_GRAPH_AUX_LOSS_WEIGHT" \
    --enable_xtc "$ENABLE_XTC" \
    --enable_xtm "$ENABLE_XTM" \
    --xtc_weight "$S2A_XTC_WEIGHT" \
    --xtm_weight "$S2A_XTM_WEIGHT" \
    --xtc_logit_scale_init "$XTC_LOGIT_SCALE_INIT" \
    --xtm_dup_thresh "$XTM_DUP_THRESH" \
    --train_node_encoder "$S2A_TRAIN_NODE_ENCODER" \
    --graph_qa_max_per_image "$S2A_GRAPH_QA_MAX_PER_IMAGE" \
    --graph_qa_repeat "$S2A_GRAPH_QA_REPEAT" \
    --pseudo_graph_qa_max_per_image "$S2A_PSEUDO_GRAPH_QA_MAX_PER_IMAGE" \
    --pseudo_graph_qa_repeat "$S2A_PSEUDO_GRAPH_QA_REPEAT" \
    --gpu "$GPU" \
    --precision "$S2A_PRECISION" \
    --max_length "$S2A_MAX_LENGTH" \
    --max_graph_tokens "$S2A_MAX_GRAPH_TOKENS" \
    --num_workers "$S2A_NUM_WORKERS" \
    --prefetch_factor "$S2A_PREFETCH_FACTOR" \
    --num_sanity_val_steps "$S2A_NUM_SANITY_VAL_STEPS" \
    --accumulate_grad_batches "$S2A_ACCUM_GRAD_BATCHES" \
    --val_ratio "$S2A_VAL_RATIO" \
    --patience "$S2A_PATIENCE" \
    --min_delta "$S2A_MIN_DELTA" \
    --lr "$S2A_LR" \
    --max_steps "$S2A_MAX_STEPS" \
    --val_check_interval "$S2A_VAL_CHECK_INTERVAL" \
    --limit_val_batches "$S2A_LIMIT_VAL_BATCHES" \
    --checkpoint_every_n_steps "$S2A_CHECKPOINT_EVERY_N_STEPS" \
    --manual_stop_file "$S2A_MANUAL_STOP_FILE" \
    --save_dir "$STAGE2A_DIR")
  if [ "$REQUIRE_STAGE1_CKPT" = "1" ]; then
    S2A_CMD+=(--graph_tokenizer_ckpt "$STAGE1_QFORMER_CKPT")
  fi
  run_with_auto_batch_retry "GraphBootstrap" "S2A_BATCH_SIZE" "${S2A_CMD[@]}"
  resolve_stage2a_ckpt
elif [ "$RUN_STAGE2B" = "1" ]; then
  resolve_stage2a_ckpt
fi

if [ "$RUN_STAGE2B" = "1" ]; then
  echo "[GraphRefine] start"
  S2B_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_graph_refine.py" \
    --scene_graphs "$VG_SCENE_GRAPHS" \
    --regions "$VG_REGIONS" \
    --extra_scene_graphs "$EXTRA_SCENE_GRAPHS" \
    --graph_bootstrap_ckpt "$STAGE2A_CKPT" \
    --llm "$LLM_MODEL" \
    --llm_dtype "$LLM_DTYPE" \
    --llm_attn_implementation "$LLM_ATTN_IMPL" \
    --enable_llm_gradient_checkpointing "$S2B_ENABLE_LLM_GRADIENT_CHECKPOINTING" \
    --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
    --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
    --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
    --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
    --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
    --perceiver_dropout "$PERCEIVER_DROPOUT" \
    --node_encoder_type "$NODE_ENCODER_TYPE" \
    --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
    --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
    --use_qa_type_token "$USE_QA_TYPE_TOKEN" \
    --enable_gvl_adapter "$ENABLE_GVL_ADAPTER" \
    --gvl_adapter_gate_init "$GVL_ADAPTER_GATE_INIT" \
    --enable_graph_aux_head "$ENABLE_GRAPH_AUX_HEAD" \
    --graph_aux_loss_weight "$S2B_GRAPH_AUX_LOSS_WEIGHT" \
    --enable_xtc "$ENABLE_XTC" \
    --enable_xtm "$ENABLE_XTM" \
    --xtc_weight "$S2B_XTC_WEIGHT" \
    --xtm_weight "$S2B_XTM_WEIGHT" \
    --xtc_logit_scale_init "$XTC_LOGIT_SCALE_INIT" \
    --xtm_dup_thresh "$XTM_DUP_THRESH" \
    --train_node_encoder "$S2B_TRAIN_NODE_ENCODER" \
    --graph_qa_max_per_image "$S2B_GRAPH_QA_MAX_PER_IMAGE" \
    --graph_qa_repeat "$S2B_GRAPH_QA_REPEAT" \
    --pseudo_graph_qa_max_per_image "$S2B_PSEUDO_GRAPH_QA_MAX_PER_IMAGE" \
    --pseudo_graph_qa_repeat "$S2B_PSEUDO_GRAPH_QA_REPEAT" \
    --gpu "$GPU" \
    --precision "$S2B_PRECISION" \
    --max_length "$S2B_MAX_LENGTH" \
    --max_graph_tokens "$S2B_MAX_GRAPH_TOKENS" \
    --num_workers "$S2B_NUM_WORKERS" \
    --prefetch_factor "$S2B_PREFETCH_FACTOR" \
    --num_sanity_val_steps "$S2B_NUM_SANITY_VAL_STEPS" \
    --accumulate_grad_batches "$S2B_ACCUM_GRAD_BATCHES" \
    --lr "$S2B_LR" \
    --max_steps "$S2B_MAX_STEPS" \
    --val_ratio "$S2B_VAL_RATIO" \
    --val_check_interval "$S2B_VAL_CHECK_INTERVAL" \
    --limit_val_batches "$S2B_LIMIT_VAL_BATCHES" \
    --checkpoint_every_n_steps "$S2B_CHECKPOINT_EVERY_N_STEPS" \
    --manual_stop_file "$S2B_MANUAL_STOP_FILE" \
    --patience "$S2B_PATIENCE" \
    --min_delta "$S2B_MIN_DELTA" \
    --save_dir "$STAGE2B_DIR")
  run_with_auto_batch_retry "GraphRefine" "S2B_BATCH_SIZE" "${S2B_CMD[@]}"
  resolve_stage2b_ckpt
elif [ "$RUN_STAGE3" = "1" ]; then
  resolve_stage2b_ckpt
fi

if [ "$RUN_STAGE3" = "1" ]; then
  echo "[MultiModalTune] start"
  S3_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_multimodal_tune.py" \
    --scene_graphs "$VG_SCENE_GRAPHS" \
    --regions "$VG_REGIONS" \
    --extra_scene_graphs "$EXTRA_SCENE_GRAPHS" \
    --image_root "$VG_IMAGE_ROOT" \
    --graph_refine_ckpt "$STAGE2B_CKPT" \
    --llm "$LLM_MODEL" \
    --llm_dtype "$LLM_DTYPE" \
    --llm_attn_implementation "$LLM_ATTN_IMPL" \
    --enable_llm_gradient_checkpointing "$S3_ENABLE_LLM_GRADIENT_CHECKPOINTING" \
    --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
    --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
    --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
    --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
    --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
    --perceiver_dropout "$PERCEIVER_DROPOUT" \
    --node_encoder_type "$NODE_ENCODER_TYPE" \
    --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
    --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
    --use_qa_type_token "$USE_QA_TYPE_TOKEN" \
    --enable_gvl_adapter "$ENABLE_GVL_ADAPTER" \
    --gvl_adapter_gate_init "$GVL_ADAPTER_GATE_INIT" \
    --enable_graph_aux_head "$ENABLE_GRAPH_AUX_HEAD" \
    --graph_aux_loss_weight "$S3_GRAPH_AUX_LOSS_WEIGHT" \
    --enable_xtc "$ENABLE_XTC" \
    --enable_xtm "$ENABLE_XTM" \
    --xtc_weight "$S3_XTC_WEIGHT" \
    --xtm_weight "$S3_XTM_WEIGHT" \
    --xtc_logit_scale_init "$XTC_LOGIT_SCALE_INIT" \
    --xtm_dup_thresh "$XTM_DUP_THRESH" \
    --enable_xgv "$S3_ENABLE_XGV" \
    --xgv_weight "$S3_XGV_WEIGHT" \
    --enable_stage3_curriculum "$S3_ENABLE_STAGE3_CURRICULUM" \
    --curriculum_phase_a_steps "$S3_CURRICULUM_PHASE_A_STEPS" \
    --curriculum_phase_b_steps "$S3_CURRICULUM_PHASE_B_STEPS" \
    --phase_a_align_scale "$S3_PHASE_A_ALIGN_SCALE" \
    --phase_c_align_scale "$S3_PHASE_C_ALIGN_SCALE" \
    --phase_a_aux_scale "$S3_PHASE_A_AUX_SCALE" \
    --phase_c_aux_scale "$S3_PHASE_C_AUX_SCALE" \
    --train_graph_drop_prob "$S3_TRAIN_GRAPH_DROP_PROB" \
    --train_vision_drop_prob "$S3_TRAIN_VISION_DROP_PROB" \
    --enable_qa_type_modality_gate "$S3_ENABLE_QA_TYPE_MODALITY_GATE" \
    --train_node_encoder "$S3_TRAIN_NODE_ENCODER" \
    --vision "$VISION_MODEL" \
    --graph_qa_max_per_image "$S3_GRAPH_QA_MAX_PER_IMAGE" \
    --graph_qa_repeat "$S3_GRAPH_QA_REPEAT" \
    --pseudo_graph_qa_max_per_image "$S3_PSEUDO_GRAPH_QA_MAX_PER_IMAGE" \
    --pseudo_graph_qa_repeat "$S3_PSEUDO_GRAPH_QA_REPEAT" \
    --gpu "$GPU" \
    --precision "$S3_PRECISION" \
    --max_length "$S3_MAX_LENGTH" \
    --max_graph_tokens "$S3_MAX_GRAPH_TOKENS" \
    --max_vision_tokens "$S3_MAX_VISION_TOKENS" \
    --num_workers "$S3_NUM_WORKERS" \
    --prefetch_factor "$S3_PREFETCH_FACTOR" \
    --num_sanity_val_steps "$S3_NUM_SANITY_VAL_STEPS" \
    --accumulate_grad_batches "$S3_ACCUM_GRAD_BATCHES" \
    --lr "$S3_LR" \
    --max_steps "$S3_MAX_STEPS" \
    --val_ratio "$S3_VAL_RATIO" \
    --val_check_interval "$S3_VAL_CHECK_INTERVAL" \
    --limit_val_batches "$S3_LIMIT_VAL_BATCHES" \
    --checkpoint_every_n_steps "$S3_CHECKPOINT_EVERY_N_STEPS" \
    --manual_stop_file "$S3_MANUAL_STOP_FILE" \
    --patience "$S3_PATIENCE" \
    --min_delta "$S3_MIN_DELTA" \
    --save_dir "$STAGE3_DIR")
  run_with_auto_batch_retry "MultiModalTune" "S3_BATCH_SIZE" "${S3_CMD[@]}"
fi

if [ "$RUN_EVAL" != "1" ]; then
  echo "[Done] mode=${PIPELINE_MODE} completed (training-only)."
  exit 0
fi

if [ ! -f "$STAGE3_STATE_DICT" ]; then
  LEGACY_STAGE3_STATE_DICT="$STAGE3_DIR/omnigraph_stage3_state_dict.pt"
  if [ -f "$LEGACY_STAGE3_STATE_DICT" ]; then
    STAGE3_STATE_DICT="$LEGACY_STAGE3_STATE_DICT"
  else
    echo "[GQA] multimodal_tune state_dict not found: $STAGE3_STATE_DICT"
    exit 1
  fi
fi

echo "[GQA] prepare converted files"
if [ ! -f "$GQA_SCENE_VG" ]; then
  "$PYTHON_BIN" "$REPO/scripts/data_prep/convert_gqa_scene_graphs.py" \
    --input "$GQA_SCENE_RAW" \
    --output "$GQA_SCENE_VG"
fi

if [ ! -f "$GQA_QUESTIONS_JSONL" ]; then
  "$PYTHON_BIN" "$REPO/scripts/data_prep/convert_gqa_questions.py" \
    --input "$GQA_QUESTIONS_JSON" \
    --output "$GQA_QUESTIONS_JSONL" \
    --image_root "$GQA_IMAGE_ROOT"
fi

echo "[GQA] infer paper run"
"$PYTHON_BIN" "$REPO/scripts/eval/infer_gqa.py" \
  --questions "$GQA_QUESTIONS_JSONL" \
  --scene_graphs "$GQA_SCENE_VG" \
  --image_root "$GQA_IMAGE_ROOT" \
  --ckpt "$STAGE3_STATE_DICT" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
  --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
  --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
  --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
  --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
  --perceiver_dropout "$PERCEIVER_DROPOUT" \
  --node_encoder_type "$NODE_ENCODER_TYPE" \
  --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
  --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
  --output "$GQA_PRED_PAPER" \
  --batch_size 1 \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu "$GPU" \
  --max_samples 0 \
  --log_every 500

echo "[GQA] eval paper run"
"$PYTHON_BIN" "$REPO/scripts/eval/eval_gqa_accuracy.py" \
  --gt "$GQA_QUESTIONS_JSONL" \
  --pred "$GQA_PRED_PAPER" \
  --query_target "$QUERY_TARGET" \
  --query_sprint_target "$QUERY_SPRINT_TARGET" | tee "$GQA_EVAL_PAPER"

read -r STRICT_SCORE COVERAGE_SCORE QUERY_SCORE <<<"$("$PYTHON_BIN" - "$GQA_EVAL_PAPER" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
strict = 0.0
coverage = 0.0
query = 0.0

strict_patterns = [
    r"Accuracy \(strict, all GT\):\s*([0-9]*\.?[0-9]+)",
    r"Accuracy:\s*([0-9]*\.?[0-9]+)",
]
coverage_patterns = [
    r"Coverage:\s*([0-9]*\.?[0-9]+)",
]
query_patterns = [
    r"Query accuracy \(strict\):\s*([0-9]*\.?[0-9]+)",
    r"^\s*query:\s*([0-9]*\.?[0-9]+)",
]

for p in strict_patterns:
    m = re.search(p, text)
    if m:
        strict = float(m.group(1))
        break
for p in coverage_patterns:
    m = re.search(p, text)
    if m:
        coverage = float(m.group(1))
        break
for p in query_patterns:
    m = re.search(p, text, flags=re.MULTILINE)
    if m:
        query = float(m.group(1))
        break

print(f"{strict} {coverage} {query}")
PY
)"

echo "[Result] strict=${STRICT_SCORE} coverage=${COVERAGE_SCORE} query=${QUERY_SCORE} baseline=${STRICT_BASELINE} target=${STRICT_TARGET} sprint=${STRICT_SPRINT_TARGET} query_target=${QUERY_TARGET} query_sprint=${QUERY_SPRINT_TARGET}"

COVERAGE_OK=$("$PYTHON_BIN" - "$COVERAGE_SCORE" "$COVERAGE_TARGET" <<'PY'
import sys
coverage = float(sys.argv[1])
target = float(sys.argv[2])
print("1" if coverage >= target else "0")
PY
)
if [ "$COVERAGE_OK" != "1" ]; then
  echo "[Fail] coverage below target: ${COVERAGE_SCORE} < ${COVERAGE_TARGET}"
  exit 2
fi

QUERY_OK=$("$PYTHON_BIN" - "$QUERY_SCORE" "$QUERY_TARGET" <<'PY'
import sys
query = float(sys.argv[1])
target = float(sys.argv[2])
print("1" if query >= target else "0")
PY
)
if [ "$QUERY_OK" != "1" ]; then
  echo "[Warn] query below target: ${QUERY_SCORE} < ${QUERY_TARGET}"
fi

RUN_R2=$("$PYTHON_BIN" - "$STRICT_SCORE" "$STRICT_TARGET" "$QUERY_SCORE" "$QUERY_TARGET" "$RUN_ROUND2_ON_FAIL" <<'PY'
import sys
strict = float(sys.argv[1])
target = float(sys.argv[2])
query = float(sys.argv[3])
query_target = float(sys.argv[4])
flag = int(sys.argv[5])
need_retry = (strict < target) or (query < query_target)
print("1" if (flag == 1 and need_retry) else "0")
PY
)

if [ "$RUN_R2" != "1" ]; then
  echo "[Done] round-2 skipped."
  exit 0
fi

echo "[Round2] strict/query below target, start fixed round-2 recipe."
S2B_R2_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_graph_refine.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --extra_scene_graphs "$EXTRA_SCENE_GRAPHS" \
  --graph_bootstrap_ckpt "$STAGE2A_CKPT" \
  --llm "$LLM_MODEL" \
  --llm_dtype "$LLM_DTYPE" \
  --llm_attn_implementation "$LLM_ATTN_IMPL" \
  --enable_llm_gradient_checkpointing "$S2B_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING" \
  --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
  --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
  --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
  --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
  --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
  --perceiver_dropout "$PERCEIVER_DROPOUT" \
  --node_encoder_type "$NODE_ENCODER_TYPE" \
  --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
  --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
  --use_qa_type_token "$USE_QA_TYPE_TOKEN" \
  --enable_gvl_adapter "$ENABLE_GVL_ADAPTER" \
  --gvl_adapter_gate_init "$GVL_ADAPTER_GATE_INIT" \
  --enable_graph_aux_head "$ENABLE_GRAPH_AUX_HEAD" \
  --graph_aux_loss_weight "$S2B_R2_GRAPH_AUX_LOSS_WEIGHT" \
  --enable_xtc "$ENABLE_XTC" \
  --enable_xtm "$ENABLE_XTM" \
  --xtc_weight "$S2B_R2_XTC_WEIGHT" \
  --xtm_weight "$S2B_R2_XTM_WEIGHT" \
  --xtc_logit_scale_init "$XTC_LOGIT_SCALE_INIT" \
  --xtm_dup_thresh "$XTM_DUP_THRESH" \
  --train_node_encoder "$S2B_R2_TRAIN_NODE_ENCODER" \
  --graph_qa_max_per_image "$S2B_R2_GRAPH_QA_MAX_PER_IMAGE" \
  --graph_qa_repeat "$S2B_R2_GRAPH_QA_REPEAT" \
  --pseudo_graph_qa_max_per_image "$S2B_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE" \
  --pseudo_graph_qa_repeat "$S2B_R2_PSEUDO_GRAPH_QA_REPEAT" \
  --gpu "$GPU" \
  --precision "$S2B_R2_PRECISION" \
  --max_length "$S2B_R2_MAX_LENGTH" \
  --max_graph_tokens "$S2B_R2_MAX_GRAPH_TOKENS" \
  --num_workers "$S2B_R2_NUM_WORKERS" \
  --prefetch_factor "$S2B_R2_PREFETCH_FACTOR" \
  --num_sanity_val_steps "$S2B_R2_NUM_SANITY_VAL_STEPS" \
  --accumulate_grad_batches "$S2B_R2_ACCUM_GRAD_BATCHES" \
  --lr "$S2B_R2_LR" \
  --max_steps "$S2B_R2_MAX_STEPS" \
  --val_ratio "$S2B_R2_VAL_RATIO" \
  --val_check_interval "$S2B_R2_VAL_CHECK_INTERVAL" \
  --limit_val_batches "$S2B_R2_LIMIT_VAL_BATCHES" \
  --checkpoint_every_n_steps "$S2B_R2_CHECKPOINT_EVERY_N_STEPS" \
  --manual_stop_file "$S2B_R2_MANUAL_STOP_FILE" \
  --patience "$S2B_R2_PATIENCE" \
  --min_delta "$S2B_R2_MIN_DELTA" \
  --save_dir "$STAGE2B_R2_DIR")
run_with_auto_batch_retry "GraphRefine-R2" "S2B_R2_BATCH_SIZE" "${S2B_R2_CMD[@]}"

S2B_R2_META="$STAGE2B_R2_DIR/graph_refine_meta.json"
if [ ! -f "$S2B_R2_META" ]; then
  S2B_R2_META="$STAGE2B_R2_DIR/stage2B_meta.json"
fi
STAGE2B_R2_CKPT=$("$PYTHON_BIN" "$SELECT_CKPT" \
  --meta "$S2B_R2_META" \
  --fallback "$STAGE2B_R2_DIR/last.ckpt")
echo "[Round2] GraphRefine ckpt: $STAGE2B_R2_CKPT"
test -f "$STAGE2B_R2_CKPT"

S3_R2_CMD=("$PYTHON_BIN" "$REPO/omnigraph/train/train_multimodal_tune.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --extra_scene_graphs "$EXTRA_SCENE_GRAPHS" \
  --image_root "$VG_IMAGE_ROOT" \
  --graph_refine_ckpt "$STAGE2B_R2_CKPT" \
  --llm "$LLM_MODEL" \
  --llm_dtype "$LLM_DTYPE" \
  --llm_attn_implementation "$LLM_ATTN_IMPL" \
  --enable_llm_gradient_checkpointing "$S3_R2_ENABLE_LLM_GRADIENT_CHECKPOINTING" \
  --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
  --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
  --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
  --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
  --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
  --perceiver_dropout "$PERCEIVER_DROPOUT" \
  --node_encoder_type "$NODE_ENCODER_TYPE" \
  --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
  --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
  --use_qa_type_token "$USE_QA_TYPE_TOKEN" \
  --enable_gvl_adapter "$ENABLE_GVL_ADAPTER" \
  --gvl_adapter_gate_init "$GVL_ADAPTER_GATE_INIT" \
  --enable_graph_aux_head "$ENABLE_GRAPH_AUX_HEAD" \
  --graph_aux_loss_weight "$S3_R2_GRAPH_AUX_LOSS_WEIGHT" \
  --enable_xtc "$ENABLE_XTC" \
  --enable_xtm "$ENABLE_XTM" \
  --xtc_weight "$S3_R2_XTC_WEIGHT" \
  --xtm_weight "$S3_R2_XTM_WEIGHT" \
  --xtc_logit_scale_init "$XTC_LOGIT_SCALE_INIT" \
  --xtm_dup_thresh "$XTM_DUP_THRESH" \
  --enable_xgv "$S3_R2_ENABLE_XGV" \
  --xgv_weight "$S3_R2_XGV_WEIGHT" \
  --enable_stage3_curriculum "$S3_R2_ENABLE_STAGE3_CURRICULUM" \
  --curriculum_phase_a_steps "$S3_R2_CURRICULUM_PHASE_A_STEPS" \
  --curriculum_phase_b_steps "$S3_R2_CURRICULUM_PHASE_B_STEPS" \
  --phase_a_align_scale "$S3_R2_PHASE_A_ALIGN_SCALE" \
  --phase_c_align_scale "$S3_R2_PHASE_C_ALIGN_SCALE" \
  --phase_a_aux_scale "$S3_R2_PHASE_A_AUX_SCALE" \
  --phase_c_aux_scale "$S3_R2_PHASE_C_AUX_SCALE" \
  --train_graph_drop_prob "$S3_R2_TRAIN_GRAPH_DROP_PROB" \
  --train_vision_drop_prob "$S3_R2_TRAIN_VISION_DROP_PROB" \
  --enable_qa_type_modality_gate "$S3_R2_ENABLE_QA_TYPE_MODALITY_GATE" \
  --train_node_encoder "$S3_R2_TRAIN_NODE_ENCODER" \
  --vision "$VISION_MODEL" \
  --graph_qa_max_per_image "$S3_R2_GRAPH_QA_MAX_PER_IMAGE" \
  --graph_qa_repeat "$S3_R2_GRAPH_QA_REPEAT" \
  --pseudo_graph_qa_max_per_image "$S3_R2_PSEUDO_GRAPH_QA_MAX_PER_IMAGE" \
  --pseudo_graph_qa_repeat "$S3_R2_PSEUDO_GRAPH_QA_REPEAT" \
  --gpu "$GPU" \
  --precision "$S3_R2_PRECISION" \
  --max_length "$S3_R2_MAX_LENGTH" \
  --max_graph_tokens "$S3_R2_MAX_GRAPH_TOKENS" \
  --max_vision_tokens "$S3_R2_MAX_VISION_TOKENS" \
  --num_workers "$S3_R2_NUM_WORKERS" \
  --prefetch_factor "$S3_R2_PREFETCH_FACTOR" \
  --num_sanity_val_steps "$S3_R2_NUM_SANITY_VAL_STEPS" \
  --accumulate_grad_batches "$S3_R2_ACCUM_GRAD_BATCHES" \
  --lr "$S3_R2_LR" \
  --max_steps "$S3_R2_MAX_STEPS" \
  --val_ratio "$S3_R2_VAL_RATIO" \
  --val_check_interval "$S3_R2_VAL_CHECK_INTERVAL" \
  --limit_val_batches "$S3_R2_LIMIT_VAL_BATCHES" \
  --checkpoint_every_n_steps "$S3_R2_CHECKPOINT_EVERY_N_STEPS" \
  --manual_stop_file "$S3_R2_MANUAL_STOP_FILE" \
  --patience "$S3_R2_PATIENCE" \
  --min_delta "$S3_R2_MIN_DELTA" \
  --save_dir "$STAGE3_R2_DIR")
run_with_auto_batch_retry "MultiModalTune-R2" "S3_R2_BATCH_SIZE" "${S3_R2_CMD[@]}"

if [ ! -f "$STAGE3_R2_STATE_DICT" ]; then
  LEGACY_STAGE3_R2_STATE_DICT="$STAGE3_R2_DIR/omnigraph_stage3_state_dict.pt"
  if [ -f "$LEGACY_STAGE3_R2_STATE_DICT" ]; then
    STAGE3_R2_STATE_DICT="$LEGACY_STAGE3_R2_STATE_DICT"
  fi
fi

"$PYTHON_BIN" "$REPO/scripts/eval/infer_gqa.py" \
  --questions "$GQA_QUESTIONS_JSONL" \
  --scene_graphs "$GQA_SCENE_VG" \
  --image_root "$GQA_IMAGE_ROOT" \
  --ckpt "$STAGE3_R2_STATE_DICT" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --graph_tokenizer_type "$GRAPH_TOKENIZER_TYPE" \
  --perceiver_num_latents "$PERCEIVER_NUM_LATENTS" \
  --perceiver_num_layers "$PERCEIVER_NUM_LAYERS" \
  --perceiver_num_heads "$PERCEIVER_NUM_HEADS" \
  --perceiver_ff_mult "$PERCEIVER_FF_MULT" \
  --perceiver_dropout "$PERCEIVER_DROPOUT" \
  --node_encoder_type "$NODE_ENCODER_TYPE" \
  --node_encoder_alpha_init "$NODE_ENCODER_ALPHA_INIT" \
  --node_encoder_out_dim "$NODE_ENCODER_OUT_DIM" \
  --output "$GQA_PRED_R2" \
  --batch_size 1 \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu "$GPU" \
  --max_samples 0 \
  --log_every 500

"$PYTHON_BIN" "$REPO/scripts/eval/eval_gqa_accuracy.py" \
  --gt "$GQA_QUESTIONS_JSONL" \
  --pred "$GQA_PRED_R2" \
  --query_target "$QUERY_TARGET" \
  --query_sprint_target "$QUERY_SPRINT_TARGET" | tee "$GQA_EVAL_R2"

read -r STRICT_SCORE_R2 COVERAGE_SCORE_R2 QUERY_SCORE_R2 <<<"$("$PYTHON_BIN" - "$GQA_EVAL_R2" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
strict = 0.0
coverage = 0.0
query = 0.0

for p in (
    r"Accuracy \(strict, all GT\):\s*([0-9]*\.?[0-9]+)",
    r"Accuracy:\s*([0-9]*\.?[0-9]+)",
):
    m = re.search(p, text)
    if m:
        strict = float(m.group(1))
        break

for p in (r"Coverage:\s*([0-9]*\.?[0-9]+)",):
    m = re.search(p, text)
    if m:
        coverage = float(m.group(1))
        break

for p in (
    r"Query accuracy \(strict\):\s*([0-9]*\.?[0-9]+)",
    r"^\s*query:\s*([0-9]*\.?[0-9]+)",
):
    m = re.search(p, text, flags=re.MULTILINE)
    if m:
        query = float(m.group(1))
        break

print(f"{strict} {coverage} {query}")
PY
)"

echo "[Round2 Result] strict=${STRICT_SCORE_R2} coverage=${COVERAGE_SCORE_R2} query=${QUERY_SCORE_R2} strict_sprint=${STRICT_SPRINT_TARGET} query_sprint=${QUERY_SPRINT_TARGET}"

R2_SPRINT_OK=$("$PYTHON_BIN" - "$STRICT_SCORE_R2" "$STRICT_SPRINT_TARGET" "$QUERY_SCORE_R2" "$QUERY_SPRINT_TARGET" <<'PY'
import sys
strict = float(sys.argv[1])
strict_target = float(sys.argv[2])
query = float(sys.argv[3])
query_target = float(sys.argv[4])
print("1" if (strict >= strict_target and query >= query_target) else "0")
PY
)
if [ "$R2_SPRINT_OK" != "1" ]; then
  echo "[Warn] round-2 did not reach sprint gate."
fi

echo "[Done] round-2 completed."
