#!/usr/bin/env bash
set -euo pipefail

# End-to-end strict pipeline:
# Stage2A -> Stage2B -> Stage3 -> GQA infer/eval
# If strict accuracy < threshold, auto-run fixed round-2 recipe.

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-"$WORKDIR"}
GPU=${GPU:-0}
ISOLATE_GPU=${ISOLATE_GPU:-1}
ORIG_GPU=${GPU}
if [ "$ISOLATE_GPU" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
  GPU=0
fi
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF

STRICT_TARGET=${STRICT_TARGET:-0.4200}
STRICT_SPRINT_TARGET=${STRICT_SPRINT_TARGET:-0.4400}
STRICT_BASELINE=${STRICT_BASELINE:-0.3912}
COVERAGE_TARGET=${COVERAGE_TARGET:-0.9990}
RUN_ROUND2_ON_FAIL=${RUN_ROUND2_ON_FAIL:-1}
INSTALL_DEPS=${INSTALL_DEPS:-0}
LOW_VRAM_4090=${LOW_VRAM_4090:-1}

LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
VISION_MODEL=${VISION_MODEL:-Salesforce/blip2-flan-t5-xl}

if [ "$LOW_VRAM_4090" = "1" ]; then
  # Conservative profile for RTX 4090 24GB.
  : "${S2A_BATCH_SIZE:=1}"
  : "${S2A_MAX_LENGTH:=192}"
  : "${S2A_NUM_WORKERS:=4}"
  : "${S2A_PRECISION:=16}"

  : "${S2B_BATCH_SIZE:=1}"
  : "${S2B_MAX_LENGTH:=192}"
  : "${S2B_NUM_WORKERS:=4}"
  : "${S2B_PRECISION:=16}"

  : "${S3_BATCH_SIZE:=1}"
  : "${S3_MAX_LENGTH:=160}"
  : "${S3_NUM_WORKERS:=2}"
  : "${S3_PRECISION:=16}"

  : "${S2B_R2_BATCH_SIZE:=1}"
  : "${S2B_R2_MAX_LENGTH:=192}"
  : "${S2B_R2_NUM_WORKERS:=4}"
  : "${S2B_R2_PRECISION:=16}"

  : "${S3_R2_BATCH_SIZE:=1}"
  : "${S3_R2_MAX_LENGTH:=160}"
  : "${S3_R2_NUM_WORKERS:=2}"
  : "${S3_R2_PRECISION:=16}"
else
  : "${S2A_BATCH_SIZE:=3}"
  : "${S2A_MAX_LENGTH:=256}"
  : "${S2A_NUM_WORKERS:=8}"
  : "${S2A_PRECISION:=16}"

  : "${S2B_BATCH_SIZE:=3}"
  : "${S2B_MAX_LENGTH:=256}"
  : "${S2B_NUM_WORKERS:=8}"
  : "${S2B_PRECISION:=16}"

  : "${S3_BATCH_SIZE:=2}"
  : "${S3_MAX_LENGTH:=256}"
  : "${S3_NUM_WORKERS:=4}"
  : "${S3_PRECISION:=16}"

  : "${S2B_R2_BATCH_SIZE:=3}"
  : "${S2B_R2_MAX_LENGTH:=256}"
  : "${S2B_R2_NUM_WORKERS:=8}"
  : "${S2B_R2_PRECISION:=16}"

  : "${S3_R2_BATCH_SIZE:=2}"
  : "${S3_R2_MAX_LENGTH:=256}"
  : "${S3_R2_NUM_WORKERS:=4}"
  : "${S3_R2_PRECISION:=16}"
fi

VG_SCENE_GRAPHS=${VG_SCENE_GRAPHS:-"$REPO/data/vg/contents/sceneGraphs/scene_graphs.json"}
VG_REGIONS=${VG_REGIONS:-"$REPO/data/vg/contents/regionDescriptions/region_descriptions.json"}
VG_IMAGE_ROOT=${VG_IMAGE_ROOT:-"$REPO/data/vg"}
STAGE1_QFORMER_CKPT=${STAGE1_QFORMER_CKPT:-"$REPO/graph_qformer_stage1.pt"}

STAGE2A_DIR=${STAGE2A_DIR:-"$REPO/checkpoints_projector_vg/stage2A_paper"}
STAGE2B_DIR=${STAGE2B_DIR:-"$REPO/checkpoints_projector_vg/stage2B_paper"}
STAGE3_DIR=${STAGE3_DIR:-"$REPO/checkpoints_stage3_paper"}
STAGE2B_R2_DIR=${STAGE2B_R2_DIR:-"$REPO/checkpoints_projector_vg/stage2B_round2"}
STAGE3_R2_DIR=${STAGE3_R2_DIR:-"$REPO/checkpoints_stage3_round2"}

GQA_QUESTIONS_JSON=${GQA_QUESTIONS_JSON:-"$REPO/data/gqa/contents/questions/val_balanced_questions.json"}
GQA_QUESTIONS_JSONL=${GQA_QUESTIONS_JSONL:-"$REPO/data/gqa/val_balanced.jsonl"}
GQA_SCENE_RAW=${GQA_SCENE_RAW:-"$REPO/data/gqa/contents/sceneGraphs/val_sceneGraphs.json"}
GQA_SCENE_VG=${GQA_SCENE_VG:-"$REPO/data/gqa/scene_graphs_val_vg.json"}
GQA_IMAGE_ROOT=${GQA_IMAGE_ROOT:-"$REPO/data/gqa/contents/images"}
GQA_PRED_PAPER=${GQA_PRED_PAPER:-"$REPO/data/gqa/pred_val_balanced_paper.jsonl"}
GQA_EVAL_PAPER=${GQA_EVAL_PAPER:-"$REPO/data/gqa/eval_val_balanced_paper.txt"}
GQA_PRED_R2=${GQA_PRED_R2:-"$REPO/data/gqa/pred_val_balanced_round2.jsonl"}
GQA_EVAL_R2=${GQA_EVAL_R2:-"$REPO/data/gqa/eval_val_balanced_round2.txt"}

SELECT_CKPT="${REPO}/scripts/train/select_best_ckpt.py"

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
echo "[Config] LOW_VRAM_4090=${LOW_VRAM_4090} LLM_MODEL=${LLM_MODEL} VISION_MODEL=${VISION_MODEL}"
echo "[Config] S2A bs=${S2A_BATCH_SIZE} max_len=${S2A_MAX_LENGTH} workers=${S2A_NUM_WORKERS} prec=${S2A_PRECISION}"
echo "[Config] S2B bs=${S2B_BATCH_SIZE} max_len=${S2B_MAX_LENGTH} workers=${S2B_NUM_WORKERS} prec=${S2B_PRECISION}"
echo "[Config] S3  bs=${S3_BATCH_SIZE} max_len=${S3_MAX_LENGTH} workers=${S3_NUM_WORKERS} prec=${S3_PRECISION}"
test -f "$STAGE1_QFORMER_CKPT"
test -f "$VG_SCENE_GRAPHS"
test -f "$VG_REGIONS"
test -f "$GQA_QUESTIONS_JSON"
test -f "$GQA_SCENE_RAW"
test -f "$SELECT_CKPT"

echo "[Stage2A] start"
"$PYTHON_BIN" "$REPO/omnigraph/train/train_projector.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --stage1_qformer_ckpt "$STAGE1_QFORMER_CKPT" \
  --llm "$LLM_MODEL" \
  --graph_qa_max_per_image 5 \
  --graph_qa_repeat 3 \
  --gpu "$GPU" \
  --batch_size "$S2A_BATCH_SIZE" \
  --precision "$S2A_PRECISION" \
  --max_length "$S2A_MAX_LENGTH" \
  --num_workers "$S2A_NUM_WORKERS" \
  --val_ratio 0.02 \
  --patience 16 \
  --min_delta 0.0005 \
  --lr 3e-5 \
  --max_steps 120000 \
  --val_check_interval 1000 \
  --save_dir "$STAGE2A_DIR"

STAGE2A_CKPT=$("$PYTHON_BIN" "$SELECT_CKPT" \
  --meta "$STAGE2A_DIR/stage2A_meta.json" \
  --fallback "$STAGE2A_DIR/last.ckpt")
echo "[Stage2A] using ckpt: $STAGE2A_CKPT"
test -f "$STAGE2A_CKPT"

echo "[Stage2B] start"
"$PYTHON_BIN" "$REPO/omnigraph/train/train_stage2B.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --stage2A_ckpt "$STAGE2A_CKPT" \
  --llm "$LLM_MODEL" \
  --graph_qa_max_per_image 6 \
  --graph_qa_repeat 4 \
  --gpu "$GPU" \
  --batch_size "$S2B_BATCH_SIZE" \
  --precision "$S2B_PRECISION" \
  --max_length "$S2B_MAX_LENGTH" \
  --num_workers "$S2B_NUM_WORKERS" \
  --lr 1.2e-5 \
  --max_steps 60000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 16 \
  --min_delta 0.0005 \
  --save_dir "$STAGE2B_DIR"

STAGE2B_CKPT=$("$PYTHON_BIN" "$SELECT_CKPT" \
  --meta "$STAGE2B_DIR/stage2B_meta.json" \
  --fallback "$STAGE2B_DIR/last.ckpt")
echo "[Stage2B] using ckpt: $STAGE2B_CKPT"
test -f "$STAGE2B_CKPT"

echo "[Stage3] start"
"$PYTHON_BIN" "$REPO/omnigraph/train/train_stage3.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --image_root "$VG_IMAGE_ROOT" \
  --stage2B_ckpt "$STAGE2B_CKPT" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --graph_qa_max_per_image 4 \
  --graph_qa_repeat 2 \
  --gpu "$GPU" \
  --batch_size "$S3_BATCH_SIZE" \
  --precision "$S3_PRECISION" \
  --max_length "$S3_MAX_LENGTH" \
  --num_workers "$S3_NUM_WORKERS" \
  --lr 2e-5 \
  --max_steps 50000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 14 \
  --min_delta 0.0005 \
  --save_dir "$STAGE3_DIR"

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
  --ckpt "$STAGE3_DIR/omnigraph_stage3_state_dict.pt" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
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
  --pred "$GQA_PRED_PAPER" | tee "$GQA_EVAL_PAPER"

read -r STRICT_SCORE COVERAGE_SCORE <<<"$("$PYTHON_BIN" - "$GQA_EVAL_PAPER" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
strict = 0.0
coverage = 0.0

strict_patterns = [
    r"Accuracy \(strict, all GT\):\s*([0-9]*\.?[0-9]+)",
    r"Accuracy:\s*([0-9]*\.?[0-9]+)",
]
coverage_patterns = [
    r"Coverage:\s*([0-9]*\.?[0-9]+)",
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

print(f"{strict} {coverage}")
PY
)"

echo "[Result] strict=${STRICT_SCORE} coverage=${COVERAGE_SCORE} baseline=${STRICT_BASELINE} target=${STRICT_TARGET} sprint=${STRICT_SPRINT_TARGET}"

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

RUN_R2=$("$PYTHON_BIN" - "$STRICT_SCORE" "$STRICT_TARGET" "$RUN_ROUND2_ON_FAIL" <<'PY'
import sys
strict = float(sys.argv[1])
target = float(sys.argv[2])
flag = int(sys.argv[3])
print("1" if (flag == 1 and strict < target) else "0")
PY
)

if [ "$RUN_R2" != "1" ]; then
  echo "[Done] round-2 skipped."
  exit 0
fi

echo "[Round2] strict below target, start fixed round-2 recipe."
"$PYTHON_BIN" "$REPO/omnigraph/train/train_stage2B.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --stage2A_ckpt "$STAGE2A_CKPT" \
  --llm "$LLM_MODEL" \
  --graph_qa_max_per_image 8 \
  --graph_qa_repeat 5 \
  --gpu "$GPU" \
  --batch_size "$S2B_R2_BATCH_SIZE" \
  --precision "$S2B_R2_PRECISION" \
  --max_length "$S2B_R2_MAX_LENGTH" \
  --num_workers "$S2B_R2_NUM_WORKERS" \
  --lr 8e-6 \
  --max_steps 80000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 20 \
  --min_delta 0.0003 \
  --save_dir "$STAGE2B_R2_DIR"

STAGE2B_R2_CKPT=$("$PYTHON_BIN" "$SELECT_CKPT" \
  --meta "$STAGE2B_R2_DIR/stage2B_meta.json" \
  --fallback "$STAGE2B_R2_DIR/last.ckpt")
echo "[Round2] Stage2B ckpt: $STAGE2B_R2_CKPT"
test -f "$STAGE2B_R2_CKPT"

"$PYTHON_BIN" "$REPO/omnigraph/train/train_stage3.py" \
  --scene_graphs "$VG_SCENE_GRAPHS" \
  --regions "$VG_REGIONS" \
  --image_root "$VG_IMAGE_ROOT" \
  --stage2B_ckpt "$STAGE2B_R2_CKPT" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --graph_qa_max_per_image 5 \
  --graph_qa_repeat 3 \
  --gpu "$GPU" \
  --batch_size "$S3_R2_BATCH_SIZE" \
  --precision "$S3_R2_PRECISION" \
  --max_length "$S3_R2_MAX_LENGTH" \
  --num_workers "$S3_R2_NUM_WORKERS" \
  --lr 1.5e-5 \
  --max_steps 70000 \
  --val_ratio 0.02 \
  --val_check_interval 1000 \
  --patience 18 \
  --min_delta 0.0003 \
  --save_dir "$STAGE3_R2_DIR"

"$PYTHON_BIN" "$REPO/scripts/eval/infer_gqa.py" \
  --questions "$GQA_QUESTIONS_JSONL" \
  --scene_graphs "$GQA_SCENE_VG" \
  --image_root "$GQA_IMAGE_ROOT" \
  --ckpt "$STAGE3_R2_DIR/omnigraph_stage3_state_dict.pt" \
  --llm "$LLM_MODEL" \
  --vision "$VISION_MODEL" \
  --output "$GQA_PRED_R2" \
  --batch_size 1 \
  --max_length 128 \
  --max_new_tokens 12 \
  --gpu "$GPU" \
  --max_samples 0 \
  --log_every 500

"$PYTHON_BIN" "$REPO/scripts/eval/eval_gqa_accuracy.py" \
  --gt "$GQA_QUESTIONS_JSONL" \
  --pred "$GQA_PRED_R2" | tee "$GQA_EVAL_R2"

echo "[Done] round-2 completed."
