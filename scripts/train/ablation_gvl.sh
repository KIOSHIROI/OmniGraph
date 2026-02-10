#!/usr/bin/env bash
set -euo pipefail

# Fixed ablation groups for graph-vision-language node encoder variants.
# Runs strict pipeline for:
#   1) legacy_vg
#   2) open_vocab
#   3) hybrid

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WORKDIR"

PYTHON_BIN=${PYTHON_BIN:-python}
REPO=${REPO:-"$WORKDIR"}
OUT_ROOT=${OUT_ROOT:-"$REPO/experiments/gvl_ablation"}
RUN_ROUND2_ON_FAIL=${RUN_ROUND2_ON_FAIL:-0}
LOW_VRAM_4090=${LOW_VRAM_4090:-1}
ISOLATE_GPU=${ISOLATE_GPU:-1}
GPU=${GPU:-0}

mkdir -p "$OUT_ROOT"
SUMMARY_TSV="$OUT_ROOT/summary.tsv"
SUMMARY_MD="$OUT_ROOT/summary.md"

echo -e "variant\tstrict\tcoverage\tquery\teval_file" > "$SUMMARY_TSV"

run_variant() {
  local variant="$1"
  local vdir="$OUT_ROOT/$variant"
  mkdir -p "$vdir"

  echo "[Ablation] start variant=$variant"

  NODE_ENCODER_TYPE="$variant" \
  RUN_ROUND2_ON_FAIL="$RUN_ROUND2_ON_FAIL" \
  LOW_VRAM_4090="$LOW_VRAM_4090" \
  ISOLATE_GPU="$ISOLATE_GPU" \
  GPU="$GPU" \
  STAGE2A_DIR="$vdir/stage2A" \
  STAGE2B_DIR="$vdir/stage2B" \
  STAGE3_DIR="$vdir/stage3" \
  STAGE2B_R2_DIR="$vdir/stage2B_round2" \
  STAGE3_R2_DIR="$vdir/stage3_round2" \
  GQA_PRED_PAPER="$vdir/pred_val_balanced.jsonl" \
  GQA_EVAL_PAPER="$vdir/eval_val_balanced.txt" \
  GQA_PRED_R2="$vdir/pred_val_balanced_round2.jsonl" \
  GQA_EVAL_R2="$vdir/eval_val_balanced_round2.txt" \
  bash "$REPO/scripts/train/run_4090_gqa_sprint.sh"

  local eval_file="$vdir/eval_val_balanced.txt"
  if [ ! -f "$eval_file" ]; then
    echo "[Ablation] missing eval file for $variant: $eval_file" >&2
    return 1
  fi

  read -r strict coverage query <<<"$("$PYTHON_BIN" - "$eval_file" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")

def first_float(patterns):
    for p in patterns:
        m = re.search(p, text, flags=re.MULTILINE)
        if m:
            return float(m.group(1))
    return 0.0

strict = first_float([
    r"Accuracy \(strict, all GT\):\s*([0-9]*\.?[0-9]+)",
    r"Accuracy:\s*([0-9]*\.?[0-9]+)",
])
coverage = first_float([r"Coverage:\s*([0-9]*\.?[0-9]+)"])
query = first_float([
    r"Query accuracy \(strict\):\s*([0-9]*\.?[0-9]+)",
    r"^\s*query:\s*([0-9]*\.?[0-9]+)",
])
print(f"{strict:.4f} {coverage:.4f} {query:.4f}")
PY
)"

  echo -e "${variant}\t${strict}\t${coverage}\t${query}\t${eval_file}" >> "$SUMMARY_TSV"
  echo "[Ablation] done variant=$variant strict=$strict coverage=$coverage query=$query"
}

run_variant "legacy_vg"
run_variant "open_vocab"
run_variant "hybrid"

"$PYTHON_BIN" - "$SUMMARY_TSV" "$SUMMARY_MD" <<'PY'
import csv
import sys
from pathlib import Path

tsv_path = Path(sys.argv[1])
md_path = Path(sys.argv[2])

rows = list(csv.DictReader(tsv_path.open("r", encoding="utf-8"), delimiter="\t"))
lines = []
lines.append("# GVL Node Encoder Ablation Summary")
lines.append("")
lines.append("| Variant | Strict | Coverage | Query | Eval File |")
lines.append("| --- | ---: | ---: | ---: | --- |")
for r in rows:
    lines.append(
        f"| {r['variant']} | {r['strict']} | {r['coverage']} | {r['query']} | `{r['eval_file']}` |"
    )
lines.append("")
lines.append("## Notes")
lines.append("- Use strict + query together for final selection.")
lines.append("- Prefer `hybrid` when strict is close but query is better.")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote markdown summary -> {md_path}")
PY

echo "[Ablation] summary tsv -> $SUMMARY_TSV"
echo "[Ablation] summary md  -> $SUMMARY_MD"
