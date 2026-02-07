#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate GQA accuracy.")
    ap.add_argument("--gt", required=True, help="GT jsonl (fields: id, answer, types)")
    ap.add_argument("--pred", required=True, help="Pred jsonl (fields: id, pred)")
    args = ap.parse_args()

    gt = load_jsonl(Path(args.gt))
    pred = load_jsonl(Path(args.pred))

    gt_map = {}
    for x in gt:
        key = str(x.get("id", ""))
        if not key:
            continue
        gt_map[key] = {
            "answer": normalize(str(x.get("answer", ""))),
            "types": x.get("types", {}),
        }

    pred_map = {}
    for x in pred:
        key = str(x.get("id", ""))
        if not key:
            continue
        pred_map[key] = normalize(str(x.get("pred", x.get("text", ""))))

    total = 0
    correct = 0
    by_struct = {}

    for qid, gt_item in gt_map.items():
        if qid not in pred_map:
            continue
        total += 1
        if pred_map[qid] == gt_item["answer"]:
            correct += 1
            hit = 1
        else:
            hit = 0
        st = str(gt_item.get("types", {}).get("structural", ""))
        if st:
            c, t = by_struct.get(st, (0, 0))
            by_struct[st] = (c + hit, t + 1)

    acc = correct / total if total > 0 else 0.0
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")
    if by_struct:
        print("By structural type:")
        for k in sorted(by_struct.keys()):
            c, t = by_struct[k]
            print(f"  {k}: {c/t:.4f} ({c}/{t})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
