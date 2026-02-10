#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set


_NUM_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


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


def canonicalize(s: str) -> str:
    s = normalize(s)
    toks = [t for t in s.split(" ") if t and t not in {"a", "an", "the"}]
    toks = [_NUM_WORD_TO_DIGIT.get(t, t) for t in toks]
    return " ".join(toks)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate GQA accuracy.")
    ap.add_argument("--gt", required=True, help="GT jsonl (fields: id, answer, types)")
    ap.add_argument("--pred", required=True, help="Pred jsonl (fields: id, pred)")
    ap.add_argument("--query_target", type=float, default=-1.0, help="Optional query strict target gate.")
    ap.add_argument("--query_sprint_target", type=float, default=-1.0, help="Optional query strict sprint target gate.")
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
            "answer_canonical": canonicalize(str(x.get("answer", ""))),
            "types": x.get("types", {}),
        }

    pred_map = {}
    for x in pred:
        key = str(x.get("id", ""))
        if not key:
            continue
        raw = str(x.get("pred", x.get("text", "")))
        pred_map[key] = {
            "pred": normalize(raw),
            "pred_canonical": canonicalize(raw),
        }

    gt_ids: Set[str] = set(gt_map.keys())
    pred_ids: Set[str] = set(pred_map.keys())
    answered_ids = gt_ids.intersection(pred_ids)
    missing_ids = gt_ids.difference(pred_ids)
    extra_ids = pred_ids.difference(gt_ids)

    total_gt = len(gt_ids)
    answered = len(answered_ids)
    correct_all_strict = 0
    correct_all_canonical = 0
    by_struct_strict = {}
    by_struct_canonical = {}

    for qid, gt_item in gt_map.items():
        pred_ans = pred_map.get(qid, None)
        hit_strict = int(pred_ans is not None and pred_ans["pred"] == gt_item["answer"])
        hit_canonical = int(pred_ans is not None and pred_ans["pred_canonical"] == gt_item["answer_canonical"])
        correct_all_strict += hit_strict
        correct_all_canonical += hit_canonical

        st = str(gt_item.get("types", {}).get("structural", ""))
        if st:
            c1, t1 = by_struct_strict.get(st, (0, 0))
            by_struct_strict[st] = (c1 + hit_strict, t1 + 1)
            c2, t2 = by_struct_canonical.get(st, (0, 0))
            by_struct_canonical[st] = (c2 + hit_canonical, t2 + 1)

    acc_all = correct_all_strict / total_gt if total_gt > 0 else 0.0
    acc_answered = correct_all_strict / answered if answered > 0 else 0.0
    acc_all_canonical = correct_all_canonical / total_gt if total_gt > 0 else 0.0
    acc_answered_canonical = correct_all_canonical / answered if answered > 0 else 0.0
    coverage = answered / total_gt if total_gt > 0 else 0.0

    print(f"Accuracy (strict, all GT): {acc_all:.4f} ({correct_all_strict}/{total_gt})")
    print(f"Coverage: {coverage:.4f} ({answered}/{total_gt})")
    print(f"Accuracy (strict, answered only): {acc_answered:.4f} ({correct_all_strict}/{answered})")
    print(f"Accuracy (canonical, all GT): {acc_all_canonical:.4f} ({correct_all_canonical}/{total_gt})")
    print(f"Accuracy (canonical, answered only): {acc_answered_canonical:.4f} ({correct_all_canonical}/{answered})")
    print(f"Missing predictions: {len(missing_ids)}")
    print(f"Extra predictions (id not in GT): {len(extra_ids)}")

    if missing_ids:
        sample_missing = sorted(missing_ids)[:10]
        print("Sample missing ids:", ", ".join(sample_missing))
    if extra_ids:
        sample_extra = sorted(extra_ids)[:10]
        print("Sample extra ids:", ", ".join(sample_extra))

    if by_struct_strict:
        print("By structural type (strict, all GT denominator per type):")
        for k in sorted(by_struct_strict.keys()):
            c, t = by_struct_strict[k]
            print(f"  {k}: {c/t:.4f} ({c}/{t})")
    if by_struct_canonical:
        print("By structural type (canonical, all GT denominator per type):")
        for k in sorted(by_struct_canonical.keys()):
            c, t = by_struct_canonical[k]
            print(f"  {k}: {c/t:.4f} ({c}/{t})")

    if "query" in by_struct_strict:
        qc, qt = by_struct_strict["query"]
        qacc = qc / qt if qt > 0 else 0.0
        print(f"Query accuracy (strict): {qacc:.4f} ({qc}/{qt})")
        if float(args.query_target) >= 0:
            print(
                f"Query target gate ({float(args.query_target):.4f}): "
                f"{'PASS' if qacc >= float(args.query_target) else 'FAIL'}"
            )
        if float(args.query_sprint_target) >= 0:
            print(
                f"Query sprint gate ({float(args.query_sprint_target):.4f}): "
                f"{'PASS' if qacc >= float(args.query_sprint_target) else 'FAIL'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
