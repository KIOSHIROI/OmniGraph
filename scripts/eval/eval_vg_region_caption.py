#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate VG region captioning with COCO metrics.")
    ap.add_argument("--gt", required=True, help="GT jsonl (fields: id|image_id, answer)")
    ap.add_argument("--pred", required=True, help="Pred jsonl (fields: id|image_id, pred)")
    args = ap.parse_args()

    gt = load_jsonl(Path(args.gt))
    pred = load_jsonl(Path(args.pred))

    gt_map = {}
    for x in gt:
        key = str(x.get("id", x.get("image_id")))
        ans = str(x.get("answer", "")).strip()
        if not ans:
            continue
        gt_map.setdefault(key, []).append(ans)

    pred_map = {}
    for x in pred:
        key = str(x.get("id", x.get("image_id")))
        p = str(x.get("pred", x.get("text", ""))).strip()
        if not p:
            continue
        pred_map[key] = p

    # build COCO-style
    images = []
    annotations = []
    ann_id = 1
    for k, refs in gt_map.items():
        images.append({"id": k})
        for r in refs:
            annotations.append({"id": ann_id, "image_id": k, "caption": r})
            ann_id += 1

    preds = []
    for k, p in pred_map.items():
        preds.append({"image_id": k, "caption": p})

    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except Exception:
        print("Missing pycocotools/pycocoevalcap. Install: pip install pycocotools git+https://github.com/salaniz/pycocoevalcap")
        return 2

    coco = COCO()
    coco.dataset = {"images": images, "annotations": annotations, "type": "captions"}
    coco.createIndex()

    coco_res = coco.loadRes(preds)
    coco_eval = COCOEvalCap(coco, coco_res)
    coco_eval.evaluate()

    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
