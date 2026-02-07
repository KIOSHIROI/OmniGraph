#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _build_image_path(image_root: Optional[str], image_id: str) -> Optional[str]:
    if not image_root:
        return None
    root = Path(image_root)
    candidates = [
        root / f"{image_id}.jpg",
        root / f"{image_id}.png",
        root / "images" / f"{image_id}.jpg",
        root / "images" / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _match_type(item: Dict[str, Any], semantic: List[str], structural: List[str], detailed: List[str]) -> bool:
    if not (semantic or structural or detailed):
        return True
    t = item.get("types", {}) or {}
    sem = str(t.get("semantic", ""))
    st = str(t.get("structural", ""))
    det = str(t.get("detailed", ""))
    if semantic and sem not in semantic:
        return False
    if structural and st not in structural:
        return False
    if detailed and det not in detailed:
        return False
    return True


def convert_questions(
    input_path: Path,
    image_root: Optional[str],
    semantic: List[str],
    structural: List[str],
    detailed: List[str],
) -> Iterable[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    for qid, item in data.items():
        if not _match_type(item, semantic, structural, detailed):
            continue
        image_id = str(item.get("imageId", ""))
        yield {
            "id": str(qid),
            "image_id": image_id,
            "image_path": _build_image_path(image_root, image_id),
            "question": str(item.get("question", "")),
            "answer": str(item.get("answer", "")),
            "types": item.get("types", {}),
            "isBalanced": bool(item.get("isBalanced", False)),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert GQA questions to JSONL for VQA/SG reasoning eval.")
    ap.add_argument("--input", required=True, help="GQA questions json (e.g., val_balanced_questions.json)")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    ap.add_argument("--image_root", default="", help="Root for images (optional)")
    ap.add_argument("--semantic", default="", help="comma-separated types.semantic filter")
    ap.add_argument("--structural", default="", help="comma-separated types.structural filter")
    ap.add_argument("--detailed", default="", help="comma-separated types.detailed filter")
    args = ap.parse_args()

    semantic = [s.strip() for s in args.semantic.split(",") if s.strip()]
    structural = [s.strip() for s in args.structural.split(",") if s.strip()]
    detailed = [s.strip() for s in args.detailed.split(",") if s.strip()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in convert_questions(Path(args.input), args.image_root or None, semantic, structural, detailed):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} questions -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
