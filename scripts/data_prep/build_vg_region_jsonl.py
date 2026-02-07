#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_region_records(region_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(region_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return out

    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = item.get("id", item.get("image_id"))
        if image_id is None:
            continue
        image_id = int(image_id)

        regions = item.get("regions", [])
        if not isinstance(regions, list):
            continue

        for r in regions:
            if not isinstance(r, dict):
                continue
            phrase = str(r.get("phrase", "")).strip()
            if not phrase:
                continue

            x = r.get("x", 0)
            y = r.get("y", 0)
            w = r.get("width", 0)
            h = r.get("height", 0)
            try:
                x = int(x) if x is not None else 0
                y = int(y) if y is not None else 0
                w = int(w) if w is not None else 0
                h = int(h) if h is not None else 0
            except Exception:
                x, y, w, h = 0, 0, 0, 0

            out.append({"image_id": image_id, "phrase": phrase, "x": x, "y": y, "w": w, "h": h})
    return out


def find_image_path(image_root: Path, image_id: int) -> Optional[str]:
    alt_root = image_root / "contents" / "images"
    img_root = image_root / "images"
    candidates = [
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.png",
        img_root / f"{image_id}.jpg",
        img_root / f"{image_id}.png",
        image_root / "VG_100K" / f"{image_id}.jpg",
        image_root / "VG_100K" / f"{image_id}.png",
        image_root / "VG_100K_2" / f"{image_id}.jpg",
        image_root / "VG_100K_2" / f"{image_id}.png",
        img_root / "VG_100K" / f"{image_id}.jpg",
        img_root / "VG_100K" / f"{image_id}.png",
        img_root / "VG_100K_2" / f"{image_id}.jpg",
        img_root / "VG_100K_2" / f"{image_id}.png",
        alt_root / f"{image_id}.jpg",
        alt_root / f"{image_id}.png",
        alt_root / "VG_100K" / f"{image_id}.jpg",
        alt_root / "VG_100K" / f"{image_id}.png",
        alt_root / "VG_100K_2" / f"{image_id}.jpg",
        alt_root / "VG_100K_2" / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Build VG region captioning jsonl for evaluation.")
    ap.add_argument("--regions", required=True, help="VG region_descriptions.json")
    ap.add_argument("--image_root", required=True, help="VG image root (e.g., data/vg)")
    ap.add_argument("--output", required=True, help="Output jsonl path")
    args = ap.parse_args()

    region_records = load_region_records(Path(args.regions))
    image_root = Path(args.image_root)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rr in region_records:
            img_path = find_image_path(image_root, int(rr["image_id"]))
            if img_path is None:
                continue
            rec = {
                "image_id": rr["image_id"],
                "image_path": img_path,
                "bbox": [rr["x"], rr["y"], rr["w"], rr["h"]],
                "answer": rr["phrase"],
                "prompt": "Describe the region.",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} region samples -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
