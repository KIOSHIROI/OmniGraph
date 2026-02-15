#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _norm_caption(x: Any) -> str:
    return " ".join(str(x or "").split())


def _resolve_coco_image_path(coco_root: Path, split: str, file_name: str) -> Optional[str]:
    candidates = [
        coco_root / split / file_name,
        coco_root / "images" / split / file_name,
        coco_root / file_name,
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _load_coco_rows(coco_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ann_dir = coco_root / "annotations"
    ann_specs: List[Tuple[str, Path]] = [
        ("train2017", ann_dir / "captions_train2017.json"),
        ("val2017", ann_dir / "captions_val2017.json"),
    ]

    for split, ann_path in ann_specs:
        if not ann_path.is_file():
            continue
        obj = _read_json(ann_path)
        images = obj.get("images", []) if isinstance(obj, dict) else []
        anns = obj.get("annotations", []) if isinstance(obj, dict) else []
        id2file: Dict[int, str] = {}
        for it in images:
            if not isinstance(it, dict):
                continue
            try:
                iid = int(it.get("id"))
            except Exception:
                continue
            fn = str(it.get("file_name", "")).strip()
            if not fn:
                continue
            id2file[iid] = fn

        for ann in anns:
            if not isinstance(ann, dict):
                continue
            try:
                image_id = int(ann.get("image_id"))
            except Exception:
                continue
            file_name = id2file.get(image_id, "")
            if not file_name:
                continue
            img_path = _resolve_coco_image_path(coco_root, split=split, file_name=file_name)
            if not img_path:
                continue
            ann_id = ann.get("id", f"{split}_{image_id}")
            caption = _norm_caption(ann.get("caption", ""))
            if not caption:
                continue
            rows.append(
                {
                    "id": f"coco_{ann_id}",
                    "image_id": f"coco_{image_id}",
                    "image_path": img_path,
                    "caption": caption,
                }
            )
    return rows


def _build_region_caption_map(vg_regions_path: Path) -> Dict[int, str]:
    region_map: Dict[int, str] = {}
    if not vg_regions_path.is_file():
        return region_map
    data = _read_json(vg_regions_path)
    if not isinstance(data, list):
        return region_map
    for item in data:
        if not isinstance(item, dict):
            continue
        iid_raw = item.get("id", item.get("image_id"))
        try:
            iid = int(iid_raw)
        except Exception:
            continue
        regions = item.get("regions", [])
        if not isinstance(regions, list):
            continue
        best = ""
        for r in regions:
            if not isinstance(r, dict):
                continue
            phrase = _norm_caption(r.get("phrase", ""))
            if not phrase:
                continue
            if len(phrase) > len(best):
                best = phrase
        if best:
            region_map[iid] = best
    return region_map


def _resolve_vg_image_path(vg_image_root: Path, image_id: int) -> Optional[str]:
    candidates = [
        vg_image_root / f"{image_id}.jpg",
        vg_image_root / f"{image_id}.png",
        vg_image_root / "images" / f"{image_id}.jpg",
        vg_image_root / "images" / f"{image_id}.png",
        vg_image_root / "VG_100K" / f"{image_id}.jpg",
        vg_image_root / "VG_100K_2" / f"{image_id}.jpg",
        vg_image_root / "images" / "VG_100K" / f"{image_id}.jpg",
        vg_image_root / "images" / "VG_100K_2" / f"{image_id}.jpg",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def _fallback_graph_caption(objects: List[Dict[str, Any]]) -> str:
    names: List[str] = []
    for o in objects[:12]:
        if not isinstance(o, dict):
            continue
        name = _norm_caption(o.get("name", ""))
        if not name:
            name = _norm_caption(o.get("names", [""])[0] if isinstance(o.get("names"), list) and o.get("names") else "")
        if name:
            names.append(name)
    if not names:
        return "A scene with multiple objects."
    uniq = []
    seen = set()
    for n in names:
        k = n.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    return "Scene graph objects: " + ", ".join(uniq[:10]) + "."


def _load_vg_rows(vg_scene_graphs_path: Path, vg_regions_path: Path, vg_image_root: Path) -> List[Dict[str, Any]]:
    data = _read_json(vg_scene_graphs_path)
    if not isinstance(data, list):
        raise RuntimeError(f"Unsupported VG scene graph format (expected list): {vg_scene_graphs_path}")

    region_caption_map = _build_region_caption_map(vg_regions_path)
    rows: List[Dict[str, Any]] = []

    for it in data:
        if not isinstance(it, dict):
            continue
        iid_raw = it.get("image_id", it.get("id"))
        try:
            image_id = int(iid_raw)
        except Exception:
            continue

        objects = it.get("objects", [])
        relationships = it.get("relationships", [])
        if not isinstance(objects, list) or not isinstance(relationships, list):
            continue
        if len(objects) < 1:
            continue

        img_path = _resolve_vg_image_path(vg_image_root=vg_image_root, image_id=image_id)
        if not img_path:
            continue

        caption = region_caption_map.get(image_id, "")
        if not caption:
            caption = _fallback_graph_caption(objects)

        graph_json = {
            "image_id": image_id,
            "width": int(it.get("width", 0) or 0),
            "height": int(it.get("height", 0) or 0),
            "objects": objects,
            "relationships": relationships,
        }
        rows.append(
            {
                "id": f"vg_{image_id}",
                "image_id": image_id,
                "image_path": img_path,
                "caption": caption,
                "graph_json": graph_json,
            }
        )

    return rows


def _slice_by_max(rows: List[Dict[str, Any]], max_samples: int) -> List[Dict[str, Any]]:
    if int(max_samples) <= 0 or len(rows) <= int(max_samples):
        return rows
    return rows[: int(max_samples)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build vt_manifest_round1/round2 and v2g_train_manifest from COCO+VG."
    )
    ap.add_argument("--coco_root", required=True, help="COCO root containing annotations/ and images.")
    ap.add_argument("--vg_scene_graphs", required=True, help="VG scene_graphs.json path.")
    ap.add_argument("--vg_regions", default="", help="VG region_descriptions.json path (optional but recommended).")
    ap.add_argument("--vg_image_root", required=True, help="VG image root.")

    ap.add_argument("--out_vt_round1", required=True, help="Output JSONL for vt manifest round1.")
    ap.add_argument("--out_vt_round2", required=True, help="Output JSONL for vt manifest round2.")
    ap.add_argument("--out_v2g_train", required=True, help="Output JSONL for v2g train manifest.")

    ap.add_argument("--vt_round1_ratio", type=float, default=0.5, help="Split ratio for vt round1 (0~1).")
    ap.add_argument("--max_vt_samples", type=int, default=0, help="<=0 means all.")
    ap.add_argument("--max_v2g_samples", type=int, default=0, help="<=0 means all.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco_root = Path(args.coco_root)
    vg_scene_graphs = Path(args.vg_scene_graphs)
    vg_regions = Path(str(args.vg_regions).strip()) if str(args.vg_regions).strip() else Path("")
    vg_image_root = Path(args.vg_image_root)

    if not coco_root.exists():
        raise FileNotFoundError(f"coco_root not found: {coco_root}")
    if not vg_scene_graphs.is_file():
        raise FileNotFoundError(f"vg_scene_graphs not found: {vg_scene_graphs}")
    if not vg_image_root.exists():
        raise FileNotFoundError(f"vg_image_root not found: {vg_image_root}")

    rng = random.Random(int(args.seed))

    vt_rows = _load_coco_rows(coco_root)
    if not vt_rows:
        raise RuntimeError(f"No usable COCO caption rows under: {coco_root}")
    rng.shuffle(vt_rows)
    vt_rows = _slice_by_max(vt_rows, int(args.max_vt_samples))

    ratio = min(0.95, max(0.05, float(args.vt_round1_ratio)))
    cut = int(len(vt_rows) * ratio)
    cut = min(max(1, cut), len(vt_rows) - 1) if len(vt_rows) > 1 else len(vt_rows)
    vt_r1 = vt_rows[:cut]
    vt_r2 = vt_rows[cut:] if cut < len(vt_rows) else list(vt_rows)

    v2g_rows = _load_vg_rows(vg_scene_graphs, vg_regions, vg_image_root)
    if not v2g_rows:
        raise RuntimeError(f"No usable VG rows from: {vg_scene_graphs}")
    rng.shuffle(v2g_rows)
    v2g_rows = _slice_by_max(v2g_rows, int(args.max_v2g_samples))

    n_r1 = _write_jsonl(Path(args.out_vt_round1), vt_r1)
    n_r2 = _write_jsonl(Path(args.out_vt_round2), vt_r2)
    n_v2g = _write_jsonl(Path(args.out_v2g_train), v2g_rows)

    print(
        "[BuildManifests] "
        f"vt_round1={n_r1} vt_round2={n_r2} v2g_train={n_v2g} "
        f"out_r1={args.out_vt_round1} out_r2={args.out_vt_round2} out_v2g={args.out_v2g_train}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

