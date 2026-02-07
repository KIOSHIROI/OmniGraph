#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def convert_gqa_scene_graphs(input_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []

    for image_id, sg in data.items():
        objects = sg.get("objects", {}) or {}
        obj_list = []
        rel_list = []

        # map object_id to int
        for obj_id_str, obj in objects.items():
            try:
                obj_id = int(obj_id_str)
            except Exception:
                obj_id = int(obj.get("id", 0)) if isinstance(obj, dict) else 0

            name = obj.get("name", "object") if isinstance(obj, dict) else "object"
            attrs = obj.get("attributes", []) if isinstance(obj, dict) else []
            if attrs is None:
                attrs = []

            obj_list.append(
                {
                    "object_id": obj_id,
                    "names": [str(name)],
                    "attributes": [str(a) for a in attrs],
                    "x": float(obj.get("x", 0.0)) if isinstance(obj, dict) else 0.0,
                    "y": float(obj.get("y", 0.0)) if isinstance(obj, dict) else 0.0,
                    "w": float(obj.get("w", 0.0)) if isinstance(obj, dict) else 0.0,
                    "h": float(obj.get("h", 0.0)) if isinstance(obj, dict) else 0.0,
                }
            )

            # relations
            rels = obj.get("relations", []) if isinstance(obj, dict) else []
            for r in rels or []:
                try:
                    obj2 = int(r.get("object"))
                except Exception:
                    obj2 = 0
                pred = r.get("name", None)
                if pred is None:
                    continue
                rel_list.append(
                    {
                        "subject_id": obj_id,
                        "object_id": obj2,
                        "predicate": str(pred),
                    }
                )

        out.append(
            {
                "image_id": int(image_id),
                "width": int(sg.get("width", 0)),
                "height": int(sg.get("height", 0)),
                "objects": obj_list,
                "relationships": rel_list,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert GQA scene graphs to VG-style list format.")
    ap.add_argument("--input", required=True, help="GQA sceneGraphs json (train_sceneGraphs.json / val_sceneGraphs.json)")
    ap.add_argument("--output", required=True, help="Output VG-style scene_graphs.json")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    out = convert_gqa_scene_graphs(input_path)
    output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"Converted {len(out)} images -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
