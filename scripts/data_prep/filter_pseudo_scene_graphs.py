#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
from PIL import Image

from omnigraph.data.graph_canonical import (
    aggregate_pseudo_score,
    canonical_graph_hash,
    canonicalize_scene_graph,
    graph_structural_validity,
    graph_to_text,
    parse_graph_json,
    token_jaccard_similarity,
)


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _load_clip(model_name: str, device: torch.device):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


@torch.no_grad()
def clip_image_text_similarity(
    *,
    image_path: str,
    text: str,
    clip_bundle: Optional[Tuple[Any, Any]],
    device: torch.device,
) -> float:
    if not clip_bundle:
        return -1.0
    if not image_path:
        return 0.0
    p = Path(image_path)
    if not p.exists():
        return 0.0
    processor, model = clip_bundle
    try:
        image = Image.open(str(p)).convert("RGB")
    except Exception:
        return 0.0

    enc = processor(text=[str(text)], images=[image], return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}

    try:
        out = model(**enc)
        im = out.image_embeds
        tx = out.text_embeds
        im = torch.nn.functional.normalize(im, dim=-1)
        tx = torch.nn.functional.normalize(tx, dim=-1)
        sim = float((im * tx).sum(dim=-1).item())
    except Exception:
        return 0.0

    score = (sim + 1.0) * 0.5
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return float(score)


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter pseudo scene graphs with strict quality gates.")
    ap.add_argument("--input_raw", required=True, help="raw pseudo candidate jsonl")
    ap.add_argument("--output_filtered", required=True, help="filtered VG-style scene_graphs.json")
    ap.add_argument("--output_report", required=True, help="filter report json")

    ap.add_argument("--min_nodes", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=36)
    ap.add_argument("--min_rels", type=int, default=1)
    ap.add_argument("--max_rels", type=int, default=72)
    ap.add_argument("--avg_logprob_thresh", type=float, default=-1.10)
    ap.add_argument("--text_sim_thresh", type=float, default=0.55)
    ap.add_argument("--vision_sim_thresh", type=float, default=0.25)
    ap.add_argument("--score_thresh", type=float, default=0.72)
    ap.add_argument("--keep_ratio_min", type=float, default=0.05, help="Report-only expected minimum retention ratio.")
    ap.add_argument("--keep_ratio_max", type=float, default=0.35, help="Report-only expected maximum retention ratio.")

    ap.add_argument("--enable_clip", type=int, default=1, choices=[0, 1])
    ap.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    use_gpu = torch.cuda.is_available() and int(args.gpu) >= 0
    device = torch.device(f"cuda:{int(args.gpu)}" if use_gpu else "cpu")

    clip_bundle = None
    clip_status = "disabled"
    if bool(int(args.enable_clip)):
        try:
            clip_bundle = _load_clip(str(args.clip_model), device=device)
            clip_status = f"loaded:{args.clip_model}"
        except Exception as e:
            clip_status = f"failed:{e}"
            clip_bundle = None

    reason_counter: Counter[str] = Counter()
    total = 0
    parse_ok = 0
    pass_count = 0
    score_values: List[float] = []

    best_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for row in read_jsonl(args.input_raw):
        total += 1
        image_id = str(row.get("image_id", "")).strip()
        image_path = str(row.get("image_path", "")).strip()
        caption = " ".join(str(row.get("caption", "")).split())
        avg_logprob = _to_float(row.get("avg_logprob", row.get("scores", {}).get("avg_logprob", -10.0)), -10.0)

        graph_obj = row.get("graph_json", None)
        if graph_obj is None:
            raw_text = str(row.get("raw_text", ""))
            try:
                graph_obj = parse_graph_json(raw_text)
            except Exception:
                reason_counter["parse_error"] += 1
                continue

        try:
            graph = canonicalize_scene_graph(graph_obj, image_id=image_id, image_path=image_path)
        except Exception:
            reason_counter["canonicalize_error"] += 1
            continue
        parse_ok += 1

        ok, reasons, stats = graph_structural_validity(
            graph,
            min_nodes=int(args.min_nodes),
            max_nodes=int(args.max_nodes),
            min_rels=int(args.min_rels),
            max_rels=int(args.max_rels),
        )
        if not ok:
            for r in reasons:
                reason_counter[r] += 1
            continue

        if avg_logprob < float(args.avg_logprob_thresh):
            reason_counter["low_logprob"] += 1
            continue

        gtext = graph_to_text(graph)
        text_sim = token_jaccard_similarity(gtext, caption)
        if text_sim < float(args.text_sim_thresh):
            reason_counter["low_text_similarity"] += 1
            continue

        vis_sim = clip_image_text_similarity(
            image_path=image_path,
            text=gtext,
            clip_bundle=clip_bundle,
            device=device,
        )
        if vis_sim < 0.0:
            # Fallback proxy when CLIP is unavailable in offline envs.
            vis_sim = float(text_sim)
        if vis_sim < float(args.vision_sim_thresh):
            reason_counter["low_vision_similarity"] += 1
            continue

        score = aggregate_pseudo_score(avg_logprob, text_sim, vis_sim)
        score_values.append(score)
        if score < float(args.score_thresh):
            reason_counter["low_total_score"] += 1
            continue

        h = canonical_graph_hash(graph)
        key = (str(graph.get("image_id", image_id)), str(h))
        cand = {
            "graph": graph,
            "hash": h,
            "avg_logprob": float(avg_logprob),
            "text_similarity": float(text_sim),
            "vision_similarity": float(vis_sim),
            "score": float(score),
            "source_id": str(row.get("source_id", row.get("id", ""))),
            "stats": stats,
        }
        prev = best_by_key.get(key)
        if prev is None or float(cand["score"]) > float(prev["score"]):
            best_by_key[key] = cand

    filtered_graphs: List[Dict[str, Any]] = [v["graph"] for v in best_by_key.values()]
    pass_count = len(filtered_graphs)

    out_filtered = Path(args.output_filtered)
    out_filtered.parent.mkdir(parents=True, exist_ok=True)
    out_filtered.write_text(json.dumps(filtered_graphs, ensure_ascii=False), encoding="utf-8")

    keep_ratio = (float(pass_count) / float(total)) if total > 0 else 0.0
    keep_ratio_in_range = float(args.keep_ratio_min) <= keep_ratio <= float(args.keep_ratio_max)
    report = {
        "input_raw": str(args.input_raw),
        "output_filtered": str(out_filtered),
        "output_count": int(pass_count),
        "input_count": int(total),
        "parse_ok_count": int(parse_ok),
        "keep_ratio": float(keep_ratio),
        "keep_ratio_gate": {
            "min": float(args.keep_ratio_min),
            "max": float(args.keep_ratio_max),
            "pass": bool(keep_ratio_in_range),
        },
        "clip_status": clip_status,
        "thresholds": {
            "min_nodes": int(args.min_nodes),
            "max_nodes": int(args.max_nodes),
            "min_rels": int(args.min_rels),
            "max_rels": int(args.max_rels),
            "avg_logprob_thresh": float(args.avg_logprob_thresh),
            "text_sim_thresh": float(args.text_sim_thresh),
            "vision_sim_thresh": float(args.vision_sim_thresh),
            "score_thresh": float(args.score_thresh),
        },
        "failure_reasons": dict(reason_counter),
        "score_stats": {
            "count": int(len(score_values)),
            "mean": float(statistics.mean(score_values)) if score_values else 0.0,
            "median": float(statistics.median(score_values)) if score_values else 0.0,
            "min": float(min(score_values)) if score_values else 0.0,
            "max": float(max(score_values)) if score_values else 0.0,
        },
    }

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[Filter] "
        f"input={total} parse_ok={parse_ok} kept={pass_count} keep_ratio={keep_ratio:.4f} "
        f"clip={clip_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
