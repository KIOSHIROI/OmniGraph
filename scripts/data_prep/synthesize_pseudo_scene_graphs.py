#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
from PIL import Image

from omnigraph.data.graph_canonical import canonicalize_scene_graph, parse_graph_json
from omnigraph.model.v2g import V2GSynthesizerConfig, VisionGraphSynthesizer, load_v2g_state_dict


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def resolve_image_path(image_path: str, image_root: str) -> str:
    raw = str(image_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_file():
        return str(p)
    rp = Path(image_root) / raw
    if rp.is_file():
        return str(rp)
    return ""


def batched(xs: Sequence[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    bsz = max(1, int(batch_size))
    return [list(xs[i : i + bsz]) for i in range(0, len(xs), bsz)]


def _load_model_name_from_meta(ckpt_path: str, fallback: str) -> str:
    p = Path(ckpt_path)
    cands = [p.parent / "v2g_meta.json", p.parent.parent / "v2g_meta.json"]
    for mp in cands:
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
            name = str(meta.get("model_name", "")).strip()
            if name:
                return name
        except Exception:
            continue
    return str(fallback)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate pseudo VG-style scene graphs from image+caption.")
    ap.add_argument("--vt_manifest", required=True, help="jsonl: id,image_id,image_path,caption")
    ap.add_argument("--image_root", default="", help="base dir for relative image_path")
    ap.add_argument("--ckpt", required=True, help="v2g checkpoint (.pt or lightning .ckpt)")
    ap.add_argument("--model_name", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_candidates", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", type=int, default=1, choices=[0, 1])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--output_raw", required=True, help="output jsonl with candidate pseudo graphs")
    args = ap.parse_args()

    records = read_jsonl(args.vt_manifest)
    if not records:
        raise RuntimeError(f"Empty vt_manifest: {args.vt_manifest}")

    kept: List[Dict[str, Any]] = []
    for rec in records:
        img_path = resolve_image_path(str(rec.get("image_path", "")), str(args.image_root))
        if not img_path:
            continue
        kept.append(
            {
                "id": str(rec.get("id", "")).strip() or str(rec.get("image_id", "")).strip() or Path(img_path).stem,
                "image_id": str(rec.get("image_id", "")).strip() or Path(img_path).stem,
                "image_path": img_path,
                "caption": " ".join(str(rec.get("caption", "")).split()),
            }
        )

    if not kept:
        raise RuntimeError("No usable rows after resolving image_path.")

    model_name = _load_model_name_from_meta(args.ckpt, args.model_name)
    cfg = V2GSynthesizerConfig(
        model_name=str(model_name),
        torch_dtype=str(args.torch_dtype),
    )
    synth = VisionGraphSynthesizer(cfg)
    load_info = load_v2g_state_dict(synth, args.ckpt, map_location="cpu")

    use_gpu = torch.cuda.is_available() and int(args.gpu) >= 0
    device = torch.device(f"cuda:{int(args.gpu)}" if use_gpu else "cpu")
    synth.to(device)
    synth.eval()

    out_path = Path(args.output_raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_parse_ok = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for bi, chunk in enumerate(batched(kept, int(args.batch_size))):
            images = [Image.open(str(x["image_path"])).convert("RGB") for x in chunk]
            prompts = [VisionGraphSynthesizer.build_prompt(caption=str(x.get("caption", ""))) for x in chunk]

            try:
                cand_rows = synth.generate_candidates(
                    images=images,
                    prompts=prompts,
                    device=device,
                    num_candidates=max(1, int(args.num_candidates)),
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(int(args.do_sample)),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                )
            finally:
                for img in images:
                    try:
                        img.close()
                    except Exception:
                        pass

            for rec, cands in zip(chunk, cand_rows):
                for ci, cand in enumerate(cands):
                    raw_text = str(cand.get("text", ""))
                    avg_logprob = float(cand.get("avg_logprob", -10.0))
                    parse_ok = False
                    graph_obj = None
                    parse_error = ""
                    try:
                        graph_obj = parse_graph_json(raw_text)
                        graph_obj = canonicalize_scene_graph(
                            graph_obj,
                            image_id=rec.get("image_id"),
                            image_path=rec.get("image_path"),
                        )
                        parse_ok = True
                    except Exception as e:
                        parse_error = str(e)

                    row = {
                        "id": f"{rec['id']}::cand{ci+1}",
                        "source_id": rec["id"],
                        "image_id": rec["image_id"],
                        "image_path": rec["image_path"],
                        "caption": rec["caption"],
                        "graph_json": graph_obj,
                        "raw_text": raw_text,
                        "avg_logprob": avg_logprob,
                        "scores": {
                            "avg_logprob": avg_logprob,
                        },
                        "passed": False,
                        "candidate_rank": int(ci + 1),
                        "parse_ok": bool(parse_ok),
                        "parse_error": parse_error,
                    }
                    if parse_ok:
                        n_parse_ok += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1

            if (bi + 1) % max(1, int(args.log_every)) == 0:
                print(f"[Synthesize] batches={bi+1} rows={n_written} parse_ok={n_parse_ok}")

    print(
        "[Synthesize] done "
        f"input={len(kept)} candidates_per_image={max(1, int(args.num_candidates))} "
        f"rows={n_written} parse_ok={n_parse_ok} output={out_path}"
    )
    print(f"[Synthesize] load_info={load_info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
