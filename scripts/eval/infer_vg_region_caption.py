#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor

from omnigraph.data.vg_scene_graph_dataset import build_vg_vocabs_from_file, VGSceneGraphDataset
from omnigraph.model.OmniGraphModel import OmniGraphModel
from omnigraph.train.train_stage3 import VGTriModalRegionDataset, collate_tri


def _build_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _load_stage3_meta(ckpt_path: str) -> Dict[str, Any]:
    ckpt = Path(ckpt_path)
    meta_path = ckpt.parent / "stage3_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer VG region captions with OmniGraph stage3.")
    ap.add_argument("--scene_graphs", required=True)
    ap.add_argument("--regions", required=True)
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--ckpt", required=True, help="stage3 state_dict (omnigraph_stage3_state_dict.pt)")
    ap.add_argument("--output", required=True, help="pred jsonl")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--vision", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--graph_model", default="clip_gt_arxiv_pub")
    ap.add_argument("--node_encoder_type", default="auto", choices=["auto", "hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_alpha_init", type=float, default=-1.0, help="<0 means read from stage3_meta or fallback.")
    ap.add_argument("--node_encoder_out_dim", type=int, default=0, help="<=0 means read from stage3_meta or fallback 128.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    stage3_meta = _load_stage3_meta(args.ckpt)
    stage3_node_cfg = stage3_meta.get("node_encoder_config", {}) if isinstance(stage3_meta, dict) else {}
    resolved_node_encoder_type = (
        str(stage3_node_cfg.get("type", "hybrid"))
        if str(args.node_encoder_type).strip().lower() == "auto"
        else str(args.node_encoder_type).strip().lower()
    )
    resolved_node_encoder_alpha = (
        float(stage3_node_cfg.get("alpha_init", 0.3))
        if float(args.node_encoder_alpha_init) < 0
        else float(args.node_encoder_alpha_init)
    )
    resolved_node_encoder_out_dim = (
        int(stage3_node_cfg.get("out_dim", 128))
        if int(args.node_encoder_out_dim) <= 0
        else int(args.node_encoder_out_dim)
    )
    print(
        "[Config] "
        f"llm={args.llm} vision={args.vision} node_encoder={resolved_node_encoder_type} "
        f"alpha_init={resolved_node_encoder_alpha} out_dim={resolved_node_encoder_out_dim}"
    )

    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_file(args.scene_graphs, min_freq=2)
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)

    sg_dataset = VGSceneGraphDataset(
        scene_graphs_path=args.scene_graphs,
        obj_vocab=obj_vocab,
        pred_vocab=pred_vocab,
        attr_vocab=attr_vocab,
        max_nodes=80,
        max_attrs=6,
        add_reverse_edges=True,
        use_bbox_max_norm=True,
    )

    from omnigraph.train.train_stage3 import load_region_records
    region_records = load_region_records(args.regions)

    dataset = VGTriModalRegionDataset(
        sg_dataset=sg_dataset,
        region_records=region_records,
        image_root=args.image_root,
        prompt="Describe the region.",
    )

    processor = AutoProcessor.from_pretrained(args.vision)
    tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _collate(batch):
        return collate_tri(batch, processor=processor)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=_collate,
    )

    model = OmniGraphModel(
        graph_model_name=args.graph_model,
        vision_model_name=args.vision,
        llm_model_name=args.llm,
        enable_vision=True,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        node_encoder_type=resolved_node_encoder_type,
        node_encoder_alpha_init=resolved_node_encoder_alpha,
        node_encoder_out_dim=resolved_node_encoder_out_dim,
        node_encoder_trainable=False,
    )
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f, torch.no_grad():
        for batch in loader:
            prompts = batch["prompts"]
            prompt_texts = [_build_prompt(tokenizer, p) for p in prompts]
            tok = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_length),
            )
            input_ids = tok["input_ids"].to(device)

            graph_data = batch["graph_data"].to(device)
            pixel_values = batch["pixel_values"].to(device)

            gen = model.generate(
                input_ids=input_ids,
                graph_data=graph_data,
                pixel_values=pixel_values,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
            )
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for i, text in enumerate(texts):
                rec = {
                    "id": batch["ids"][i],
                    "image_id": batch["image_ids"][i],
                    "pred": text.strip(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
