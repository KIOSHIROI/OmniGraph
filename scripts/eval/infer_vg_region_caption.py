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


def _normalize_graph_tokenizer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = str(cfg.get("type", "qformer")).strip().lower()
    if t not in {"qformer", "perceiver"}:
        t = "qformer"
    return {
        "type": t,
        "num_latents": int(cfg.get("num_latents", cfg.get("num_query_tokens", 32))),
        "hidden_dim": int(cfg.get("hidden_dim", cfg.get("qformer_hidden_dim", 768))),
        "num_layers": int(cfg.get("num_layers", 3)),
        "num_heads": int(cfg.get("num_heads", 8)),
        "ff_mult": int(cfg.get("ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.0)),
    }


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
    ap.add_argument("--graph_tokenizer_type", default="auto", choices=["auto", "qformer", "perceiver"])
    ap.add_argument("--perceiver_num_latents", type=int, default=-1, help="<=0 means read from stage3_meta or fallback.")
    ap.add_argument("--perceiver_num_layers", type=int, default=-1, help="<=0 means read from stage3_meta or fallback.")
    ap.add_argument("--perceiver_num_heads", type=int, default=-1, help="<=0 means read from stage3_meta or fallback.")
    ap.add_argument("--perceiver_ff_mult", type=int, default=-1, help="<=0 means read from stage3_meta or fallback.")
    ap.add_argument("--perceiver_dropout", type=float, default=-1.0, help="<0 means read from stage3_meta or fallback.")
    ap.add_argument("--node_encoder_type", default="auto", choices=["auto", "hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_alpha_init", type=float, default=-1.0, help="<0 means read from stage3_meta or fallback.")
    ap.add_argument("--node_encoder_out_dim", type=int, default=0, help="<=0 means read from stage3_meta or fallback 128.")
    ap.add_argument("--enable_gvl_adapter", type=int, default=-1, choices=[-1, 0, 1], help="-1 means read from stage3_meta or fallback.")
    ap.add_argument("--gvl_adapter_gate_init", type=float, default=-1.0, help="<0 means read from stage3_meta or fallback.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    stage3_meta = _load_stage3_meta(args.ckpt)
    stage3_node_cfg = stage3_meta.get("node_encoder_config", {}) if isinstance(stage3_meta, dict) else {}
    stage3_arch_cfg = stage3_meta.get("architecture_config", {}) if isinstance(stage3_meta, dict) else {}
    stage3_graph_cfg = stage3_meta.get("graph_tokenizer_config", {}) if isinstance(stage3_meta, dict) else {}
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
    resolved_enable_gvl_adapter = (
        bool(stage3_arch_cfg.get("enable_gvl_adapter", True))
        if int(args.enable_gvl_adapter) < 0
        else bool(int(args.enable_gvl_adapter))
    )
    resolved_gvl_adapter_gate_init = (
        float(stage3_arch_cfg.get("gvl_adapter_gate_init", 0.1))
        if float(args.gvl_adapter_gate_init) < 0
        else float(args.gvl_adapter_gate_init)
    )
    stage3_graph_cfg = _normalize_graph_tokenizer_config(stage3_graph_cfg if isinstance(stage3_graph_cfg, dict) else {})
    resolved_graph_tokenizer_type = (
        str(stage3_graph_cfg.get("type", "qformer"))
        if str(args.graph_tokenizer_type).strip().lower() == "auto"
        else str(args.graph_tokenizer_type).strip().lower()
    )
    resolved_perceiver_num_latents = (
        int(stage3_graph_cfg.get("num_latents", 32))
        if int(args.perceiver_num_latents) <= 0
        else int(args.perceiver_num_latents)
    )
    resolved_perceiver_num_layers = (
        int(stage3_graph_cfg.get("num_layers", 3))
        if int(args.perceiver_num_layers) <= 0
        else int(args.perceiver_num_layers)
    )
    resolved_perceiver_num_heads = (
        int(stage3_graph_cfg.get("num_heads", 8))
        if int(args.perceiver_num_heads) <= 0
        else int(args.perceiver_num_heads)
    )
    resolved_perceiver_ff_mult = (
        int(stage3_graph_cfg.get("ff_mult", 4))
        if int(args.perceiver_ff_mult) <= 0
        else int(args.perceiver_ff_mult)
    )
    resolved_perceiver_dropout = (
        float(stage3_graph_cfg.get("dropout", 0.0))
        if float(args.perceiver_dropout) < 0
        else float(args.perceiver_dropout)
    )
    print(
        "[Config] "
        f"llm={args.llm} vision={args.vision} node_encoder={resolved_node_encoder_type} "
        f"alpha_init={resolved_node_encoder_alpha} out_dim={resolved_node_encoder_out_dim} "
        f"graph_tokenizer={resolved_graph_tokenizer_type} "
        f"perceiver(latents={resolved_perceiver_num_latents},layers={resolved_perceiver_num_layers},heads={resolved_perceiver_num_heads}) "
        f"gvl_adapter={resolved_enable_gvl_adapter} gvl_gate={resolved_gvl_adapter_gate_init}"
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
        graph_tokenizer_type=resolved_graph_tokenizer_type,
        perceiver_num_latents=int(resolved_perceiver_num_latents),
        perceiver_num_layers=int(resolved_perceiver_num_layers),
        perceiver_num_heads=int(resolved_perceiver_num_heads),
        perceiver_ff_mult=int(resolved_perceiver_ff_mult),
        perceiver_dropout=float(resolved_perceiver_dropout),
        node_encoder_type=resolved_node_encoder_type,
        node_encoder_alpha_init=resolved_node_encoder_alpha,
        node_encoder_out_dim=resolved_node_encoder_out_dim,
        node_encoder_trainable=False,
        enable_gvl_adapter=resolved_enable_gvl_adapter,
        gvl_adapter_gate_init=resolved_gvl_adapter_gate_init,
        enable_graph_aux_head=False,
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
