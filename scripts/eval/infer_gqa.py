#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from transformers import AutoProcessor, AutoTokenizer

from omnigraph.data.vg_scene_graph_dataset import (
    VGSceneGraphDataset,
    build_vg_vocabs_from_file,
    build_vg_vocabs_from_items,
    load_vg_scene_graph_items,
    merge_scene_graph_items,
)
from omnigraph.model.OmniGraphModel import OmniGraphModel


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def find_image_path(image_root: Path, image_id: str) -> Optional[str]:
    candidates = [
        image_root / f"{image_id}.jpg",
        image_root / f"{image_id}.png",
        image_root / "images" / f"{image_id}.jpg",
        image_root / "images" / f"{image_id}.png",
        image_root / "contents" / "images" / f"{image_id}.jpg",
        image_root / "contents" / "images" / f"{image_id}.png",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def build_short_answer_prompt(question: str) -> str:
    q = " ".join((question or "").split())
    return (
        "Answer with the shortest possible phrase.\n"
        "Do not explain.\n"
        "Output only the short answer.\n"
        f"Question: {q}\n"
        "Short answer:"
    )


def extract_short_answer(text: str) -> str:
    t = (text or "").strip()
    lower = t.lower()
    for marker in ("short answer:", "answer:"):
        idx = lower.rfind(marker)
        if idx >= 0:
            t = t[idx + len(marker):].strip()
            lower = t.lower()

    t = re.split(r"[\r\n]", t)[0].strip()
    t = re.split(r"[.?!]", t)[0].strip()
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split(" ")
    if len(words) > 8:
        t = " ".join(words[:8]).strip()
    return t


def build_prompt_for_tokenizer(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def _load_stage3_meta(ckpt_path: str) -> Dict[str, Any]:
    ckpt = Path(ckpt_path)
    candidates = [
        ckpt.parent / "multimodal_tune_meta.json",
        ckpt.parent / "stage3_meta.json",
    ]
    for meta_path in candidates:
        if not meta_path.exists():
            continue
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
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


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _extract_train_vocab_sources_from_meta(stage3_meta: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
    if not isinstance(stage3_meta, dict):
        return None, []
    src = stage3_meta.get("scene_graph_sources", {})
    if not isinstance(src, dict):
        return None, []
    base = str(src.get("base_scene_graphs", "")).strip() or None
    extras_raw = src.get("extra_scene_graphs", [])
    extras: List[str] = []
    if isinstance(extras_raw, list):
        for p in extras_raw:
            t = str(p).strip()
            if t:
                extras.append(t)
    return base, extras


def _build_vocabs_with_train_alignment(
    eval_scene_graphs: str,
    min_freq: int,
    stage3_meta: Dict[str, Any],
    prefer_train_vocab_from_meta: bool,
):
    if bool(prefer_train_vocab_from_meta):
        base_sg, extra_sg = _extract_train_vocab_sources_from_meta(stage3_meta)
        if base_sg:
            src_paths = [base_sg] + list(extra_sg)
            missing = [p for p in src_paths if not Path(p).exists()]
            if not missing:
                base_items = load_vg_scene_graph_items(base_sg)
                extra_items_list = [load_vg_scene_graph_items(p) for p in extra_sg]
                merged_items, merge_stats = merge_scene_graph_items(base_items, extra_items_list)
                vocabs = build_vg_vocabs_from_items(merged_items, min_freq=int(min_freq))
                info = {
                    "source": "train_meta",
                    "base_scene_graphs": base_sg,
                    "extra_scene_graphs_count": len(extra_sg),
                    "merged_kept": int(merge_stats.get("kept", 0)),
                }
                return vocabs, info
            print(f"[Vocab][Warn] train-meta scene graph paths missing -> fallback to eval scene_graphs: {missing}")

    vocabs = build_vg_vocabs_from_file(eval_scene_graphs, min_freq=int(min_freq))
    info = {
        "source": "eval_scene_graphs",
        "base_scene_graphs": str(eval_scene_graphs),
        "extra_scene_graphs_count": 0,
        "merged_kept": -1,
    }
    return vocabs, info


def _is_critical_weight_key(k: str) -> bool:
    critical_prefixes = (
        "llm.",
        "node_encoder.",
        "vg_adapter.",
        "graph_qformer.",
        "graph_branch.",
        "gl_projector.",
        "vision_branch.",
        "vl_projector.",
        "gvl_adapter.",
    )
    return str(k).startswith(critical_prefixes)


class GQATriModalDataset(Dataset):
    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        questions: List[Dict[str, Any]],
        image_root: str,
        use_vision: bool = True,
        allow_missing_modalities: bool = False,
    ):
        self.sg = sg_dataset
        self.image_root = Path(image_root)
        self.use_vision = bool(use_vision)
        self.allow_missing_modalities = bool(allow_missing_modalities)

        raw_items = getattr(self.sg, "items", None)
        if raw_items is None:
            raise AttributeError("VGSceneGraphDataset must expose .items for image_id join.")

        image2idx: Dict[str, int] = {}
        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                image2idx[str(it["image_id"])] = i
        self.image2idx = image2idx

        self.samples: List[Dict[str, Any]] = []
        for rec in questions:
            qid = str(rec.get("id", "")).strip()
            image_id = str(rec.get("image_id", "")).strip()
            question = str(rec.get("question", "")).strip()
            if not qid or not image_id or not question:
                continue
            has_graph = image_id in self.image2idx

            image_path = str(rec.get("image_path", "")).strip() or None
            has_image = False
            if self.use_vision:
                if image_path is None or not Path(image_path).exists():
                    image_path = find_image_path(self.image_root, image_id)
                has_image = image_path is not None

            if not self.allow_missing_modalities:
                if not has_graph:
                    continue
                if self.use_vision and not has_image:
                    continue

            self.samples.append(
                {
                    "id": qid,
                    "image_id": image_id,
                    "sg_idx": int(self.image2idx[image_id]) if has_graph else None,
                    "question": question,
                    "image_path": image_path if has_image else None,
                    "has_graph": bool(has_graph),
                    "has_image": bool(has_image) if self.use_vision else False,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        sg_item = self.sg[int(s["sg_idx"])] if s.get("sg_idx") is not None else None
        out = {
            "id": str(s["id"]),
            "image_id": str(s["image_id"]),
            "graph_data": sg_item["graph_data"] if sg_item is not None else None,
            "prompt": build_short_answer_prompt(str(s["question"])),
            "has_graph": bool(s.get("has_graph", False)),
            "has_image": bool(s.get("has_image", False)),
        }
        if self.use_vision and s.get("image_path"):
            from PIL import Image

            img = Image.open(str(s["image_path"])).convert("RGB")
            out["pil_image"] = img
        return out


class GQATriModalIterableDataset(IterableDataset):
    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        questions_path: Path,
        image_root: str,
        use_vision: bool = True,
        allow_missing_modalities: bool = False,
        max_samples: int = 0,
    ):
        super().__init__()
        self.sg = sg_dataset
        self.questions_path = Path(questions_path)
        self.image_root = Path(image_root)
        self.use_vision = bool(use_vision)
        self.allow_missing_modalities = bool(allow_missing_modalities)
        self.max_samples = int(max_samples)

        raw_items = getattr(self.sg, "items", None)
        if raw_items is None:
            raise AttributeError("VGSceneGraphDataset must expose .items for image_id join.")

        image2idx: Dict[str, int] = {}
        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                image2idx[str(it["image_id"])] = i
        self.image2idx = image2idx

    def __iter__(self):
        yielded = 0
        for rec in iter_jsonl(self.questions_path):
            qid = str(rec.get("id", "")).strip()
            image_id = str(rec.get("image_id", "")).strip()
            question = str(rec.get("question", "")).strip()
            if not qid or not image_id or not question:
                continue

            has_graph = image_id in self.image2idx
            image_path = str(rec.get("image_path", "")).strip() or None
            has_image = False
            if self.use_vision:
                if image_path is None or not Path(image_path).exists():
                    image_path = find_image_path(self.image_root, image_id)
                has_image = image_path is not None

            if not self.allow_missing_modalities:
                if not has_graph:
                    continue
                if self.use_vision and not has_image:
                    continue

            sg_item = self.sg[int(self.image2idx[image_id])] if has_graph else None
            out = {
                "id": qid,
                "image_id": image_id,
                "graph_data": sg_item["graph_data"] if sg_item is not None else None,
                "prompt": build_short_answer_prompt(question),
                "has_graph": bool(has_graph),
                "has_image": bool(has_image) if self.use_vision else False,
            }
            if self.use_vision and image_path:
                from PIL import Image

                img = Image.open(str(image_path)).convert("RGB")
                out["pil_image"] = img
            yield out
            yielded += 1
            if self.max_samples > 0 and yielded >= self.max_samples:
                break


def collate_gqa(batch: List[Dict[str, Any]], processor: Any, use_vision: bool) -> Dict[str, Any]:
    ids = [x["id"] for x in batch]
    image_ids = [x["image_id"] for x in batch]
    prompts = [x["prompt"] for x in batch]
    has_graph = [bool(x.get("has_graph", False)) for x in batch]
    has_image = [bool(x.get("has_image", False)) for x in batch]
    graph_list = [x.get("graph_data", None) for x in batch]
    pil_images = [x.get("pil_image", None) for x in batch]

    out = {
        "ids": ids,
        "image_ids": image_ids,
        "prompts": prompts,
        "graph_list": graph_list,
        "pil_images": pil_images,
        "has_graph": has_graph,
        "has_image": has_image,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Infer GQA predictions with OmniGraph.")
    ap.add_argument("--questions", required=True, help="GQA question jsonl from convert_gqa_questions.py")
    ap.add_argument("--scene_graphs", required=True, help="VG-style scene graph json converted from GQA")
    ap.add_argument("--image_root", required=True, help="image root for GQA images")
    ap.add_argument("--ckpt", required=True, help="OmniGraph model state_dict (.pt), preferably multimodal_tune export.")
    ap.add_argument("--output", required=True, help="prediction jsonl path")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--vision", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--graph_model", default="clip_gt_arxiv_pub")
    ap.add_argument("--graph_tokenizer_type", default="auto", choices=["auto", "qformer", "perceiver"])
    ap.add_argument("--perceiver_num_latents", type=int, default=-1, help="<=0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--perceiver_num_layers", type=int, default=-1, help="<=0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--perceiver_num_heads", type=int, default=-1, help="<=0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--perceiver_ff_mult", type=int, default=-1, help="<=0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--perceiver_dropout", type=float, default=-1.0, help="<0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--node_encoder_type", default="auto", choices=["auto", "hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_alpha_init", type=float, default=-1.0, help="<0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--node_encoder_out_dim", type=int, default=0, help="<=0 means read from multimodal_tune_meta (or legacy stage3_meta) fallback 128.")
    ap.add_argument("--enable_gvl_adapter", type=int, default=-1, choices=[-1, 0, 1], help="-1 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--gvl_adapter_gate_init", type=float, default=-1.0, help="<0 means read from multimodal_tune_meta (or legacy stage3_meta).")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument(
        "--prefer_train_vocab_from_meta",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: prefer rebuilding vocab from multimodal_tune_meta scene_graph_sources for ckpt alignment.",
    )
    ap.add_argument(
        "--strict_vocab_match",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: fail if rebuilt vocab size differs from multimodal_tune_meta num_obj/num_attr.",
    )
    ap.add_argument(
        "--strict_weight_load",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: fail-fast on critical key mismatches while loading checkpoint weights.",
    )
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)
    ap.add_argument("--disable_vision", action="store_true", help="disable vision branch at inference")
    ap.add_argument("--allow_missing_modalities", action="store_true", help="allow missing graph/image inputs")
    ap.add_argument("--stream_questions", action="store_true", help="stream questions jsonl to reduce RAM")
    ap.add_argument("--max_samples", type=int, default=0, help="limit number of samples for quick eval")
    ap.add_argument("--log_every", type=int, default=200, help="log progress every N samples")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--persistent_workers", action="store_true")
    args = ap.parse_args()

    print("[Pipeline] GQA inference start (short-answer constrained decoding).")
    use_vision = not bool(args.disable_vision)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and int(args.gpu) >= 0 else "cpu")

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

    (obj_vocab, pred_vocab, attr_vocab), vocab_info = _build_vocabs_with_train_alignment(
        eval_scene_graphs=str(args.scene_graphs),
        min_freq=int(args.min_freq),
        stage3_meta=stage3_meta,
        prefer_train_vocab_from_meta=bool(int(args.prefer_train_vocab_from_meta)),
    )
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)
    expected_num_obj = _safe_int(stage3_meta.get("num_obj"), num_obj) if isinstance(stage3_meta, dict) else num_obj
    expected_num_attr = _safe_int(stage3_meta.get("num_attr"), num_attr) if isinstance(stage3_meta, dict) else num_attr
    print(
        "[Vocab] "
        f"source={vocab_info.get('source')} NUM_OBJ={num_obj} NUM_ATTR={num_attr} "
        f"expected_from_meta=({expected_num_obj},{expected_num_attr})"
    )
    if (num_obj != expected_num_obj) or (num_attr != expected_num_attr):
        msg = (
            "Vocabulary size mismatch against training meta: "
            f"rebuilt=({num_obj},{num_attr}) vs meta=({expected_num_obj},{expected_num_attr}). "
            "This usually causes checkpoint embedding mismatch and accuracy drop."
        )
        if int(args.strict_vocab_match) == 1:
            raise RuntimeError(msg)
        print(f"[Vocab][Warn] {msg}")

    sg_dataset = VGSceneGraphDataset(
        scene_graphs_path=args.scene_graphs,
        obj_vocab=obj_vocab,
        pred_vocab=pred_vocab,
        attr_vocab=attr_vocab,
        max_nodes=int(args.max_nodes),
        max_attrs=int(args.max_attrs),
        add_reverse_edges=True,
        use_bbox_max_norm=True,
    )
    if bool(args.stream_questions):
        dataset = GQATriModalIterableDataset(
            sg_dataset=sg_dataset,
            questions_path=Path(args.questions),
            image_root=args.image_root,
            use_vision=use_vision,
            allow_missing_modalities=bool(args.allow_missing_modalities),
            max_samples=int(args.max_samples),
        )
        print("[Join] usable_questions=streaming")
    else:
        questions = load_jsonl(Path(args.questions))
        dataset = GQATriModalDataset(
            sg_dataset=sg_dataset,
            questions=questions,
            image_root=args.image_root,
            use_vision=use_vision,
            allow_missing_modalities=bool(args.allow_missing_modalities),
        )
        if int(args.max_samples) > 0:
            dataset = Subset(dataset, list(range(min(int(args.max_samples), len(dataset)))))
        print(f"[Join] usable_questions={len(dataset)}")
        if len(dataset) == 0:
            raise RuntimeError("No usable GQA samples after joining questions/scene-graphs/images.")

    processor = AutoProcessor.from_pretrained(args.vision) if use_vision else None
    tokenizer = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_gqa(batch, processor=processor, use_vision=use_vision)

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
        persistent_workers=bool(args.persistent_workers) and int(args.num_workers) > 0,
    )

    model = OmniGraphModel(
        graph_model_name=args.graph_model,
        vision_model_name=args.vision,
        llm_model_name=args.llm,
        enable_vision=use_vision,
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
    raw_sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(raw_sd, dict) and "state_dict" in raw_sd and isinstance(raw_sd["state_dict"], dict):
        ckpt_sd = raw_sd["state_dict"]
        if any(str(k).startswith("model.") for k in ckpt_sd.keys()):
            sd = {str(k)[len("model.") :] if str(k).startswith("model.") else str(k): v for k, v in ckpt_sd.items()}
        else:
            sd = {str(k): v for k, v in ckpt_sd.items()}
    elif isinstance(raw_sd, dict):
        sd = {str(k): v for k, v in raw_sd.items()}
    else:
        raise RuntimeError(f"Unsupported checkpoint format for {args.ckpt}: expected dict/state_dict.")

    model_sd = model.state_dict()
    filtered: Dict[str, Any] = {}
    skipped = 0
    mismatched: List[Tuple[str, Any, Any]] = []
    unexpected_in_ckpt: List[str] = []
    for k, v in sd.items():
        if k not in model_sd:
            unexpected_in_ckpt.append(str(k))
            continue
        if hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape != model_sd[k].shape:
            skipped += 1
            mismatched.append((str(k), tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v
    critical_mismatched = [m for m in mismatched if _is_critical_weight_key(m[0])]
    critical_unexpected = [k for k in unexpected_in_ckpt if _is_critical_weight_key(k)]
    if int(args.strict_weight_load) == 1 and (critical_mismatched or critical_unexpected):
        msg_lines = ["Critical checkpoint/model key mismatch detected."]
        if critical_mismatched:
            msg_lines.append(f"critical shape mismatches={len(critical_mismatched)} (showing up to 8):")
            for k, s1, s2 in critical_mismatched[:8]:
                msg_lines.append(f"  - {k}: ckpt={s1} model={s2}")
        if critical_unexpected:
            msg_lines.append(f"critical unexpected ckpt keys={len(critical_unexpected)} (showing up to 8):")
            for k in critical_unexpected[:8]:
                msg_lines.append(f"  - {k}")
        raise RuntimeError("\n".join(msg_lines))

    if mismatched:
        print(f"[Init][Warn] shape mismatches skipped={len(mismatched)} (critical={len(critical_mismatched)})")
    if unexpected_in_ckpt:
        print(f"[Init][Warn] unexpected ckpt keys ignored={len(unexpected_in_ckpt)} (critical={len(critical_unexpected)})")

    missing, unexpected_after = model.load_state_dict(filtered, strict=False)
    critical_missing = [k for k in missing if _is_critical_weight_key(str(k))]
    critical_unexpected_after = [k for k in unexpected_after if _is_critical_weight_key(str(k))]
    if int(args.strict_weight_load) == 1 and (critical_missing or critical_unexpected_after):
        msg_lines = ["Critical missing/unexpected keys after load_state_dict."]
        if critical_missing:
            msg_lines.append(f"critical missing={len(critical_missing)} (showing up to 8):")
            for k in critical_missing[:8]:
                msg_lines.append(f"  - {k}")
        if critical_unexpected_after:
            msg_lines.append(f"critical unexpected={len(critical_unexpected_after)} (showing up to 8):")
            for k in critical_unexpected_after[:8]:
                msg_lines.append(f"  - {k}")
        raise RuntimeError("\n".join(msg_lines))

    if missing:
        print(f"[Init][Warn] missing keys={len(missing)}")
    if unexpected_after:
        print(f"[Init][Warn] unexpected keys={len(unexpected_after)}")

    model.to(device)
    model.eval()

    if device.type == "cuda":
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    total = None
    try:
        total = len(dataset)
    except Exception:
        pass

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=total, desc="Infer GQA")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    autocast_ctx = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    with out_path.open("w", encoding="utf-8") as f, torch.inference_mode():
        seen = 0
        for batch in iterator:
            ids = batch["ids"]
            image_ids = batch["image_ids"]
            prompts = batch["prompts"]
            graph_list = batch["graph_list"]
            pil_images = batch["pil_images"]
            has_graph = batch["has_graph"]
            has_image = batch["has_image"]

            groups = {
                "both": [i for i in range(len(ids)) if has_graph[i] and has_image[i]],
                "graph_only": [i for i in range(len(ids)) if has_graph[i] and not has_image[i]],
                "vision_only": [i for i in range(len(ids)) if (not has_graph[i]) and has_image[i]],
                "text_only": [i for i in range(len(ids)) if (not has_graph[i]) and (not has_image[i])],
            }

            def run_group(idxs: List[int], use_graph: bool, use_vision_branch: bool) -> List[str]:
                if not idxs:
                    return []
                prompt_texts = [build_prompt_for_tokenizer(tokenizer, prompts[i]) for i in idxs]
                tok = tokenizer(
                    prompt_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=int(args.max_length),
                )
                input_ids = tok["input_ids"].to(device)

                graph_data = None
                if use_graph:
                    from torch_geometric.data import Batch as GeoBatch

                    graphs = [graph_list[i] for i in idxs]
                    graph_data = GeoBatch.from_data_list(graphs).to(device)
                    if hasattr(graph_data, "obj_id") and hasattr(graph_data, "attr_id") and hasattr(graph_data, "bbox"):
                        try:
                            graph_node = model.vg_adapter(
                                graph_data.obj_id,
                                graph_data.attr_id,
                                graph_data.bbox,
                                obj_hash_id=getattr(graph_data, "obj_hash_id", None),
                                attr_hash_id=getattr(graph_data, "attr_hash_id", None),
                                edge_index=getattr(graph_data, "edge_index", None),
                                edge_pred_id=getattr(graph_data, "edge_pred_id", None),
                                edge_pred_hash_id=getattr(graph_data, "edge_pred_hash_id", None),
                            )
                            graph_data.graph_node = graph_node
                            graph_data.x = graph_node
                        except Exception:
                            pass

                pixel_values = None
                if use_vision_branch:
                    images = [pil_images[i] for i in idxs]
                    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)

                with autocast_ctx(dtype=amp_dtype, enabled=amp_dtype is not None):
                    inputs_embeds = model.llm.model.get_input_embeddings()(input_ids)
                    target_dtype = inputs_embeds.dtype
                    embeds_list = []

                    if graph_data is not None:
                        graph_tokens = model.graph_branch(graph_data)
                        graph_embeds = model.gl_projector(graph_tokens)
                        if graph_embeds.dtype != target_dtype:
                            graph_embeds = graph_embeds.to(target_dtype)
                        embeds_list.append(graph_embeds)

                    if pixel_values is not None and model.vision_branch is not None and model.vl_projector is not None:
                        vision_tokens = model.vision_branch(pixel_values)
                        vision_embeds = model.vl_projector(vision_tokens)
                        if vision_embeds.dtype != target_dtype:
                            vision_embeds = vision_embeds.to(target_dtype)
                        embeds_list.append(vision_embeds)

                    embeds_list.append(inputs_embeds)
                    combined_embeds = torch.cat(embeds_list, dim=1)
                    attention_mask = torch.ones(
                        combined_embeds.size(0),
                        combined_embeds.size(1),
                        dtype=torch.long,
                        device=combined_embeds.device,
                    )

                    gen = model.llm.generate(
                        inputs_embeds=combined_embeds,
                        attention_mask=attention_mask,
                        max_new_tokens=int(args.max_new_tokens),
                        do_sample=False,
                    )
                return tokenizer.batch_decode(gen, skip_special_tokens=True)

            decoded_map: Dict[int, str] = {}
            for group_name, idxs in groups.items():
                if not idxs:
                    continue
                use_graph = group_name in {"both", "graph_only"}
                use_vision_branch = group_name in {"both", "vision_only"}
                decoded = run_group(idxs, use_graph=use_graph, use_vision_branch=use_vision_branch)
                for i, text in zip(idxs, decoded):
                    decoded_map[i] = text

            for i in range(len(ids)):
                text = decoded_map.get(i, "")
                pred = extract_short_answer(text)
                rec = {
                    "id": ids[i],
                    "pred": pred,
                    "image_id": image_ids[i],
                    "has_graph": bool(has_graph[i]),
                    "has_image": bool(has_image[i]),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                seen += 1

            if int(args.log_every) > 0 and seen % int(args.log_every) == 0:
                print(f"Processed {seen} samples...")

    print(f"Wrote GQA predictions -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
