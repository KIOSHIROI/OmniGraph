#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Subset
from transformers import AutoProcessor, AutoTokenizer

from omnigraph.data.vg_scene_graph_dataset import VGSceneGraphDataset, build_vg_vocabs_from_file
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
    meta_path = ckpt.parent / "stage3_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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
    ap.add_argument("--ckpt", required=True, help="OmniGraph model state_dict (.pt) or stage3 exported state dict")
    ap.add_argument("--output", required=True, help="prediction jsonl path")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--vision", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--graph_model", default="clip_gt_arxiv_pub")
    ap.add_argument("--node_encoder_type", default="auto", choices=["auto", "hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_alpha_init", type=float, default=-1.0, help="<0 means read from stage3_meta or fallback.")
    ap.add_argument("--node_encoder_out_dim", type=int, default=0, help="<=0 means read from stage3_meta or fallback 128.")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--min_freq", type=int, default=2)
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

    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_file(args.scene_graphs, min_freq=int(args.min_freq))
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)
    print(f"[Vocab] NUM_OBJ={num_obj} NUM_ATTR={num_attr}")

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
        node_encoder_type=resolved_node_encoder_type,
        node_encoder_alpha_init=resolved_node_encoder_alpha,
        node_encoder_out_dim=resolved_node_encoder_out_dim,
        node_encoder_trainable=False,
    )
    sd = torch.load(args.ckpt, map_location="cpu")
    model_sd = model.state_dict()
    filtered = {}
    skipped = 0
    for k, v in sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
            if v.shape != model_sd[k].shape:
                skipped += 1
                continue
        filtered[k] = v
    if skipped:
        print(f"[Init] Skipped {skipped} mismatched keys (vocab/adapter resize).")
    model.load_state_dict(filtered, strict=False)
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
