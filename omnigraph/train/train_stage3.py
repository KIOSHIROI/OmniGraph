# omnigraph/train/train_stage3.py
from __future__ import annotations

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Repo bootstrap + env (MUST be before transformers import)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from transformers import AutoTokenizer, AutoProcessor
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from torch_geometric.data import Batch as GeoBatch

# Dataset (your VG scene graph builder)
from omnigraph.data.vg_scene_graph_dataset import (  # noqa: E402
    build_vg_vocabs_from_file,
    VGSceneGraphDataset,
)
from omnigraph.data.vg_graph_qa import build_vg_graph_qa_records  # noqa: E402

# Model
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402


# ---------------------------------------------------------------------------
# Utils: load regions (with bbox) + find image path
# ---------------------------------------------------------------------------

def load_region_records(region_path: str) -> List[Dict[str, Any]]:
    """
    region_descriptions.json format (typical):
    [
      {"regions":[{"image_id":26,"phrase":"...", "x":..., "y":..., "width":..., "height":...}, ...], "id":26},
      ...
    ]
    Returns list of dict:
      {"image_id": int, "phrase": str, "x": int, "y": int, "w": int, "h": int}
    """
    with open(region_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    if not isinstance(data, list):
        return out

    for item in data:
        if not isinstance(item, dict):
            continue
        image_id = item.get("id", None)
        if image_id is None:
            image_id = item.get("image_id", None)
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

            # bbox fields
            x = r.get("x", None)
            y = r.get("y", None)
            w = r.get("width", None)
            h = r.get("height", None)

            # Some dumps may store as strings
            try:
                x = int(x) if x is not None else 0
                y = int(y) if y is not None else 0
                w = int(w) if w is not None else 0
                h = int(h) if h is not None else 0
            except Exception:
                x, y, w, h = 0, 0, 0, 0

            out.append({"image_id": image_id, "phrase": phrase, "x": x, "y": y, "w": w, "h": h})
    return out


def find_image_path(image_root: str, image_id: int) -> Optional[str]:
    """
    VG images are commonly stored as {image_id}.jpg under VG_100K / VG_100K_2.
    This tries several common layouts.
    """
    root = Path(image_root)
    alt_root = root / "contents" / "images"
    img_root = root / "images"

    candidates = [
        root / f"{image_id}.jpg",
        root / f"{image_id}.png",
        img_root / f"{image_id}.jpg",
        img_root / f"{image_id}.png",
        root / "VG_100K" / f"{image_id}.jpg",
        root / "VG_100K" / f"{image_id}.png",
        root / "VG_100K_2" / f"{image_id}.jpg",
        root / "VG_100K_2" / f"{image_id}.png",
        img_root / "VG_100K" / f"{image_id}.jpg",
        img_root / "VG_100K" / f"{image_id}.png",
        img_root / "VG_100K_2" / f"{image_id}.jpg",
        img_root / "VG_100K_2" / f"{image_id}.png",
        # common in this repo: data/vg/contents/images
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


# ---------------------------------------------------------------------------
# Dataset: (graph + cropped region image + region phrase)
# ---------------------------------------------------------------------------

class VGTriModalRegionDataset(Dataset):
    """
    One sample:
      - graph_data (pyg Data) from scene graph
      - image (PIL) cropped to region bbox (or full if bbox invalid)
      - prompt + answer (region phrase)
    Output dict:
      { id, image_id, graph_data, pil_image, prompt, answer }
    """

    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        region_records: List[Dict[str, Any]],
        image_root: str,
        qa_records: Optional[List[Dict[str, Any]]] = None,
        prompt: str = "Describe the region.",
        min_phrase_len: int = 1,
    ):
        self.sg = sg_dataset
        self.image_root = image_root
        self.prompt = prompt

        # build image_id -> sg_dataset index
        raw_items = None
        if hasattr(self.sg, "items"):
            raw_items = getattr(self.sg, "items")
        elif hasattr(self.sg, "scene_graphs"):
            raw_items = getattr(self.sg, "scene_graphs")
        if raw_items is None:
            raise AttributeError(
                "VGSceneGraphDataset must expose raw scene graph list as .items or .scene_graphs"
            )

        self.image2idx: Dict[int, int] = {}
        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                self.image2idx[int(it["image_id"])] = i

        # join & filter
        self.samples: List[Dict[str, Any]] = []
        for rr in region_records:
            iid = int(rr["image_id"])
            if iid not in self.image2idx:
                continue
            phrase = str(rr["phrase"]).strip()
            if len(phrase) < int(min_phrase_len):
                continue
            img_path = find_image_path(self.image_root, iid)
            if img_path is None:
                continue
            self.samples.append(
                {
                    "image_id": iid,
                    "sg_idx": int(self.image2idx[iid]),
                    "img_path": img_path,
                    "phrase": phrase,
                    "prompt": self.prompt,
                    "x": int(rr.get("x", 0)),
                    "y": int(rr.get("y", 0)),
                    "w": int(rr.get("w", rr.get("width", 0)) if rr.get("w", None) is not None else 0),
                    "h": int(rr.get("h", rr.get("height", 0)) if rr.get("h", None) is not None else 0),
                    "source": "region",
                }
            )

        if qa_records:
            for qa in qa_records:
                try:
                    iid = int(qa.get("image_id"))
                except Exception:
                    continue
                if iid not in self.image2idx:
                    continue
                q = str(qa.get("question", "")).strip()
                a = str(qa.get("answer", "")).strip()
                if not q or not a:
                    continue
                img_path = find_image_path(self.image_root, iid)
                if img_path is None:
                    continue
                self.samples.append(
                    {
                        "image_id": iid,
                        "sg_idx": int(self.image2idx[iid]),
                        "img_path": img_path,
                        "phrase": a,
                        "prompt": q,
                        "x": 0,
                        "y": 0,
                        "w": 0,
                        "h": 0,
                        "source": "graph_qa",
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image_id = int(s["image_id"])
        sg_idx = int(s["sg_idx"])

        sg_item = self.sg[sg_idx]
        graph_data = sg_item["graph_data"]

        # image load + crop
        from PIL import Image  # local import (safe)
        img = Image.open(s["img_path"]).convert("RGB")

        x, y, w, h = int(s["x"]), int(s["y"]), int(s["w"]), int(s["h"])
        if w > 1 and h > 1:
            # clamp crop to image boundary
            W, H = img.size
            x0 = max(0, min(x, W - 1))
            y0 = max(0, min(y, H - 1))
            x1 = max(x0 + 1, min(x0 + w, W))
            y1 = max(y0 + 1, min(y0 + h, H))
            img = img.crop((x0, y0, x1, y1))

        return {
            "id": f"{image_id}_{s.get('source', 'mix')}_{idx}",
            "image_id": image_id,
            "graph_data": graph_data,
            "pil_image": img,
            "prompt": str(s.get("prompt", self.prompt)),
            "answer": str(s["phrase"]),
        }


def split_by_image_id(dataset: VGTriModalRegionDataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    """
    Split by image_id: all regions of a given image go either train or val.
    """
    image_ids = sorted({int(s["image_id"]) for s in dataset.samples})
    rng = random.Random(int(seed))
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        val_ids = set(image_ids)
    else:
        val_n = max(1, int(len(image_ids) * float(val_ratio)))
        val_ids = set(image_ids[:val_n])

    train_idx, val_idx = [], []
    for i, s in enumerate(dataset.samples):
        if int(s["image_id"]) in val_ids:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_tri(batch: List[Dict[str, Any]], processor: Any) -> Dict[str, Any]:
    graphs = [b["graph_data"] for b in batch]
    batch_graph = GeoBatch.from_data_list(graphs)

    from PIL import Image
    import numpy as np
    import torch

    def _ensure_pil(img: Any) -> Image.Image:
        try:
            if isinstance(img, Image.Image):
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.size[0] < 1 or img.size[1] < 1:
                    return Image.new("RGB", (224, 224), (0, 0, 0))
                return img

            if torch.is_tensor(img):
                img = img.detach().cpu().numpy()

            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                elif img.ndim == 3:
                    # try channel-last first
                    if img.shape[-1] not in (1, 3, 4) and img.shape[0] in (1, 3, 4):
                        img = np.transpose(img, (1, 2, 0))
                    if img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    elif img.shape[-1] > 4:
                        img = img[..., :3]
                else:
                    return Image.new("RGB", (224, 224), (0, 0, 0))

                return Image.fromarray(img.astype("uint8"), mode="RGB")
        except Exception:
            return Image.new("RGB", (224, 224), (0, 0, 0))

        return Image.new("RGB", (224, 224), (0, 0, 0))

    pil_images = [_ensure_pil(b["pil_image"]) for b in batch]

    safe_image = Image.new("RGB", (224, 224), (0, 0, 0))
    try:
        pixel_values = processor(images=pil_images, return_tensors="pt")["pixel_values"]
    except Exception:
        pixel_values_list = []
        for img in pil_images:
            try:
                pv = processor(images=img, return_tensors="pt")["pixel_values"][0]
            except Exception:
                pv = processor(images=safe_image, return_tensors="pt")["pixel_values"][0]
            pixel_values_list.append(pv)
        pixel_values = torch.stack(pixel_values_list, dim=0)

    prompts = [b["prompt"] for b in batch]
    answers = [b["answer"] for b in batch]
    ids = [b["id"] for b in batch]
    image_ids = [b["image_id"] for b in batch]

    return {
        "ids": ids,
        "image_ids": image_ids,
        "graph_data": batch_graph,
        "pixel_values": pixel_values,
        "prompts": prompts,
        "answers": answers,
    }


# ---------------------------------------------------------------------------
# Tokenization: chat inputs + labels masking prompt tokens
# ---------------------------------------------------------------------------

def build_chat_inputs_and_labels(
    tokenizer: Any,
    prompts: List[str],
    answers: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_chat = hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None

    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []

    for p, a in zip(prompts, answers):
        p = p or ""
        a = a or ""

        if use_chat:
            messages_full = [{"role": "user", "content": p}, {"role": "assistant", "content": a}]
            full_text = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)

            messages_prompt = [{"role": "user", "content": p}]
            prompt_text = tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True)

            full = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
            prompt_tok = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_length)

            full_ids = full["input_ids"][0]
            prompt_len = int(prompt_tok["input_ids"].shape[1])

            labels = full_ids.clone()
            labels[:prompt_len] = -100

            attn = full.get("attention_mask", torch.ones_like(full_ids))
            if attn.dim() > 1:
                attn = attn[0]
        else:
            # fallback: train only answer tokens
            full = tokenizer(a, return_tensors="pt", truncation=True, max_length=max_length)
            full_ids = full["input_ids"][0]
            labels = full_ids.clone()
            attn = full.get("attention_mask", torch.ones_like(full_ids))
            if attn.dim() > 1:
                attn = attn[0]

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append(attn)

    pad_id = tokenizer.pad_token_id
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(device)
    labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100).to(device)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attn_list, batch_first=True, padding_value=0).to(device)
    return input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# Checkpoint loading: strict Stage2B ckpt only
# ---------------------------------------------------------------------------

def load_stage2b_weights_only_into_model(model: OmniGraphModel, stage2b_ckpt_path: str) -> Dict[str, Any]:
    p = Path(stage2b_ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"--stage2B_ckpt not found: {stage2b_ckpt_path}")

    obj = torch.load(str(p), map_location="cpu")
    if not isinstance(obj, dict) or "state_dict" not in obj:
        raise RuntimeError("Stage3 requires a Stage2B Lightning checkpoint (.ckpt), not a raw state_dict.")

    hp = obj.get("hyper_parameters", {}) or {}
    stage2a_ckpt = hp.get("stage2A_ckpt")
    if not stage2a_ckpt:
        raise RuntimeError(
            "Stage2B checkpoint metadata missing 'stage2A_ckpt'. "
            "Use Stage2B checkpoint generated from strict pipeline."
        )

    sd = obj["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            new_sd[k] = v

    # Keep strict pipeline robust against arch changes.
    model_sd = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in new_sd.items():
        if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape") and v.shape != model_sd[k].shape:
            skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
            continue
        filtered[k] = v
    if skipped:
        print(f"[Stage3] Skipped {len(skipped)} mismatched keys (example: {skipped[0][0]}).")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Stage3] loaded weights-only from Stage2B ckpt: {p}")
    print(f"[Stage3] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[Stage3] Stage2B provenance: stage2A_ckpt={stage2a_ckpt}")
    return {
        "stage2B_ckpt": str(p),
        "stage2A_ckpt": str(stage2a_ckpt),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


# ---------------------------------------------------------------------------
# LightningModule (Stage3 projector-only)
# ---------------------------------------------------------------------------

class Stage3PL(pl.LightningModule):
    """
    Stage3: tri-modal (graph + image + text), train ONLY:
      - gl_projector
      - vl_projector
    """

    def __init__(
        self,
        llm_model_name: str,
        vision_model_name: str,
        graph_model_name: str,
        num_obj: int,
        num_attr: int,
        lr: float,
        max_length: int,
        stage2B_ckpt: str,
        max_graph_tokens: int | None = None,
        max_vision_tokens: int | None = None,
        llm_dtype: str = "bfloat16",
        llm_attn_implementation: str = "sdpa",
        node_encoder_type: str = "hybrid",
        node_encoder_out_dim: int = 128,
        train_node_encoder: bool = False,
        node_encoder_alpha_init: float = 0.3,
        auto_resize_token_embeddings: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = OmniGraphModel(
            graph_model_name=graph_model_name,
            vision_model_name=vision_model_name,
            llm_model_name=llm_model_name,
            enable_vision=True,
            num_obj=int(num_obj),
            num_attr=int(num_attr),
            max_graph_tokens=max_graph_tokens,
            max_vision_tokens=max_vision_tokens,
            llm_dtype=llm_dtype,
            llm_attn_implementation=llm_attn_implementation,
            node_encoder_type=node_encoder_type,
            node_encoder_out_dim=int(node_encoder_out_dim),
            node_encoder_trainable=bool(train_node_encoder),
            node_encoder_alpha_init=float(node_encoder_alpha_init),
        )

        # strict pipeline: Stage3 always initializes from Stage2B ckpt
        self.stage2B_provenance = load_stage2b_weights_only_into_model(self.model, stage2B_ckpt)

        # Freeze everything except projectors
        self.train_node_encoder = bool(train_node_encoder)
        for name, p in self.model.named_parameters():
            p.requires_grad = (
                ("gl_projector" in name)
                or ("vl_projector" in name)
                or (self.train_node_encoder and (name.startswith("node_encoder.") or ("vg_adapter" in name)))
            )
        print(
            f"[Stage3] node_encoder_type={self.model.node_encoder_type} "
            f"train_node_encoder={self.train_node_encoder}"
        )

        # LLM memory-saving knobs
        if hasattr(self.model.llm.model, "gradient_checkpointing_enable"):
            self.model.llm.model.gradient_checkpointing_enable()
        if hasattr(self.model.llm.model, "config"):
            self.model.llm.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = min(getattr(self.tokenizer, "model_max_length", 2048), int(max_length))

        if auto_resize_token_embeddings:
            try:
                self.model.llm.model.resize_token_embeddings(len(self.tokenizer))
            except Exception:
                pass

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return AdamW(params, lr=float(self.hparams.lr))

    def _step(self, batch: Dict[str, Any]) -> torch.Tensor:
        graph_data = batch["graph_data"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        prompts = batch["prompts"]
        answers = batch["answers"]

        input_ids, attention_mask, labels = build_chat_inputs_and_labels(
            tokenizer=self.tokenizer,
            prompts=prompts,
            answers=answers,
            device=self.device,
            max_length=self.max_length,
        )

        outputs = self.model(
            graph_data=graph_data,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_debug=False,
        )

        loss = outputs.loss
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            # fallback: fp32 CE
            logits = getattr(outputs, "logits", None)
            if logits is None:
                return torch.tensor(0.0, device=self.device)
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()
            L = min(shift_logits.size(1), shift_labels.size(1))
            shift_logits = shift_logits[:, :L, :]
            shift_labels = shift_labels[:, :L]
            if (shift_labels != -100).any():
                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                )
            else:
                loss = torch.tensor(0.0, device=self.device)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["prompts"]))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["prompts"]))
        return loss


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def _parse_val_check_interval(x: float) -> float | int:
    x = float(x)
    if x >= 1.0:
        return int(x)
    if x <= 0.0:
        return 1.0
    return x


def _parse_precision(v: str) -> int | str:
    s = str(v).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return s


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scene_graphs", type=str, required=True)
    ap.add_argument("--regions", type=str, required=True)
    ap.add_argument("--image_root", type=str, required=True)

    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--vision", type=str, default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--graph_model", type=str, default="clip_gt_arxiv_pub")

    ap.add_argument("--stage2B_ckpt", type=str, required=True, help="best ckpt from stage2-B (.ckpt)")

    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_graph_tokens", type=int, default=0, help="If >0, truncate graph prefix tokens before LLM.")
    ap.add_argument("--max_vision_tokens", type=int, default=0, help="If >0, truncate vision prefix tokens before LLM.")
    ap.add_argument("--llm_dtype", type=str, default="bfloat16", help="LLM load dtype: bfloat16/float16/float32.")
    ap.add_argument("--llm_attn_implementation", type=str, default="sdpa", help="LLM attention backend (e.g., sdpa/flash_attention_2).")
    ap.add_argument("--node_encoder_type", type=str, default="hybrid", choices=["hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_out_dim", type=int, default=128, help="Graph node encoder output dim.")
    ap.add_argument("--train_node_encoder", type=int, default=0, choices=[0, 1], help="1=train node encoder, 0=freeze.")
    ap.add_argument("--node_encoder_alpha_init", type=float, default=0.3, help="Hybrid node encoder alpha init.")

    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=20000)

    ap.add_argument("--val_ratio", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=0.01)

    ap.add_argument("--val_check_interval", type=float, default=2000,
                    help=">=1: validate every N steps; (0,1]: validate every fraction of epoch")
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--num_sanity_val_steps", type=int, default=0)
    ap.add_argument("--accumulate_grad_batches", type=int, default=4)

    ap.add_argument("--save_dir", type=str, default="checkpoints_stage3")

    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)
    ap.add_argument("--prompt", type=str, default="Describe the region.")
    ap.add_argument("--disable_graph_qa", action="store_true", help="Disable synthetic graph QA from VG scene graphs.")
    ap.add_argument("--graph_qa_max_per_image", type=int, default=2, help="Max synthetic QA pairs per image.")
    ap.add_argument("--graph_qa_repeat", type=int, default=1, help="Repeat factor for synthetic graph QA samples.")
    ap.add_argument("--graph_qa_seed", type=int, default=42, help="Random seed for synthetic graph QA generation.")

    args = ap.parse_args()
    pl.seed_everything(int(args.seed), workers=True)
    print("[Pipeline] Stage3 start: requires Stage2B .ckpt from strict pipeline.")
    if not Path(args.stage2B_ckpt).exists():
        raise FileNotFoundError(f"--stage2B_ckpt not found: {args.stage2B_ckpt}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) build vocabs
    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_file(args.scene_graphs, min_freq=int(args.min_freq))
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)
    print(f"[Vocab] NUM_OBJ={num_obj} NUM_ATTR={num_attr}")

    # 2) scene graph dataset
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

    # 3) regions + join
    region_records = load_region_records(args.regions)
    print(f"[Regions] loaded records={len(region_records)}")

    qa_records: List[Dict[str, Any]] = []
    if not bool(args.disable_graph_qa):
        qa_records = build_vg_graph_qa_records(
            scene_graph_items=sg_dataset.items,
            max_per_image=int(args.graph_qa_max_per_image),
            seed=int(args.graph_qa_seed),
        )
        repeat = max(1, int(args.graph_qa_repeat))
        if repeat > 1 and len(qa_records) > 0:
            qa_records = qa_records * repeat
        print(f"[GraphQA] synthetic_pairs={len(qa_records)} (repeat={repeat})")
    else:
        print("[GraphQA] disabled.")

    full_dataset = VGTriModalRegionDataset(
        sg_dataset=sg_dataset,
        region_records=region_records,
        image_root=args.image_root,
        qa_records=qa_records,
        prompt=str(args.prompt),
    )
    print(f"[Join] usable_samples={len(full_dataset)}")

    if len(full_dataset) == 0:
        raise RuntimeError("No usable samples after joining regions with scene graphs & images. Check paths/ids.")

    # 4) split train/val by image_id
    train_ds, val_ds = split_by_image_id(full_dataset, val_ratio=float(args.val_ratio), seed=int(args.seed))
    print(f"[Split] train={len(train_ds)} val={len(val_ds)} (val_ratio={args.val_ratio})")

    # 5) vision processor (to pixel_values)
    processor = AutoProcessor.from_pretrained(args.vision)

    def _collate(batch):
        return collate_tri(batch, processor=processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=_collate,
    )

    vci = _parse_val_check_interval(args.val_check_interval)

    # 6) model
    pl_model = Stage3PL(
        llm_model_name=args.llm,
        vision_model_name=args.vision,
        graph_model_name=args.graph_model,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        lr=float(args.lr),
        max_length=int(args.max_length),
        stage2B_ckpt=str(args.stage2B_ckpt),
        max_graph_tokens=(int(args.max_graph_tokens) if int(args.max_graph_tokens) > 0 else None),
        max_vision_tokens=(int(args.max_vision_tokens) if int(args.max_vision_tokens) > 0 else None),
        llm_dtype=str(args.llm_dtype),
        llm_attn_implementation=str(args.llm_attn_implementation),
        node_encoder_type=str(args.node_encoder_type),
        node_encoder_out_dim=int(args.node_encoder_out_dim),
        train_node_encoder=bool(int(args.train_node_encoder)),
        node_encoder_alpha_init=float(args.node_encoder_alpha_init),
        auto_resize_token_embeddings=True,
    )

    # 7) callbacks
    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="stage3-step={step:07d}-val_loss={val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        verbose=True,
    )

    use_gpu = torch.cuda.is_available() and int(args.gpu) >= 0
    devices = [int(args.gpu)] if use_gpu else 1
    if use_gpu:
        if int(args.gpu) >= torch.cuda.device_count():
            raise ValueError(f"--gpu={args.gpu} out of range (device_count={torch.cuda.device_count()}).")
        torch.cuda.set_device(int(args.gpu))

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        precision=_parse_precision(args.precision),
        max_steps=int(args.max_steps),
        max_epochs=999999,  # controlled by max_steps + early stopping
        callbacks=[ckpt_cb, lr_monitor, es],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
        check_val_every_n_epoch=1,
        val_check_interval=vci,
        limit_val_batches=float(args.limit_val_batches),
        num_sanity_val_steps=max(0, int(args.num_sanity_val_steps)),
        enable_checkpointing=True,
        enable_model_summary=False,
    )

    # IMPORTANT: do NOT pass ckpt_path here (we already loaded weights-only)
    trainer.fit(pl_model, train_loader, val_loader)

    # Export clean OmniGraphModel state_dict (easy to load later)
    export_path = save_dir / "omnigraph_stage3_state_dict.pt"
    torch.save(pl_model.model.state_dict(), str(export_path))
    stage_meta = {
        "stage": "stage3",
        "num_obj": int(num_obj),
        "num_attr": int(num_attr),
        "stage2B_provenance": pl_model.stage2B_provenance,
        "node_encoder_config": pl_model.model.node_encoder_config,
        "best_ckpt": ckpt_cb.best_model_path or "",
        "export_path": str(export_path),
    }
    (save_dir / "stage3_meta.json").write_text(json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Export] stage3 state_dict -> {export_path}")
    print(f"[Best] {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
