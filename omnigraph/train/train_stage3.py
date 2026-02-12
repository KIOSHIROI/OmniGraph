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
import torch.nn as nn
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
from omnigraph.train.common import (  # noqa: E402
    build_binary_aux_targets,
    build_chat_inputs_and_labels,
    format_prompt_with_qa_type as _format_prompt_with_qa_type,
    parse_precision as _parse_precision,
    parse_val_check_interval as _parse_val_check_interval,
)

# Model
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402
from omnigraph.model.losses.multimodal_align import (  # noqa: E402
    build_pooled_features,
    mine_hard_negatives,
    xtc_bidirectional_loss,
    xtm_pair_loss,
)


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
        use_qa_type_token: bool = True,
    ):
        self.sg = sg_dataset
        self.image_root = image_root
        self.prompt = prompt
        self.use_qa_type_token = bool(use_qa_type_token)

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
                    "qa_type": "region_caption",
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
                        "qa_type": str(qa.get("qa_type", "graph_qa")),
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
            "prompt": _format_prompt_with_qa_type(
                str(s.get("prompt", self.prompt)),
                str(s.get("qa_type", "unknown")),
                self.use_qa_type_token,
            ),
            "answer": str(s["phrase"]),
            "qa_type": str(s.get("qa_type", "unknown")),
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
    qa_types = [str(b.get("qa_type", "unknown")) for b in batch]

    return {
        "ids": ids,
        "image_ids": image_ids,
        "graph_data": batch_graph,
        "pixel_values": pixel_values,
        "prompts": prompts,
        "answers": answers,
        "qa_type": qa_types,
    }


# ---------------------------------------------------------------------------
# Checkpoint loading: strict Stage2B ckpt only
# ---------------------------------------------------------------------------
def _normalize_graph_tokenizer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    t = str(cfg.get("type", "qformer")).strip().lower()
    if t not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Unsupported graph tokenizer type in provenance: {t}")
    out = {
        "type": t,
        "num_latents": int(cfg.get("num_latents", cfg.get("num_query_tokens", 32))),
        "hidden_dim": int(cfg.get("hidden_dim", cfg.get("qformer_hidden_dim", 768))),
        "num_layers": int(cfg.get("num_layers", 3)),
        "num_heads": int(cfg.get("num_heads", 8)),
        "ff_mult": int(cfg.get("ff_mult", 4)),
        "dropout": float(cfg.get("dropout", 0.0)),
    }
    return out


def _assert_graph_tokenizer_match(expected_cfg: Dict[str, Any], got_cfg: Dict[str, Any], stage_name: str) -> None:
    if str(expected_cfg.get("type")) != str(got_cfg.get("type")):
        raise RuntimeError(
            f"{stage_name} graph tokenizer type mismatch: expected={expected_cfg.get('type')} got={got_cfg.get('type')}"
        )
    if str(got_cfg.get("type")) == "perceiver":
        for k in ("num_latents", "hidden_dim", "num_layers", "num_heads", "ff_mult"):
            if int(expected_cfg.get(k)) != int(got_cfg.get(k)):
                raise RuntimeError(
                    f"{stage_name} perceiver config mismatch on {k}: "
                    f"expected={expected_cfg.get(k)} got={got_cfg.get(k)}"
                )
        if abs(float(expected_cfg.get("dropout")) - float(got_cfg.get("dropout"))) > 1e-6:
            raise RuntimeError(
                f"{stage_name} perceiver config mismatch on dropout: "
                f"expected={expected_cfg.get('dropout')} got={got_cfg.get('dropout')}"
            )


def _extract_stage2b_provenance_from_ckpt(stage2b_ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
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

    stage2a_bootstrap = hp.get("stage2A_bootstrap")
    if not isinstance(stage2a_bootstrap, dict):
        stage1_qformer_ckpt = hp.get("stage1_qformer_ckpt", None)
        guessed_mode = "legacy_stage1" if stage1_qformer_ckpt else "no_stage1"
        guessed_tok = str(hp.get("graph_tokenizer_type", "qformer" if guessed_mode == "legacy_stage1" else "perceiver")).strip().lower()
        stage2a_bootstrap = {
            "mode": guessed_mode,
            "graph_tokenizer_type": guessed_tok,
            "stage1_qformer_ckpt": stage1_qformer_ckpt,
        }
    mode = str(stage2a_bootstrap.get("mode", "")).strip().lower()
    tok = str(stage2a_bootstrap.get("graph_tokenizer_type", "")).strip().lower()
    if mode not in {"legacy_stage1", "no_stage1"}:
        raise RuntimeError(f"Invalid Stage2B->Stage3 bootstrap mode: {mode}")
    if tok not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Invalid Stage2B->Stage3 graph tokenizer type: {tok}")

    raw_cfg = hp.get("graph_tokenizer_config")
    if isinstance(raw_cfg, dict):
        graph_tokenizer_config = _normalize_graph_tokenizer_config(raw_cfg)
    else:
        graph_tokenizer_config = _normalize_graph_tokenizer_config(
            {
                "type": tok,
                "num_latents": hp.get("perceiver_num_latents", 32),
                "hidden_dim": hp.get("perceiver_hidden_dim", 768),
                "num_layers": hp.get("perceiver_num_layers", 3),
                "num_heads": hp.get("perceiver_num_heads", 8),
                "ff_mult": hp.get("perceiver_ff_mult", 4),
                "dropout": hp.get("perceiver_dropout", 0.0),
            }
        )
    if graph_tokenizer_config["type"] != tok:
        raise RuntimeError(
            "Stage2B provenance mismatch: "
            f"bootstrap type={tok}, graph_tokenizer_config type={graph_tokenizer_config['type']}"
        )
    return obj, stage2a_bootstrap, graph_tokenizer_config


def load_stage2b_weights_only_into_model(
    model: OmniGraphModel,
    stage2b_ckpt_path: str,
    expected_graph_tokenizer_config: Dict[str, Any],
) -> Dict[str, Any]:
    p = Path(stage2b_ckpt_path)
    obj, stage2a_bootstrap, stage2b_graph_tokenizer_config = _extract_stage2b_provenance_from_ckpt(stage2b_ckpt_path)
    _assert_graph_tokenizer_match(
        expected_cfg=expected_graph_tokenizer_config,
        got_cfg=stage2b_graph_tokenizer_config,
        stage_name="Stage3",
    )
    hp = obj.get("hyper_parameters", {}) or {}
    stage2a_ckpt = hp.get("stage2A_ckpt")

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
    print(
        "[Stage3] Stage2B provenance: "
        f"stage2A_ckpt={stage2a_ckpt} "
        f"bootstrap={stage2a_bootstrap} graph_tokenizer={stage2b_graph_tokenizer_config}"
    )
    return {
        "stage2B_ckpt": str(p),
        "stage2A_ckpt": str(stage2a_ckpt),
        "stage2A_bootstrap": stage2a_bootstrap,
        "graph_tokenizer_config": stage2b_graph_tokenizer_config,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def _resolve_graph_tokenizer_from_stage2b(
    args: argparse.Namespace,
    stage2b_graph_tokenizer_config: Dict[str, Any],
) -> Dict[str, Any]:
    req_type = str(args.graph_tokenizer_type).strip().lower()
    upstream_type = str(stage2b_graph_tokenizer_config.get("type")).strip().lower()
    if req_type == "auto":
        resolved_type = upstream_type
    else:
        resolved_type = req_type
    if resolved_type not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Unsupported graph_tokenizer_type for Stage3: {resolved_type}")
    if resolved_type != upstream_type:
        raise RuntimeError(
            "Stage3 tokenizer type must match Stage2B provenance: "
            f"requested={resolved_type} upstream={upstream_type}"
        )

    resolved_cfg = {
        "type": resolved_type,
        "num_latents": int(stage2b_graph_tokenizer_config.get("num_latents", 32)),
        "hidden_dim": int(stage2b_graph_tokenizer_config.get("hidden_dim", 768)),
        "num_layers": int(stage2b_graph_tokenizer_config.get("num_layers", 3)),
        "num_heads": int(stage2b_graph_tokenizer_config.get("num_heads", 8)),
        "ff_mult": int(stage2b_graph_tokenizer_config.get("ff_mult", 4)),
        "dropout": float(stage2b_graph_tokenizer_config.get("dropout", 0.0)),
    }
    if resolved_type == "perceiver":
        if int(args.perceiver_num_latents) > 0:
            resolved_cfg["num_latents"] = int(args.perceiver_num_latents)
        if int(args.perceiver_num_layers) > 0:
            resolved_cfg["num_layers"] = int(args.perceiver_num_layers)
        if int(args.perceiver_num_heads) > 0:
            resolved_cfg["num_heads"] = int(args.perceiver_num_heads)
        if int(args.perceiver_ff_mult) > 0:
            resolved_cfg["ff_mult"] = int(args.perceiver_ff_mult)
        if float(args.perceiver_dropout) >= 0:
            resolved_cfg["dropout"] = float(args.perceiver_dropout)
        _assert_graph_tokenizer_match(
            expected_cfg=stage2b_graph_tokenizer_config,
            got_cfg=resolved_cfg,
            stage_name="Stage3",
        )
    return resolved_cfg


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
        graph_tokenizer_type: str = "qformer",
        perceiver_num_latents: int = 32,
        perceiver_num_layers: int = 3,
        perceiver_num_heads: int = 8,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
        node_encoder_type: str = "hybrid",
        node_encoder_out_dim: int = 128,
        train_node_encoder: bool = False,
        node_encoder_alpha_init: float = 0.3,
        enable_gvl_adapter: bool = True,
        gvl_adapter_gate_init: float = 0.1,
        enable_graph_aux_head: bool = True,
        graph_aux_loss_weight: float = 0.0,
        enable_xtc: bool = True,
        enable_xtm: bool = True,
        xtc_weight: float = 0.05,
        xtm_weight: float = 0.03,
        xtc_logit_scale_init: float = 2.66,
        xtm_dup_thresh: float = 0.98,
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
            graph_tokenizer_type=str(graph_tokenizer_type),
            perceiver_num_latents=int(perceiver_num_latents),
            perceiver_num_layers=int(perceiver_num_layers),
            perceiver_num_heads=int(perceiver_num_heads),
            perceiver_ff_mult=int(perceiver_ff_mult),
            perceiver_dropout=float(perceiver_dropout),
            node_encoder_type=node_encoder_type,
            node_encoder_out_dim=int(node_encoder_out_dim),
            node_encoder_trainable=bool(train_node_encoder),
            node_encoder_alpha_init=float(node_encoder_alpha_init),
            enable_gvl_adapter=bool(enable_gvl_adapter),
            gvl_adapter_num_heads=8,
            gvl_adapter_gate_init=float(gvl_adapter_gate_init),
            enable_graph_aux_head=bool(enable_graph_aux_head),
            graph_aux_dropout=0.1,
        )
        self.graph_aux_loss_weight = max(0.0, float(graph_aux_loss_weight))
        self.enable_xtc = bool(enable_xtc)
        self.enable_xtm = bool(enable_xtm)
        self.xtc_weight = max(0.0, float(xtc_weight))
        self.xtm_weight = max(0.0, float(xtm_weight))
        self.xtm_dup_thresh = float(xtm_dup_thresh)
        self.xtc_logit_scale = nn.Parameter(torch.tensor(float(xtc_logit_scale_init), dtype=torch.float32))
        self.xtc_logit_scale.requires_grad = bool(self.enable_xtc)
        self.alignment_config = {
            "enable_xtc": bool(self.enable_xtc),
            "enable_xtm": bool(self.enable_xtm),
            "xtc_weight": float(self.xtc_weight),
            "xtm_weight": float(self.xtm_weight),
            "xtc_logit_scale_init": float(xtc_logit_scale_init),
            "xtm_dup_thresh": float(self.xtm_dup_thresh),
        }
        self._xtm_stats_accum: Dict[str, float] = {
            "steps": 0.0,
            "hard_valid_ratio_sum": 0.0,
            "fallback_count_sum": 0.0,
            "same_image_blocked_pairs_sum": 0.0,
            "near_dup_blocked_pairs_sum": 0.0,
        }

        # strict pipeline: Stage3 always initializes from Stage2B ckpt
        self.stage2B_provenance = load_stage2b_weights_only_into_model(
            model=self.model,
            stage2b_ckpt_path=stage2B_ckpt,
            expected_graph_tokenizer_config=self.model.graph_tokenizer_config,
        )

        # Freeze everything except projectors
        self.train_node_encoder = bool(train_node_encoder)
        for name, p in self.model.named_parameters():
            p.requires_grad = (
                ("gl_projector" in name)
                or ("vl_projector" in name)
                or ("gvl_adapter" in name)
                or ("graph_aux_head" in name)
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
        if self.xtc_logit_scale.requires_grad:
            params.append(self.xtc_logit_scale)
        return AdamW(params, lr=float(self.hparams.lr))

    def _update_xtm_stats(self, stats: Dict[str, Any]) -> None:
        self._xtm_stats_accum["steps"] += 1.0
        self._xtm_stats_accum["hard_valid_ratio_sum"] += float(stats.get("hard_valid_ratio", 0.0))
        self._xtm_stats_accum["fallback_count_sum"] += float(stats.get("fallback_count", 0.0))
        self._xtm_stats_accum["same_image_blocked_pairs_sum"] += float(stats.get("same_image_blocked_pairs", 0.0))
        self._xtm_stats_accum["near_dup_blocked_pairs_sum"] += float(stats.get("near_dup_blocked_pairs", 0.0))

    def get_xtm_stats_summary(self) -> Dict[str, float]:
        steps = max(1.0, float(self._xtm_stats_accum.get("steps", 0.0)))
        if self._xtm_stats_accum.get("steps", 0.0) <= 0:
            return {
                "steps": 0.0,
                "avg_hard_valid_ratio": 0.0,
                "avg_fallback_count": 0.0,
                "avg_same_image_blocked_pairs": 0.0,
                "avg_near_dup_blocked_pairs": 0.0,
            }
        return {
            "steps": float(self._xtm_stats_accum["steps"]),
            "avg_hard_valid_ratio": float(self._xtm_stats_accum["hard_valid_ratio_sum"] / steps),
            "avg_fallback_count": float(self._xtm_stats_accum["fallback_count_sum"] / steps),
            "avg_same_image_blocked_pairs": float(self._xtm_stats_accum["same_image_blocked_pairs_sum"] / steps),
            "avg_near_dup_blocked_pairs": float(self._xtm_stats_accum["near_dup_blocked_pairs_sum"] / steps),
        }

    def _step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        graph_data = batch["graph_data"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        image_ids: List[int] = [int(x) for x in batch.get("image_ids", [])]
        prompts = batch["prompts"]
        answers = batch["answers"]
        qa_types = batch.get("qa_type", ["unknown"] * len(prompts))

        input_ids, attention_mask, labels = build_chat_inputs_and_labels(
            tokenizer=self.tokenizer,
            prompts=prompts,
            answers=answers,
            device=self.device,
            max_length=self.max_length,
        )
        aux_binary_labels, aux_binary_mask = build_binary_aux_targets(
            qa_types=qa_types,
            answers=answers,
            device=self.device,
        )

        outputs = self.model(
            graph_data=graph_data,
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            aux_binary_labels=aux_binary_labels,
            aux_binary_mask=aux_binary_mask,
            aux_loss_weight=float(self.graph_aux_loss_weight),
            return_alignment_features=bool(self.enable_xtc or self.enable_xtm),
            return_debug=False,
        )

        loss = outputs.loss
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            # fallback: fp32 CE
            logits = getattr(outputs, "logits", None)
            if logits is None:
                loss = torch.tensor(0.0, device=self.device)
            else:
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
        base_loss = loss

        aux_loss = getattr(outputs, "aux_loss", None)
        aux_w = float(getattr(outputs, "aux_loss_weight", 0.0))
        if aux_loss is None:
            aux_loss = torch.tensor(0.0, device=self.device)
            aux_w = 0.0
        lm_loss = base_loss - (aux_loss * aux_w)

        loss_xtc = torch.tensor(0.0, device=self.device)
        loss_xtm = torch.tensor(0.0, device=self.device)
        xtm_acc = torch.tensor(0.0, device=self.device)
        valid_neg_ratio = torch.tensor(0.0, device=self.device)

        do_align = bool(self.enable_xtc or self.enable_xtm)
        if do_align and len(prompts) >= 2:
            pooled = build_pooled_features(
                graph_embeds=getattr(outputs, "graph_embeds", None),
                text_embeds=getattr(outputs, "text_embeds", None),
                text_attention_mask=getattr(outputs, "text_attention_mask", None),
            )
            z_graph = pooled.get("graph", None)
            z_text = pooled.get("text", None)
            if z_graph is not None and z_text is not None and z_graph.size(0) >= 2 and z_text.size(0) >= 2:
                if self.enable_xtc and self.xtc_weight > 0.0:
                    logit_scale = self.xtc_logit_scale.exp().clamp(max=100.0)
                    loss_xtc, _ = xtc_bidirectional_loss(z_graph, z_text, logit_scale)
                if self.enable_xtm and self.xtm_weight > 0.0:
                    text_sim = z_text @ z_text.t()
                    neg_idx, hard_valid_mask, stats = mine_hard_negatives(
                        sim=text_sim,
                        image_ids=image_ids if image_ids else list(range(z_text.size(0))),
                        qa_types=qa_types,
                        dup_thresh=float(self.xtm_dup_thresh),
                    )
                    self._update_xtm_stats(stats)
                    valid_neg_ratio = hard_valid_mask.float().mean()
                    pos_logits = (z_graph * z_text).sum(dim=-1)
                    neg_logits = (z_graph * z_text[neg_idx]).sum(dim=-1)
                    loss_xtm, xtm_acc = xtm_pair_loss(pos_logits, neg_logits)

        total_loss = base_loss + (self.xtc_weight * loss_xtc) + (self.xtm_weight * loss_xtm)
        metrics = {
            "loss_lm": lm_loss.detach(),
            "loss_aux": aux_loss.detach(),
            "loss_xtc": loss_xtc.detach(),
            "loss_xtm": loss_xtm.detach(),
            "xtm_acc": xtm_acc.detach(),
            "xtm_valid_neg_ratio": valid_neg_ratio.detach(),
        }
        return total_loss, metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._step(batch)
        bsz = len(batch["prompts"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._step(batch)
        bsz = len(batch["prompts"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        return loss


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
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
    ap.add_argument("--graph_tokenizer_type", type=str, default="auto", choices=["auto", "qformer", "perceiver"])
    ap.add_argument("--perceiver_num_latents", type=int, default=-1, help="<=0 means inherit from Stage2B provenance.")
    ap.add_argument("--perceiver_num_layers", type=int, default=-1, help="<=0 means inherit from Stage2B provenance.")
    ap.add_argument("--perceiver_num_heads", type=int, default=-1, help="<=0 means inherit from Stage2B provenance.")
    ap.add_argument("--perceiver_ff_mult", type=int, default=-1, help="<=0 means inherit from Stage2B provenance.")
    ap.add_argument("--perceiver_dropout", type=float, default=-1.0, help="<0 means inherit from Stage2B provenance.")
    ap.add_argument("--node_encoder_type", type=str, default="hybrid", choices=["hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_out_dim", type=int, default=128, help="Graph node encoder output dim.")
    ap.add_argument("--train_node_encoder", type=int, default=0, choices=[0, 1], help="1=train node encoder, 0=freeze.")
    ap.add_argument("--node_encoder_alpha_init", type=float, default=0.3, help="Hybrid node encoder alpha init.")
    ap.add_argument("--use_qa_type_token", type=int, default=1, choices=[0, 1], help="Prefix prompt with QA type control token.")
    ap.add_argument("--enable_gvl_adapter", type=int, default=1, choices=[0, 1], help="Enable graph-vision-language cross-attn adapter.")
    ap.add_argument("--gvl_adapter_gate_init", type=float, default=0.1, help="Initial gate for cross-modal adapter.")
    ap.add_argument("--enable_graph_aux_head", type=int, default=1, choices=[0, 1], help="Enable auxiliary graph reasoning binary head.")
    ap.add_argument("--graph_aux_loss_weight", type=float, default=0.05, help="Auxiliary binary loss weight.")
    ap.add_argument("--enable_xtc", type=int, default=1, choices=[0, 1], help="Enable bidirectional graph-text contrastive loss.")
    ap.add_argument("--enable_xtm", type=int, default=1, choices=[0, 1], help="Enable graph-text matching loss with hard negatives.")
    ap.add_argument("--xtc_weight", type=float, default=0.05, help="Weight for XTC loss.")
    ap.add_argument("--xtm_weight", type=float, default=0.03, help="Weight for XTM loss.")
    ap.add_argument("--xtc_logit_scale_init", type=float, default=2.66, help="Initial logit scale parameter (log space).")
    ap.add_argument("--xtm_dup_thresh", type=float, default=0.98, help="Cosine similarity threshold for duplicate-text negative filtering.")

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
    _, stage2b_bootstrap, stage2b_graph_tokenizer_config = _extract_stage2b_provenance_from_ckpt(args.stage2B_ckpt)
    resolved_graph_tokenizer_config = _resolve_graph_tokenizer_from_stage2b(
        args=args,
        stage2b_graph_tokenizer_config=stage2b_graph_tokenizer_config,
    )
    print(
        "[Stage3] resolved tokenizer from Stage2B provenance: "
        f"bootstrap={stage2b_bootstrap} graph_tokenizer={resolved_graph_tokenizer_config}"
    )

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
        if qa_records:
            type_counts: Dict[str, int] = {}
            for rec in qa_records:
                qa_type = str(rec.get("qa_type", "unknown"))
                type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
            summary = ", ".join(f"{k}:{type_counts[k]}" for k in sorted(type_counts.keys()))
            print(f"[GraphQA] type_dist={summary}")
    else:
        print("[GraphQA] disabled.")

    full_dataset = VGTriModalRegionDataset(
        sg_dataset=sg_dataset,
        region_records=region_records,
        image_root=args.image_root,
        qa_records=qa_records,
        prompt=str(args.prompt),
        use_qa_type_token=bool(int(args.use_qa_type_token)),
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
        graph_tokenizer_type=str(resolved_graph_tokenizer_config.get("type")),
        perceiver_num_latents=int(resolved_graph_tokenizer_config.get("num_latents", 32)),
        perceiver_num_layers=int(resolved_graph_tokenizer_config.get("num_layers", 3)),
        perceiver_num_heads=int(resolved_graph_tokenizer_config.get("num_heads", 8)),
        perceiver_ff_mult=int(resolved_graph_tokenizer_config.get("ff_mult", 4)),
        perceiver_dropout=float(resolved_graph_tokenizer_config.get("dropout", 0.0)),
        node_encoder_type=str(args.node_encoder_type),
        node_encoder_out_dim=int(args.node_encoder_out_dim),
        train_node_encoder=bool(int(args.train_node_encoder)),
        node_encoder_alpha_init=float(args.node_encoder_alpha_init),
        enable_gvl_adapter=bool(int(args.enable_gvl_adapter)),
        gvl_adapter_gate_init=float(args.gvl_adapter_gate_init),
        enable_graph_aux_head=bool(int(args.enable_graph_aux_head)),
        graph_aux_loss_weight=float(args.graph_aux_loss_weight),
        enable_xtc=bool(int(args.enable_xtc)),
        enable_xtm=bool(int(args.enable_xtm)),
        xtc_weight=float(args.xtc_weight),
        xtm_weight=float(args.xtm_weight),
        xtc_logit_scale_init=float(args.xtc_logit_scale_init),
        xtm_dup_thresh=float(args.xtm_dup_thresh),
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
        "graph_tokenizer_config": pl_model.model.graph_tokenizer_config,
        "node_encoder_config": pl_model.model.node_encoder_config,
        "architecture_config": pl_model.model.architecture_config,
        "alignment_config": pl_model.alignment_config,
        "xtm_stats_summary": pl_model.get_xtm_stats_summary(),
        "best_ckpt": ckpt_cb.best_model_path or "",
        "export_path": str(export_path),
    }
    (save_dir / "stage3_meta.json").write_text(json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Export] stage3 state_dict -> {export_path}")
    print(f"[Best] {ckpt_cb.best_model_path}")


if __name__ == "__main__":
    main()
