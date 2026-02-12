# omnigraph/train/train_projector.py
from __future__ import annotations

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Repo bootstrap + env (must be before transformers import)
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

from transformers import AutoTokenizer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

# Dataset
from omnigraph.data.vg_scene_graph_dataset import (  # noqa: E402
    build_vg_vocabs_from_file,
    VGSceneGraphDataset,
)
from omnigraph.data.vg_graph_qa import build_vg_graph_qa_records  # noqa: E402
from omnigraph.train.common import (  # noqa: E402
    build_binary_aux_targets,
    build_chat_inputs_and_labels,
    format_prompt_with_qa_type as _format_prompt_with_qa_type,
    load_region_pairs,
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
# Regions + dataset wrapper
# ---------------------------------------------------------------------------


class VGGraphRegionTextDataset(Dataset):
    """
    One sample = (scene graph of image_id) + (one region phrase).
    Output: {id, image_id, graph_data, text, answer}
    """

    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        region_pairs: List[Tuple[int, str]],
        qa_records: Optional[List[Dict[str, Any]]] = None,
        prompt: str = "Describe the region.",
        use_qa_type_token: bool = True,
    ):
        self.sg = sg_dataset
        self.prompt = prompt
        self.use_qa_type_token = bool(use_qa_type_token)

        raw_items = None
        if hasattr(self.sg, "items"):
            raw_items = getattr(self.sg, "items")
        elif hasattr(self.sg, "scene_graphs"):
            raw_items = getattr(self.sg, "scene_graphs")

        if raw_items is None:
            raise AttributeError(
                "VGSceneGraphDataset must expose raw scene graph list as .items or .scene_graphs for image_id join."
            )

        self.image2idx: Dict[int, int] = {}
        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                self.image2idx[int(it["image_id"])] = i

        self.samples: List[Dict[str, Any]] = []
        for (iid, phr) in region_pairs:
            sg_idx = self.image2idx.get(int(iid), None)
            if sg_idx is None:
                continue
            self.samples.append(
                {
                    "image_id": int(iid),
                    "answer": str(phr),
                    "prompt": self.prompt,
                    "sg_idx": int(sg_idx),
                    "source": "region",
                    "qa_type": "region_caption",
                }
            )

        if qa_records:
            for rec in qa_records:
                iid = int(rec.get("image_id", -1))
                sg_idx = self.image2idx.get(iid, None)
                if sg_idx is None:
                    continue
                q = str(rec.get("question", "")).strip()
                a = str(rec.get("answer", "")).strip()
                if not q or not a:
                    continue
                self.samples.append(
                    {
                        "image_id": int(iid),
                        "answer": a,
                        "prompt": q,
                        "sg_idx": int(sg_idx),
                        "source": "graph_qa",
                        "qa_type": str(rec.get("qa_type", "graph_qa")),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        image_id = int(s["image_id"])
        phrase = str(s["answer"])
        sg_idx = int(s["sg_idx"])
        prompt = str(s["prompt"])
        source = str(s["source"])
        qa_type = str(s.get("qa_type", "unknown"))
        sg_item = self.sg[sg_idx]
        return {
            "id": f"{image_id}_{source}_{idx}",
            "image_id": int(image_id),
            "graph_data": sg_item["graph_data"],
            "text": _format_prompt_with_qa_type(prompt, qa_type, self.use_qa_type_token),
            "answer": phrase,
            "qa_type": qa_type,
        }


def split_by_image_id(
    dataset: VGGraphRegionTextDataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """
    Split by image_id to avoid leakage:
    all phrases under an image_id go either train or val.
    """
    image_ids = sorted({int(s["image_id"]) for s in dataset.samples})
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        # degenerate: still create a tiny val
        val_set = set(image_ids)
    else:
        val_n = max(1, int(len(image_ids) * float(val_ratio)))
        val_set = set(image_ids[:val_n])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, s in enumerate(dataset.samples):
        iid = int(s["image_id"])
        if iid in val_set:
            val_idx.append(i)
        else:
            train_idx.append(i)

    if len(train_idx) == 0:
        # ensure non-empty train
        train_idx = val_idx[:-1]
        val_idx = val_idx[-1:]

    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_graph_text(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    from torch_geometric.data import Batch as GeoBatch

    ids: List[str] = []
    image_ids: List[int] = []
    texts: List[str] = []
    answers: List[str] = []
    qa_types: List[str] = []
    graphs = []

    for item in batch:
        ids.append(str(item.get("id", "unknown")))
        image_ids.append(int(item.get("image_id", -1)))
        texts.append(item.get("text", ""))
        answers.append(item.get("answer", item.get("text", "")))
        qa_types.append(str(item.get("qa_type", "unknown")))
        graphs.append(item["graph_data"])

    batch_graph = GeoBatch.from_data_list(graphs)
    return {
        "ids": ids,
        "image_ids": image_ids,
        "graph_data": batch_graph,
        "text": texts,
        "answer": answers,
        "qa_type": qa_types,
    }


# ---------------------------------------------------------------------------
# Upstream checkpoint loading
# ---------------------------------------------------------------------------

def _is_state_dict_like(x: Any) -> bool:
    if not isinstance(x, dict) or len(x) == 0:
        return False
    tensor_like = 0
    for v in x.values():
        if hasattr(v, "shape"):
            tensor_like += 1
    return tensor_like > 0


def _collect_stage1_state_dict_candidates(obj: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Support common checkpoint layouts:
      - raw state_dict
      - lightning checkpoint with state_dict/model_state_dict
      - nested graph_qformer_state_dict
    """
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    if not isinstance(obj, dict):
        return candidates

    if _is_state_dict_like(obj):
        candidates.append(("root", obj))

    for key in ("state_dict", "model_state_dict", "graph_qformer_state_dict", "graph_qformer"):
        v = obj.get(key, None)
        if _is_state_dict_like(v):
            candidates.append((key, v))
    return candidates


def _normalize_qformer_keys(
    sd: Dict[str, Any],
    model_keys: set[str],
) -> Tuple[Dict[str, Any], int]:
    prefixes = (
        "graph_qformer.",
        "model.graph_qformer.",
        "module.graph_qformer.",
        "model.graph_branch.qformer.",
        "graph_branch.qformer.",
        "qformer.",
    )
    out: Dict[str, Any] = {}
    matched_raw = 0
    for k, v in sd.items():
        if not hasattr(v, "shape"):
            continue

        candidate_keys = [k]
        for p in prefixes:
            if k.startswith(p):
                candidate_keys.append(k[len(p):])

        picked = None
        for ck in candidate_keys:
            if ck in model_keys:
                picked = ck
                break
        if picked is None:
            continue
        if picked not in out:
            out[picked] = v
            matched_raw += 1
    return out, matched_raw


def load_stage1_qformer_weights(model: OmniGraphModel, stage1_qformer_ckpt: str) -> Dict[str, Any]:
    ckpt_path = Path(stage1_qformer_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--stage1_qformer_ckpt not found: {stage1_qformer_ckpt}")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid stage1 qformer checkpoint format: {stage1_qformer_ckpt}")

    model_state = model.graph_qformer.state_dict()
    model_keys = set(model_state.keys())

    candidates = _collect_stage1_state_dict_candidates(obj)
    best_name = ""
    best_sd: Dict[str, Any] = {}
    best_matched = 0
    for name, raw_sd in candidates:
        norm_sd, matched = _normalize_qformer_keys(raw_sd, model_keys)
        if matched > best_matched:
            best_name = name
            best_sd = norm_sd
            best_matched = matched

    overlap = sorted(model_keys.intersection(best_sd.keys()))
    if not overlap:
        cand_desc = ", ".join(f"{n}:{len(sd)}" for n, sd in candidates) if candidates else "none"
        raise RuntimeError(
            "Stage1 checkpoint has no overlapping GraphQFormer keys. "
            f"Expected checkpoint from train_graph_qfromer.py output. candidates={cand_desc}"
        )

    adapted = []
    skipped = []
    filtered_sd = {}
    for k, v in best_sd.items():
        if k not in model_state:
            continue
        tgt = model_state[k]
        if getattr(v, "shape", None) == getattr(tgt, "shape", None):
            filtered_sd[k] = v
            continue

        # Compatibility shim for Stage1 ckpt with different max position length.
        if k.endswith("bert.embeddings.position_embeddings.weight"):
            if v.dim() == 2 and tgt.dim() == 2 and v.size(1) == tgt.size(1):
                new_v = tgt.clone()
                n = min(v.size(0), tgt.size(0))
                new_v[:n] = v[:n]
                filtered_sd[k] = new_v
                adapted.append((k, tuple(v.shape), tuple(tgt.shape)))
                continue
        if k.endswith("bert.embeddings.position_ids"):
            if v.dim() == 2 and tgt.dim() == 2:
                new_v = tgt.clone()
                n = min(v.size(1), tgt.size(1))
                new_v[:, :n] = v[:, :n]
                filtered_sd[k] = new_v
                adapted.append((k, tuple(v.shape), tuple(tgt.shape)))
                continue

        skipped.append((k, tuple(v.shape), tuple(tgt.shape)))

    missing, unexpected = model.graph_qformer.load_state_dict(filtered_sd, strict=False)
    loaded = len(filtered_sd)
    total = len(model_keys)
    if loaded == 0 or total == 0:
        raise RuntimeError("Failed to load GraphQFormer weights from Stage1 checkpoint.")

    required_keys = [k for k in model_state.keys() if not k.endswith("position_ids")]
    loaded_required = sum(1 for k in filtered_sd.keys() if k in required_keys)
    key_coverage = float(loaded_required) / float(max(1, len(required_keys)))
    missing_required = [k for k in missing if not k.endswith("position_ids")]

    # Guardrail for silent bad checkpoints.
    if len(unexpected) > 0 or key_coverage < 0.98 or len(missing_required) > 8:
        raise RuntimeError(
            "Stage1 GraphQFormer load failed strict coverage gate: "
            f"coverage={key_coverage:.4f}, missing_required={len(missing_required)}, unexpected={len(unexpected)}. "
            "Likely wrong ckpt path or incompatible Stage1 architecture."
        )

    info = {
        "stage1_qformer_ckpt": str(ckpt_path),
        "source_state_dict": best_name,
        "loaded_keys": loaded,
        "total_model_keys": total,
        "key_coverage": key_coverage,
        "missing_keys": len(missing),
        "missing_required_keys": len(missing_required),
        "unexpected_keys": len(unexpected),
        "adapted_keys": len(adapted),
        "skipped_shape_mismatch_keys": len(skipped),
    }
    print(
        "[Stage2A] Loaded Stage1 GraphQFormer: "
        f"source={best_name} loaded={loaded}/{total} coverage={key_coverage:.4f} "
        f"missing={len(missing)} missing_required={len(missing_required)} unexpected={len(unexpected)} "
        f"adapted={len(adapted)} skipped={len(skipped)}"
    )
    if adapted:
        print(f"[Stage2A] Adapted keys (example): {adapted[0][0]} {adapted[0][1]} -> {adapted[0][2]}")
    if skipped:
        print(f"[Stage2A] Skipped mismatched keys (example): {skipped[0][0]} {skipped[0][1]} -> {skipped[0][2]}")
    if missing_required:
        print(f"[Stage2A] Missing required keys (sample): {missing_required[:8]}")
    return info


# ---------------------------------------------------------------------------
# LightningModule (Stage2-A)
# ---------------------------------------------------------------------------

class ProjectorPL(pl.LightningModule):
    """
    Stage2-A:
      Train:
        - vg_adapter
        - gl_projector
      Freeze:
        - graph_qformer (explicit)
        - LLM
        - graphgpt
        - vision (disabled)
    """

    def __init__(
        self,
        llm_model_name: str,
        graph_model_name: str,
        num_obj: int,
        num_attr: int,
        stage1_qformer_ckpt: str,
        lr: float,
        max_length: int,
        max_graph_tokens: int | None = None,
        llm_dtype: str = "bfloat16",
        llm_attn_implementation: str = "sdpa",
        node_encoder_type: str = "hybrid",
        node_encoder_out_dim: int = 128,
        train_node_encoder: bool = True,
        node_encoder_alpha_init: float = 0.3,
        enable_gvl_adapter: bool = True,
        gvl_adapter_gate_init: float = 0.1,
        enable_graph_aux_head: bool = True,
        graph_aux_loss_weight: float = 0.0,
        enable_xtc: bool = True,
        enable_xtm: bool = True,
        xtc_weight: float = 0.08,
        xtm_weight: float = 0.05,
        xtc_logit_scale_init: float = 2.66,
        xtm_dup_thresh: float = 0.98,
        auto_resize_token_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = OmniGraphModel(
            graph_model_name=graph_model_name,
            llm_model_name=llm_model_name,
            enable_vision=False,
            num_obj=int(num_obj),
            num_attr=int(num_attr),
            max_graph_tokens=max_graph_tokens,
            llm_dtype=llm_dtype,
            llm_attn_implementation=llm_attn_implementation,
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
        self.stage1_load_info = load_stage1_qformer_weights(self.model, stage1_qformer_ckpt)
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

        # Stage2-A trainable selection
        self.train_node_encoder = bool(train_node_encoder)
        for name, p in self.model.named_parameters():
            trainable = ("gl_projector" in name) or (
                self.train_node_encoder and (name.startswith("node_encoder.") or ("vg_adapter" in name))
            )
            if ("gvl_adapter" in name) or ("graph_aux_head" in name):
                trainable = True
            # graph_qformer must be frozen in stage2A
            if "graph_qformer" in name:
                trainable = False
            p.requires_grad = bool(trainable)
        print(
            f"[Stage2A] node_encoder_type={self.model.node_encoder_type} "
            f"train_node_encoder={self.train_node_encoder}"
        )

        # LLM memory-saving settings
        if hasattr(self.model.llm.model, "gradient_checkpointing_enable"):
            self.model.llm.model.gradient_checkpointing_enable()
        if hasattr(self.model.llm.model, "config"):
            self.model.llm.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = min(getattr(self.tokenizer, "model_max_length", 2048), max_length)

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

    def _forward_and_loss(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        graph_data = batch["graph_data"].to(self.device)
        image_ids: List[int] = [int(x) for x in batch.get("image_ids", [])]
        prompts: List[str] = batch["text"]
        answers: List[str] = batch["answer"]
        qa_types: List[str] = batch.get("qa_type", ["unknown"] * len(prompts))

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
            pixel_values=None,
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
            logits = getattr(outputs, "logits", None)
            if logits is not None:
                shift_logits = logits[:, :-1, :].float().contiguous()
                shift_labels = labels[:, 1:].contiguous()
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
                if (shift_labels != -100).any():
                    loss = F.cross_entropy(
                        shift_logits.reshape(-1, shift_logits.size(-1)),
                        shift_labels.reshape(-1),
                        ignore_index=-100,
                    )
                else:
                    loss = torch.tensor(0.0, device=self.device)
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
        loss, metrics = self._forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        self.log("train_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        self.log("val_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        return loss


# ---------------------------------------------------------------------------
# Entry (Stage2-A only)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scene_graphs", type=str, required=True)
    ap.add_argument("--regions", type=str, required=True, help="VG region descriptions json")

    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--graph_model", type=str, default="clip_gt_arxiv_pub")

    ap.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_graph_tokens", type=int, default=0, help="If >0, truncate graph prefix tokens before LLM.")
    ap.add_argument("--llm_dtype", type=str, default="bfloat16", help="LLM load dtype: bfloat16/float16/float32.")
    ap.add_argument("--llm_attn_implementation", type=str, default="sdpa", help="LLM attention backend (e.g., sdpa/flash_attention_2).")
    ap.add_argument("--node_encoder_type", type=str, default="hybrid", choices=["hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_out_dim", type=int, default=128, help="Graph node encoder output dim.")
    ap.add_argument("--train_node_encoder", type=int, default=1, choices=[0, 1], help="1=train node encoder, 0=freeze.")
    ap.add_argument("--node_encoder_alpha_init", type=float, default=0.3, help="Hybrid node encoder alpha init.")
    ap.add_argument("--use_qa_type_token", type=int, default=1, choices=[0, 1], help="Prefix prompt with QA type control token.")
    ap.add_argument("--enable_gvl_adapter", type=int, default=1, choices=[0, 1], help="Enable graph-vision-language cross-attn adapter.")
    ap.add_argument("--gvl_adapter_gate_init", type=float, default=0.1, help="Initial gate for cross-modal adapter.")
    ap.add_argument("--enable_graph_aux_head", type=int, default=1, choices=[0, 1], help="Enable auxiliary graph reasoning binary head.")
    ap.add_argument("--graph_aux_loss_weight", type=float, default=0.03, help="Auxiliary binary loss weight.")
    ap.add_argument("--enable_xtc", type=int, default=1, choices=[0, 1], help="Enable bidirectional graph-text contrastive loss.")
    ap.add_argument("--enable_xtm", type=int, default=1, choices=[0, 1], help="Enable graph-text matching loss with hard negatives.")
    ap.add_argument("--xtc_weight", type=float, default=0.08, help="Weight for XTC loss.")
    ap.add_argument("--xtm_weight", type=float, default=0.05, help="Weight for XTM loss.")
    ap.add_argument("--xtc_logit_scale_init", type=float, default=2.66, help="Initial logit scale parameter (log space).")
    ap.add_argument("--xtm_dup_thresh", type=float, default=0.98, help="Cosine similarity threshold for duplicate-text negative filtering.")

    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)

    ap.add_argument("--val_ratio", type=float, default=0.001, help="split by image_id")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=0.01)

    ap.add_argument("--val_check_interval", type=float, default=2000, help=">=1: every N steps; (0,1]: fraction epoch")
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--num_sanity_val_steps", type=int, default=0)
    ap.add_argument("--accumulate_grad_batches", type=int, default=4)

    # Stage2-A hyperparams
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=80000)

    ap.add_argument("--save_dir", type=str, default="checkpoints_projector_vg/stage2A")
    ap.add_argument("--disable_graph_qa", action="store_true", help="Disable synthetic graph QA from VG scene graphs.")
    ap.add_argument("--graph_qa_max_per_image", type=int, default=3, help="Max synthetic QA pairs per image.")
    ap.add_argument("--graph_qa_repeat", type=int, default=2, help="Repeat factor for synthetic graph QA samples.")
    ap.add_argument("--graph_qa_seed", type=int, default=42, help="Random seed for synthetic graph QA generation.")

    ap.add_argument(
        "--stage1_qformer_ckpt",
        type=str,
        required=True,
        help="Path to Stage1 graph_qformer checkpoint (graph_qformer_stage1.pt)",
    )

    args = ap.parse_args()
    pl.seed_everything(int(args.seed), workers=True)
    print("[Pipeline] Stage2A start: requires Stage1 GraphQFormer checkpoint.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stage1_ckpt = Path(args.stage1_qformer_ckpt)
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"--stage1_qformer_ckpt not found: {stage1_ckpt}")

    # 1) build vocabs
    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_file(args.scene_graphs, min_freq=args.min_freq)
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)
    print(f"[Vocab] NUM_OBJ={num_obj} NUM_ATTR={num_attr}")

    # 2) scene graph dataset
    sg_dataset = VGSceneGraphDataset(
        scene_graphs_path=args.scene_graphs,
        obj_vocab=obj_vocab,
        pred_vocab=pred_vocab,
        attr_vocab=attr_vocab,
        max_nodes=args.max_nodes,
        max_attrs=args.max_attrs,
        add_reverse_edges=True,
        use_bbox_max_norm=True,
    )

    # 3) region pairs -> joined dataset
    region_pairs = load_region_pairs(args.regions)
    print(f"[Regions] loaded pairs={len(region_pairs)} from {args.regions}")

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

    full_dataset = VGGraphRegionTextDataset(
        sg_dataset=sg_dataset,
        region_pairs=region_pairs,
        qa_records=qa_records,
        prompt="Describe the region.",
        use_qa_type_token=bool(int(args.use_qa_type_token)),
    )
    print(f"[Join] usable_pairs={len(full_dataset)}")
    if len(full_dataset) == 0:
        raise RuntimeError("No usable (scene graph, region phrase) pairs after join. Check image_id alignment.")

    # 4) split train/val by image_id
    train_ds, val_ds = split_by_image_id(full_dataset, val_ratio=float(args.val_ratio), seed=int(args.seed))
    print(f"[Split] train={len(train_ds)} val={len(val_ds)} (val_ratio={args.val_ratio})")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_graph_text,
        persistent_workers=(int(args.num_workers) > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_graph_text,
        persistent_workers=(int(args.num_workers) > 0),
    )

    vci = _parse_val_check_interval(args.val_check_interval)

    train_node_encoder = bool(int(args.train_node_encoder))

    # 5) model + callbacks
    pl_model = ProjectorPL(
        llm_model_name=args.llm,
        graph_model_name=args.graph_model,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        stage1_qformer_ckpt=str(stage1_ckpt),
        lr=float(args.lr),
        max_length=int(args.max_length),
        max_graph_tokens=(int(args.max_graph_tokens) if int(args.max_graph_tokens) > 0 else None),
        llm_dtype=str(args.llm_dtype),
        llm_attn_implementation=str(args.llm_attn_implementation),
        node_encoder_type=str(args.node_encoder_type),
        node_encoder_out_dim=int(args.node_encoder_out_dim),
        train_node_encoder=bool(train_node_encoder),
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

    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="stage2A-step={step:07d}-val_loss={val_loss:.3f}",
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
        max_epochs=999999,
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

    trainer.fit(pl_model, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path or ""
    export_path = save_dir / "model_state_dict.pt"
    torch.save(pl_model.model.state_dict(), str(export_path))
    stage_meta = {
        "stage": "stage2A",
        "stage1_qformer_ckpt": str(stage1_ckpt),
        "num_obj": int(num_obj),
        "num_attr": int(num_attr),
        "stage1_load_info": pl_model.stage1_load_info,
        "node_encoder_config": pl_model.model.node_encoder_config,
        "architecture_config": pl_model.model.architecture_config,
        "alignment_config": pl_model.alignment_config,
        "xtm_stats_summary": pl_model.get_xtm_stats_summary(),
        "best_ckpt": best_path,
        "export_path": str(export_path),
    }
    (save_dir / "stage2A_meta.json").write_text(json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage2A] exported state_dict -> {export_path}")
    if best_path:
        print(f"[stage2A] best ckpt -> {best_path}")
    else:
        print("[stage2A] warning: best ckpt not found (check monitor='val_loss' and val loader).")


if __name__ == "__main__":
    main()
