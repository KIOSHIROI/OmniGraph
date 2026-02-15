# omnigraph/train/train_projector.py
from __future__ import annotations

import errno

import sys
import json
import argparse
import shutil
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
    TQDMProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Dataset
from omnigraph.data.vg_scene_graph_dataset import (  # noqa: E402
    build_vg_vocabs_from_items,
    VGSceneGraphDataset,
)
from omnigraph.train.common import (  # noqa: E402
    build_binary_aux_targets,
    build_chat_inputs_and_labels,
    format_prompt_with_qa_type as _format_prompt_with_qa_type,
    load_region_pairs,
    parse_precision as _parse_precision,
    parse_val_check_interval as _parse_val_check_interval,
)
from omnigraph.train.pipeline_data import (  # noqa: E402
    build_graph_qa_records_with_pseudo,
    load_merged_scene_graph_items,
    split_indices_by_image_id,
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
    train_idx, val_idx, _, _ = split_indices_by_image_id(
        dataset.samples,
        val_ratio=float(val_ratio),
        seed=int(seed),
        image_id_key="image_id",
        fallback_when_train_empty=True,
        require_non_empty_train=False,
        require_non_empty_val=False,
        error_prefix="Stage2A split",
    )
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_graph_text(
    batch: List[Dict[str, Any]],
    tokenizer: Optional[Any] = None,
    max_length: Optional[int] = None,
) -> Dict[str, Any]:
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
    out = {
        "ids": ids,
        "image_ids": image_ids,
        "graph_data": batch_graph,
        "text": texts,
        "answer": answers,
        "qa_type": qa_types,
    }
    if tokenizer is not None and max_length is not None and int(max_length) > 0:
        input_ids, attention_mask, labels = build_chat_inputs_and_labels(
            tokenizer=tokenizer,
            prompts=texts,
            answers=answers,
            device=None,
            max_length=int(max_length),
        )
        out["input_ids"] = input_ids
        out["attention_mask"] = attention_mask
        out["labels"] = labels
    return out


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


def _resolve_stage2a_bootstrap_mode(
    requested_mode: str,
    graph_tokenizer_type: str,
    stage1_qformer_ckpt: Optional[str],
) -> str:
    mode = str(requested_mode or "auto").strip().lower()
    tok = str(graph_tokenizer_type or "qformer").strip().lower()
    if tok not in {"qformer", "perceiver"}:
        raise ValueError(f"Unsupported graph_tokenizer_type: {graph_tokenizer_type}")

    if mode == "auto":
        mode = "legacy_stage1" if tok == "qformer" else "no_stage1"

    if mode not in {"legacy_stage1", "no_stage1"}:
        raise ValueError(f"Unsupported stage2A bootstrap mode: {requested_mode}")

    if mode == "legacy_stage1":
        if tok != "qformer":
            raise ValueError("legacy_stage1 mode requires graph_tokenizer_type=qformer.")
        if not stage1_qformer_ckpt:
            raise ValueError("legacy_stage1 mode requires --stage1_qformer_ckpt.")
    return mode


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
        - graph_qformer in legacy_stage1 bootstrap mode
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
        lr: float,
        max_length: int,
        stage1_qformer_ckpt: Optional[str] = None,
        stage2A_bootstrap_mode: str = "auto",
        max_graph_tokens: int | None = None,
        llm_dtype: str = "bfloat16",
        llm_attn_implementation: str = "sdpa",
        enable_llm_gradient_checkpointing: bool = True,
        graph_tokenizer_type: str = "perceiver",
        perceiver_num_latents: int = 32,
        perceiver_num_layers: int = 3,
        perceiver_num_heads: int = 8,
        perceiver_ff_mult: int = 4,
        perceiver_dropout: float = 0.0,
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
        self.stage2A_bootstrap_mode = _resolve_stage2a_bootstrap_mode(
            requested_mode=str(stage2A_bootstrap_mode),
            graph_tokenizer_type=str(graph_tokenizer_type),
            stage1_qformer_ckpt=stage1_qformer_ckpt,
        )
        self.stage1_load_info: Optional[Dict[str, Any]] = None
        if self.stage2A_bootstrap_mode == "legacy_stage1":
            self.stage1_load_info = load_stage1_qformer_weights(self.model, str(stage1_qformer_ckpt))
        self.stage2A_bootstrap = {
            "mode": str(self.stage2A_bootstrap_mode),
            "graph_tokenizer_type": str(self.model.graph_tokenizer_config.get("type", str(graph_tokenizer_type))),
            "stage1_qformer_ckpt": str(stage1_qformer_ckpt) if stage1_qformer_ckpt else None,
        }
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
        self.enable_llm_gradient_checkpointing = bool(enable_llm_gradient_checkpointing)

        # Stage2-A trainable selection
        self.train_node_encoder = bool(train_node_encoder)
        self.train_graph_tokenizer = bool(self.stage2A_bootstrap_mode == "no_stage1")
        for name, p in self.model.named_parameters():
            trainable = ("gl_projector" in name) or (
                self.train_node_encoder and (name.startswith("node_encoder.") or ("vg_adapter" in name))
            )
            if ("gvl_adapter" in name) or ("graph_aux_head" in name):
                trainable = True
            # Legacy Stage1 bootstrapping keeps graph tokenizer frozen.
            if "graph_qformer" in name:
                trainable = bool(self.train_graph_tokenizer)
            p.requires_grad = bool(trainable)
        print(
            f"[Stage2A] node_encoder_type={self.model.node_encoder_type} "
            f"train_node_encoder={self.train_node_encoder} "
            f"graph_tokenizer={self.model.graph_tokenizer_config.get('type')} "
            f"bootstrap_mode={self.stage2A_bootstrap_mode} "
            f"train_graph_tokenizer={self.train_graph_tokenizer}"
        )

        # LLM memory-saving / speed trade-off
        if self.enable_llm_gradient_checkpointing:
            if hasattr(self.model.llm.model, "gradient_checkpointing_enable"):
                self.model.llm.model.gradient_checkpointing_enable()
        else:
            if hasattr(self.model.llm.model, "gradient_checkpointing_disable"):
                self.model.llm.model.gradient_checkpointing_disable()
        if hasattr(self.model.llm.model, "config"):
            self.model.llm.model.config.use_cache = False
        print(f"[Stage2A] llm_gradient_checkpointing={self.enable_llm_gradient_checkpointing}")

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

        if "input_ids" in batch and "attention_mask" in batch and "labels" in batch:
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
        else:
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


class PeriodicSaveLastCheckpoint(pl.Callback):
    """Force-save trainer state to last.ckpt every N steps for OOM recovery."""

    def __init__(self, every_n_steps: int, ckpt_path: str, save_weights_only: bool = True):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.ckpt_path = str(ckpt_path)
        self.save_weights_only = bool(save_weights_only)
        self._disk_full_warned = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore[override]
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if step % self.every_n_steps != 0:
            return
        try:
            trainer.save_checkpoint(self.ckpt_path, weights_only=self.save_weights_only)
            self._disk_full_warned = False
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.ENOSPC:
                if not self._disk_full_warned:
                    print(
                        f"[GraphBootstrap][Warn] skip periodic checkpoint at step={step}: "
                        f"no space left on device for {self.ckpt_path}"
                    )
                    self._disk_full_warned = True
                return
            raise


class ManualEarlyStopByFile(pl.Callback):
    """Allow manual early-stop by creating a signal file during training."""

    def __init__(self, stop_file: str, stage_tag: str, save_ckpt_path: Optional[str] = None):
        super().__init__()
        self.stop_file = str(stop_file)
        self.stage_tag = str(stage_tag)
        self.save_ckpt_path = str(save_ckpt_path) if save_ckpt_path else ""
        self._triggered = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore[override]
        if self._triggered:
            return
        if not Path(self.stop_file).exists():
            return
        step = int(getattr(trainer, "global_step", 0))
        print(f"[{self.stage_tag}] manual early-stop signal detected: {self.stop_file} (step={step})")
        if self.save_ckpt_path:
            try:
                trainer.save_checkpoint(self.save_ckpt_path)
                print(f"[{self.stage_tag}] saved manual-stop checkpoint: {self.save_ckpt_path}")
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.ENOSPC:
                    print(f"[{self.stage_tag}][Warn] skip manual-stop checkpoint (disk full): {self.save_ckpt_path}")
                else:
                    print(f"[{self.stage_tag}][Warn] failed to save manual-stop checkpoint: {exc}")
            except Exception as exc:
                print(f"[{self.stage_tag}][Warn] failed to save manual-stop checkpoint: {exc}")
        trainer.should_stop = True
        self._triggered = True


# ---------------------------------------------------------------------------
# Entry (Stage2-A only)
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scene_graphs", type=str, required=True)
    ap.add_argument("--regions", type=str, required=True, help="VG region descriptions json")

    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--graph_model", type=str, default="clip_gt_arxiv_pub")
    ap.add_argument(
        "--stage2A_bootstrap_mode",
        type=str,
        default="auto",
        choices=["auto", "no_stage1", "legacy_stage1"],
        help="auto: perceiver=>no_stage1, qformer=>legacy_stage1",
    )
    ap.add_argument(
        "--bootstrap_mode",
        type=str,
        default="",
        help="Alias of --stage2A_bootstrap_mode.",
    )

    ap.add_argument("--gpu", type=int, default=0, help="GPU index; use -1 for CPU")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_graph_tokens", type=int, default=0, help="If >0, truncate graph prefix tokens before LLM.")
    ap.add_argument("--llm_dtype", type=str, default="bfloat16", help="LLM load dtype: bfloat16/float16/float32.")
    ap.add_argument("--llm_attn_implementation", type=str, default="sdpa", help="LLM attention backend (e.g., sdpa/flash_attention_2).")
    ap.add_argument("--enable_llm_gradient_checkpointing", type=int, default=1, choices=[0, 1], help="Enable LLM gradient checkpointing.")
    ap.add_argument("--graph_tokenizer_type", type=str, default="perceiver", choices=["qformer", "perceiver"])
    ap.add_argument("--perceiver_num_latents", type=int, default=32)
    ap.add_argument("--perceiver_num_layers", type=int, default=3)
    ap.add_argument("--perceiver_num_heads", type=int, default=8)
    ap.add_argument("--perceiver_ff_mult", type=int, default=4)
    ap.add_argument("--perceiver_dropout", type=float, default=0.0)
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
    ap.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch_factor when num_workers > 0.")

    # Stage2-A hyperparams
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=80000)

    ap.add_argument("--save_dir", type=str, default="checkpoints_projector_vg/graph_bootstrap")
    ap.add_argument("--resume_from_checkpoint", type=str, default="", help="Resume full trainer state from this checkpoint.")
    ap.add_argument("--checkpoint_every_n_steps", type=int, default=1000, help="Force-save last.ckpt every N train steps (0 to disable).")
    ap.add_argument("--manual_stop_file", type=str, default="", help="If this file appears during training, trigger graceful early stop.")
    ap.add_argument("--disable_graph_qa", action="store_true", help="Disable synthetic graph QA from VG scene graphs.")
    ap.add_argument("--graph_qa_max_per_image", type=int, default=3, help="Max synthetic QA pairs per image.")
    ap.add_argument("--graph_qa_repeat", type=int, default=2, help="Repeat factor for synthetic graph QA samples.")
    ap.add_argument("--extra_scene_graphs", type=str, default="", help="Comma-separated extra VG-style scene graph JSON files (pseudo graphs).")
    ap.add_argument("--pseudo_graph_qa_max_per_image", type=int, default=2, help="Max synthetic QA pairs per pseudo-graph image.")
    ap.add_argument("--pseudo_graph_qa_repeat", type=int, default=1, help="Repeat factor for pseudo-graph synthetic QA.")
    ap.add_argument("--graph_qa_seed", type=int, default=42, help="Random seed for synthetic graph QA generation.")

    ap.add_argument(
        "--stage1_qformer_ckpt",
        type=str,
        default="",
        help="Path to Stage1 graph_qformer checkpoint (graph_qformer_stage1.pt)",
    )
    ap.add_argument(
        "--graph_tokenizer_ckpt",
        type=str,
        default="",
        help="Alias of --stage1_qformer_ckpt for legacy qformer bootstrap.",
    )

    args = ap.parse_args()
    resume_ckpt = str(args.resume_from_checkpoint).strip()
    if resume_ckpt and not Path(resume_ckpt).exists():
        raise FileNotFoundError(f"--resume_from_checkpoint not found: {resume_ckpt}")
    if str(args.bootstrap_mode).strip():
        args.stage2A_bootstrap_mode = str(args.bootstrap_mode).strip()
    if str(args.graph_tokenizer_ckpt).strip():
        args.stage1_qformer_ckpt = str(args.graph_tokenizer_ckpt).strip()
    pl.seed_everything(int(args.seed), workers=True)
    stage1_ckpt_str = str(args.stage1_qformer_ckpt).strip()
    resolved_bootstrap = _resolve_stage2a_bootstrap_mode(
        requested_mode=str(args.stage2A_bootstrap_mode),
        graph_tokenizer_type=str(args.graph_tokenizer_type),
        stage1_qformer_ckpt=stage1_ckpt_str if stage1_ckpt_str else None,
    )
    if resolved_bootstrap == "legacy_stage1":
        print("[Pipeline] GraphBootstrap start: legacy_stage1 bootstrap enabled.")
    else:
        print("[Pipeline] GraphBootstrap start: no_stage1 bootstrap enabled (default new pipeline).")
    print(
        f"[GraphBootstrap] bootstrap_mode={resolved_bootstrap} "
        f"graph_tokenizer_type={args.graph_tokenizer_type}"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stage1_ckpt: Optional[Path] = None
    if stage1_ckpt_str:
        stage1_ckpt = Path(stage1_ckpt_str)
    if resolved_bootstrap == "legacy_stage1":
        if stage1_ckpt is None or not stage1_ckpt.exists():
            raise FileNotFoundError(f"--stage1_qformer_ckpt not found: {stage1_ckpt_str}")

    # 1) load+merge scene graphs and build vocabs
    base_items, merged_items, pseudo_items, extra_paths, merge_stats = load_merged_scene_graph_items(
        scene_graphs_path=str(args.scene_graphs),
        extra_scene_graphs=str(args.extra_scene_graphs),
    )
    print(
        "[SceneGraphs] "
        f"base={merge_stats.get('base_items', 0)} "
        f"extra_files={len(extra_paths)} extra_items={merge_stats.get('extra_items', 0)} "
        f"kept={merge_stats.get('kept', 0)} dropped_dup={merge_stats.get('dropped_duplicate_image_id', 0)}"
    )
    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_items(merged_items, min_freq=args.min_freq)
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)
    print(f"[Vocab] NUM_OBJ={num_obj} NUM_ATTR={num_attr}")

    # 2) scene graph dataset
    sg_dataset = VGSceneGraphDataset(
        scene_graphs_path=None,
        scene_graph_items=merged_items,
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

    qa_records, qa_summary = build_graph_qa_records_with_pseudo(
        disable_graph_qa=bool(args.disable_graph_qa),
        base_scene_graph_items=base_items,
        pseudo_scene_graph_items=pseudo_items,
        graph_qa_max_per_image=int(args.graph_qa_max_per_image),
        graph_qa_repeat=int(args.graph_qa_repeat),
        pseudo_graph_qa_max_per_image=int(args.pseudo_graph_qa_max_per_image),
        pseudo_graph_qa_repeat=int(args.pseudo_graph_qa_repeat),
        graph_qa_seed=int(args.graph_qa_seed),
    )
    if bool(qa_summary.get("enabled", False)):
        print(
            "[GraphQA] "
            f"real_pairs={int(qa_summary.get('real_pairs', 0))}(repeat={int(qa_summary.get('real_repeat', 1))}) "
            f"pseudo_pairs={int(qa_summary.get('pseudo_pairs', 0))}(repeat={int(qa_summary.get('pseudo_repeat', 1))}) "
            f"total={int(qa_summary.get('total_pairs', len(qa_records)))}"
        )
        type_counts = dict(qa_summary.get("type_counts", {}) or {})
        if type_counts:
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

    tokenizer_for_collate = AutoTokenizer.from_pretrained(args.llm, use_fast=True)
    if tokenizer_for_collate.pad_token is None:
        tokenizer_for_collate.pad_token = tokenizer_for_collate.eos_token
    max_length_for_collate = min(
        int(getattr(tokenizer_for_collate, "model_max_length", 2048)),
        int(args.max_length),
    )

    def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_graph_text(
            batch=batch,
            tokenizer=tokenizer_for_collate,
            max_length=max_length_for_collate,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=(int(args.prefetch_factor) if int(args.num_workers) > 0 else None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=_collate,
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=(int(args.prefetch_factor) if int(args.num_workers) > 0 else None),
    )

    vci = _parse_val_check_interval(args.val_check_interval)

    train_node_encoder = bool(int(args.train_node_encoder))

    # 5) model + callbacks
    pl_model = ProjectorPL(
        llm_model_name=args.llm,
        graph_model_name=args.graph_model,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        lr=float(args.lr),
        max_length=int(args.max_length),
        stage1_qformer_ckpt=(str(stage1_ckpt) if stage1_ckpt else None),
        stage2A_bootstrap_mode=str(args.stage2A_bootstrap_mode),
        max_graph_tokens=(int(args.max_graph_tokens) if int(args.max_graph_tokens) > 0 else None),
        llm_dtype=str(args.llm_dtype),
        llm_attn_implementation=str(args.llm_attn_implementation),
        enable_llm_gradient_checkpointing=bool(int(args.enable_llm_gradient_checkpointing)),
        graph_tokenizer_type=str(args.graph_tokenizer_type),
        perceiver_num_latents=int(args.perceiver_num_latents),
        perceiver_num_layers=int(args.perceiver_num_layers),
        perceiver_num_heads=int(args.perceiver_num_heads),
        perceiver_ff_mult=int(args.perceiver_ff_mult),
        perceiver_dropout=float(args.perceiver_dropout),
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
        filename="graph_bootstrap-step={step:07d}-val_loss={val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=False,
        save_weights_only=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        verbose=True,
    )
    try:
        tb_logger = TensorBoardLogger(
            save_dir=str(save_dir),
            name="tb_logs",
            version="",
        )
    except ModuleNotFoundError as exc:
        print(f"[GraphBootstrap][Warn] TensorBoard unavailable, fallback to CSV logger: {exc}")
        tb_logger = CSVLogger(
            save_dir=str(save_dir),
            name="csv_logs",
            version="",
        )
    callbacks = [ckpt_cb, lr_monitor, es, TQDMProgressBar(refresh_rate=10)]
    manual_stop_file = str(args.manual_stop_file).strip() or str(save_dir / "STOP_EARLY")
    callbacks.append(
        ManualEarlyStopByFile(
            stop_file=manual_stop_file,
            stage_tag="GraphBootstrap",
            save_ckpt_path=str(save_dir / "manual_stop.ckpt"),
        )
    )
    print(f"[GraphBootstrap] manual early-stop enabled: touch file -> {manual_stop_file}")
    ckpt_every_n = max(0, int(args.checkpoint_every_n_steps))
    if ckpt_every_n > 0:
        periodic_ckpt_path = save_dir / "last.ckpt"
        callbacks.append(
            PeriodicSaveLastCheckpoint(
                every_n_steps=ckpt_every_n,
                ckpt_path=str(periodic_ckpt_path),
                save_weights_only=True,
            )
        )
        print(f"[GraphBootstrap] periodic last.ckpt save enabled: every_n_steps={ckpt_every_n} path={periodic_ckpt_path}")

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
        logger=tb_logger,
        callbacks=callbacks,
        enable_progress_bar=True,
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

    if resume_ckpt:
        print(f"[GraphBootstrap] resume trainer state from ckpt: {resume_ckpt}")
        trainer.fit(pl_model, train_loader, val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(pl_model, train_loader, val_loader)

    best_path = ckpt_cb.best_model_path or ""
    export_path = save_dir / "model_state_dict.pt"
    export_saved = False
    try:
        torch.save(pl_model.model.state_dict(), str(export_path))
        export_saved = True
    except Exception as exc:
        msg = str(exc).lower()
        is_enospc = isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.ENOSPC
        free_bytes = shutil.disk_usage(str(save_dir)).free
        is_stream_write_fail = isinstance(exc, RuntimeError) and (
            "pytorchstreamwriter failed writing" in msg
            or "file write failed" in msg
            or "unexpected pos" in msg
            or "no space left on device" in msg
            or ("cannot be opened" in msg and free_bytes < (512 * 1024 * 1024))
        )
        if is_enospc or is_stream_write_fail:
            print(f"[GraphBootstrap][Warn] skip model_state_dict export (disk full): {export_path}")
        else:
            raise
    stage_meta = {
        "stage": "graph_bootstrap",
        "legacy_stage": "stage2A",
        "stage1_qformer_ckpt": str(stage1_ckpt) if stage1_ckpt else None,
        "num_obj": int(num_obj),
        "num_attr": int(num_attr),
        "scene_graph_sources": {
            "base_scene_graphs": str(args.scene_graphs),
            "extra_scene_graphs": list(extra_paths),
            "merge_stats": merge_stats,
        },
        "pseudo_graph_qa_config": {
            "max_per_image": int(args.pseudo_graph_qa_max_per_image),
            "repeat": int(args.pseudo_graph_qa_repeat),
        },
        "stage2A_bootstrap": pl_model.stage2A_bootstrap,
        "graph_bootstrap_config": pl_model.stage2A_bootstrap,
        "stage1_load_info": pl_model.stage1_load_info,
        "graph_tokenizer_config": pl_model.model.graph_tokenizer_config,
        "node_encoder_config": pl_model.model.node_encoder_config,
        "architecture_config": pl_model.model.architecture_config,
        "alignment_config": pl_model.alignment_config,
        "xtm_stats_summary": pl_model.get_xtm_stats_summary(),
        "best_ckpt": best_path,
        "export_path": (str(export_path) if export_saved else None),
    }
    stage_meta_text = json.dumps(stage_meta, ensure_ascii=False, indent=2)
    (save_dir / "graph_bootstrap_meta.json").write_text(stage_meta_text, encoding="utf-8")
    (save_dir / "stage2A_meta.json").write_text(stage_meta_text, encoding="utf-8")
    if export_saved:
        print(f"[GraphBootstrap] exported state_dict -> {export_path}")
    if best_path:
        print(f"[GraphBootstrap] best ckpt -> {best_path}")
    else:
        print("[GraphBootstrap] warning: best ckpt not found (check monitor='val_loss' and val loader).")


if __name__ == "__main__":
    main()
