# omnigraph/train/train_stage2B.py
from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Repo bootstrap + env
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
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Dataset (你已有)
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
# Wrapper dataset
# ---------------------------------------------------------------------------


class VGGraphRegionTextDataset(torch.utils.data.Dataset):
    """
    sample = (scene graph of image_id) + (one region phrase of same image_id)
    output: {id, graph_data, text, answer}
    """

    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        region_pairs: List[Tuple[int, str]],
        qa_records: List[Dict[str, Any]] | None = None,
        prompt: str = "Describe the region.",
        use_qa_type_token: bool = True,
    ):
        self.sg = sg_dataset
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
                "VGSceneGraphDataset must expose raw list as .items or .scene_graphs to join by image_id."
            )

        self.image2idx: Dict[int, int] = {}
        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                self.image2idx[int(it["image_id"])] = i

        self.samples: List[Dict[str, Any]] = []
        for iid, phr in region_pairs:
            if iid in self.image2idx:
                self.samples.append(
                    {
                        "image_id": int(iid),
                        "prompt": self.prompt,
                        "answer": str(phr),
                        "source": "region",
                        "qa_type": "region_caption",
                    }
                )
        if qa_records:
            for rec in qa_records:
                iid = int(rec.get("image_id", -1))
                if iid not in self.image2idx:
                    continue
                q = str(rec.get("question", "")).strip()
                a = str(rec.get("answer", "")).strip()
                if not q or not a:
                    continue
                self.samples.append(
                    {
                        "image_id": int(iid),
                        "prompt": q,
                        "answer": a,
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
        prompt = str(s["prompt"])
        source = str(s["source"])
        qa_type = str(s.get("qa_type", "unknown"))
        sg_idx = self.image2idx[image_id]
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
) -> Tuple[Subset, Subset, Set[int], Set[int]]:
    image_ids = sorted({int(s["image_id"]) for s in dataset.samples})
    rng = random.Random(int(seed))
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        val_ids = set(image_ids)
    else:
        n_val = max(1, int(len(image_ids) * float(val_ratio)))
        val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids) - val_ids

    train_indices: List[int] = []
    val_indices: List[int] = []
    for idx, s in enumerate(dataset.samples):
        iid = int(s["image_id"])
        if iid in val_ids:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    if not train_indices and len(val_indices) > 1:
        train_indices = val_indices[:-1]
        val_indices = val_indices[-1:]
        train_ids = {int(dataset.samples[i]["image_id"]) for i in train_indices}
        val_ids = {int(dataset.samples[i]["image_id"]) for i in val_indices}

    if not train_indices:
        raise RuntimeError("Stage2B split failed: empty train set after image_id split.")
    if not val_indices:
        raise RuntimeError("Stage2B split failed: empty val set after image_id split.")

    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_ids, val_ids


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
# LightningModule: Stage2-B only
# ---------------------------------------------------------------------------
class Stage2BProjectorPL(pl.LightningModule):
    """
    Stage2-B:
      train: vg_adapter + gl_projector + graph_qformer
      freeze: LLM + graphgpt (+ vision)
    """

    def __init__(
        self,
        llm_model_name: str,
        graph_model_name: str,
        num_obj: int,
        num_attr: int,
        stage2A_ckpt: str,
        lr: float,
        max_length: int,
        stage2A_bootstrap: Optional[Dict[str, Any]] = None,
        max_graph_tokens: int | None = None,
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
        train_node_encoder: bool = True,
        node_encoder_alpha_init: float = 0.3,
        enable_gvl_adapter: bool = True,
        gvl_adapter_gate_init: float = 0.1,
        enable_graph_aux_head: bool = True,
        graph_aux_loss_weight: float = 0.0,
        enable_xtc: bool = True,
        enable_xtm: bool = True,
        xtc_weight: float = 0.15,
        xtm_weight: float = 0.10,
        xtc_logit_scale_init: float = 2.66,
        xtm_dup_thresh: float = 0.98,
        auto_resize_token_embeddings: bool = True,
    ):
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
        self.stage2A_bootstrap = stage2A_bootstrap or {}

        # trainable params (Stage2-B)
        self.train_node_encoder = bool(train_node_encoder)
        for name, p in self.model.named_parameters():
            if (
                (self.train_node_encoder and (name.startswith("node_encoder.") or ("vg_adapter" in name)))
                or ("gl_projector" in name)
                or ("graph_qformer" in name)
                or ("gvl_adapter" in name)
                or ("graph_aux_head" in name)
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False
        print(
            f"[Stage2B] node_encoder_type={self.model.node_encoder_type} "
            f"train_node_encoder={self.train_node_encoder}"
        )

        # LLM config
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
        return AdamW(params, lr=self.hparams.lr)

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

    def _compute_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        loss, metrics = self._compute_step(batch)
        bsz = len(batch["text"])
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        self.log("train_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=True, on_epoch=True, batch_size=bsz)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._compute_step(batch)
        bsz = len(batch["text"])
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_lm", metrics["loss_lm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_aux", metrics["loss_aux"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_xtc", metrics["loss_xtc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_loss_xtm", metrics["loss_xtm"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_xtm_acc", metrics["xtm_acc"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        self.log("val_xtm_valid_neg_ratio", metrics["xtm_valid_neg_ratio"], prog_bar=False, on_step=False, on_epoch=True, batch_size=bsz)
        return loss


# ---------------------------------------------------------------------------
# Load stage2-A weights only (no optimizer restore)
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


def _extract_stage2a_provenance_from_ckpt(ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            "Stage2B requires a Stage2A Lightning checkpoint (.ckpt) from the strict pipeline."
        )
    hp = ckpt.get("hyper_parameters", {}) or {}

    bootstrap = hp.get("stage2A_bootstrap")
    if not isinstance(bootstrap, dict):
        mode = str(hp.get("stage2A_bootstrap_mode", "")).strip().lower()
        tok = str(hp.get("graph_tokenizer_type", "")).strip().lower()
        stage1_qformer_ckpt = hp.get("stage1_qformer_ckpt", None)
        if mode not in {"legacy_stage1", "no_stage1"}:
            if stage1_qformer_ckpt:
                mode = "legacy_stage1"
            elif tok in {"qformer", "perceiver"}:
                mode = "legacy_stage1" if tok == "qformer" else "no_stage1"
        if mode not in {"legacy_stage1", "no_stage1"}:
            raise RuntimeError(
                "Stage2A checkpoint metadata missing valid bootstrap info. "
                "Expected stage2A_bootstrap or stage2A_bootstrap_mode."
            )
        if tok not in {"qformer", "perceiver"}:
            tok = "qformer" if mode == "legacy_stage1" else "perceiver"
        bootstrap = {
            "mode": mode,
            "graph_tokenizer_type": tok,
            "stage1_qformer_ckpt": stage1_qformer_ckpt,
        }

    mode = str(bootstrap.get("mode", "")).strip().lower()
    tok = str(bootstrap.get("graph_tokenizer_type", "")).strip().lower()
    if mode not in {"legacy_stage1", "no_stage1"}:
        raise RuntimeError(f"Invalid Stage2A bootstrap mode in checkpoint: {mode}")
    if tok not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Invalid Stage2A graph tokenizer type in checkpoint: {tok}")
    if mode == "legacy_stage1" and not bootstrap.get("stage1_qformer_ckpt"):
        raise RuntimeError(
            "Stage2A checkpoint has legacy_stage1 bootstrap but missing stage1_qformer_ckpt."
        )

    raw_cfg = hp.get("graph_tokenizer_config")
    if isinstance(raw_cfg, dict):
        tokenizer_cfg = _normalize_graph_tokenizer_config(raw_cfg)
    else:
        tokenizer_cfg = _normalize_graph_tokenizer_config(
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
    if tokenizer_cfg["type"] != tok:
        raise RuntimeError(
            f"Stage2A provenance mismatch: bootstrap type={tok}, tokenizer_config type={tokenizer_cfg['type']}"
        )
    return ckpt, bootstrap, tokenizer_cfg


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


def load_stage2A_weights_only(
    pl_model: Stage2BProjectorPL,
    ckpt_path: str,
    expected_bootstrap: Dict[str, Any],
    expected_graph_tokenizer_config: Dict[str, Any],
):
    ckpt, stage2a_bootstrap, stage2a_graph_tokenizer_config = _extract_stage2a_provenance_from_ckpt(ckpt_path)

    if str(expected_bootstrap.get("mode")) != str(stage2a_bootstrap.get("mode")):
        raise RuntimeError(
            "Stage2B bootstrap mismatch with Stage2A provenance: "
            f"expected_mode={expected_bootstrap.get('mode')} got_mode={stage2a_bootstrap.get('mode')}"
        )
    _assert_graph_tokenizer_match(
        expected_cfg=expected_graph_tokenizer_config,
        got_cfg=stage2a_graph_tokenizer_config,
        stage_name="Stage2B",
    )

    sd = ckpt["state_dict"]
    # strip "model." prefix -> match OmniGraphModel keys
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = pl_model.model.load_state_dict(sd, strict=False)
    print(f"[Stage2B] loaded weights-only from: {ckpt_path}")
    print(f"[Stage2B] missing={len(missing)} unexpected={len(unexpected)}")
    print(
        "[Stage2B] Stage2A provenance: "
        f"bootstrap={stage2a_bootstrap} graph_tokenizer={stage2a_graph_tokenizer_config}"
    )
    if missing:
        print("  missing head:", missing[:20])
    if unexpected:
        print("  unexpected head:", unexpected[:20])
    return {
        "stage2A_ckpt": str(ckpt_path),
        "stage2A_bootstrap": stage2a_bootstrap,
        "graph_tokenizer_config": stage2a_graph_tokenizer_config,
        "stage1_qformer_ckpt": stage2a_bootstrap.get("stage1_qformer_ckpt"),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


def _resolve_graph_tokenizer_from_stage2a(
    args: argparse.Namespace,
    stage2a_bootstrap: Dict[str, Any],
    stage2a_graph_tokenizer_config: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    req_type = str(args.graph_tokenizer_type).strip().lower()
    upstream_type = str(stage2a_graph_tokenizer_config.get("type")).strip().lower()
    if req_type == "auto":
        resolved_type = upstream_type
    else:
        resolved_type = req_type
    if resolved_type not in {"qformer", "perceiver"}:
        raise RuntimeError(f"Unsupported graph_tokenizer_type for Stage2B: {resolved_type}")
    if resolved_type != upstream_type:
        raise RuntimeError(
            "Stage2B tokenizer type must match Stage2A provenance: "
            f"requested={resolved_type} upstream={upstream_type}"
        )

    resolved_cfg = {
        "type": resolved_type,
        "num_latents": int(stage2a_graph_tokenizer_config.get("num_latents", 32)),
        "hidden_dim": int(stage2a_graph_tokenizer_config.get("hidden_dim", 768)),
        "num_layers": int(stage2a_graph_tokenizer_config.get("num_layers", 3)),
        "num_heads": int(stage2a_graph_tokenizer_config.get("num_heads", 8)),
        "ff_mult": int(stage2a_graph_tokenizer_config.get("ff_mult", 4)),
        "dropout": float(stage2a_graph_tokenizer_config.get("dropout", 0.0)),
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
            expected_cfg=stage2a_graph_tokenizer_config,
            got_cfg=resolved_cfg,
            stage_name="Stage2B",
        )
    resolved_bootstrap = {
        "mode": str(stage2a_bootstrap.get("mode", "no_stage1")),
        "graph_tokenizer_type": resolved_type,
        "stage1_qformer_ckpt": stage2a_bootstrap.get("stage1_qformer_ckpt"),
    }
    return resolved_bootstrap, resolved_cfg


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_graphs", type=str, required=True)
    ap.add_argument("--regions", type=str, required=True)
    ap.add_argument("--stage2A_ckpt", type=str, required=True, help="best ckpt from stage2-A (weights init)")
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--graph_model", type=str, default="clip_gt_arxiv_pub")

    ap.add_argument("--gpu", type=int, default=0, help="gpu index; -1 for cpu")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--max_graph_tokens", type=int, default=0, help="If >0, truncate graph prefix tokens before LLM.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", type=str, default="32")
    ap.add_argument("--llm_dtype", type=str, default="bfloat16", help="LLM load dtype: bfloat16/float16/float32.")
    ap.add_argument("--llm_attn_implementation", type=str, default="sdpa", help="LLM attention backend (e.g., sdpa/flash_attention_2).")
    ap.add_argument("--graph_tokenizer_type", type=str, default="auto", choices=["auto", "qformer", "perceiver"])
    ap.add_argument("--perceiver_num_latents", type=int, default=-1, help="<=0 means inherit from Stage2A provenance.")
    ap.add_argument("--perceiver_num_layers", type=int, default=-1, help="<=0 means inherit from Stage2A provenance.")
    ap.add_argument("--perceiver_num_heads", type=int, default=-1, help="<=0 means inherit from Stage2A provenance.")
    ap.add_argument("--perceiver_ff_mult", type=int, default=-1, help="<=0 means inherit from Stage2A provenance.")
    ap.add_argument("--perceiver_dropout", type=float, default=-1.0, help="<0 means inherit from Stage2A provenance.")
    ap.add_argument("--node_encoder_type", type=str, default="hybrid", choices=["hybrid", "open_vocab", "legacy_vg"])
    ap.add_argument("--node_encoder_out_dim", type=int, default=128, help="Graph node encoder output dim.")
    ap.add_argument("--train_node_encoder", type=int, default=1, choices=[0, 1], help="1=train node encoder, 0=freeze.")
    ap.add_argument("--node_encoder_alpha_init", type=float, default=0.3, help="Hybrid node encoder alpha init.")
    ap.add_argument("--use_qa_type_token", type=int, default=1, choices=[0, 1], help="Prefix prompt with QA type control token.")
    ap.add_argument("--enable_gvl_adapter", type=int, default=1, choices=[0, 1], help="Enable graph-vision-language cross-attn adapter.")
    ap.add_argument("--gvl_adapter_gate_init", type=float, default=0.1, help="Initial gate for cross-modal adapter.")
    ap.add_argument("--enable_graph_aux_head", type=int, default=1, choices=[0, 1], help="Enable auxiliary graph reasoning binary head.")
    ap.add_argument("--graph_aux_loss_weight", type=float, default=0.05, help="Auxiliary binary loss weight.")
    ap.add_argument("--enable_xtc", type=int, default=1, choices=[0, 1], help="Enable bidirectional graph-text contrastive loss.")
    ap.add_argument("--enable_xtm", type=int, default=1, choices=[0, 1], help="Enable graph-text matching loss with hard negatives.")
    ap.add_argument("--xtc_weight", type=float, default=0.15, help="Weight for XTC loss.")
    ap.add_argument("--xtm_weight", type=float, default=0.10, help="Weight for XTM loss.")
    ap.add_argument("--xtc_logit_scale_init", type=float, default=2.66, help="Initial logit scale parameter (log space).")
    ap.add_argument("--xtm_dup_thresh", type=float, default=0.98, help="Cosine similarity threshold for duplicate-text negative filtering.")

    ap.add_argument("--save_dir", type=str, default="checkpoints_stage2B")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)
    ap.add_argument("--disable_graph_qa", action="store_true", help="Disable synthetic graph QA from VG scene graphs.")
    ap.add_argument("--graph_qa_max_per_image", type=int, default=3, help="Max synthetic QA pairs per image.")
    ap.add_argument("--graph_qa_repeat", type=int, default=2, help="Repeat factor for synthetic graph QA samples.")
    ap.add_argument("--graph_qa_seed", type=int, default=42, help="Random seed for synthetic graph QA generation.")

    ap.add_argument("--val_ratio", type=float, default=0.001)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_check_interval", type=float, default=2000,
                    help=">=1: validate every N steps; (0,1]: fraction epoch")
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--num_sanity_val_steps", type=int, default=0)
    ap.add_argument("--accumulate_grad_batches", type=int, default=4)
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    print("[Pipeline] Stage2B start: requires strict Stage2A .ckpt with bootstrap provenance.")
    if not Path(args.stage2A_ckpt).exists():
        raise FileNotFoundError(f"--stage2A_ckpt not found: {args.stage2A_ckpt}")
    _, stage2a_bootstrap, stage2a_graph_tokenizer_config = _extract_stage2a_provenance_from_ckpt(args.stage2A_ckpt)
    resolved_bootstrap, resolved_graph_tokenizer_config = _resolve_graph_tokenizer_from_stage2a(
        args=args,
        stage2a_bootstrap=stage2a_bootstrap,
        stage2a_graph_tokenizer_config=stage2a_graph_tokenizer_config,
    )
    print(
        "[Stage2B] resolved tokenizer from Stage2A provenance: "
        f"bootstrap={resolved_bootstrap} graph_tokenizer={resolved_graph_tokenizer_config}"
    )

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

    # 3) region phrases dataset
    region_pairs = load_region_pairs(args.regions)
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
        raise RuntimeError("No usable pairs for Stage2B after scene-graph/region join.")

    train_ds, val_ds, train_ids, val_ids = split_by_image_id(
        full_dataset,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    overlap = train_ids.intersection(val_ids)
    print(f"[Split] train={len(train_ds)} val={len(val_ds)} train_images={len(train_ids)} val_images={len(val_ids)}")
    print(f"[SplitCheck] train_image_ids ∩ val_image_ids = {len(overlap)}")
    if overlap:
        raise RuntimeError(f"Leakage detected in Stage2B split: overlap image_ids={list(sorted(overlap))[:10]}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text,
    )
    vci = _parse_val_check_interval(args.val_check_interval)

    train_node_encoder = bool(int(args.train_node_encoder))

    # 4) stage2-B model
    pl_model = Stage2BProjectorPL(
        llm_model_name=args.llm,
        graph_model_name=args.graph_model,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        stage2A_ckpt=str(args.stage2A_ckpt),
        stage2A_bootstrap=dict(resolved_bootstrap),
        lr=args.lr,
        max_length=args.max_length,
        max_graph_tokens=(int(args.max_graph_tokens) if int(args.max_graph_tokens) > 0 else None),
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

    # 5) init from stage2-A (weights only)
    stage2a_provenance = load_stage2A_weights_only(
        pl_model=pl_model,
        ckpt_path=args.stage2A_ckpt,
        expected_bootstrap=resolved_bootstrap,
        expected_graph_tokenizer_config=resolved_graph_tokenizer_config,
    )

    # 6) callbacks
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="stage2B-step={step:07d}-val_loss={val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience,
        min_delta=args.min_delta,
    )

    use_gpu = torch.cuda.is_available() and args.gpu >= 0
    devices = [int(args.gpu)] if use_gpu else 1
    if use_gpu and int(args.gpu) >= torch.cuda.device_count():
        raise ValueError(f"--gpu={args.gpu} out of range (count={torch.cuda.device_count()})")

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        precision=_parse_precision(args.precision),
        max_steps=args.max_steps,
        max_epochs=999999,  # max_steps 控制
        callbacks=[ckpt, lr_monitor, early],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=max(1, int(args.accumulate_grad_batches)),
        check_val_every_n_epoch=1,
        val_check_interval=vci,
        limit_val_batches=float(args.limit_val_batches),
        num_sanity_val_steps=max(0, int(args.num_sanity_val_steps)),
        enable_model_summary=False,
    )

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 7) export pure weights (OmniGraphModel)
    out_pt = save_dir / "stage2B_model_state_dict.pt"
    torch.save(pl_model.model.state_dict(), str(out_pt))
    stage_meta = {
        "stage": "stage2B",
        "num_obj": int(num_obj),
        "num_attr": int(num_attr),
        "stage2A_provenance": stage2a_provenance,
        "stage2A_bootstrap": resolved_bootstrap,
        "graph_tokenizer_config": pl_model.model.graph_tokenizer_config,
        "node_encoder_config": pl_model.model.node_encoder_config,
        "architecture_config": pl_model.model.architecture_config,
        "alignment_config": pl_model.alignment_config,
        "xtm_stats_summary": pl_model.get_xtm_stats_summary(),
        "best_ckpt": ckpt.best_model_path or "",
        "export_path": str(out_pt),
    }
    (save_dir / "stage2B_meta.json").write_text(json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] weights -> {out_pt}")


if __name__ == "__main__":
    main()
