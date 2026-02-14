# omnigraph/train/train_stage2B.py
from __future__ import annotations

import sys
import json
import argparse
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
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Dataset (你已有)
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
from omnigraph.train.alignment_step import (  # noqa: E402
    compute_alignment_losses,
    compute_base_lm_aux_losses,
    init_xtm_stats_accum,
    summarize_xtm_stats_accum,
    update_xtm_stats_accum,
)
from omnigraph.train.stage_reporting import (  # noqa: E402
    build_stage_meta,
    log_alignment_step_metrics,
)
from omnigraph.train.graph_tokenizer_provenance import (  # noqa: E402
    assert_graph_tokenizer_match,
    parse_bootstrap_and_graph_tokenizer_from_hparams,
    resolve_graph_tokenizer_from_upstream,
)

# Model
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402


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
    train_indices, val_indices, train_ids, val_ids = split_indices_by_image_id(
        dataset.samples,
        val_ratio=float(val_ratio),
        seed=int(seed),
        image_id_key="image_id",
        fallback_when_train_empty=True,
        require_non_empty_train=True,
        require_non_empty_val=True,
        error_prefix="Stage2B split",
    )
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
        self._xtm_stats_accum: Dict[str, float] = init_xtm_stats_accum()
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
        update_xtm_stats_accum(self._xtm_stats_accum, stats)

    def get_xtm_stats_summary(self) -> Dict[str, float]:
        return summarize_xtm_stats_accum(self._xtm_stats_accum)

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

        base_loss, lm_loss, aux_loss = compute_base_lm_aux_losses(
            outputs=outputs,
            labels=labels,
            device=self.device,
        )
        weighted_align_loss, align_metrics = compute_alignment_losses(
            outputs=outputs,
            device=self.device,
            batch_size=len(prompts),
            image_ids=image_ids,
            qa_types=qa_types,
            enable_xtc=bool(self.enable_xtc),
            enable_xtm=bool(self.enable_xtm),
            enable_xgv=False,
            xtc_weight=float(self.xtc_weight),
            xtm_weight=float(self.xtm_weight),
            xgv_weight=0.0,
            xtm_dup_thresh=float(self.xtm_dup_thresh),
            xtc_logit_scale=self.xtc_logit_scale,
            update_xtm_stats=self._update_xtm_stats,
        )
        total_loss = base_loss + weighted_align_loss
        metrics = {
            "loss_lm": lm_loss.detach(),
            "loss_aux": aux_loss.detach(),
            "loss_xtc": align_metrics["loss_xtc"],
            "loss_xtm": align_metrics["loss_xtm"],
            "xtm_acc": align_metrics["xtm_acc"],
            "xtm_valid_neg_ratio": align_metrics["xtm_valid_neg_ratio"],
        }
        return total_loss, metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._compute_step(batch)
        bsz = len(batch["text"])
        log_alignment_step_metrics(
            self,
            phase="train",
            loss=loss,
            metrics=metrics,
            batch_size=bsz,
            include_xgv=False,
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss, metrics = self._compute_step(batch)
        bsz = len(batch["text"])
        log_alignment_step_metrics(
            self,
            phase="val",
            loss=loss,
            metrics=metrics,
            batch_size=bsz,
            include_xgv=False,
        )
        return loss


class PeriodicSaveLastCheckpoint(pl.Callback):
    """Force-save trainer state to last.ckpt every N steps for OOM recovery."""

    def __init__(self, every_n_steps: int, ckpt_path: str):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.ckpt_path = str(ckpt_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):  # type: ignore[override]
        step = int(getattr(trainer, "global_step", 0))
        if step <= 0:
            return
        if step % self.every_n_steps != 0:
            return
        trainer.save_checkpoint(self.ckpt_path)


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
            except Exception as exc:
                print(f"[{self.stage_tag}][Warn] failed to save manual-stop checkpoint: {exc}")
        trainer.should_stop = True
        self._triggered = True


# ---------------------------------------------------------------------------
# Load stage2-A weights only (no optimizer restore)
# ---------------------------------------------------------------------------


def _extract_stage2a_provenance_from_ckpt(ckpt_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Stage2B failed to load Stage2A checkpoint/state_dict: unsupported object type.")

    if "state_dict" in ckpt:
        hp = ckpt.get("hyper_parameters", {}) or {}
        bootstrap, tokenizer_cfg = parse_bootstrap_and_graph_tokenizer_from_hparams(
            hp=hp,
            bootstrap_field="stage2A_bootstrap",
            bootstrap_mode_field="stage2A_bootstrap_mode",
            context_label="Stage2A",
            require_legacy_stage1_qformer_ckpt=True,
        )
        return ckpt, bootstrap, tokenizer_cfg

    # Fallback: raw OmniGraphModel state_dict exported by GraphBootstrap
    if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
        ckpt_file = Path(ckpt_path)
        meta_candidates = [
            ckpt_file.parent / "graph_bootstrap_meta.json",
            ckpt_file.parent / "stage2A_meta.json",
        ]
        meta_obj: Dict[str, Any] = {}
        for meta_path in meta_candidates:
            if meta_path.exists():
                try:
                    meta_obj = json.loads(meta_path.read_text(encoding="utf-8"))
                    break
                except Exception:
                    continue

        bootstrap = meta_obj.get("stage2A_bootstrap") or meta_obj.get("graph_bootstrap_config")
        tokenizer_cfg = meta_obj.get("graph_tokenizer_config")
        if not isinstance(bootstrap, dict) or not isinstance(tokenizer_cfg, dict):
            raise RuntimeError(
                "Stage2B got Stage2A raw state_dict but missing provenance in graph_bootstrap_meta.json/stage2A_meta.json."
            )
        return {"state_dict": ckpt}, dict(bootstrap), dict(tokenizer_cfg)

    raise RuntimeError(
        "Stage2B requires a Stage2A Lightning checkpoint (.ckpt) or GraphBootstrap raw state_dict export."
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
    assert_graph_tokenizer_match(
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
        "graph_bootstrap_ckpt": str(ckpt_path),
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
    resolved_cfg = resolve_graph_tokenizer_from_upstream(
        args=args,
        upstream_graph_tokenizer_config=stage2a_graph_tokenizer_config,
        stage_name="Stage2B",
        upstream_stage_name="Stage2A",
    )
    resolved_type = str(resolved_cfg.get("type", "qformer"))
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
    ap.add_argument("--stage2A_ckpt", type=str, default="", help="Legacy arg: best ckpt from stage2-A (weights init)")
    ap.add_argument("--graph_bootstrap_ckpt", type=str, default="", help="Preferred arg: graph_bootstrap ckpt (weights init).")
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
    ap.add_argument("--perceiver_num_latents", type=int, default=-1, help="<=0 means inherit from GraphBootstrap provenance.")
    ap.add_argument("--perceiver_num_layers", type=int, default=-1, help="<=0 means inherit from GraphBootstrap provenance.")
    ap.add_argument("--perceiver_num_heads", type=int, default=-1, help="<=0 means inherit from GraphBootstrap provenance.")
    ap.add_argument("--perceiver_ff_mult", type=int, default=-1, help="<=0 means inherit from GraphBootstrap provenance.")
    ap.add_argument("--perceiver_dropout", type=float, default=-1.0, help="<0 means inherit from GraphBootstrap provenance.")
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

    ap.add_argument("--save_dir", type=str, default="checkpoints_graph_refine")
    ap.add_argument("--resume_from_checkpoint", type=str, default="", help="Resume full trainer state from this checkpoint.")
    ap.add_argument("--checkpoint_every_n_steps", type=int, default=1000, help="Force-save last.ckpt every N train steps (0 to disable).")
    ap.add_argument("--manual_stop_file", type=str, default="", help="If this file appears during training, trigger graceful early stop.")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)
    ap.add_argument("--disable_graph_qa", action="store_true", help="Disable synthetic graph QA from VG scene graphs.")
    ap.add_argument("--graph_qa_max_per_image", type=int, default=3, help="Max synthetic QA pairs per image.")
    ap.add_argument("--graph_qa_repeat", type=int, default=2, help="Repeat factor for synthetic graph QA samples.")
    ap.add_argument("--extra_scene_graphs", type=str, default="", help="Comma-separated extra VG-style scene graph JSON files (pseudo graphs).")
    ap.add_argument("--pseudo_graph_qa_max_per_image", type=int, default=2, help="Max synthetic QA pairs per pseudo-graph image.")
    ap.add_argument("--pseudo_graph_qa_repeat", type=int, default=1, help="Repeat factor for pseudo-graph synthetic QA.")
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
    resume_ckpt = str(args.resume_from_checkpoint).strip()
    if resume_ckpt and not Path(resume_ckpt).exists():
        raise FileNotFoundError(f"--resume_from_checkpoint not found: {resume_ckpt}")
    stage2a_ckpt = str(args.graph_bootstrap_ckpt).strip() or str(args.stage2A_ckpt).strip()
    if not stage2a_ckpt:
        raise ValueError("Missing upstream checkpoint: provide --graph_bootstrap_ckpt (or legacy --stage2A_ckpt).")
    args.stage2A_ckpt = stage2a_ckpt

    pl.seed_everything(args.seed, workers=True)
    print("[Pipeline] GraphRefine start: requires strict GraphBootstrap .ckpt with bootstrap provenance.")
    if not Path(args.stage2A_ckpt).exists():
        raise FileNotFoundError(f"--graph_bootstrap_ckpt/--stage2A_ckpt not found: {args.stage2A_ckpt}")
    _, stage2a_bootstrap, stage2a_graph_tokenizer_config = _extract_stage2a_provenance_from_ckpt(args.stage2A_ckpt)
    resolved_bootstrap, resolved_graph_tokenizer_config = _resolve_graph_tokenizer_from_stage2a(
        args=args,
        stage2a_bootstrap=stage2a_bootstrap,
        stage2a_graph_tokenizer_config=stage2a_graph_tokenizer_config,
    )
    print(
        "[GraphRefine] resolved tokenizer from GraphBootstrap provenance: "
        f"bootstrap={resolved_bootstrap} graph_tokenizer={resolved_graph_tokenizer_config}"
    )

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

    # 3) region phrases dataset
    region_pairs = load_region_pairs(args.regions)
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
        raise RuntimeError("No usable pairs for GraphRefine after scene-graph/region join.")

    train_ds, val_ds, train_ids, val_ids = split_by_image_id(
        full_dataset,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    overlap = train_ids.intersection(val_ids)
    print(f"[Split] train={len(train_ds)} val={len(val_ds)} train_images={len(train_ids)} val_images={len(val_ids)}")
    print(f"[SplitCheck] train_image_ids ∩ val_image_ids = {len(overlap)}")
    if overlap:
        raise RuntimeError(f"Leakage detected in GraphRefine split: overlap image_ids={list(sorted(overlap))[:10]}")

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
        filename="graph_refine-step={step:07d}-val_loss={val_loss:.3f}",
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
    try:
        tb_logger = TensorBoardLogger(
            save_dir=str(save_dir),
            name="tb_logs",
            version="",
        )
    except ModuleNotFoundError as exc:
        print(f"[GraphRefine][Warn] TensorBoard unavailable, fallback to CSV logger: {exc}")
        tb_logger = CSVLogger(
            save_dir=str(save_dir),
            name="csv_logs",
            version="",
        )
    callbacks = [ckpt, lr_monitor, early, TQDMProgressBar(refresh_rate=10)]
    manual_stop_file = str(args.manual_stop_file).strip() or str(save_dir / "STOP_EARLY")
    callbacks.append(
        ManualEarlyStopByFile(
            stop_file=manual_stop_file,
            stage_tag="GraphRefine",
            save_ckpt_path=str(save_dir / "manual_stop.ckpt"),
        )
    )
    print(f"[GraphRefine] manual early-stop enabled: touch file -> {manual_stop_file}")
    ckpt_every_n = max(0, int(args.checkpoint_every_n_steps))
    if ckpt_every_n > 0:
        periodic_ckpt_path = save_dir / "last.ckpt"
        callbacks.append(PeriodicSaveLastCheckpoint(every_n_steps=ckpt_every_n, ckpt_path=str(periodic_ckpt_path)))
        print(f"[GraphRefine] periodic last.ckpt save enabled: every_n_steps={ckpt_every_n} path={periodic_ckpt_path}")

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
        enable_model_summary=False,
    )

    if resume_ckpt:
        print(f"[GraphRefine] resume trainer state from ckpt: {resume_ckpt}")
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=resume_ckpt)
    else:
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 7) export pure weights (OmniGraphModel)
    out_pt = save_dir / "graph_refine_model_state_dict.pt"
    out_pt_legacy = save_dir / "stage2B_model_state_dict.pt"
    torch.save(pl_model.model.state_dict(), str(out_pt))
    torch.save(pl_model.model.state_dict(), str(out_pt_legacy))
    stage_meta = build_stage_meta(
        stage="graph_refine",
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        base_scene_graphs=str(args.scene_graphs),
        extra_scene_graphs=list(extra_paths),
        merge_stats=merge_stats,
        pseudo_graph_qa_max_per_image=int(args.pseudo_graph_qa_max_per_image),
        pseudo_graph_qa_repeat=int(args.pseudo_graph_qa_repeat),
        provenance_key="graph_bootstrap_provenance",
        provenance_value=stage2a_provenance,
        graph_tokenizer_config=pl_model.model.graph_tokenizer_config,
        node_encoder_config=pl_model.model.node_encoder_config,
        architecture_config=pl_model.model.architecture_config,
        alignment_config=pl_model.alignment_config,
        xtm_stats_summary=pl_model.get_xtm_stats_summary(),
        best_ckpt=ckpt.best_model_path or "",
        export_path=str(out_pt),
        extra_fields={
            "legacy_stage": "stage2B",
            "stage2A_provenance": stage2a_provenance,
            "stage2A_bootstrap": resolved_bootstrap,
        },
    )
    stage_meta_text = json.dumps(stage_meta, ensure_ascii=False, indent=2)
    (save_dir / "graph_refine_meta.json").write_text(stage_meta_text, encoding="utf-8")
    (save_dir / "stage2B_meta.json").write_text(stage_meta_text, encoding="utf-8")
    print(f"[GraphRefine] saved weights -> {out_pt} (legacy copy: {out_pt_legacy})")


if __name__ == "__main__":
    main()
