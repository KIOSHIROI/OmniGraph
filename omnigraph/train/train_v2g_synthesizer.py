#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# repo bootstrap
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from omnigraph.data.graph_canonical import canonicalize_scene_graph
from omnigraph.model.v2g import V2GSynthesizerConfig, VisionGraphSynthesizer


def _parse_precision(x: str):
    s = str(x).strip().lower()
    if s in {"16", "16-mixed", "bf16-mixed", "32", "64"}:
        return s
    if s in {"fp16", "float16"}:
        return "16-mixed"
    if s in {"bf16", "bfloat16"}:
        return "bf16-mixed"
    if s in {"fp32", "float32"}:
        return "32"
    return "16-mixed"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _resolve_image_path(image_path: str, image_root: str) -> Optional[str]:
    raw = str(image_path or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if p.is_file():
        return str(p)
    if image_root:
        rp = Path(image_root) / raw
        if rp.is_file():
            return str(rp)
    return None


def _extract_graph_obj(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    g = rec.get("graph_json")
    if g is None:
        g = rec.get("scene_graph")
    if g is None and ("objects" in rec or "relationships" in rec):
        g = {
            "image_id": rec.get("image_id", rec.get("id")),
            "width": rec.get("width", 0),
            "height": rec.get("height", 0),
            "objects": rec.get("objects", []),
            "relationships": rec.get("relationships", []),
        }
    if g is None:
        return None
    try:
        return canonicalize_scene_graph(
            g,
            image_id=rec.get("image_id", rec.get("id")),
            image_path=str(rec.get("image_path", "") or ""),
            width=rec.get("width", 0),
            height=rec.get("height", 0),
        )
    except Exception:
        return None


class V2GTrainDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], image_root: str):
        self.image_root = str(image_root or "")
        self.items: List[Dict[str, Any]] = []

        for rec in records:
            if not isinstance(rec, dict):
                continue
            img_path = _resolve_image_path(str(rec.get("image_path", "")), self.image_root)
            if not img_path:
                continue
            graph = _extract_graph_obj(rec)
            if graph is None:
                continue
            caption = " ".join(str(rec.get("caption", "")).split())
            prompt = VisionGraphSynthesizer.build_prompt(caption=caption)
            target = VisionGraphSynthesizer.graph_to_target_text(graph)
            self.items.append(
                {
                    "id": str(rec.get("id", rec.get("image_id", ""))),
                    "image_path": img_path,
                    "prompt": prompt,
                    "target": target,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = Image.open(str(it["image_path"])).convert("RGB")
        return {
            "id": str(it["id"]),
            "image": img,
            "prompt": str(it["prompt"]),
            "target": str(it["target"]),
        }


def collate_v2g(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "ids": [str(x["id"]) for x in batch],
        "images": [x["image"] for x in batch],
        "prompts": [str(x["prompt"]) for x in batch],
        "targets": [str(x["target"]) for x in batch],
    }


class V2GSynthPL(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        torch_dtype: str,
        max_length: int,
        max_new_tokens: int,
        lr: float,
        weight_decay: float,
        freeze_vision: bool,
        freeze_language: bool,
        train_qformer: bool,
        train_language_projection: bool,
        train_lm_head: bool,
    ):
        super().__init__()
        self.save_hyperparameters()

        cfg = V2GSynthesizerConfig(
            model_name=str(model_name),
            torch_dtype=str(torch_dtype),
            max_length=int(max_length),
            max_new_tokens=int(max_new_tokens),
            freeze_vision=bool(freeze_vision),
            freeze_language=bool(freeze_language),
            train_qformer=bool(train_qformer),
            train_language_projection=bool(train_language_projection),
            train_lm_head=bool(train_lm_head),
        )
        self.synthesizer = VisionGraphSynthesizer(cfg)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)

        trainable = self.synthesizer.count_trainable_params()
        total = int(sum(p.numel() for p in self.synthesizer.parameters()))
        print(f"[V2G] trainable_params={trainable} total_params={total}")

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self.synthesizer.forward_train(
            images=batch["images"],
            prompts=batch["prompts"],
            targets=batch["targets"],
            device=self.device,
            max_length=int(self.hparams.max_length),
        )
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["images"]))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self.synthesizer.forward_train(
            images=batch["images"],
            prompts=batch["prompts"],
            targets=batch["targets"],
            device=self.device,
            max_length=int(self.hparams.max_length),
        )
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["images"]))
        return loss

    def configure_optimizers(self):
        params = [p for p in self.synthesizer.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("No trainable params in V2G synthesizer.")
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(100, int(self.trainer.max_steps)))
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train V2G synthesizer on image->scene-graph supervision.")
    ap.add_argument("--train_manifest", required=True, help="jsonl with id,image_path,caption,graph_json|scene_graph")
    ap.add_argument("--val_manifest", default="", help="optional val jsonl; empty means split from train")
    ap.add_argument("--image_root", required=True, help="base path for relative image_path in manifests")

    ap.add_argument("--model_name", default="Salesforce/blip2-flan-t5-xl")
    ap.add_argument("--torch_dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--precision", default="16-mixed")

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--accumulate_grad_batches", type=int, default=8)

    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--max_steps", type=int, default=50000)
    ap.add_argument("--val_check_interval", type=float, default=1000)
    ap.add_argument("--limit_val_batches", type=float, default=1.0)
    ap.add_argument("--num_sanity_val_steps", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--min_delta", type=float, default=0.0005)
    ap.add_argument("--val_ratio", type=float, default=0.02)

    ap.add_argument("--freeze_vision", type=int, default=1, choices=[0, 1])
    ap.add_argument("--freeze_language", type=int, default=1, choices=[0, 1])
    ap.add_argument("--train_qformer", type=int, default=1, choices=[0, 1])
    ap.add_argument("--train_language_projection", type=int, default=1, choices=[0, 1])
    ap.add_argument("--train_lm_head", type=int, default=0, choices=[0, 1])

    ap.add_argument("--lora_r", type=int, default=0, help="Reserved for optional PEFT; 0 disables.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--save_dir", required=True)

    args = ap.parse_args()

    if int(args.lora_r) > 0:
        print("[V2G] lora_r > 0 requested, but this script currently uses projector/qformer-style lightweight tuning.")

    pl.seed_everything(int(args.seed), workers=True)

    train_records = _read_jsonl(args.train_manifest)
    if not train_records:
        raise RuntimeError(f"Empty train manifest: {args.train_manifest}")

    if str(args.val_manifest).strip():
        val_records = _read_jsonl(args.val_manifest)
        if not val_records:
            raise RuntimeError(f"Empty val manifest: {args.val_manifest}")
    else:
        n_total = len(train_records)
        n_val = max(1, int(round(n_total * float(args.val_ratio))))
        n_val = min(n_val, max(1, n_total - 1))
        val_records = train_records[:n_val]
        train_records = train_records[n_val:]

    train_ds = V2GTrainDataset(train_records, image_root=args.image_root)
    val_ds = V2GTrainDataset(val_records, image_root=args.image_root)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)}")

    if len(train_ds) <= 0 or len(val_ds) <= 0:
        raise RuntimeError("No valid training/validation samples after filtering manifests.")

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=collate_v2g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=True,
        persistent_workers=(int(args.num_workers) > 0),
        collate_fn=collate_v2g,
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pl_model = V2GSynthPL(
        model_name=str(args.model_name),
        torch_dtype=str(args.torch_dtype),
        max_length=int(args.max_length),
        max_new_tokens=int(args.max_new_tokens),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        freeze_vision=bool(int(args.freeze_vision)),
        freeze_language=bool(int(args.freeze_language)),
        train_qformer=bool(int(args.train_qformer)),
        train_language_projection=bool(int(args.train_language_projection)),
        train_lm_head=bool(int(args.train_lm_head)),
    )

    vci = float(args.val_check_interval)
    if vci > 1.0:
        vci = int(vci)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(save_dir),
        filename="v2g-step={step:07d}-val_loss={val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=max(1, int(args.patience)),
        min_delta=float(args.min_delta),
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    use_gpu = torch.cuda.is_available() and int(args.gpu) >= 0
    devices = [int(args.gpu)] if use_gpu else 1
    if use_gpu and int(args.gpu) >= torch.cuda.device_count():
        raise ValueError(f"--gpu={args.gpu} out of range (device_count={torch.cuda.device_count()}).")

    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        precision=_parse_precision(str(args.precision)),
        max_steps=max(1, int(args.max_steps)),
        max_epochs=999999,
        callbacks=[ckpt_cb, es, lr_monitor],
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

    export_path = save_dir / "v2g_state_dict.pt"
    torch.save(pl_model.synthesizer.state_dict(), str(export_path))

    meta = {
        "train_manifest": str(args.train_manifest),
        "val_manifest": str(args.val_manifest),
        "image_root": str(args.image_root),
        "model_name": str(args.model_name),
        "torch_dtype": str(args.torch_dtype),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "best_ckpt": ckpt_cb.best_model_path or "",
        "last_ckpt": str(save_dir / "last.ckpt"),
        "export_path": str(export_path),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "trainable_params": int(pl_model.synthesizer.count_trainable_params()),
        "config": pl_model.synthesizer.as_config_dict(),
    }
    (save_dir / "v2g_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Export] {export_path}")
    print(f"[Best] {ckpt_cb.best_model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
