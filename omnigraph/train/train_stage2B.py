# omnigraph/train/train_stage2B.py
from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Repo bootstrap + env
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
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Dataset (你已有)
from omnigraph.data.vg_scene_graph_dataset import (  # noqa: E402
    build_vg_vocabs_from_file,
    VGSceneGraphDataset,
)
from omnigraph.data.vg_graph_qa import build_vg_graph_qa_records  # noqa: E402

# Model
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402


# ---------------------------------------------------------------------------
# Region loader + wrapper dataset
# ---------------------------------------------------------------------------
def load_region_pairs(region_path: str) -> List[Tuple[int, str]]:
    """
    region_descriptions.json:
    [
      {"regions":[{"image_id":26,"phrase":"..."}, ...], "id":26},
      ...
    ]
    return [(image_id, phrase), ...]
    """
    with open(region_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs: List[Tuple[int, str]] = []
    if not isinstance(data, list):
        return pairs

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
            if phrase:
                pairs.append((image_id, phrase))
    return pairs


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
    ):
        self.sg = sg_dataset
        self.prompt = prompt

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
        sg_idx = self.image2idx[image_id]
        sg_item = self.sg[sg_idx]
        return {
            "id": f"{image_id}_{source}_{idx}",
            "graph_data": sg_item["graph_data"],
            "text": prompt,
            "answer": phrase,
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
    texts: List[str] = []
    answers: List[str] = []
    graphs = []

    for item in batch:
        ids.append(str(item.get("id", "unknown")))
        texts.append(item.get("text", ""))
        answers.append(item.get("answer", item.get("text", "")))
        graphs.append(item["graph_data"])

    batch_graph = GeoBatch.from_data_list(graphs)
    return {"ids": ids, "graph_data": batch_graph, "text": texts, "answer": answers}


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def build_chat_inputs_and_labels(
    tokenizer: Any,
    prompts: List[str],
    answers: List[str],
    device: torch.device,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []

    for prompt, answer in zip(prompts, answers):
        prompt = prompt or ""
        answer = answer or ""

        if use_chat:
            messages_full = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            full_text = tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)

            messages_prompt = [{"role": "user", "content": prompt}]
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
            full = tokenizer(answer, return_tensors="pt", truncation=True, max_length=max_length)
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
        )

        # trainable params (Stage2-B)
        for name, p in self.model.named_parameters():
            if ("vg_adapter" in name) or ("gl_projector" in name) or ("graph_qformer" in name):
                p.requires_grad = True
            else:
                p.requires_grad = False

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
        return AdamW(params, lr=self.hparams.lr)

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        graph_data = batch["graph_data"].to(self.device)
        prompts: List[str] = batch["text"]
        answers: List[str] = batch["answer"]

        input_ids, attention_mask, labels = build_chat_inputs_and_labels(
            tokenizer=self.tokenizer,
            prompts=prompts,
            answers=answers,
            device=self.device,
            max_length=self.max_length,
        )

        outputs = self.model(
            graph_data=graph_data,
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
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

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(prompts))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        graph_data = batch["graph_data"].to(self.device)
        prompts: List[str] = batch["text"]
        answers: List[str] = batch["answer"]

        input_ids, attention_mask, labels = build_chat_inputs_and_labels(
            tokenizer=self.tokenizer,
            prompts=prompts,
            answers=answers,
            device=self.device,
            max_length=self.max_length,
        )

        outputs = self.model(
            graph_data=graph_data,
            pixel_values=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
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

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(prompts))
        return loss


# ---------------------------------------------------------------------------
# Load stage2-A weights only (no optimizer restore)
# ---------------------------------------------------------------------------
def load_stage2A_weights_only(pl_model: Stage2BProjectorPL, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise RuntimeError(
            "Stage2B requires a Stage2A Lightning checkpoint (.ckpt) from the strict pipeline."
        )

    hp = ckpt.get("hyper_parameters", {}) or {}
    stage1_qformer_ckpt = hp.get("stage1_qformer_ckpt")
    if not stage1_qformer_ckpt:
        raise RuntimeError(
            "Stage2A checkpoint metadata missing 'stage1_qformer_ckpt'. "
            "Use Stage2A checkpoint generated with --stage1_qformer_ckpt."
        )

    sd = ckpt["state_dict"]
    # strip "model." prefix -> match OmniGraphModel keys
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = pl_model.model.load_state_dict(sd, strict=False)
    print(f"[Stage2B] loaded weights-only from: {ckpt_path}")
    print(f"[Stage2B] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[Stage2B] Stage2A provenance: stage1_qformer_ckpt={stage1_qformer_ckpt}")
    if missing:
        print("  missing head:", missing[:20])
    if unexpected:
        print("  unexpected head:", unexpected[:20])
    return {
        "stage2A_ckpt": str(ckpt_path),
        "stage1_qformer_ckpt": str(stage1_qformer_ckpt),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_val_check_interval(x: float) -> float | int:
    x = float(x)
    if x >= 1.0:
        return int(x)
    if x <= 0.0:
        return 1.0
    return x


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
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", type=int, default=32)

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
    args = ap.parse_args()

    pl.seed_everything(args.seed, workers=True)
    print("[Pipeline] Stage2B start: requires Stage2A .ckpt with Stage1 provenance metadata.")
    if not Path(args.stage2A_ckpt).exists():
        raise FileNotFoundError(f"--stage2A_ckpt not found: {args.stage2A_ckpt}")

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
    else:
        print("[GraphQA] disabled.")
    full_dataset = VGGraphRegionTextDataset(
        sg_dataset=sg_dataset,
        region_pairs=region_pairs,
        qa_records=qa_records,
        prompt="Describe the region.",
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

    # 4) stage2-B model
    pl_model = Stage2BProjectorPL(
        llm_model_name=args.llm,
        graph_model_name=args.graph_model,
        num_obj=int(num_obj),
        num_attr=int(num_attr),
        stage2A_ckpt=str(args.stage2A_ckpt),
        lr=args.lr,
        max_length=args.max_length,
        auto_resize_token_embeddings=True,
    )

    # 5) init from stage2-A (weights only)
    stage2a_provenance = load_stage2A_weights_only(pl_model, args.stage2A_ckpt)

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
        precision=args.precision,
        max_steps=args.max_steps,
        max_epochs=999999,  # max_steps 控制
        callbacks=[ckpt, lr_monitor, early],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        check_val_every_n_epoch=1,
        val_check_interval=vci,
        limit_val_batches=float(args.limit_val_batches),
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
        "best_ckpt": ckpt.best_model_path or "",
        "export_path": str(out_pt),
    }
    (save_dir / "stage2B_meta.json").write_text(json.dumps(stage_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Saved] weights -> {out_pt}")


if __name__ == "__main__":
    main()
