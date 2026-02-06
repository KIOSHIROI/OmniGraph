# train_projector.py
from __future__ import annotations

import sys
import json
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Repo bootstrap (确保本地包可导入) + env (必须在 transformers 导入前)
# ---------------------------------------------------------------------------

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env  # noqa: E402

setup_env()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import pytorch_lightning as pl  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from pytorch_lightning.callbacks import (  # noqa: E402
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)

# Dataset
from omnigraph.data.vg_scene_graph_dataset import (  # noqa: E402
    build_vg_vocabs_from_file,
    VGSceneGraphDataset,
)

# Model
import omnigraph.model.OmniGraphModel as omnigraph_model_module  # noqa: E402
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402


# ---------------------------------------------------------------------------
# Regions: loader + dataset wrapper
# ---------------------------------------------------------------------------

def load_region_pairs(region_path: str) -> List[Tuple[int, str]]:
    """
    匹配你给的格式：
    [
      {"regions":[{"image_id":26,"phrase":"..."}, ...], "id":26},
      ...
    ]
    返回: [(image_id, phrase), ...]
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
        try:
            image_id = int(image_id)
        except Exception:
            continue

        regions = item.get("regions", [])
        if not isinstance(regions, list):
            continue

        for r in regions:
            if not isinstance(r, dict):
                continue
            phrase = r.get("phrase", "")
            phrase = str(phrase).strip() if phrase is not None else ""
            if phrase:
                pairs.append((image_id, phrase))

    return pairs


class VGGraphRegionTextDataset(torch.utils.data.Dataset):
    """
    一个样本 = (image_id 对应的 scene graph) + (该 image_id 下的一条 region phrase)
    输出: {id, graph_data, text, answer}
    """

    def __init__(
        self,
        sg_dataset: VGSceneGraphDataset,
        region_pairs: List[Tuple[int, str]],
        prompt: str = "Describe the region.",
        max_samples: int = 0,
        sample_ratio: float = 1.0,
    ):
        self.sg = sg_dataset
        self.prompt = prompt

        # 建 image_id -> sg_dataset index
        self.image2idx: Dict[int, int] = {}

        raw_items = None
        if hasattr(self.sg, "items"):
            raw_items = getattr(self.sg, "items")
        elif hasattr(self.sg, "scene_graphs"):
            raw_items = getattr(self.sg, "scene_graphs")

        if raw_items is None:
            raise AttributeError(
                "VGSceneGraphDataset must expose raw scene graph list as .items or .scene_graphs for image_id join."
            )

        for i, it in enumerate(raw_items):
            if isinstance(it, dict) and "image_id" in it:
                try:
                    self.image2idx[int(it["image_id"])] = i
                except Exception:
                    continue

        # 过滤没 scene graph 的 region
        samples: List[Tuple[int, str]] = [(iid, phr) for (iid, phr) in region_pairs if iid in self.image2idx]

        # 可选：ratio / cap（用于把“epoch”变小、训练更可控）
        if sample_ratio < 1.0:
            keep = max(1, int(len(samples) * float(sample_ratio)))
            samples = samples[:keep]
        if max_samples and int(max_samples) > 0:
            samples = samples[: int(max_samples)]

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, phrase = self.samples[idx]
        sg_idx = self.image2idx[image_id]
        sg_item = self.sg[sg_idx]  # 复用建图逻辑，拿到 graph_data

        # 注意：这里 id 用 idx 只是为了唯一性；真实 image_id 已包含
        return {
            "id": f"{image_id}_r{idx}",
            "image_id": int(image_id),
            "graph_data": sg_item["graph_data"],
            "text": self.prompt,
            "answer": phrase,
        }


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
# Tokenization: build chat inputs and labels
# ---------------------------------------------------------------------------

def build_chat_inputs_and_labels(
    tokenizer: Any,
    prompts: List[str],
    answers: List[str],
    device: torch.device,
    max_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []

    for prompt, answer in zip(prompts, answers):
        prompt = prompt or ""
        answer = answer or ""

        if use_chat:
            messages_full = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )

            messages_prompt = [{"role": "user", "content": prompt}]
            prompt_text = tokenizer.apply_chat_template(
                messages_prompt,
                tokenize=False,
                add_generation_prompt=True,
            )

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
# LightningModule
# ---------------------------------------------------------------------------

class ProjectorPL(pl.LightningModule):
    """
    Stage2 训练封装：
      - trainable: vg_adapter + gl_projector (+ graph_qformer 可选)
      - frozen: LLM + graphgpt (+ vision)
    """

    def __init__(
        self,
        llm_model_name: str,
        graph_model_name: str,
        lr: float,
        max_length: int,
        enable_vision: bool,
        auto_resize_token_embeddings: bool,
        freeze_graph_qformer: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = OmniGraphModel(
            graph_model_name=graph_model_name,
            llm_model_name=llm_model_name,
            enable_vision=enable_vision,
        )

        # Trainable selection
        for name, p in self.model.named_parameters():
            trainable = ("vg_adapter" in name) or ("gl_projector" in name)
            if not freeze_graph_qformer:
                trainable = trainable or ("graph_qformer" in name)
            p.requires_grad = bool(trainable)

        # LLM: checkpointing + no cache
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

    def _forward_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
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

        # fallback: recompute in fp32 (防 bf16/数值异常)
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            logits = getattr(outputs, "logits", None)
            if logits is None:
                return torch.tensor(0.0, device=self.device)

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

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._forward_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._forward_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        return loss


# ---------------------------------------------------------------------------
# Helpers: split + stage runner
# ---------------------------------------------------------------------------

def build_train_val_loaders_by_image_id(
    dataset: VGGraphRegionTextDataset,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    以 image_id 为单位切分，避免同一张图同时在 train/val 造成泄漏。
    """
    assert 0.0 < val_ratio < 1.0

    image_ids = [iid for (iid, _) in dataset.samples]
    uniq = sorted(set(image_ids))

    rng = random.Random(seed)
    rng.shuffle(uniq)

    val_n = max(1, int(len(uniq) * float(val_ratio)))
    val_images = set(uniq[:val_n])

    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, (iid, _) in enumerate(dataset.samples):
        (val_idx if iid in val_images else train_idx).append(i)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_graph_text,
    )

    print(
        f"[Split] train={len(train_set)} val={len(val_set)} "
        f"unique_images={len(uniq)} val_images={len(val_images)}",
        flush=True,
    )
    return train_loader, val_loader


def run_stage(
    stage_name: str,
    llm_model_name: str,
    graph_model_name: str,
    lr: float,
    max_length: int,
    precision: int,
    freeze_graph_qformer: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str,
    gpu: int,
    patience: int,
    min_delta: float,
    max_steps: int,
    limit_train_batches: float,
    init_ckpt: Optional[str] = None,
) -> str:
    """
    返回 best checkpoint path（Lightning ckpt）。
    """
    stage_dir = Path(save_dir) / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    # build / optionally load from ckpt
    if init_ckpt:
        pl_model = ProjectorPL.load_from_checkpoint(
            init_ckpt,
            llm_model_name=llm_model_name,
            graph_model_name=graph_model_name,
            lr=lr,
            max_length=max_length,
            enable_vision=False,
            auto_resize_token_embeddings=True,
            freeze_graph_qformer=freeze_graph_qformer,
        )
    else:
        pl_model = ProjectorPL(
            llm_model_name=llm_model_name,
            graph_model_name=graph_model_name,
            lr=lr,
            max_length=max_length,
            enable_vision=False,
            auto_resize_token_embeddings=True,
            freeze_graph_qformer=freeze_graph_qformer,
        )

    ckpt = ModelCheckpoint(
        dirpath=str(stage_dir),
        filename=f"{stage_name}" + "-{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(patience),
        min_delta=float(min_delta),
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    use_gpu = torch.cuda.is_available() and gpu >= 0
    if use_gpu:
        if int(gpu) >= torch.cuda.device_count():
            raise ValueError(f"--gpu={gpu} is out of range (device_count={torch.cuda.device_count()}).")
        torch.cuda.set_device(int(gpu))
        devices = [int(gpu)]
    else:
        devices = 1

    trainer = pl.Trainer(
        max_epochs=999999,  # 让 early stop / max_steps 接管停止
        max_steps=(int(max_steps) if int(max_steps) > 0 else -1),
        accelerator="gpu" if use_gpu else "cpu",
        devices=devices,
        precision=int(precision),
        callbacks=[ckpt, early, lr_monitor],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        limit_train_batches=float(limit_train_batches),
        enable_checkpointing=True,
    )

    print(
        f"[{stage_name}] lr={lr} freeze_graph_qformer={freeze_graph_qformer} "
        f"max_steps={max_steps} patience={patience} min_delta={min_delta}",
        flush=True,
    )
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = ckpt.best_model_path
    if not best:
        # fallback: last.ckpt
        last = stage_dir / "last.ckpt"
        best = str(last) if last.exists() else ""
    print(f"[{stage_name}] best_ckpt={best}", flush=True)
    return best


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    # data
    ap.add_argument("--scene_graphs", type=str, default="data/vg/contents/sceneGraphs/scene_graphs.json")
    ap.add_argument("--regions", type=str, required=True, help="VG region descriptions json")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_nodes", type=int, default=80)
    ap.add_argument("--max_attrs", type=int, default=6)

    # sampling (控制 epoch 大小)
    ap.add_argument("--max_samples", type=int, default=0, help="cap #region samples (0 = no cap)")
    ap.add_argument("--sample_ratio", type=float, default=1.0, help="keep ratio of region samples (<=1.0)")

    # split / early stop
    ap.add_argument("--val_ratio", type=float, default=0.001, help="split by image_id; e.g. 0.001 = 0.1% images")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--min_delta", type=float, default=0.01)

    # model / train
    ap.add_argument("--llm", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--graph_model", type=str, default="clip_gt_arxiv_pub")
    ap.add_argument("--gpu", type=int, default=0, help="GPU index, e.g. 1 means cuda:1; use -1 for CPU")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--precision", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--save_dir", type=str, default="checkpoints_projector")
    ap.add_argument("--limit_train_batches", type=float, default=1.0)

    # staged training
    ap.add_argument("--stage", type=str, default="both", choices=["A", "B", "both"])
    ap.add_argument("--lr_A", type=float, default=5e-5)
    ap.add_argument("--lr_B", type=float, default=2e-5)
    ap.add_argument("--max_steps_A", type=int, default=-1, help="hard cap optimizer steps for stage A (-1 no cap)")
    ap.add_argument("--max_steps_B", type=int, default=-1, help="hard cap optimizer steps for stage B (-1 no cap)")

    args = ap.parse_args()
    pl.seed_everything(int(args.seed), workers=True)

    # 1) build vocabs -> NUM_OBJ/NUM_ATTR (OmniGraphModel 依赖全局常量)
    obj_vocab, pred_vocab, attr_vocab = build_vg_vocabs_from_file(args.scene_graphs, min_freq=args.min_freq)
    num_obj = len(obj_vocab.stoi)
    num_attr = len(attr_vocab.stoi)

    omnigraph_model_module.NUM_OBJ = num_obj
    omnigraph_model_module.NUM_ATTR = num_attr
    print(f"[Vocab] NUM_OBJ={num_obj} NUM_ATTR={num_attr}", flush=True)

    # 2) scene graph dataset（负责建图）
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

    # 3) region phrase (answer)
    region_pairs = load_region_pairs(args.regions)
    print(f"[Regions] loaded_pairs={len(region_pairs)}", flush=True)

    dataset = VGGraphRegionTextDataset(
        sg_dataset=sg_dataset,
        region_pairs=region_pairs,
        prompt="Describe the region.",
        max_samples=args.max_samples,
        sample_ratio=args.sample_ratio,
    )
    print(f"[Join] usable_pairs={len(dataset)} (after join/cap/ratio)", flush=True)

    if len(dataset) == 0:
        raise ValueError("No usable (image_id, phrase) pairs after join. Check scene_graphs/regions paths.")

    # 4) train/val loaders (by image_id)
    if not (0.0 < float(args.val_ratio) < 1.0):
        raise ValueError("--val_ratio must be in (0,1). Suggested: 0.001 ~ 0.01")
    train_loader, val_loader = build_train_val_loaders_by_image_id(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # 5) run stages
    save_dir = str(Path(args.save_dir))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    bestA = None
    if args.stage in ("A", "both"):
        bestA = run_stage(
            stage_name="stage2A",
            llm_model_name=args.llm,
            graph_model_name=args.graph_model,
            lr=float(args.lr_A),
            max_length=int(args.max_length),
            precision=int(args.precision),
            freeze_graph_qformer=True,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            gpu=int(args.gpu),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            max_steps=int(args.max_steps_A),
            limit_train_batches=float(args.limit_train_batches),
            init_ckpt=None,
        )

    if args.stage in ("B", "both"):
        init = bestA if bestA else None
        bestB = run_stage(
            stage_name="stage2B",
            llm_model_name=args.llm,
            graph_model_name=args.graph_model,
            lr=float(args.lr_B),
            max_length=int(args.max_length),
            precision=int(args.precision),
            freeze_graph_qformer=False,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            gpu=int(args.gpu),
            patience=int(args.patience),
            min_delta=float(args.min_delta),
            max_steps=int(args.max_steps_B),
            limit_train_batches=float(args.limit_train_batches),
            init_ckpt=init,
        )
        print(f"[Done] stage2B_best={bestB}", flush=True)
    else:
        print(f"[Done] stage2A_best={bestA}", flush=True)


if __name__ == "__main__":
    main()