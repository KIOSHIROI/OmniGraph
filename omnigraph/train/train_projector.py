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

# Model
from omnigraph.model.OmniGraphModel import OmniGraphModel  # noqa: E402


# ---------------------------------------------------------------------------
# Regions: loader + dataset wrapper
# ---------------------------------------------------------------------------

def load_region_pairs(region_path: str) -> List[Tuple[int, str]]:
    """
    region_descriptions.json format:
    [
      {"regions":[{"image_id":26,"phrase":"..."}, ...], "id":26},
      ...
    ]
    Returns: [(image_id, phrase), ...]
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
    ):
        self.sg = sg_dataset
        self.prompt = prompt

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
        sg_item = self.sg[sg_idx]
        return {
            "id": f"{image_id}_{source}_{idx}",
            "image_id": int(image_id),
            "graph_data": sg_item["graph_data"],
            "text": prompt,
            "answer": phrase,
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
# Upstream checkpoint loading
# ---------------------------------------------------------------------------

def load_stage1_qformer_weights(model: OmniGraphModel, stage1_qformer_ckpt: str) -> Dict[str, Any]:
    ckpt_path = Path(stage1_qformer_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"--stage1_qformer_ckpt not found: {stage1_qformer_ckpt}")

    obj = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid stage1 qformer checkpoint format: {stage1_qformer_ckpt}")

    # Accept both raw graph_qformer state_dict and prefixed variants.
    if any(k.startswith("graph_qformer.") for k in obj.keys()):
        qformer_sd = {k.replace("graph_qformer.", "", 1): v for k, v in obj.items() if k.startswith("graph_qformer.")}
    else:
        qformer_sd = obj

    model_state = model.graph_qformer.state_dict()
    model_keys = set(model_state.keys())
    overlap = sorted(model_keys.intersection(qformer_sd.keys()))
    if not overlap:
        raise RuntimeError(
            "Stage1 checkpoint has no overlapping GraphQFormer keys. "
            "Expected checkpoint from train_graph_qfromer.py output."
        )

    adapted = []
    skipped = []
    filtered_sd = {}
    for k, v in qformer_sd.items():
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
    loaded = len(overlap)
    total = len(model_keys)
    if loaded == 0 or total == 0:
        raise RuntimeError("Failed to load GraphQFormer weights from Stage1 checkpoint.")

    info = {
        "stage1_qformer_ckpt": str(ckpt_path),
        "loaded_keys": loaded,
        "total_model_keys": total,
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "adapted_keys": len(adapted),
        "skipped_shape_mismatch_keys": len(skipped),
    }
    print(
        "[Stage2A] Loaded Stage1 GraphQFormer: "
        f"loaded={loaded}/{total} missing={len(missing)} unexpected={len(unexpected)} "
        f"adapted={len(adapted)} skipped={len(skipped)}"
    )
    if adapted:
        print(f"[Stage2A] Adapted keys (example): {adapted[0][0]} {adapted[0][1]} -> {adapted[0][2]}")
    if skipped:
        print(f"[Stage2A] Skipped mismatched keys (example): {skipped[0][0]} {skipped[0][1]} -> {skipped[0][2]}")
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
        )
        self.stage1_load_info = load_stage1_qformer_weights(self.model, stage1_qformer_ckpt)

        # Stage2-A trainable selection
        self.train_node_encoder = bool(train_node_encoder)
        for name, p in self.model.named_parameters():
            trainable = ("gl_projector" in name) or (
                self.train_node_encoder and (name.startswith("node_encoder.") or ("vg_adapter" in name))
            )
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
        return AdamW(params, lr=float(self.hparams.lr))

    def _forward_and_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
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
            else:
                loss = torch.tensor(0.0, device=self.device)
        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._forward_and_loss(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(batch["text"]))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        loss = self._forward_and_loss(batch)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=len(batch["text"]))
        return loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_val_check_interval(x: float) -> float | int:
    """
    Lightning accepts:
      - int: validate every N train steps
      - float in (0,1]: fraction of epoch
    """
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
    ap.add_argument(
        "--freeze_vg_adapter",
        action="store_true",
        help="Deprecated compatibility flag. If set, node encoder is frozen in Stage2A.",
    )
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

    train_node_encoder = bool(int(args.train_node_encoder)) and (not bool(args.freeze_vg_adapter))

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
