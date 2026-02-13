from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from omnigraph.data.graph_canonical import canonicalize_scene_graph


@dataclass
class V2GSynthesizerConfig:
    model_name: str = "Salesforce/blip2-flan-t5-xl"
    torch_dtype: str = "bfloat16"
    max_length: int = 384
    max_new_tokens: int = 256
    freeze_vision: bool = True
    freeze_language: bool = True
    train_qformer: bool = True
    train_language_projection: bool = True
    train_lm_head: bool = False


_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name or "bfloat16").strip().lower()
    return _DTYPE_MAP.get(key, torch.bfloat16)


def _to_json_text(scene_graph: Dict[str, Any]) -> str:
    c = canonicalize_scene_graph(scene_graph)
    # Keep compact JSON to make teacher-forcing target stable.
    return json.dumps(c, ensure_ascii=False, separators=(",", ":"))


class VisionGraphSynthesizer(nn.Module):
    """
    BLIP-style Vision->Graph synthesizer.

    The model consumes image + prompt and autoregressively generates
    canonical VG-style scene graph JSON text.
    """

    def __init__(self, config: V2GSynthesizerConfig):
        super().__init__()
        self.config = config
        self.processor = Blip2Processor.from_pretrained(config.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            config.model_name,
            torch_dtype=_resolve_dtype(config.torch_dtype),
            low_cpu_mem_usage=True,
        )
        self._set_trainable(
            freeze_vision=bool(config.freeze_vision),
            freeze_language=bool(config.freeze_language),
            train_qformer=bool(config.train_qformer),
            train_language_projection=bool(config.train_language_projection),
            train_lm_head=bool(config.train_lm_head),
        )

    def extra_repr(self) -> str:
        return f"model={self.config.model_name} trainable={self.count_trainable_params()}"

    def as_config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def count_trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _set_trainable(
        self,
        *,
        freeze_vision: bool,
        freeze_language: bool,
        train_qformer: bool,
        train_language_projection: bool,
        train_lm_head: bool,
    ) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        if hasattr(self.model, "vision_model"):
            for p in self.model.vision_model.parameters():
                p.requires_grad = not freeze_vision

        if hasattr(self.model, "language_model"):
            for p in self.model.language_model.parameters():
                p.requires_grad = not freeze_language

        if train_qformer and hasattr(self.model, "qformer"):
            for p in self.model.qformer.parameters():
                p.requires_grad = True

        if train_language_projection and hasattr(self.model, "language_projection"):
            for p in self.model.language_projection.parameters():
                p.requires_grad = True

        if train_lm_head and hasattr(self.model, "lm_head"):
            for p in self.model.lm_head.parameters():
                p.requires_grad = True

    @staticmethod
    def build_prompt(caption: Optional[str] = None) -> str:
        c = " ".join(str(caption or "").strip().split())
        if c:
            hint = f"Caption hint: {c}\n"
        else:
            hint = ""
        return (
            "Generate a Visual Genome style scene graph as compact JSON.\n"
            "Schema: {\"objects\":[{\"object_id\":int,\"names\":[str],\"attributes\":[str],\"x\":float,\"y\":float,\"w\":float,\"h\":float}],\n"
            "\"relationships\":[{\"subject_id\":int,\"object_id\":int,\"predicate\":str}],\"width\":int,\"height\":int}.\n"
            "Only output valid JSON object with objects and relationships.\n"
            f"{hint}"
            "Scene graph JSON:"
        )

    @staticmethod
    def graph_to_target_text(scene_graph: Dict[str, Any]) -> str:
        return _to_json_text(scene_graph)

    def _build_model_inputs(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        enc = self.processor(
            images=list(images),
            text=list(prompts),
            padding=True,
            truncation=True,
            max_length=int(max_length or self.config.max_length),
            return_tensors="pt",
        )
        out: Dict[str, torch.Tensor] = {}
        for k, v in enc.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
        return out

    def _build_labels(
        self,
        targets: Sequence[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        tok = self.processor.tokenizer(
            list(targets),
            padding=True,
            truncation=True,
            max_length=int(max_length or self.config.max_length),
            return_tensors="pt",
        )
        labels = tok.input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return labels.to(device)

    def forward_train(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        targets: Sequence[str],
        device: torch.device,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        inputs = self._build_model_inputs(images=images, prompts=prompts, device=device, max_length=max_length)
        labels = self._build_labels(targets=targets, device=device, max_length=max_length)
        out = self.model(**inputs, labels=labels)
        return out.loss

    @torch.no_grad()
    def generate_candidates(
        self,
        images: Sequence[Image.Image],
        prompts: Sequence[str],
        *,
        device: torch.device,
        num_candidates: int = 3,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[List[Dict[str, Any]]]:
        if len(images) != len(prompts):
            raise ValueError("images/prompts length mismatch")

        bsz = len(images)
        if bsz <= 0:
            return []

        inputs = self._build_model_inputs(images=images, prompts=prompts, device=device, max_length=self.config.max_length)
        gen = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens or self.config.max_new_tokens),
            do_sample=bool(do_sample),
            top_p=float(top_p),
            temperature=float(temperature),
            num_return_sequences=max(1, int(num_candidates)),
            return_dict_in_generate=True,
            output_scores=True,
        )

        seqs = gen.sequences
        decoded = self.processor.tokenizer.batch_decode(seqs, skip_special_tokens=True)

        try:
            transition_scores = self.model.compute_transition_scores(
                sequences=gen.sequences,
                scores=gen.scores,
                normalize_logits=True,
            )
        except Exception:
            transition_scores = None

        rows: List[List[Dict[str, Any]]] = [[] for _ in range(bsz)]
        n_ret = max(1, int(num_candidates))
        for i, text in enumerate(decoded):
            bi = i // n_ret
            avg_logprob = -10.0
            if transition_scores is not None and i < int(transition_scores.size(0)):
                ts = transition_scores[i]
                ts = ts[torch.isfinite(ts)]
                if ts.numel() > 0:
                    avg_logprob = float(ts.mean().item())
            rows[bi].append(
                {
                    "text": str(text),
                    "avg_logprob": float(avg_logprob),
                }
            )

        for i in range(len(rows)):
            rows[i] = sorted(rows[i], key=lambda x: x["avg_logprob"], reverse=True)
        return rows


def load_v2g_state_dict(model: VisionGraphSynthesizer, ckpt_path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    info: Dict[str, Any] = {"loaded": 0, "missing": 0, "unexpected": 0, "source": ckpt_path}

    state_dict = None
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        if isinstance(sd, dict):
            # Lightning checkpoint convention.
            state_dict = {}
            for k, v in sd.items():
                kk = str(k)
                if kk.startswith("synthesizer."):
                    state_dict[kk[len("synthesizer.") :]] = v
                else:
                    state_dict[kk] = v
    elif isinstance(ckpt, dict):
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    info["loaded"] = int(len(state_dict))
    info["missing"] = int(len(missing))
    info["unexpected"] = int(len(unexpected))
    info["missing_keys"] = [str(x) for x in missing[:20]]
    info["unexpected_keys"] = [str(x) for x in unexpected[:20]]
    return info
