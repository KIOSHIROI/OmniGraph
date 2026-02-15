from __future__ import annotations

import json
from typing import Any, List, Sequence, Tuple

import torch


_BINARY_QA_TYPES = {
    "verify_exist",
    "verify_rel",
    "logical_and",
    "logical_or",
    "compare_count",
}


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


def format_prompt_with_qa_type(prompt: str, qa_type: str, enabled: bool) -> str:
    if not bool(enabled):
        return str(prompt)
    t = str(qa_type).strip().lower().replace(" ", "_")
    if not t:
        t = "unknown"
    return f"[QA_TYPE={t}] {str(prompt)}"


def build_binary_aux_targets(
    qa_types: Sequence[str],
    answers: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.full((len(qa_types),), -100, dtype=torch.long, device=device)
    mask = torch.zeros((len(qa_types),), dtype=torch.float32, device=device)

    for i, (qt, ans) in enumerate(zip(qa_types, answers)):
        q = str(qt).strip().lower()
        if q not in _BINARY_QA_TYPES:
            continue
        a = str(ans).strip().lower()
        if a in {"yes", "true", "1"}:
            labels[i] = 1
            mask[i] = 1.0
        elif a in {"no", "false", "0"}:
            labels[i] = 0
            mask[i] = 1.0
    return labels, mask


def build_chat_inputs_and_labels(
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    device: torch.device | str | None,
    max_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None
    prompts = [str(p or "") for p in prompts]
    answers = [str(a or "") for a in answers]

    if use_chat:
        full_texts: List[str] = []
        prompt_texts: List[str] = []
        for prompt, answer in zip(prompts, answers):
            messages_full = [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
            messages_prompt = [{"role": "user", "content": prompt}]
            full_texts.append(tokenizer.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False))
            prompt_texts.append(tokenizer.apply_chat_template(messages_prompt, tokenize=False, add_generation_prompt=True))

        full = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        prompt_tok = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        input_ids = full["input_ids"]
        attention_mask = full.get("attention_mask", torch.ones_like(input_ids))
        labels = input_ids.clone()

        prompt_attn = prompt_tok.get("attention_mask", None)
        if prompt_attn is None:
            prompt_lens = torch.sum(prompt_tok["input_ids"] != tokenizer.pad_token_id, dim=1)
        else:
            prompt_lens = prompt_attn.sum(dim=1)

        seq_len = int(labels.size(1))
        for i in range(int(labels.size(0))):
            n = int(prompt_lens[i].item())
            if n > seq_len:
                n = seq_len
            if n > 0:
                labels[i, :n] = -100
    else:
        # fallback: train only answer tokens
        full = tokenizer(
            answers,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full["input_ids"]
        attention_mask = full.get("attention_mask", torch.ones_like(input_ids))
        labels = input_ids.clone()

    if device is None:
        return input_ids, attention_mask, labels
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def parse_val_check_interval(x: float) -> float | int:
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


def parse_precision(v: str) -> int | str:
    s = str(v).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return s
