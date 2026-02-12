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
    device: torch.device,
    max_length: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    use_chat = hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None

    input_ids_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    attn_list: List[torch.Tensor] = []

    for prompt, answer in zip(prompts, answers):
        prompt = str(prompt or "")
        answer = str(answer or "")

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
            # fallback: train only answer tokens
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

