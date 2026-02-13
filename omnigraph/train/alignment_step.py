from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from omnigraph.model.losses.multimodal_align import (
    build_pooled_features,
    mine_hard_negatives,
    xtc_bidirectional_loss,
    xtm_pair_loss,
)


def init_xtm_stats_accum() -> Dict[str, float]:
    return {
        "steps": 0.0,
        "hard_valid_ratio_sum": 0.0,
        "fallback_count_sum": 0.0,
        "same_image_blocked_pairs_sum": 0.0,
        "near_dup_blocked_pairs_sum": 0.0,
    }


def update_xtm_stats_accum(accum: Dict[str, float], stats: Dict[str, Any]) -> None:
    accum["steps"] = float(accum.get("steps", 0.0)) + 1.0
    accum["hard_valid_ratio_sum"] = float(accum.get("hard_valid_ratio_sum", 0.0)) + float(
        stats.get("hard_valid_ratio", 0.0)
    )
    accum["fallback_count_sum"] = float(accum.get("fallback_count_sum", 0.0)) + float(
        stats.get("fallback_count", 0.0)
    )
    accum["same_image_blocked_pairs_sum"] = float(accum.get("same_image_blocked_pairs_sum", 0.0)) + float(
        stats.get("same_image_blocked_pairs", 0.0)
    )
    accum["near_dup_blocked_pairs_sum"] = float(accum.get("near_dup_blocked_pairs_sum", 0.0)) + float(
        stats.get("near_dup_blocked_pairs", 0.0)
    )


def summarize_xtm_stats_accum(accum: Dict[str, float]) -> Dict[str, float]:
    steps_raw = float(accum.get("steps", 0.0))
    if steps_raw <= 0.0:
        return {
            "steps": 0.0,
            "avg_hard_valid_ratio": 0.0,
            "avg_fallback_count": 0.0,
            "avg_same_image_blocked_pairs": 0.0,
            "avg_near_dup_blocked_pairs": 0.0,
        }
    steps = max(1.0, steps_raw)
    return {
        "steps": steps_raw,
        "avg_hard_valid_ratio": float(accum.get("hard_valid_ratio_sum", 0.0) / steps),
        "avg_fallback_count": float(accum.get("fallback_count_sum", 0.0) / steps),
        "avg_same_image_blocked_pairs": float(accum.get("same_image_blocked_pairs_sum", 0.0) / steps),
        "avg_near_dup_blocked_pairs": float(accum.get("near_dup_blocked_pairs_sum", 0.0) / steps),
    }


def compute_base_lm_aux_losses(
    *,
    outputs: Any,
    labels: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      base_loss, lm_loss, aux_loss
    """
    loss = outputs.loss
    if loss is None or torch.isnan(loss) or torch.isinf(loss):
        logits = getattr(outputs, "logits", None)
        if logits is None:
            loss = torch.tensor(0.0, device=device)
        else:
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
                loss = torch.tensor(0.0, device=device)

    base_loss = loss
    aux_loss = getattr(outputs, "aux_loss", None)
    aux_w = float(getattr(outputs, "aux_loss_weight", 0.0))
    if aux_loss is None:
        aux_loss = torch.tensor(0.0, device=device)
        aux_w = 0.0

    lm_loss = base_loss - (aux_loss * aux_w)
    return base_loss, lm_loss, aux_loss


def compute_alignment_losses(
    *,
    outputs: Any,
    device: torch.device,
    batch_size: int,
    image_ids: Sequence[int],
    qa_types: Sequence[str],
    enable_xtc: bool,
    enable_xtm: bool,
    enable_xgv: bool,
    xtc_weight: float,
    xtm_weight: float,
    xgv_weight: float,
    xtm_dup_thresh: float,
    xtc_logit_scale: torch.Tensor,
    update_xtm_stats: Callable[[Dict[str, Any]], None] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    loss_xtc = torch.tensor(0.0, device=device)
    loss_xtm = torch.tensor(0.0, device=device)
    loss_xgv = torch.tensor(0.0, device=device)
    xtm_acc = torch.tensor(0.0, device=device)
    valid_neg_ratio = torch.tensor(0.0, device=device)

    do_align = bool(enable_xtc or enable_xtm or enable_xgv)
    if do_align and int(batch_size) >= 2:
        pooled = build_pooled_features(
            graph_embeds=getattr(outputs, "graph_embeds", None),
            vision_embeds=getattr(outputs, "vision_embeds", None),
            text_embeds=getattr(outputs, "text_embeds", None),
            text_attention_mask=getattr(outputs, "text_attention_mask", None),
        )
        z_graph = pooled.get("graph", None)
        z_vision = pooled.get("vision", None)
        z_text = pooled.get("text", None)

        if z_graph is not None and z_text is not None and z_graph.size(0) >= 2 and z_text.size(0) >= 2:
            if bool(enable_xtc) and float(xtc_weight) > 0.0:
                logit_scale = xtc_logit_scale.exp().clamp(max=100.0)
                loss_xtc, _ = xtc_bidirectional_loss(z_graph, z_text, logit_scale)

            if bool(enable_xtm) and float(xtm_weight) > 0.0:
                text_sim = z_text @ z_text.t()
                neg_idx, hard_valid_mask, stats = mine_hard_negatives(
                    sim=text_sim,
                    image_ids=image_ids if image_ids else list(range(z_text.size(0))),
                    qa_types=qa_types,
                    dup_thresh=float(xtm_dup_thresh),
                )
                if update_xtm_stats is not None:
                    update_xtm_stats(stats)
                valid_neg_ratio = hard_valid_mask.float().mean()
                pos_logits = (z_graph * z_text).sum(dim=-1)
                neg_logits = (z_graph * z_text[neg_idx]).sum(dim=-1)
                loss_xtm, xtm_acc = xtm_pair_loss(pos_logits, neg_logits)

        if bool(enable_xgv) and float(xgv_weight) > 0.0 and z_graph is not None and z_vision is not None:
            if z_graph.size(0) >= 2 and z_vision.size(0) >= 2:
                logit_scale = xtc_logit_scale.exp().clamp(max=100.0)
                loss_xgv, _ = xtc_bidirectional_loss(z_graph, z_vision, logit_scale)

    weighted_alignment_loss = (
        (float(xtc_weight) * loss_xtc)
        + (float(xtm_weight) * loss_xtm)
        + (float(xgv_weight) * loss_xgv)
    )
    metrics = {
        "loss_xtc": loss_xtc.detach(),
        "loss_xtm": loss_xtm.detach(),
        "loss_xgv": loss_xgv.detach(),
        "xtm_acc": xtm_acc.detach(),
        "xtm_valid_neg_ratio": valid_neg_ratio.detach(),
    }
    return weighted_alignment_loss, metrics
