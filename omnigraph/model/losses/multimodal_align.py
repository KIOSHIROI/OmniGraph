from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(dtype=x.dtype, device=x.device)
    if m.dim() == 1:
        m = m.unsqueeze(0)
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1e-6)
    return (x * m).sum(dim=1) / denom


def build_pooled_features(
    graph_embeds: torch.Tensor | None = None,
    vision_embeds: torch.Tensor | None = None,
    text_embeds: torch.Tensor | None = None,
    text_attention_mask: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    """
    Build pooled + normalized features for alignment losses.
    Returns keys among {"graph", "vision", "text"} depending on availability.
    """
    out: Dict[str, torch.Tensor] = {}
    if graph_embeds is not None and graph_embeds.numel() > 0:
        out["graph"] = F.normalize(graph_embeds.mean(dim=1), dim=-1)
    if vision_embeds is not None and vision_embeds.numel() > 0:
        out["vision"] = F.normalize(vision_embeds.mean(dim=1), dim=-1)
    if text_embeds is not None and text_embeds.numel() > 0:
        pooled_text = _masked_mean(text_embeds, text_attention_mask)
        out["text"] = F.normalize(pooled_text, dim=-1)
    return out


def xtc_bidirectional_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CLIP-style bidirectional contrastive loss.
    Returns: (loss, similarity_matrix)
    """
    if z_a.dim() != 2 or z_b.dim() != 2:
        raise ValueError("xtc_bidirectional_loss expects 2D pooled features.")
    if z_a.size(0) != z_b.size(0):
        raise ValueError("xtc_bidirectional_loss expects same batch size for both modalities.")

    bsz = int(z_a.size(0))
    sim = z_a @ z_b.t()
    if isinstance(logit_scale, torch.Tensor):
        scale = logit_scale.to(device=sim.device, dtype=sim.dtype)
    else:
        scale = torch.tensor(float(logit_scale), device=sim.device, dtype=sim.dtype)
    sim = sim * scale

    if bsz < 2:
        return sim.new_zeros(()), sim

    labels = torch.arange(bsz, device=sim.device)
    loss_a2b = F.cross_entropy(sim, labels)
    loss_b2a = F.cross_entropy(sim.t(), labels)
    return 0.5 * (loss_a2b + loss_b2a), sim


def mine_hard_negatives(
    sim: torch.Tensor,
    image_ids: Sequence[int] | torch.Tensor,
    qa_types: Sequence[str] | None,
    dup_thresh: float = 0.98,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Hard negative mining with constraints:
      - exclude self pair
      - exclude same image_id
      - exclude near-duplicate text pairs (sim > dup_thresh)
    Falls back to random valid negatives if no hard negative is available.
    Returns:
      neg_idx: (B,)
      hard_valid_mask: (B,)  # whether selected from hard pool (before fallback)
      stats: dict
    """
    if sim.dim() != 2 or sim.size(0) != sim.size(1):
        raise ValueError("mine_hard_negatives expects square similarity matrix.")
    bsz = int(sim.size(0))
    device = sim.device

    if isinstance(image_ids, torch.Tensor):
        image_ids_t = image_ids.to(device=device, dtype=torch.long).view(-1)
    else:
        image_ids_t = torch.tensor([int(x) for x in image_ids], device=device, dtype=torch.long)
    if image_ids_t.numel() != bsz:
        raise ValueError("image_ids length must match sim batch size.")

    if qa_types is None:
        qa_types = ["unknown"] * bsz
    if len(qa_types) != bsz:
        raise ValueError("qa_types length must match sim batch size.")

    neg_idx = torch.arange(bsz, device=device, dtype=torch.long)
    hard_valid_mask = torch.zeros(bsz, device=device, dtype=torch.bool)

    same_image = image_ids_t.unsqueeze(1).eq(image_ids_t.unsqueeze(0))
    near_dup = sim > float(dup_thresh)
    eye = torch.eye(bsz, device=device, dtype=torch.bool)

    # Candidates for hard negative.
    valid_hard = (~eye) & (~same_image) & (~near_dup)
    masked = sim.masked_fill(~valid_hard, -1e9)
    row_max, row_argmax = masked.max(dim=1)
    has_hard = row_max > -1e8

    neg_idx[has_hard] = row_argmax[has_hard]
    hard_valid_mask[has_hard] = True

    # Fallback: random negative excluding self/same-image.
    fallback_rows = (~has_hard).nonzero(as_tuple=True)[0]
    for i in fallback_rows.tolist():
        valid = ((~eye[i]) & (~same_image[i])).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            valid = (~eye[i]).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            neg_idx[i] = i
        else:
            r = torch.randint(0, int(valid.numel()), (1,), device=device).item()
            neg_idx[i] = valid[r]

    neg_qatype_same = 0
    for i in range(bsz):
        j = int(neg_idx[i].item())
        if str(qa_types[i]) == str(qa_types[j]):
            neg_qatype_same += 1

    stats: Dict[str, Any] = {
        "batch_size": bsz,
        "hard_valid_count": int(has_hard.sum().item()),
        "hard_valid_ratio": float(has_hard.float().mean().item()) if bsz > 0 else 0.0,
        "fallback_count": int((~has_hard).sum().item()),
        "same_image_blocked_pairs": int((same_image & (~eye)).sum().item()),
        "near_dup_blocked_pairs": int((near_dup & (~eye)).sum().item()),
        "neg_same_qatype_count": int(neg_qatype_same),
    }
    return neg_idx, hard_valid_mask, stats


def xtm_pair_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Binary matching loss for positive and negative pairs.
    Returns: (loss, accuracy)
    """
    pos = pos_logits.reshape(-1)
    neg = neg_logits.reshape(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        z = pos.new_zeros(())
        return z, z

    logits = torch.cat([pos, neg], dim=0)
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)], dim=0)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    preds = (torch.sigmoid(logits) >= 0.5).to(labels.dtype)
    acc = (preds == labels).float().mean()
    return loss, acc

