from __future__ import annotations

import torch
import torch.nn as nn

from omnigraph.model.graph.node_encoder import GraphNodeEncoderBase


class OpenVocabNodeEncoder(GraphNodeEncoderBase):
    """
    Open-vocab graph node encoder.
    Uses hash + geometry + relation aggregation; no closed-vocab object/attribute embedding.
    """

    def __init__(
        self,
        out_dim: int = 128,
        hash_dim: int = 64,
        bbox_dim: int = 64,
        hash_buckets: int = 65536,
        attr_pad_id: int = 0,
    ) -> None:
        super().__init__(out_dim=int(out_dim))
        self.attr_pad_id = int(attr_pad_id)

        self.obj_hash_emb = nn.Embedding(int(hash_buckets), int(hash_dim))
        self.attr_hash_emb = nn.Embedding(int(hash_buckets), int(hash_dim))
        self.rel_hash_emb = nn.Embedding(int(hash_buckets), int(hash_dim))

        self.bbox_mlp = nn.Sequential(
            nn.Linear(8, int(bbox_dim)),
            nn.GELU(),
            nn.Linear(int(bbox_dim), int(bbox_dim)),
            nn.GELU(),
        )

        base_in = int(hash_dim) + int(hash_dim) + int(bbox_dim)
        self.proj = nn.Sequential(
            nn.Linear(base_in, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(int(hash_dim), self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
        )
        self.out_norm = nn.LayerNorm(self.out_dim)

    def _masked_attr_mean(self, attr_embed: torch.Tensor, attr_id: torch.Tensor) -> torch.Tensor:
        mask = (attr_id != self.attr_pad_id).unsqueeze(-1).to(attr_embed.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (attr_embed * mask).sum(dim=1) / denom

    @staticmethod
    def _bbox_features(bbox: torch.Tensor) -> torch.Tensor:
        x = bbox[:, 0:1]
        y = bbox[:, 1:2]
        w = bbox[:, 2:3]
        h = bbox[:, 3:4]
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        area = (w * h).clamp_min(0.0)
        aspect = w / h.clamp_min(1e-6)
        return torch.cat([x, y, w, h, cx, cy, area, aspect], dim=-1)

    def _aggregate_rel(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor | None,
        edge_pred_id: torch.Tensor | None,
        edge_pred_hash_id: torch.Tensor | None,
    ) -> torch.Tensor:
        if edge_index is None or edge_index.numel() == 0:
            return torch.zeros_like(node_feat)

        if edge_pred_hash_id is not None:
            rel_idx = edge_pred_hash_id.long().remainder(self.rel_hash_emb.num_embeddings)
            rel_e = self.rel_hash_emb(rel_idx)
        elif edge_pred_id is not None:
            rel_idx = edge_pred_id.long().remainder(self.rel_hash_emb.num_embeddings)
            rel_e = self.rel_hash_emb(rel_idx)
        else:
            return torch.zeros_like(node_feat)

        rel_n = self.rel_proj(rel_e)
        dst = edge_index[1].long()
        n = node_feat.size(0)

        out = torch.zeros_like(node_feat)
        deg = torch.zeros(n, 1, device=node_feat.device, dtype=node_feat.dtype)
        out.index_add_(0, dst, rel_n)
        deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=deg.device, dtype=deg.dtype))
        return out / deg.clamp_min(1.0)

    def forward(
        self,
        obj_id: torch.Tensor,
        attr_id: torch.Tensor,
        bbox: torch.Tensor,
        obj_hash_id: torch.Tensor | None = None,
        attr_hash_id: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_pred_id: torch.Tensor | None = None,
        edge_pred_hash_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obj_hash_id is None:
            obj_hash_id = obj_id
        if attr_hash_id is None:
            attr_hash_id = attr_id

        obj_hash_idx = obj_hash_id.long().remainder(self.obj_hash_emb.num_embeddings)
        attr_hash_idx = attr_hash_id.long().remainder(self.attr_hash_emb.num_embeddings)

        o_hash = self.obj_hash_emb(obj_hash_idx)
        a_hash = self._masked_attr_mean(self.attr_hash_emb(attr_hash_idx), attr_id)
        b = self.bbox_mlp(self._bbox_features(bbox))

        base = self.proj(torch.cat([o_hash, a_hash, b], dim=-1))
        rel = self._aggregate_rel(base, edge_index=edge_index, edge_pred_id=edge_pred_id, edge_pred_hash_id=edge_pred_hash_id)
        return self.out_norm(base + rel)
