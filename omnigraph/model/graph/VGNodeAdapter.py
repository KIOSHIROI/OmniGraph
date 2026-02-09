import torch
import torch.nn as nn


class VGNodeAdapter(nn.Module):
    """
    Generalization-oriented adapter for VG nodes.
    - Keeps vocab embeddings for in-domain fidelity.
    - Adds hash embeddings for OOV robustness.
    - Uses masked attribute pooling (ignores pad attrs).
    - Injects relation predicates via edge aggregation.
    """

    def __init__(
        self,
        num_obj: int,
        num_attr: int,
        out_dim: int = 128,
        obj_dim: int = 80,
        attr_dim: int = 56,
        hash_dim: int = 32,
        bbox_dim: int = 48,
        hash_buckets: int = 65536,
        attr_pad_id: int = 0,
    ):
        super().__init__()
        self.attr_pad_id = int(attr_pad_id)

        self.obj_emb = nn.Embedding(num_obj, obj_dim)
        self.attr_emb = nn.Embedding(num_attr, attr_dim)

        # Hash channels reduce hard dependency on closed VG vocab ids.
        self.obj_hash_emb = nn.Embedding(hash_buckets, hash_dim)
        self.attr_hash_emb = nn.Embedding(hash_buckets, hash_dim)
        self.rel_hash_emb = nn.Embedding(hash_buckets, hash_dim)

        self.bbox_mlp = nn.Sequential(
            nn.Linear(8, bbox_dim),
            nn.GELU(),
            nn.Linear(bbox_dim, bbox_dim),
            nn.GELU(),
        )

        base_in = obj_dim + attr_dim + hash_dim + hash_dim + bbox_dim
        self.proj = nn.Sequential(
            nn.Linear(base_in, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.rel_proj = nn.Sequential(
            nn.Linear(hash_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)

    def _masked_attr_mean(self, attr_embed: torch.Tensor, attr_id: torch.Tensor) -> torch.Tensor:
        # attr_embed: [N, A, D], attr_id: [N, A]
        mask = (attr_id != self.attr_pad_id).unsqueeze(-1).to(attr_embed.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (attr_embed * mask).sum(dim=1) / denom

    def _bbox_features(self, bbox: torch.Tensor) -> torch.Tensor:
        # bbox: [N,4] normalized [x,y,w,h]
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

        rel_n = self.rel_proj(rel_e)  # [E, out_dim]
        src = edge_index[0].long()
        dst = edge_index[1].long()
        n = node_feat.size(0)

        out = torch.zeros_like(node_feat)
        deg = torch.zeros(n, 1, device=node_feat.device, dtype=node_feat.dtype)
        out.index_add_(0, dst, rel_n)
        deg.index_add_(0, dst, torch.ones(dst.size(0), 1, device=deg.device, dtype=deg.dtype))
        out = out / deg.clamp_min(1.0)
        return out

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
        # obj_id: [N], attr_id: [N,A], bbox:[N,4]
        o_vocab = self.obj_emb(obj_id)  # [N, Do]
        a_vocab = self._masked_attr_mean(self.attr_emb(attr_id), attr_id)  # [N, Da]

        if obj_hash_id is None:
            obj_hash_id = obj_id
        if attr_hash_id is None:
            attr_hash_id = attr_id

        obj_hash_idx = obj_hash_id.long().remainder(self.obj_hash_emb.num_embeddings)
        attr_hash_idx = attr_hash_id.long().remainder(self.attr_hash_emb.num_embeddings)
        o_hash = self.obj_hash_emb(obj_hash_idx)  # [N, Dh]
        a_hash = self._masked_attr_mean(self.attr_hash_emb(attr_hash_idx), attr_id)  # [N, Dh]

        b = self.bbox_mlp(self._bbox_features(bbox))  # [N, Db]
        base = self.proj(torch.cat([o_vocab, a_vocab, o_hash, a_hash, b], dim=-1))  # [N, D]
        rel = self._aggregate_rel(base, edge_index=edge_index, edge_pred_id=edge_pred_id, edge_pred_hash_id=edge_pred_hash_id)
        return self.out_norm(base + rel)
