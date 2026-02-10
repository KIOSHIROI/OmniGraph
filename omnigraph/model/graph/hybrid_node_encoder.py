from __future__ import annotations

import math

import torch
import torch.nn as nn

from omnigraph.model.graph.VGNodeAdapter import LegacyVGNodeEncoder
from omnigraph.model.graph.node_encoder import GraphNodeEncoderBase
from omnigraph.model.graph.open_vocab_node_encoder import OpenVocabNodeEncoder


class HybridNodeEncoder(GraphNodeEncoderBase):
    """
    Hybrid encoder = (1 - alpha) * legacy_vg + alpha * open_vocab.
    alpha in [0, 1], default 0.3 (biased to open-vocab generalization).
    """

    def __init__(
        self,
        num_obj: int,
        num_attr: int,
        out_dim: int = 128,
        alpha_init: float = 0.3,
        alpha_trainable: bool = True,
    ) -> None:
        super().__init__(out_dim=int(out_dim))

        self.legacy_encoder = LegacyVGNodeEncoder(
            num_obj=int(num_obj),
            num_attr=int(num_attr),
            out_dim=self.out_dim,
        )
        self.open_encoder = OpenVocabNodeEncoder(
            out_dim=self.out_dim,
        )

        a = min(1.0 - 1e-5, max(1e-5, float(alpha_init)))
        init_logit = math.log(a / (1.0 - a))
        t = torch.tensor([init_logit], dtype=torch.float32)
        if bool(alpha_trainable):
            self.alpha_logit = nn.Parameter(t)
            self.register_buffer("alpha_logit_buffer", torch.zeros(1), persistent=False)
            self.alpha_trainable = True
        else:
            self.register_parameter("alpha_logit", None)
            self.register_buffer("alpha_logit_buffer", t, persistent=True)
            self.alpha_trainable = False

        self.output_norm = nn.LayerNorm(self.out_dim)

    def alpha(self) -> torch.Tensor:
        if self.alpha_trainable:
            return torch.sigmoid(self.alpha_logit)
        return torch.sigmoid(self.alpha_logit_buffer)

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
        legacy_feat = self.legacy_encoder(
            obj_id=obj_id,
            attr_id=attr_id,
            bbox=bbox,
            obj_hash_id=obj_hash_id,
            attr_hash_id=attr_hash_id,
            edge_index=edge_index,
            edge_pred_id=edge_pred_id,
            edge_pred_hash_id=edge_pred_hash_id,
        )
        open_feat = self.open_encoder(
            obj_id=obj_id,
            attr_id=attr_id,
            bbox=bbox,
            obj_hash_id=obj_hash_id,
            attr_hash_id=attr_hash_id,
            edge_index=edge_index,
            edge_pred_id=edge_pred_id,
            edge_pred_hash_id=edge_pred_hash_id,
        )
        alpha = self.alpha().to(device=legacy_feat.device, dtype=legacy_feat.dtype)
        fused = (1.0 - alpha) * legacy_feat + alpha * open_feat
        return self.output_norm(fused)
