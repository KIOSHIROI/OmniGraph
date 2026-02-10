from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class GraphNodeEncoderBase(nn.Module, ABC):
    """Unified graph node encoder interface."""

    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = int(out_dim)

    @abstractmethod
    def forward(
        self,
        obj_id: torch.Tensor,
        attr_id: torch.Tensor,
        bbox: torch.Tensor,
        obj_hash_id: Optional[torch.Tensor] = None,
        attr_hash_id: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
        edge_pred_id: Optional[torch.Tensor] = None,
        edge_pred_hash_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def build_graph_node_encoder(
    node_encoder_type: str,
    num_obj: int,
    num_attr: int,
    out_dim: int = 128,
    alpha_init: float = 0.3,
    alpha_trainable: bool = True,
) -> GraphNodeEncoderBase:
    t = str(node_encoder_type or "hybrid").strip().lower()

    if t == "legacy_vg":
        from omnigraph.model.graph.VGNodeAdapter import LegacyVGNodeEncoder

        return LegacyVGNodeEncoder(
            num_obj=int(num_obj),
            num_attr=int(num_attr),
            out_dim=int(out_dim),
        )

    if t == "open_vocab":
        from omnigraph.model.graph.open_vocab_node_encoder import OpenVocabNodeEncoder

        return OpenVocabNodeEncoder(
            out_dim=int(out_dim),
        )

    if t == "hybrid":
        from omnigraph.model.graph.hybrid_node_encoder import HybridNodeEncoder

        return HybridNodeEncoder(
            num_obj=int(num_obj),
            num_attr=int(num_attr),
            out_dim=int(out_dim),
            alpha_init=float(alpha_init),
            alpha_trainable=bool(alpha_trainable),
        )

    raise ValueError(f"Unsupported node encoder type: {node_encoder_type}")
