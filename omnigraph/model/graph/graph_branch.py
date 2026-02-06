from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


class GraphBranch(nn.Module):
    """
    Graph branch wrapper:
    - GraphCLIP-GT encoder (frozen) -> node embeddings
    - Graph Q-Former -> query tokens
    """

    def __init__(self, graphgpt_encoder: nn.Module, qformer: nn.Module) -> None:
        super().__init__()
        self.graphgpt_encoder = graphgpt_encoder
        self.qformer = qformer

        # Graph encoder is frozen; keep in eval to avoid dropout noise.
        self.graphgpt_encoder.eval()
        for p in self.graphgpt_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        graph_data: Any,
        return_debug: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
        if graph_data is None:
            raise ValueError("GraphBranch requires graph_data, got None.")

        device = next(self.qformer.parameters()).device
        if hasattr(graph_data, "to"):
            graph_data = graph_data.to(device)

        graph_node_reps = self.graphgpt_encoder(graph_data)

        # Build dense batch for Q-Former
        if hasattr(graph_data, "batch") and graph_data.batch is not None:
            batch_idx = graph_data.batch
            batch_size = getattr(graph_data, "num_graphs", None)
            graph_hidden_states, graph_attention_mask = to_dense_batch(
                graph_node_reps,
                batch_idx,
                batch_size=batch_size,
            )

            # Guard: empty graphs -> enable a dummy node
            if graph_attention_mask.sum(dim=1).eq(0).any():
                zero_mask_indices = (
                    graph_attention_mask.sum(dim=1) == 0
                ).nonzero(as_tuple=True)[0]
                graph_attention_mask[zero_mask_indices, 0] = 1
        else:
            graph_hidden_states = graph_node_reps.unsqueeze(0)
            graph_attention_mask = torch.ones(
                graph_hidden_states.shape[:-1],
                device=graph_hidden_states.device,
                dtype=torch.long,
            )

        graph_hidden_states = graph_hidden_states.to(device)
        graph_attention_mask = graph_attention_mask.to(device).long()

        query_tokens = self.qformer(
            graph_hidden_states=graph_hidden_states,
            graph_attention_mask=graph_attention_mask,
        )

        if return_debug:
            debug: Dict[str, Any] = {
                "graph_node_reps_nan": bool(torch.isnan(graph_node_reps).any().item()),
                "graph_node_reps_inf": bool(torch.isinf(graph_node_reps).any().item()),
                "graph_hidden_nan": bool(torch.isnan(graph_hidden_states).any().item()),
                "graph_hidden_inf": bool(torch.isinf(graph_hidden_states).any().item()),
                "graph_mask_sum_min": int(
                    graph_attention_mask.sum(dim=1).min().item()
                )
                if graph_attention_mask.numel() > 0
                else 0,
            }
            return query_tokens, debug

        return query_tokens