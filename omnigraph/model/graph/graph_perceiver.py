from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GraphPerceiverConfig:
    num_query_tokens: int = 32
    graph_hidden_dim: int = 128
    qformer_hidden_dim: int = 768
    num_layers: int = 3
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0


class _PerceiverBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        h = int(hidden_size)
        nh = max(1, int(num_heads))
        if h % nh != 0:
            nh = 1

        self.cross_norm_q = nn.LayerNorm(h)
        self.cross_norm_kv = nn.LayerNorm(h)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(dropout),
            batch_first=True,
        )

        self.self_norm = nn.LayerNorm(h)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(dropout),
            batch_first=True,
        )

        ff_hidden = max(h, int(h * max(1, int(ff_mult))))
        self.ffn_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, ff_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_hidden, h),
            nn.Dropout(float(dropout)),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.cross_norm_q(latents)
        kv = self.cross_norm_kv(context)
        cross_out, _ = self.cross_attn(
            q,
            kv,
            kv,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )
        latents = latents + cross_out

        s = self.self_norm(latents)
        self_out, _ = self.self_attn(s, s, s, need_weights=False)
        latents = latents + self_out

        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class GraphPerceiverResampler(nn.Module):
    """
    Perceiver-style graph token resampler.
    Input:
      graph_hidden_states: (B, N, Dg)
      graph_attention_mask: (B, N), 1=valid, 0=pad
    Output:
      latent tokens: (B, K, D)
    """

    def __init__(self, config: GraphPerceiverConfig):
        super().__init__()
        self.config = config

        h = int(config.qformer_hidden_dim)
        self.num_query_tokens = int(config.num_query_tokens)
        self.latents = nn.Parameter(torch.randn(1, self.num_query_tokens, h) * 0.02)

        if int(config.graph_hidden_dim) != h:
            self.graph_proj = nn.Linear(int(config.graph_hidden_dim), h)
        else:
            self.graph_proj = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                _PerceiverBlock(
                    hidden_size=h,
                    num_heads=int(config.num_heads),
                    ff_mult=int(config.ff_mult),
                    dropout=float(config.dropout),
                )
                for _ in range(max(1, int(config.num_layers)))
            ]
        )
        self.out_norm = nn.LayerNorm(h)

    def forward(
        self,
        graph_hidden_states: torch.Tensor | None,
        graph_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if graph_hidden_states is None:
            raise ValueError("GraphPerceiverResampler.forward requires graph_hidden_states.")

        x = self.graph_proj(graph_hidden_states)
        bsz = int(x.size(0))
        latents = self.latents.expand(bsz, -1, -1)

        key_padding_mask = None
        if graph_attention_mask is not None:
            key_padding_mask = (graph_attention_mask <= 0)

        for blk in self.blocks:
            latents = blk(latents, x, context_key_padding_mask=key_padding_mask)
        return self.out_norm(latents)
