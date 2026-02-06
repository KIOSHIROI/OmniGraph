# ============================================================
# Graph Q-Former (BLIP-2 style, Graph-conditioned)
# Safe for long training & paper-level reproducibility
# ============================================================

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel


# ----------------------------
# Config
# ----------------------------
@dataclass
class GraphQFormerConfig:
    num_query_tokens: int = 32
    graph_hidden_dim: int = 768
    qformer_hidden_dim: int = 768  # must match BERT hidden size
    llm_hidden_dim: int = 4096     # reserved for Stage-2
    max_text_len: int = 512        # GTM text length (excluding query tokens)
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1


# ----------------------------
# Model
# ----------------------------
class GraphQFormer(nn.Module):
    """
    Graph Q-Former following BLIP-2 design:
    - Learnable query tokens
    - Query self-attention + cross-attention to graph
    - Supports GTC and GTM
    - Safe for mixed batches (graph / no-graph)
    """

    def __init__(self, config: GraphQFormerConfig):
        super().__init__()
        self.config = config

        # --------------------------------------------------
        # BERT config (BLIP-2 style)
        # --------------------------------------------------
        bert_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_config.add_cross_attention = True
        bert_config.is_decoder = True
        bert_config.hidden_dropout_prob = config.dropout
        bert_config.attention_probs_dropout_prob = config.dropout

        target_max_pos = config.max_text_len + config.num_query_tokens
        if target_max_pos != bert_config.max_position_embeddings:
            bert_config.max_position_embeddings = target_max_pos

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased",
            config=bert_config,
            ignore_mismatched_sizes=True
        )

        if self.bert.config.max_position_embeddings != target_max_pos:
            base_bert = BertModel.from_pretrained("bert-base-uncased")
            old_pos_emb = base_bert.embeddings.position_embeddings.weight.data
            old_len = old_pos_emb.size(0)

            self.bert.resize_position_embeddings(target_max_pos)

            new_pos_emb = F.interpolate(
                old_pos_emb.unsqueeze(0).transpose(1, 2),
                size=target_max_pos,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).squeeze(0)
            self.bert.embeddings.position_embeddings.weight.data = new_pos_emb
            self.bert.embeddings.position_ids.data = torch.arange(target_max_pos).unsqueeze(0)

        hidden_size = bert_config.hidden_size
        assert hidden_size == config.qformer_hidden_dim, \
            "qformer_hidden_dim must equal BERT hidden size (768)."

        # --------------------------------------------------
        # Learnable Query Tokens
        # --------------------------------------------------
        self.query_tokens = nn.Parameter(
            torch.randn(1, config.num_query_tokens, hidden_size)
        )
        self.query_tokens.data.normal_(
            mean=0.0, std=bert_config.initializer_range
        )

        # --------------------------------------------------
        # Graph projection
        # --------------------------------------------------
        if config.graph_hidden_dim != hidden_size:
            self.graph_proj = nn.Linear(
                config.graph_hidden_dim, hidden_size
            )
        else:
            self.graph_proj = nn.Identity()

        # --------------------------------------------------
        # ITM / GTM Head
        # --------------------------------------------------
        self.itm_head = nn.Linear(hidden_size, 2)

    # --------------------------------------------------
    # Utility: build dummy graph (NO silent failure)
    # --------------------------------------------------
    def _build_dummy_graph(
        self,
        batch_size: int,
        device: torch.device,
    ):
        """
        Used when a batch has no graph.
        This is NOT a hack: BLIP-2 does the same for no-image cases.
        """
        graph_hidden_states = torch.zeros(
            batch_size,
            1,  # one dummy node
            self.config.graph_hidden_dim,
            device=device,
        )
        graph_attention_mask = torch.ones(
            batch_size,
            1,
            device=device,
            dtype=torch.long,
        )
        return graph_hidden_states, graph_attention_mask

    # --------------------------------------------------
    # Graph-only forward (GTC)
    # --------------------------------------------------
    def forward(
        self,
        graph_hidden_states: torch.Tensor | None,
        graph_attention_mask: torch.Tensor | None,
    ):
        """
        Returns query-level graph-conditioned representations.

        graph_hidden_states: (B, Lg, Dg) or None
        graph_attention_mask: (B, Lg) or None
        """

        if graph_hidden_states is None:
            raise ValueError(
                "GraphQFormer.forward() is GTC-only and requires graph input. "
                "Use forward_match or build dummy graph explicitly."
            )

        B = graph_hidden_states.size(0)

        # Expand queries
        query_embeds = self.query_tokens.expand(B, -1, -1)

        # Project graph
        encoder_hidden_states = self.graph_proj(graph_hidden_states)

        # Query mask
        query_attention_mask = torch.ones(
            query_embeds.size()[:-1],
            device=query_embeds.device,
            dtype=torch.long,
        )

        outputs = self.bert(
            inputs_embeds=query_embeds,
            attention_mask=query_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=graph_attention_mask,
            return_dict=True,
        )

        return outputs.last_hidden_state

    # --------------------------------------------------
    # Graph-Text Matching (GTM)
    # --------------------------------------------------
    def forward_match(
        self,
        graph_hidden_states: torch.Tensor | None,
        graph_attention_mask: torch.Tensor | None,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ):
        """
        graph_hidden_states: (B, Lg, Dg) or None
        graph_attention_mask: (B, Lg) or None
        text_input_ids: (B, Lt)
        text_attention_mask: (B, Lt)
        """

        device = text_input_ids.device
        B = text_input_ids.size(0)

        # --------------------------------------------------
        # Graph safety (explicit, reproducible)
        # --------------------------------------------------
        if graph_hidden_states is None:
            graph_hidden_states, graph_attention_mask = \
                self._build_dummy_graph(B, device)
        else:
            assert graph_hidden_states.size(0) == B, \
                "Batch size mismatch between graph and text"

        # --------------------------------------------------
        # Text embeddings
        # --------------------------------------------------
        text_embeds = self.bert.embeddings(
            input_ids=text_input_ids
        )

        # --------------------------------------------------
        # Query embeddings
        # --------------------------------------------------
        query_embeds = self.query_tokens.expand(B, -1, -1)

        # --------------------------------------------------
        # Concat: [Query | Text]
        # --------------------------------------------------
        inputs_embeds = torch.cat(
            [query_embeds, text_embeds], dim=1
        )

        query_mask = torch.ones(
            B,
            self.config.num_query_tokens,
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.cat(
            [query_mask, text_attention_mask], dim=1
        )

        # --------------------------------------------------
        # Graph projection
        # --------------------------------------------------
        encoder_hidden_states = self.graph_proj(graph_hidden_states)

        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=graph_attention_mask,
            return_dict=True,
        )

        # --------------------------------------------------
        # Text CLS (first text token)
        # --------------------------------------------------
        text_cls_index = self.config.num_query_tokens
        text_cls_embed = outputs.last_hidden_state[
            :, text_cls_index, :
        ]

        logits = self.itm_head(text_cls_embed)
        return logits
