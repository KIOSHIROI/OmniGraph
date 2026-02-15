from omnigraph.utils.env import setup_env

setup_env()
import math
import torch
import torch.nn as nn
from omnigraph.model.graph.node_encoder import build_graph_node_encoder
from omnigraph.model.graph.builder import build_graph_tower
from omnigraph.model.graph.graph_branch import GraphBranch
from omnigraph.model.graph.graph_qformer import GraphQFormer, GraphQFormerConfig
from omnigraph.model.graph.graph_perceiver import GraphPerceiverConfig, GraphPerceiverResampler
from omnigraph.model.vision.blip2_vision_qformer import BLIP2VisionQFormer
from omnigraph.model.projector import Projector
from omnigraph.model.llm.llama import LlamaLLM


class GraphLanguageCrossAttentionAdapter(nn.Module):
    """
    Lightweight graph/vision -> language adapter.
    Query: text embeddings
    Key/Value: multimodal prefix embeddings (graph + vision)
    """

    def __init__(self, hidden_size: int, num_heads: int = 8, gate_init: float = 0.1, dropout: float = 0.0) -> None:
        super().__init__()
        h = int(hidden_size)
        nh = max(1, int(num_heads))
        if h % nh != 0:
            nh = 1
        self.text_norm = nn.LayerNorm(h)
        self.ctx_norm = nn.LayerNorm(h)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h,
            num_heads=nh,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(h)
        self.ffn = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU(),
            nn.Linear(h, h),
        )

        g = min(1.0 - 1e-5, max(1e-5, float(gate_init)))
        self.gate_logit = nn.Parameter(torch.tensor(math.log(g / (1.0 - g)), dtype=torch.float32))

    def forward(self, text_embeds: torch.Tensor, context_embeds: torch.Tensor | None) -> torch.Tensor:
        if context_embeds is None or context_embeds.numel() == 0:
            return text_embeds

        q = self.text_norm(text_embeds)
        kv = self.ctx_norm(context_embeds)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)

        gate = torch.sigmoid(self.gate_logit).to(dtype=text_embeds.dtype, device=text_embeds.device)
        mixed = text_embeds + gate * attn_out
        ffn_out = self.ffn(self.ffn_norm(mixed))
        return mixed + gate * ffn_out


class GraphReasoningAuxHead(nn.Module):
    """
    Auxiliary binary head for graph reasoning types (verify/logical/compare).
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        h = int(hidden_size)
        self.graph_norm = nn.LayerNorm(h)
        self.text_norm = nn.LayerNorm(h)
        self.mlp = nn.Sequential(
            nn.Linear(h * 4, h),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.classifier = nn.Linear(h, 2)

    @staticmethod
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

    def forward(
        self,
        graph_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        g = self.graph_norm(graph_embeds).mean(dim=1)
        t = self.text_norm(text_embeds)
        t = self._masked_mean(t, text_attention_mask)
        fused = torch.cat([g, t, torch.abs(g - t), g * t], dim=-1)
        hidden = self.mlp(fused)
        return self.classifier(hidden)

    @staticmethod
    def loss(
        logits: torch.Tensor,
        labels: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if labels is None or mask is None:
            return None
        valid = (mask > 0) & (labels >= 0)
        if valid.sum().item() <= 0:
            return None
        return nn.functional.cross_entropy(logits[valid], labels[valid])


class OmniGraphModel(nn.Module):
    def __init__(self,
                 graph_model_name="clip_gt_arxiv_pub",
                 vision_model_name="Salesforce/blip2-flan-t5-xl",
                 llm_model_name="Qwen/Qwen2.5-7B-Instruct", # Qwen2.5 7B
                 pretrained_llm=None,
                 pretrained_vision=None,
                 pretrained_graph=None,
                 pretrained_graph_qformer=None,
                 use_blip2_vl_projector: bool = False,
                 blip2_vl_projector_trainable: bool = True,
                 enable_vision: bool = True,
                 num_obj: int | None = None,
                 num_attr: int | None = None,
                 max_graph_tokens: int | None = None,
                 max_vision_tokens: int | None = None,
                 llm_dtype: str = "bfloat16",
                 llm_attn_implementation: str = "sdpa",
                 node_encoder_type: str = "hybrid",
                 node_encoder_out_dim: int = 128,
                 node_encoder_trainable: bool = True,
                 node_encoder_alpha_init: float = 0.3,
                 graph_tokenizer_type: str = "qformer",
                 perceiver_num_latents: int = 32,
                 perceiver_num_layers: int = 3,
                 perceiver_num_heads: int = 8,
                 perceiver_ff_mult: int = 4,
                 perceiver_dropout: float = 0.0,
                 enable_gvl_adapter: bool = True,
                 gvl_adapter_num_heads: int = 8,
                 gvl_adapter_gate_init: float = 0.1,
                 enable_graph_aux_head: bool = True,
                 graph_aux_dropout: float = 0.1,
                 ):
        super().__init__()
        if num_obj is None or num_attr is None:
            raise ValueError("OmniGraphModel requires explicit num_obj and num_attr.")
        if int(num_obj) <= 0 or int(num_attr) <= 0:
            raise ValueError(f"Invalid vocab sizes: num_obj={num_obj}, num_attr={num_attr}")

        self.num_obj = int(num_obj)
        self.num_attr = int(num_attr)
        self.max_graph_tokens = int(max_graph_tokens) if max_graph_tokens is not None and int(max_graph_tokens) > 0 else None
        self.max_vision_tokens = int(max_vision_tokens) if max_vision_tokens is not None and int(max_vision_tokens) > 0 else None
        self.node_encoder_type = str(node_encoder_type).strip().lower() or "hybrid"
        self.node_encoder_trainable = bool(node_encoder_trainable)
        self.node_encoder_alpha_init = float(node_encoder_alpha_init)
        self.graph_tokenizer_type = str(graph_tokenizer_type).strip().lower() or "qformer"
        if self.graph_tokenizer_type not in {"qformer", "perceiver"}:
            raise ValueError(f"Unsupported graph_tokenizer_type: {graph_tokenizer_type}")
        self.perceiver_num_latents = max(1, int(perceiver_num_latents))
        self.perceiver_num_layers = max(1, int(perceiver_num_layers))
        self.perceiver_num_heads = max(1, int(perceiver_num_heads))
        self.perceiver_ff_mult = max(1, int(perceiver_ff_mult))
        self.perceiver_dropout = float(perceiver_dropout)
        self.node_encoder = build_graph_node_encoder(
            node_encoder_type=self.node_encoder_type,
            num_obj=self.num_obj,
            num_attr=self.num_attr,
            out_dim=int(node_encoder_out_dim),
            alpha_init=self.node_encoder_alpha_init,
            alpha_trainable=self.node_encoder_trainable,
        )
        for p in self.node_encoder.parameters():
            p.requires_grad = self.node_encoder_trainable
        print(
            "[NodeEncoder] "
            f"type={self.node_encoder_type} out_dim={self.node_encoder.out_dim} "
            f"trainable={self.node_encoder_trainable} alpha_init={self.node_encoder_alpha_init}"
        )
        self.node_encoder_config = {
            "type": self.node_encoder_type,
            "out_dim": int(self.node_encoder.out_dim),
            "trainable": bool(self.node_encoder_trainable),
            "alpha_init": float(self.node_encoder_alpha_init),
        }
        self.enable_gvl_adapter = bool(enable_gvl_adapter)
        self.enable_graph_aux_head = bool(enable_graph_aux_head)
        # 1. LLM (Frozen, bf16)
        print(f"Loading LLM: {llm_model_name}...")
        self.llm = LlamaLLM(
            model_name=llm_model_name,
            pretrained_model=pretrained_llm,
            dtype=llm_dtype,
            attn_implementation=llm_attn_implementation,
        )
        self.hidden_size = self.llm.config.hidden_size
        if self.enable_gvl_adapter:
            self.gvl_adapter = GraphLanguageCrossAttentionAdapter(
                hidden_size=self.hidden_size,
                num_heads=int(gvl_adapter_num_heads),
                gate_init=float(gvl_adapter_gate_init),
                dropout=0.0,
            )
        else:
            self.gvl_adapter = None

        if self.enable_graph_aux_head:
            self.graph_aux_head = GraphReasoningAuxHead(
                hidden_size=self.hidden_size,
                dropout=float(graph_aux_dropout),
            )
        else:
            self.graph_aux_head = None

        self.architecture_config = {
            "enable_gvl_adapter": bool(self.enable_gvl_adapter),
            "gvl_adapter_num_heads": int(gvl_adapter_num_heads),
            "gvl_adapter_gate_init": float(gvl_adapter_gate_init),
            "enable_graph_aux_head": bool(self.enable_graph_aux_head),
            "graph_aux_dropout": float(graph_aux_dropout),
        }
        
        # 2. Graph Branch
        print("Loading Graph Branch...")
        # GraphCLIP-GT (Frozen)
        if pretrained_graph is not None:
             self.graphgpt = pretrained_graph
        else:
             self.graphgpt = build_graph_tower(graph_model_name, "omnigraph/model/graph/clip_gt_arxiv_pub")
        
        # Freeze GraphCLIP-GT
        for p in self.graphgpt.parameters():
            p.requires_grad = False

        # Graph tokenizer (trainable): GraphQFormer or Perceiver Resampler.
        if pretrained_graph_qformer is not None:
            self.graph_qformer = pretrained_graph_qformer
        else:
            if self.graph_tokenizer_type == "perceiver":
                perceiver_cfg = GraphPerceiverConfig(
                    num_query_tokens=self.perceiver_num_latents,
                    graph_hidden_dim=int(self.node_encoder.out_dim),
                    qformer_hidden_dim=768,
                    num_layers=self.perceiver_num_layers,
                    num_heads=self.perceiver_num_heads,
                    ff_mult=self.perceiver_ff_mult,
                    dropout=self.perceiver_dropout,
                )
                self.graph_qformer = GraphPerceiverResampler(config=perceiver_cfg)
            else:
                graph_qformer_config = GraphQFormerConfig(graph_hidden_dim=int(self.node_encoder.out_dim))
                self.graph_qformer = GraphQFormer(config=graph_qformer_config)
        # Ensure Q-Former is trainable (it should be by default)
        for p in self.graph_qformer.parameters():
            p.requires_grad = True
        
        self.graph_branch = GraphBranch(graphgpt_encoder=self.graphgpt, qformer=self.graph_qformer)
        
        # GL-Projector (Trainable)
        self.gl_projector = Projector(
            input_dim=self.graph_qformer.config.qformer_hidden_dim, 
            output_dim=self.hidden_size,
            type="mlp"
        )
        self.graph_tokenizer_config = {
            "type": self.graph_tokenizer_type,
            "num_latents": int(getattr(self.graph_qformer.config, "num_query_tokens", 32)),
            "hidden_dim": int(getattr(self.graph_qformer.config, "qformer_hidden_dim", 768)),
            "num_layers": int(getattr(self.graph_qformer.config, "num_layers", self.perceiver_num_layers)),
            "num_heads": int(getattr(self.graph_qformer.config, "num_heads", self.perceiver_num_heads)),
            "ff_mult": int(getattr(self.graph_qformer.config, "ff_mult", self.perceiver_ff_mult)),
            "dropout": float(getattr(self.graph_qformer.config, "dropout", self.perceiver_dropout)),
        }
        print(
            "[GraphTokenizer] "
            f"type={self.graph_tokenizer_config['type']} "
            f"latents={self.graph_tokenizer_config['num_latents']} "
            f"layers={self.graph_tokenizer_config['num_layers']} "
            f"heads={self.graph_tokenizer_config['num_heads']}"
        )

        # 3. Vision Branch (optional)
        if enable_vision:
            print("Loading Vision Branch...")
            # ViT-g + BLIP2 Q-Former (Frozen, except maybe projector but user said VL-Projector is separate)
            # BLIP2VisionQFormer loads both.
            # User: "Vision Encoder为ViT-g", "BLIP2 Q-Former直接使用BLIP2的Q-Former"
            # "需要训练的模块分别为... VL-Projector"
            # So we freeze everything inside BLIP2VisionQFormer
            self.vision_branch = BLIP2VisionQFormer(
                model_name=vision_model_name,
                pretrained_model=pretrained_vision,
                freeze_vision=True,
                freeze_qformer=True # Freeze Q-Former as per requirements
            )

            # VL-Projector (Trainable)
            # BLIP2VisionQFormer returns tokens of dim 768
            if use_blip2_vl_projector and self.vision_branch.has_language_projection():
                blip2_proj = self.vision_branch.language_projection
                if not blip2_vl_projector_trainable:
                    for p in blip2_proj.parameters():
                        p.requires_grad = False

                proj_out_dim = getattr(blip2_proj, "out_features", None)
                if proj_out_dim is None:
                    proj_out_dim = self.hidden_size

                if proj_out_dim != self.hidden_size:
                    # Add a lightweight adapter to match LLM hidden size
                    self.vl_projector = nn.Sequential(
                        blip2_proj,
                        Projector(
                            input_dim=proj_out_dim,
                            output_dim=self.hidden_size,
                            type="mlp"
                        )
                    )
                else:
                    # Directly use BLIP2 VL-Projector
                    self.vl_projector = blip2_proj
            else:
                self.vl_projector = Projector(
                    input_dim=self.vision_branch.get_qformer_hidden_dim(),
                    output_dim=self.hidden_size,
                    type="mlp"
                )
        else:
            self.vision_branch = None
            self.vl_projector = None

    @staticmethod
    def _truncate_prefix_tokens(tokens: torch.Tensor | None, max_tokens: int | None) -> torch.Tensor | None:
        if tokens is None or max_tokens is None:
            return tokens
        if tokens.size(1) <= max_tokens:
            return tokens
        return tokens[:, :max_tokens, :]

    @staticmethod
    def _attach_output_field(outputs, key: str, value):
        try:
            outputs[key] = value
        except Exception:
            pass
        try:
            setattr(outputs, key, value)
        except Exception:
            pass

    @staticmethod
    def _resolve_context_scale(
        scale: torch.Tensor | float | int | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if scale is None:
            return None
        if isinstance(scale, torch.Tensor):
            s = scale.to(device=device, dtype=dtype).view(-1)
        else:
            s = torch.tensor([float(scale)], device=device, dtype=dtype).view(-1)
        if s.numel() == 1:
            s = s.expand(int(batch_size))
        elif s.numel() != int(batch_size):
            raise ValueError(
                f"Context scale length mismatch: got {int(s.numel())}, expected {int(batch_size)}."
            )
        return s.clamp(min=0.0, max=2.0)

    @staticmethod
    def _apply_context_scale(
        embeds: torch.Tensor | None,
        scale: torch.Tensor | float | int | None,
        *,
        batch_size: int,
    ) -> torch.Tensor | None:
        if embeds is None:
            return None
        s = OmniGraphModel._resolve_context_scale(
            scale=scale,
            batch_size=batch_size,
            device=embeds.device,
            dtype=embeds.dtype,
        )
        if s is None:
            return embeds
        return embeds * s.view(-1, 1, 1)

    @property
    def vg_adapter(self):
        # Backward-compatible view for legacy call-sites.
        return self.node_encoder

    def _inject_graph_node_features(self, graph_data):
        if graph_data is None:
            return None
        if not (hasattr(graph_data, "obj_id") and hasattr(graph_data, "attr_id") and hasattr(graph_data, "bbox")):
            return graph_data

        x = self.node_encoder(
            graph_data.obj_id,
            graph_data.attr_id,
            graph_data.bbox,
            obj_hash_id=getattr(graph_data, "obj_hash_id", None),
            attr_hash_id=getattr(graph_data, "attr_hash_id", None),
            edge_index=getattr(graph_data, "edge_index", None),
            edge_pred_id=getattr(graph_data, "edge_pred_id", None),
            edge_pred_hash_id=getattr(graph_data, "edge_pred_hash_id", None),
        )
        graph_data.x = x
        graph_data.graph_node = x
        return graph_data
        
    def forward(self, 
                input_ids, 
                attention_mask=None,
                graph_data=None, # Expecting object with input_ids, attention_mask for GraphGPT
                pixel_values=None, # (B, 3, H, W)
                graph_context_scale: torch.Tensor | float | None = None,
                vision_context_scale: torch.Tensor | float | None = None,
                labels=None,
                aux_binary_labels: torch.Tensor | None = None,
                aux_binary_mask: torch.Tensor | None = None,
                aux_loss_weight: float = 0.0,
                return_alignment_features: bool = False,
                return_debug: bool = False
                ):
        
        # Get LLM token embeddings
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        target_dtype = inputs_embeds.dtype
        
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        text_attention_mask = attention_mask
        
        # 1. Process Graph Data
        # Support missing modality: if graph_data is None, skip adding tokens
        debug = {}
        graph_embeds = None
        if graph_data is not None:
            graph_data = self._inject_graph_node_features(graph_data)
            # graph_embeds: (B, Nq_graph, 4096)
            if return_debug:
                graph_tokens, graph_dbg = self.graph_branch(graph_data, return_debug=True)  # (B, Nq, d_qformer)
                debug.update({f"graph_{k}": v for k, v in graph_dbg.items()})
            else:
                graph_tokens = self.graph_branch(graph_data)     # (B, Nq, d_qformer)
            graph_embeds = self.gl_projector(graph_tokens)      # (B, Nq, 4096)
            graph_embeds = self._truncate_prefix_tokens(graph_embeds, self.max_graph_tokens)
            if return_debug:
                debug["graph_embeds_nan"] = bool(torch.isnan(graph_embeds).any().item())
                debug["graph_embeds_inf"] = bool(torch.isinf(graph_embeds).any().item())
                debug["graph_prefix_tokens"] = int(graph_embeds.shape[1])
            if graph_embeds.dtype != target_dtype:
                graph_embeds = graph_embeds.to(target_dtype)
            graph_embeds = self._apply_context_scale(
                graph_embeds,
                graph_context_scale,
                batch_size=batch_size,
            )

        # 2. Process Vision Data
        # Support missing modality: if pixel_values is None, skip adding tokens
        vision_embeds = None
        if pixel_values is not None and self.vision_branch is not None and self.vl_projector is not None:
            # vision_tokens: (B, Nq_vision, 768)
            vision_tokens = self.vision_branch(pixel_values)
            # vision_embeds: (B, Nq_vision, 4096)
            vision_embeds = self.vl_projector(vision_tokens)
            vision_embeds = self._truncate_prefix_tokens(vision_embeds, self.max_vision_tokens)
            if return_debug:
                debug["vision_tokens_nan"] = bool(torch.isnan(vision_tokens).any().item())
                debug["vision_tokens_inf"] = bool(torch.isinf(vision_tokens).any().item())
                debug["vision_embeds_nan"] = bool(torch.isnan(vision_embeds).any().item())
                debug["vision_embeds_inf"] = bool(torch.isinf(vision_embeds).any().item())
                debug["vision_prefix_tokens"] = int(vision_embeds.shape[1])
            if vision_embeds.dtype != target_dtype:
                vision_embeds = vision_embeds.to(target_dtype)
            vision_embeds = self._apply_context_scale(
                vision_embeds,
                vision_context_scale,
                batch_size=batch_size,
            )

        context_tokens = []
        if graph_embeds is not None:
            context_tokens.append(graph_embeds)
        if vision_embeds is not None:
            context_tokens.append(vision_embeds)
        context_embeds = torch.cat(context_tokens, dim=1) if context_tokens else None

        text_embeds = inputs_embeds
        if self.gvl_adapter is not None and context_embeds is not None:
            text_embeds = self.gvl_adapter(text_embeds, context_embeds)
            if return_debug:
                debug["text_after_adapter_nan"] = bool(torch.isnan(text_embeds).any().item())
                debug["text_after_adapter_inf"] = bool(torch.isinf(text_embeds).any().item())

        aux_logits = None
        aux_loss = None
        if self.graph_aux_head is not None and graph_embeds is not None:
            aux_logits = self.graph_aux_head(
                graph_embeds=graph_embeds,
                text_embeds=text_embeds,
                text_attention_mask=text_attention_mask,
            )
            aux_loss = self.graph_aux_head.loss(
                logits=aux_logits,
                labels=aux_binary_labels,
                mask=aux_binary_mask,
            )
            if return_debug:
                debug["aux_logits_nan"] = bool(torch.isnan(aux_logits).any().item())
                debug["aux_logits_inf"] = bool(torch.isinf(aux_logits).any().item())

        # 3. Concatenate embeddings [Graph, Vision, Text]
        embeds_list = []
        if graph_embeds is not None:
            embeds_list.append(graph_embeds)
        if vision_embeds is not None:
            embeds_list.append(vision_embeds)
        embeds_list.append(text_embeds)
        combined_embeds = torch.cat(embeds_list, dim=1)
        if return_debug:
            debug["combined_embeds_nan"] = bool(torch.isnan(combined_embeds).any().item())
            debug["combined_embeds_inf"] = bool(torch.isinf(combined_embeds).any().item())
        
        # Update attention mask and labels
        if attention_mask is not None:
            prefix_len = 0
            if graph_embeds is not None:
                prefix_len += graph_embeds.shape[1]
            if vision_embeds is not None:
                prefix_len += vision_embeds.shape[1]

            if prefix_len > 0:
                prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
                
        if labels is not None:
            # Recalculate prefix length for labels based on actual concatenated embeddings
            prefix_len = combined_embeds.shape[1] - inputs_embeds.shape[1]
            if prefix_len > 0:
                # Pad labels with -100 for prefix tokens so they are ignored in loss
                prefix_labels = torch.full((batch_size, prefix_len), -100, device=device, dtype=labels.dtype)
                labels = torch.cat([prefix_labels, labels], dim=1)
        
        outputs = self.llm(
            input_ids=None,
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        if return_alignment_features:
            if graph_embeds is not None:
                self._attach_output_field(outputs, "graph_embeds", graph_embeds)
            if vision_embeds is not None:
                self._attach_output_field(outputs, "vision_embeds", vision_embeds)
            self._attach_output_field(outputs, "text_embeds", text_embeds)
            if text_attention_mask is not None:
                self._attach_output_field(outputs, "text_attention_mask", text_attention_mask)

        if aux_logits is not None:
            self._attach_output_field(outputs, "aux_logits", aux_logits)
        if aux_loss is not None:
            self._attach_output_field(outputs, "aux_loss", aux_loss)
            self._attach_output_field(outputs, "aux_loss_weight", float(aux_loss_weight))
            w = max(0.0, float(aux_loss_weight))
            if w > 0.0:
                base_loss = getattr(outputs, "loss", None)
                aux_term = aux_loss * w
                if base_loss is None:
                    outputs.loss = aux_term
                else:
                    outputs.loss = base_loss + aux_term
        
        if return_debug:
            return outputs, debug
        return outputs

    def generate(
        self,
        input_ids,
        graph_data=None,
        pixel_values=None,
        graph_context_scale: torch.Tensor | float | None = None,
        vision_context_scale: torch.Tensor | float | None = None,
        **kwargs,
    ):
        # Get LLM token embeddings
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        target_dtype = inputs_embeds.dtype

        graph_embeds = None
        if graph_data is not None:
            graph_data = self._inject_graph_node_features(graph_data)
            graph_tokens = self.graph_branch(graph_data)
            graph_embeds = self.gl_projector(graph_tokens)
            graph_embeds = self._truncate_prefix_tokens(graph_embeds, self.max_graph_tokens)
            if graph_embeds.dtype != target_dtype:
                graph_embeds = graph_embeds.to(target_dtype)
            graph_embeds = self._apply_context_scale(
                graph_embeds,
                graph_context_scale,
                batch_size=int(inputs_embeds.shape[0]),
            )

        vision_embeds = None
        if pixel_values is not None and self.vision_branch is not None and self.vl_projector is not None:
            vision_tokens = self.vision_branch(pixel_values)
            vision_embeds = self.vl_projector(vision_tokens)
            vision_embeds = self._truncate_prefix_tokens(vision_embeds, self.max_vision_tokens)
            if vision_embeds.dtype != target_dtype:
                vision_embeds = vision_embeds.to(target_dtype)
            vision_embeds = self._apply_context_scale(
                vision_embeds,
                vision_context_scale,
                batch_size=int(inputs_embeds.shape[0]),
            )

        context_tokens = []
        if graph_embeds is not None:
            context_tokens.append(graph_embeds)
        if vision_embeds is not None:
            context_tokens.append(vision_embeds)
        context_embeds = torch.cat(context_tokens, dim=1) if context_tokens else None

        text_embeds = inputs_embeds
        if self.gvl_adapter is not None and context_embeds is not None:
            text_embeds = self.gvl_adapter(text_embeds, context_embeds)

        embeds_list = []
        if graph_embeds is not None:
            embeds_list.append(graph_embeds)
        if vision_embeds is not None:
            embeds_list.append(vision_embeds)
        embeds_list.append(text_embeds)
        combined_embeds = torch.cat(embeds_list, dim=1)
            
        return self.llm.generate(inputs_embeds=combined_embeds, **kwargs)

if __name__ == "__main__":
    pass
