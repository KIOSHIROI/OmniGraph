from omnigraph.utils.env import setup_env

setup_env()
import torch
import torch.nn as nn
from omnigraph.model.graph.VGNodeAdapter import VGNodeAdapter
from omnigraph.model.graph.graph_transformer import graph_transformer
from omnigraph.model.graph.builder import build_graph_tower
from omnigraph.model.graph.graph_branch import GraphBranch
from omnigraph.model.graph.graph_qformer import GraphQFormer, GraphQFormerConfig
from omnigraph.model.vision.blip2_vision_qformer import BLIP2VisionQFormer
from omnigraph.model.projector import Projector, ProjectorConfig
from omnigraph.model.llm.llama import LlamaLLM

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
                 enable_vision: bool = True
                 ):
        super().__init__()
        self.vg_adapter = VGNodeAdapter(num_obj=NUM_OBJ, num_attr=NUM_ATTR, out_dim=128)
        # 1. LLM (Frozen, bf16)
        print(f"Loading LLM: {llm_model_name}...")
        self.llm = LlamaLLM(model_name=llm_model_name, pretrained_model=pretrained_llm)
        self.hidden_size = self.llm.config.hidden_size
        
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

        # Graph Q-Former (Trainable)
        if pretrained_graph_qformer is not None:
             self.graph_qformer = pretrained_graph_qformer
        else:
             graph_qformer_config = GraphQFormerConfig(graph_hidden_dim=128)
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
        
    def forward(self, 
                input_ids, 
                attention_mask=None,
                graph_data=None, # Expecting object with input_ids, attention_mask for GraphGPT
                pixel_values=None, # (B, 3, H, W)
                labels=None,
                return_debug: bool = False
                ):
        
        # Get LLM token embeddings
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        target_dtype = inputs_embeds.dtype
        
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        # List to hold embeddings to concatenate: [Graph, Vision, Text]
        embeds_list = []
        
        # 1. Process Graph Data
        # Support missing modality: if graph_data is None, skip adding tokens
        debug = {}
        graph_embeds = None
        if graph_data is not None:
            # 如果 dataset 输出了 obj_id/attr_id/bbox，则用 adapter 生成 x
            if hasattr(graph_data, "obj_id") and hasattr(graph_data, "attr_id") and hasattr(graph_data, "bbox"):
                x = self.vg_adapter(graph_data.obj_id, graph_data.attr_id, graph_data.bbox)
                graph_data.x = x
                graph_data.graph_node = x
            # graph_embeds: (B, Nq_graph, 4096)
            if return_debug:
                graph_tokens, graph_dbg = self.graph_branch(graph_data, return_debug=True)  # (B, Nq, d_qformer)
                debug.update({f"graph_{k}": v for k, v in graph_dbg.items()})
            else:
                graph_tokens = self.graph_branch(graph_data)     # (B, Nq, d_qformer)
            graph_embeds = self.gl_projector(graph_tokens)      # (B, Nq, 4096)
            if return_debug:
                debug["graph_embeds_nan"] = bool(torch.isnan(graph_embeds).any().item())
                debug["graph_embeds_inf"] = bool(torch.isinf(graph_embeds).any().item())
            if graph_embeds.dtype != target_dtype:
                graph_embeds = graph_embeds.to(target_dtype)
            embeds_list.append(graph_embeds)

        # 2. Process Vision Data
        # Support missing modality: if pixel_values is None, skip adding tokens
        vision_embeds = None
        if pixel_values is not None and self.vision_branch is not None and self.vl_projector is not None:
            # vision_tokens: (B, Nq_vision, 768)
            vision_tokens = self.vision_branch(pixel_values)
            # vision_embeds: (B, Nq_vision, 4096)
            vision_embeds = self.vl_projector(vision_tokens)
            if return_debug:
                debug["vision_tokens_nan"] = bool(torch.isnan(vision_tokens).any().item())
                debug["vision_tokens_inf"] = bool(torch.isinf(vision_tokens).any().item())
                debug["vision_embeds_nan"] = bool(torch.isnan(vision_embeds).any().item())
                debug["vision_embeds_inf"] = bool(torch.isinf(vision_embeds).any().item())
            if vision_embeds.dtype != target_dtype:
                vision_embeds = vision_embeds.to(target_dtype)
            embeds_list.append(vision_embeds)
            
        # 3. Add Text Embeddings
        embeds_list.append(inputs_embeds)
        
        # Concatenate
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
        
        if return_debug:
            return outputs, debug
        return outputs

    def generate(self, input_ids, graph_data=None, pixel_values=None, **kwargs):
        # Get LLM token embeddings
        inputs_embeds = self.llm.model.get_input_embeddings()(input_ids)
        target_dtype = inputs_embeds.dtype
        
        embeds_list = []
        
        if graph_data is not None:
            graph_tokens = self.graph_branch(graph_data)
            graph_embeds = self.gl_projector(graph_tokens)
            if graph_embeds.dtype != target_dtype:
                graph_embeds = graph_embeds.to(target_dtype)
            embeds_list.append(graph_embeds)
            
        if pixel_values is not None and self.vision_branch is not None and self.vl_projector is not None:
            vision_tokens = self.vision_branch(pixel_values)
            vision_embeds = self.vl_projector(vision_tokens)
            if vision_embeds.dtype != target_dtype:
                vision_embeds = vision_embeds.to(target_dtype)
            embeds_list.append(vision_embeds)
            
        embeds_list.append(inputs_embeds)
        combined_embeds = torch.cat(embeds_list, dim=1)
            
        return self.llm.generate(inputs_embeds=combined_embeds, **kwargs)

if __name__ == "__main__":
    pass
