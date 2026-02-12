import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from omnigraph.utils.env import setup_env
setup_env()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor


from omnigraph.model.graph.builder import build_graph_tower
from omnigraph.model.graph.graph_qformer import GraphQFormer, GraphQFormerConfig
from omnigraph.data.utils import preprocess_graph, DEFAULT_GRAPH_TOKEN

import json
import copy
import time
import argparse

import random
random.seed(42)
torch.manual_seed(42)

class GraphQFormerStage1PL((pl.LightningModule if pl is not None else nn.Module)):
    def __init__(
        self, 
        graph_model_name="clip_gt_arxiv_pub",
        text_model_name="sentence-transformers/all-mpnet-base-v2", # Strong semantic text encoder
        embed_dim=256,
        num_query_tokens=32,
        graph_hidden_dim=128, # Updated to 128 as per clip_gt_arxiv_pub config
        qformer_hidden_dim=768,
        pretrained_graph=None,
        lr=1e-4,
        weight_decay=0.01,
        aux_node_text_weight=0.0,
        aux_neigh_text_weight=0.0,
        text_max_len=512,
        gtm_max_len=768
    ):
        super().__init__()
        if pl is not None:
            self.save_hyperparameters(ignore=['pretrained_graph'])
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_node_text_weight = aux_node_text_weight
        self.aux_neigh_text_weight = aux_neigh_text_weight
        self.text_max_len = text_max_len
        self.gtm_max_len = gtm_max_len
        
        # 1. Graph Encoder (Frozen)
        print(f"Loading Graph Encoder: {graph_model_name}")
        if pretrained_graph is not None:
            self.graph_encoder = pretrained_graph
        else:
            # Assumes clip_gt_arxiv_pub folder is in current directory or relative path
            self.graph_encoder = build_graph_tower(graph_model_name, "omnigraph/model/graph/clip_gt_arxiv_pub")
        
        for p in self.graph_encoder.parameters():
            p.requires_grad = False
        self.graph_encoder.eval()
            
        # 2. Text Encoder (Frozen) - For Contrastive Loss (GTC)
        print(f"Loading Text Encoder: {text_model_name}")
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()
            
        text_hidden_dim = self.text_encoder.config.hidden_size
            
        # 3. Graph Q-Former (Trainable)
        print("Initializing Graph Q-Former...")
        config = GraphQFormerConfig(
            num_query_tokens=num_query_tokens,
            graph_hidden_dim=graph_hidden_dim,
            qformer_hidden_dim=qformer_hidden_dim,
            llm_hidden_dim=embed_dim, # Not used for LLM here, but used for internal sizing if needed
            max_text_len=gtm_max_len
        )
        self.graph_qformer = GraphQFormer(config)
        # Tokenizer for Q-Former (BERT Base)
        self.qformer_tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            model_max_length=gtm_max_len
        )
        
        # 4. Projection Heads (Trainable)
        # Graph Q-Former outputs (B, K, qformer_hidden_dim) -> Mean Pool -> (B, qformer_hidden_dim) -> Proj -> (B, embed_dim)
        self.graph_proj = nn.Linear(qformer_hidden_dim, embed_dim)
        
        # Text Encoder outputs (B, L, text_hidden_dim) -> Mean Pool or CLS -> (B, text_hidden_dim) -> Proj -> (B, embed_dim)
        self.text_proj = nn.Linear(text_hidden_dim, embed_dim)
        
        # Temperature parameter for CLIP loss
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) # init log(14.28) standard CLIP value
    
    def train(self, mode: bool = True):
        super().train(mode)
        self.graph_encoder.eval()
        self.text_encoder.eval()
        return self
        
    def _encode_text_list(self, texts):
        device = self.logit_scale.device
        if texts is None or len(texts) == 0:
            return None
        
        # Ensure all elements are strings
        cleaned_texts = [t if t is not None else "" for t in texts]
        
        text_tokens = self.tokenizer(
            cleaned_texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_len,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_outputs = self.text_encoder(**text_tokens)
            if hasattr(text_outputs, 'last_hidden_state'):
                # For BERT-like models
                mask = text_tokens['attention_mask'].unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
                text_repr = torch.sum(text_outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            elif hasattr(text_outputs, 'pooler_output'):
                # For models that return pooler_output directly
                text_repr = text_outputs.pooler_output
            else:
                # Fallback for models that might return a tuple or other structure
                # Assuming index 0 is last_hidden_state
                last_hidden_state = text_outputs[0]
                mask = text_tokens['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                text_repr = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                
        text_feat = self.text_proj(text_repr)
        text_feat = F.normalize(text_feat, dim=-1)
        return text_feat
    
    def _clip_contrastive_loss(self, a_feat, b_feat, logit_scale):
        logits_a2b = logit_scale * a_feat @ b_feat.t()
        logits_b2a = logit_scale * b_feat @ a_feat.t()
        
        # Guard against NaN/Inf in logits (crucial for fp16)
        if torch.isnan(logits_a2b).any() or torch.isinf(logits_a2b).any():
            logits_a2b = torch.nan_to_num(logits_a2b, nan=0.0, posinf=0.0, neginf=0.0)
        if torch.isnan(logits_b2a).any() or torch.isinf(logits_b2a).any():
            logits_b2a = torch.nan_to_num(logits_b2a, nan=0.0, posinf=0.0, neginf=0.0)
            
        batch_size = logits_a2b.shape[0]
        labels = torch.arange(batch_size, device=logits_a2b.device)
        loss_a = F.cross_entropy(logits_a2b, labels)
        loss_b = F.cross_entropy(logits_b2a, labels)
        return (loss_a + loss_b) / 2

    def forward(self, graph_data, text_input, aux_node_text_input=None, aux_neigh_texts=None):
        """
        graph_data: input for GraphGPT (Batch object or Data object)
        text_input: input list of strings
        """
        device = self.logit_scale.device
        if hasattr(graph_data, "to"):
            graph_data = graph_data.to(device)

        # --- Graph Forward ---
        with torch.no_grad():
            # GraphGPT returns raw node embeddings (Total_Nodes, D)
            graph_node_reps = self.graph_encoder(graph_data)
            
            # Reconstruct batch structure
        if hasattr(graph_data, 'batch') and graph_data.batch is not None:
            batch_idx = graph_data.batch
            # Use max_num_nodes=None, batch_size=batch_size from data
            # Infer batch size from text_input length to ensure consistency
            target_batch_size = len(text_input) if isinstance(text_input, list) else 1

            # (B, Max_Nodes, D), (B, Max_Nodes)
            # FORCE batch_size to match text input. 
            # If graph batch is smaller (e.g. empty graphs filtered out?), this will pad with zeros.
            # If graph batch is larger (unlikely?), it might truncate? No, to_dense_batch doesn't truncate B.
            graph_hidden_states, graph_attention_mask = to_dense_batch(graph_node_reps, batch_idx, batch_size=target_batch_size)
            
            # Check for empty graphs (all masked) which cause NaN in attention
            if not graph_attention_mask.sum(dim=1).gt(0).all():
                 # Handle empty graphs by ensuring at least one dummy node is active
                 # This can happen if a graph has 0 nodes (unlikely but possible in some data pipelines)
                 # or if batch alignment issues occur.
                 zero_mask_indices = (graph_attention_mask.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
                 # Force the first node to be valid (even if it's padding 0) to avoid NaN
                 graph_attention_mask[zero_mask_indices, 0] = 1
        else:
            graph_hidden_states = graph_node_reps.unsqueeze(0)
            graph_attention_mask = torch.ones(
                graph_hidden_states.shape[:-1],
                device=device,
                dtype=torch.long,
            )

        graph_hidden_states = graph_hidden_states.to(device)
        graph_attention_mask = graph_attention_mask.to(device).long()

        # Q-Former Forward (Unimodal Graph)
        # Output: (B, K, Dq)
        query_tokens = self.graph_qformer(
            graph_hidden_states=graph_hidden_states,
            graph_attention_mask=graph_attention_mask
        )
        
        # Pooling & Projection for GTC
        graph_repr = query_tokens.mean(dim=1) # (B, Dq)
        graph_feat = self.graph_proj(graph_repr) # (B, D)
        graph_feat = F.normalize(graph_feat, dim=-1)
        
        # --- Text Forward (Text Encoder for GTC) ---
        if isinstance(text_input, list):
            text_feat = self._encode_text_list(text_input)
            text_tokens_for_gtm = None
        else:
            text_tokens_for_gtm = text_input
            with torch.no_grad():
                text_outputs = self.text_encoder(**text_tokens_for_gtm)
                if hasattr(text_outputs, 'last_hidden_state'):
                    mask = text_tokens_for_gtm['attention_mask'].unsqueeze(-1).expand(text_outputs.last_hidden_state.size()).float()
                    text_repr = torch.sum(text_outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
                elif hasattr(text_outputs, 'pooler_output'):
                    text_repr = text_outputs.pooler_output
                else:
                    last_hidden_state = text_outputs[0]
                    mask = text_tokens_for_gtm['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                    text_repr = torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            text_feat = self.text_proj(text_repr)
            text_feat = F.normalize(text_feat, dim=-1)
        
        # --- Grounding Loss (formerly GTC) ---
        # User requested to re-enable GTC logic.
        # Fix: logit_scale.exp() in fp16 might overflow/underflow, compute in float
        logit_scale = self.logit_scale.float().exp().clamp(max=100.0) 
        logits_g2t = logit_scale * graph_feat @ text_feat.t()
        
        # Guard against NaN/Inf in logits
        if torch.isnan(logits_g2t).any() or torch.isinf(logits_g2t).any():
            logits_g2t = torch.nan_to_num(logits_g2t, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size = logits_g2t.shape[0]
        labels = torch.arange(batch_size, device=device)
        loss_gtc = self._clip_contrastive_loss(graph_feat, text_feat, logit_scale)
        
        # --- GTM (Graph-Text Matching) Loss ---
        # Select hard negatives or random negatives
        # Compute logits just for mining (without optimization)

        with torch.no_grad():
            # logits_g2t: (B, B)
            B = logits_g2t.size(0)

            # 构造 mask：不能采自己
            neg_mask = torch.ones_like(logits_g2t, dtype=torch.bool)
            neg_mask.fill_diagonal_(False)

            # --- 关键修正：过滤掉文本内容相同/极相似的负样本 ---
            # 如果 batch 中存在多张图对应相同的文本（或极相似文本），
            # GTC 会将它们拉近，Hard Negative Mining 可能会选中它们作为负样本。
            # 但这对 GTM 来说是 "False Negative"（标签是负，内容是正），导致 Loss 无法下降。
            with torch.no_grad():
                text_sim = text_feat @ text_feat.t()
                # 阈值设为 0.98，过滤掉几乎一样的文本
                is_duplicate_text = text_sim > 0.98
                # 从 neg_mask 中去除这些重复文本对
                neg_mask = neg_mask & (~is_duplicate_text)

            # 在 logits 层面 mask
            masked_logits = logits_g2t.float().masked_fill(~neg_mask, -1e9)

            # softmax（强制 fp32，避免 half underflow）
            weights_g2t = F.softmax(masked_logits, dim=1)

            # 行和检查（非常重要）
            row_sum = weights_g2t.sum(dim=1)

            # 找到“没有任何负样本”的 graph
            invalid = row_sum <= 1e-6

            if invalid.any():
                # fallback：对这些样本均匀随机采
                weights_g2t[invalid] = neg_mask[invalid].float()
                weights_g2t[invalid] /= weights_g2t[invalid].sum(dim=1, keepdim=True)

            neg_text_idx = torch.multinomial(weights_g2t, 1).squeeze(1)
            
        # Tokenize for Q-Former (BERT) - Need separate tokenizer
        # Because Text Encoder might differ from Q-Former's BERT
        # Ensure total length (queries + text) <= Q-Former position limit
        max_text_len = self.graph_qformer.config.max_text_len
        qformer_text_tokens = self.qformer_tokenizer(
            text_input,
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt"
        ).to(device)
        
        pos_text_ids = qformer_text_tokens.input_ids
        pos_text_att = qformer_text_tokens.attention_mask
        
        neg_text_ids = pos_text_ids[neg_text_idx]
        neg_text_att = pos_text_att[neg_text_idx]
        
        # Concat Positive and Negative
        # [Pos, Neg]
        total_text_ids = torch.cat([pos_text_ids, neg_text_ids], dim=0)
        total_text_att = torch.cat([pos_text_att, neg_text_att], dim=0)
        
        # Double Graph inputs
        total_graph_states = torch.cat([graph_hidden_states, graph_hidden_states], dim=0)
        total_graph_mask = torch.cat([graph_attention_mask, graph_attention_mask], dim=0)
        
        # Labels: 1 for pos, 0 for neg
        gtm_labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).long().to(device)
        
        # Forward GTM
        gtm_logits = self.graph_qformer.forward_match(
            total_graph_states,
            total_graph_mask,
            total_text_ids,
            total_text_att
        )
        
        loss_gtm = F.cross_entropy(gtm_logits, gtm_labels)

        # 计算 GTM Accuracy 用于监控
        with torch.no_grad():
            gtm_preds = gtm_logits.argmax(dim=1)
            gtm_acc = (gtm_preds == gtm_labels).float().mean()
        
        loss_aux_node = torch.zeros([], device=device)
        loss_aux_neigh = torch.zeros([], device=device)
        
        if aux_node_text_input is not None and self.aux_node_text_weight > 0:
            node_text_feat = self._encode_text_list(aux_node_text_input)
            loss_aux_node = self._clip_contrastive_loss(graph_feat, node_text_feat, logit_scale)
        
        if aux_neigh_texts is not None and self.aux_neigh_text_weight > 0:
            valid_indices = [i for i, t in enumerate(aux_neigh_texts) if t is not None and len(t) > 0]
            if len(valid_indices) > 1:
                flat_texts = []
                sizes = []
                for i in valid_indices:
                    texts_i = aux_neigh_texts[i]
                    sizes.append(len(texts_i))
                    flat_texts.extend(texts_i)
                flat_feat = self._encode_text_list(flat_texts)
                splits = list(flat_feat.split(sizes, dim=0))
                neigh_feat = torch.stack([s.mean(dim=0) for s in splits], dim=0)
                graph_feat_valid = graph_feat[torch.tensor(valid_indices, device=device)]
                loss_aux_neigh = self._clip_contrastive_loss(graph_feat_valid, neigh_feat, logit_scale)
        
        loss = loss_gtc + loss_gtm + self.aux_node_text_weight * loss_aux_node + self.aux_neigh_text_weight * loss_aux_neigh
        
        return {
            "loss": loss,
            "loss_gtc": loss_gtc,
            "loss_gtm": loss_gtm,
            "acc_gtm": gtm_acc,
            "loss_aux_node": loss_aux_node,
            "loss_aux_neigh": loss_aux_neigh
        }

    def training_step(self, batch, batch_idx):
        if pl is None:
            raise RuntimeError("pytorch_lightning is required for training_step")
        graph_data = batch['graph_data']
        text_input = batch['text_input']
        aux_node_text_input = batch.get('aux_node_text_input', None)
        aux_neigh_texts = batch.get('aux_neigh_texts', None)
        
        outputs = self(graph_data, text_input, aux_node_text_input=aux_node_text_input, aux_neigh_texts=aux_neigh_texts)
        loss = outputs['loss']
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        self.log("train_loss_gtm", outputs['loss_gtm'], prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        self.log("train_acc_gtm", outputs['acc_gtm'], prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        self.log("train_loss_gtc", outputs['loss_gtc'], prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        if self.aux_node_text_weight > 0:
            self.log("train_loss_aux_node", outputs['loss_aux_node'], prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        if self.aux_neigh_text_weight > 0:
            self.log("train_loss_aux_neigh", outputs['loss_aux_neigh'], prog_bar=True, on_step=True, on_epoch=True, batch_size=len(text_input))
        
        return loss

    def configure_optimizers(self):
        if pl is None:
            raise RuntimeError("pytorch_lightning is required for configure_optimizers")
        # Optimize only trainable params (Q-Former + Proj Heads + Logit Scale)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        # Can add scheduler here if needed
        return optimizer

# Real Dataset Implementation
class GraphTextDataset(Dataset):
    def __init__(self, data_path, graph_data_path="omnigraph/data/graph_data_all.pt", max_len=128, node_text_json_path=None, node_feat_npy_path=None, neigh_text_k=8):
        """
        data_path: Path or list of paths to dataset json files
        max_len: Max sequence length for tokenizer
        """
        self.max_len = max_len
        self.neigh_text_k = neigh_text_k
        
        if isinstance(data_path, str):
            data_paths = [data_path]
        else:
            data_paths = data_path
            
        self.data = []
        for path in data_paths:
            print(f"Loading dataset from {path}...")
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                        self.data.extend(data)
                    print(f"Loaded {len(data)} samples from {path}.")
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from {path}: {e}")
                    print("Skipping this file.")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"Warning: Dataset file not found: {path}")
                
        print(f"Total samples loaded: {len(self.data)}")
        
        print(f"Loading graph features from {graph_data_path}...")
        try:
            self.graph_data_all = torch.load(graph_data_path, map_location='cpu', weights_only=False)
            print("Graph features loaded successfully.")
        except Exception as e:
            print(f"Error loading graph data from {graph_data_path}: {e}")
            print("Please ensure omnigraph/data/graph_data_all.pt exists.")
            # For robustness in case file is missing during dev
            self.graph_data_all = None
        
        self.node_text_map = None
        if node_text_json_path is not None and os.path.exists(node_text_json_path):
            try:
                with open(node_text_json_path, "r") as f:
                    self.node_text_map = json.load(f)
                print(f"Loaded node text map from {node_text_json_path}.")
            except Exception as e:
                print(f"Error loading node text map from {node_text_json_path}: {e}")
                self.node_text_map = None
        
        self.node_feat = None
        if node_feat_npy_path is not None and os.path.exists(node_feat_npy_path):
            try:
                self.node_feat = np.load(node_feat_npy_path)
                print(f"Loaded node features from {node_feat_npy_path}: {self.node_feat.shape}.")
            except Exception as e:
                print(f"Error loading node features from {node_feat_npy_path}: {e}")
                self.node_feat = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Process Graph
        graph_data = item['graph']
        node_list = graph_data.get('node_list', [])
        edge_index = graph_data.get('edge_index', [[], []])
        
        # Get features
        sample_id = item.get('id', 'arxiv_unknown')
        graph_type = sample_id.split('_')[0] # e.g. 'arxiv'
        
        if self.graph_data_all is not None and graph_type in self.graph_data_all:
             full_graph = self.graph_data_all[graph_type]
             # Ensure node_list indices are valid
             # Assuming node_list are indices into full_graph.x
             graph_node = full_graph.x[node_list]
        elif self.node_feat is not None:
             graph_node = torch.from_numpy(self.node_feat[node_list]).float()
        else:
             # Fallback: Random features matching 128 dim (config)
             graph_node = torch.randn(len(node_list), 128)
        
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        # Create Data object
        # User fix: use standard key 'x' for PyG compatibility, and alias 'graph_node' for clip_graph.py
        data_obj = Data(x=graph_node, edge_index=edge_index_tensor)
        data_obj.graph_node = graph_node
        
        # Determine graph token length (num nodes)
        cur_token_len = len(node_list)
        
        # Prepare sources for text preprocessing
        sources = [item['conversations']]
        
        # Configuration for preprocessing
        graph_cfg = {
            'is_graph': True,
            'sep_graph_conv_front': False, # Default
            'use_graph_start_end': False # Default
        }
        
        # Apply preprocessing (text replacement)
        sources = preprocess_graph(copy.deepcopy(sources), graph_cfg, cur_token_len)
        
        # Now we have the modified text with <g_patch> tokens
        # We need to extract the instruction text (Human)
        conversations = sources[0]
        text_content = ""
        answer_content = ""
        if len(conversations) > 0:
             # Find human message
             for msg in conversations:
                 if msg['from'] == 'human':
                     text_content = msg['value']
                     break
             for msg in conversations:
                 if msg['from'] == 'gpt':
                     answer_content = msg['value']
                     break
        
        text_content = text_content.replace("<g_patch>", " ").replace("<graph>", " ")
        text_content = " ".join(text_content.split())
        answer_content = answer_content.replace("<g_patch>", " ").replace("<graph>", " ")
        answer_content = " ".join(answer_content.split())
        
        aux_node_text = None
        aux_neigh_texts = None
        if self.node_text_map is not None and len(node_list) > 0:
            aux_node_text = self.node_text_map.get(str(node_list[0]), "")
            neigh_ids = node_list[1:1 + self.neigh_text_k]
            aux_neigh_texts = [self.node_text_map.get(str(nid), "") for nid in neigh_ids if str(nid) in self.node_text_map]
        
        return {
            "graph_data": data_obj,
            "text": text_content,
            "answer": answer_content,
            "aux_node_text": aux_node_text,
            "aux_neigh_texts": aux_neigh_texts
        }

def collate_fn(batch):
    # Batch graph inputs
    graph_data_list = [item['graph_data'] for item in batch]
    batch_graph = Batch.from_data_list(graph_data_list)
    
    texts = [item['text'] for item in batch]
    answers = [item.get('answer', '') for item in batch]
    aux_node_text_input = [item.get('aux_node_text', None) for item in batch]
    aux_neigh_texts = [item.get('aux_neigh_texts', None) for item in batch]
    
    if all(t is None for t in aux_node_text_input):
        aux_node_text_input = None
    if all(t is None for t in aux_neigh_texts):
        aux_neigh_texts = None
    
    return {
        "graph_data": batch_graph,
        "text_input": texts,
        "answer_input": answers,
        "aux_node_text_input": aux_node_text_input,
        "aux_neigh_texts": aux_neigh_texts
    }

def main():
    torch.set_float32_matmul_precision('medium')
    print("[Pipeline] Stage1 start: no upstream checkpoint dependencies.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_endpoint", type=str, default=None)
    parser.add_argument("--hf_cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dataset_paths", nargs="+", default=[
        "data/train_instruct_graphmatch.json",
        "data/arxiv_pub_node_st_cot_link_mix.json"
    ])
    parser.add_argument("--node_text_json_path", type=str, default=None)
    parser.add_argument("--node_feat_npy_path", type=str, default=None)
    parser.add_argument("--aux_node_text_weight", type=float, default=0.0)
    parser.add_argument("--aux_neigh_text_weight", type=float, default=0.0)
    parser.add_argument("--text_model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--text_max_len", type=int, default=512)
    parser.add_argument("--gtm_max_len", type=int, default=512)
    parser.add_argument("--max_steps", type=int, default=-1, help="If >0, hard stop after this many optimizer steps.")
    parser.add_argument("--enable_early_stop", type=int, default=1, choices=[0, 1], help="Enable early stopping on train_loss_epoch.")
    parser.add_argument("--early_stop_patience", type=int, default=2, help="Early stop patience in epochs.")
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-3, help="Minimum train loss improvement.")
    parser.add_argument("--early_stop_mode", type=str, default="min", choices=["min", "max"], help="Early stop mode.")
    args = parser.parse_args()

    setup_env(hf_endpoint=args.hf_endpoint, hf_cache_dir=args.hf_cache_dir)
    
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    MAX_STEPS = int(args.max_steps)
    ENABLE_EARLY_STOP = bool(int(args.enable_early_stop))
    
    dataset_paths = args.dataset_paths
    # Check if at least one exists
    valid_paths = [p for p in dataset_paths if os.path.exists(p)]
    if not valid_paths:
        print(f"Warning: No valid dataset files found in {dataset_paths}.")
        return

    dataset = GraphTextDataset(
        valid_paths,
        node_text_json_path=args.node_text_json_path,
        node_feat_npy_path=args.node_feat_npy_path
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = GraphQFormerStage1PL(
        lr=LR,
        aux_node_text_weight=args.aux_node_text_weight,
        aux_neigh_text_weight=args.aux_neigh_text_weight,
        text_model_name=args.text_model_name,
        text_max_len=args.text_max_len,
        gtm_max_len=args.gtm_max_len
    )

    if pl is not None:
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints_stage1",
            filename="graph_qformer_stage1-{epoch:02d}-{train_loss_epoch:.2f}",
            save_top_k=1,
            monitor="train_loss_epoch",
            mode="min"
        )
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks = [checkpoint_callback, lr_monitor]
        if ENABLE_EARLY_STOP:
            callbacks.append(
                EarlyStopping(
                    monitor="train_loss_epoch",
                    mode=str(args.early_stop_mode),
                    patience=int(args.early_stop_patience),
                    min_delta=float(args.early_stop_min_delta),
                    check_on_train_epoch_end=True,
                    verbose=True,
                )
            )
        
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            max_steps=MAX_STEPS if MAX_STEPS > 0 else -1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision="16-mixed",
            callbacks=callbacks,
            log_every_n_steps=10,
            gradient_clip_val=1.0,
            accumulate_grad_batches=8,
            # limit_train_batches=50
        )
        
        trainer.fit(model, dataloader)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params, lr=LR)
        model.train()
        global_step = 0
        best_epoch_loss = float("inf")
        no_improve_epochs = 0
        
        for epoch in range(EPOCHS):
            t0 = time.time()
            running = 0.0
            steps = 0
            for batch_idx, batch in enumerate(dataloader):
                graph_data = batch["graph_data"].to(device)
                text_input = batch["text_input"]
                aux_node_text_input = batch.get("aux_node_text_input", None)
                aux_neigh_texts = batch.get("aux_neigh_texts", None)
                out = model(graph_data, text_input, aux_node_text_input=aux_node_text_input, aux_neigh_texts=aux_neigh_texts)
                loss = out["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                running += float(loss.detach().cpu())
                steps += 1
                global_step += 1
                if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                    print(f"[Stage1] Reached max_steps={MAX_STEPS}, stopping.")
                    break
                # if batch_idx >= 50:
                #     break
            dt = time.time() - t0
            epoch_loss = running / max(steps, 1)
            print(f"epoch={epoch} loss={epoch_loss:.4f} time={dt:.1f}s")
            if ENABLE_EARLY_STOP:
                improved = (best_epoch_loss - epoch_loss) > float(args.early_stop_min_delta)
                if improved:
                    best_epoch_loss = epoch_loss
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    if no_improve_epochs >= int(args.early_stop_patience):
                        print(
                            f"[Stage1] Early stop triggered: no improvement for "
                            f"{args.early_stop_patience} epoch(s)."
                        )
                        break
            if MAX_STEPS > 0 and global_step >= MAX_STEPS:
                break
    
    # Save final model
    print("Training finished.")
    torch.save(model.graph_qformer.state_dict(), "graph_qformer_stage1.pt")
    print("Checkpoint saved to graph_qformer_stage1.pt")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Training failed: {e}")
        print(
            "Hint: if this is a model download/network error, set "
            "STAGE1_TEXT_MODEL_NAME to a local model path, and/or configure "
            "OMNIGRAPH_HF_ENDPOINT / OMNIGRAPH_HF_CACHE."
        )
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
