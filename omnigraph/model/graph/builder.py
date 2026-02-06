import torch
import torch.nn as nn
import json
import os.path as osp
import glob
from omnigraph.model.graph.clip_graph import CLIP
from omnigraph.model.graph.graph_transformer import graph_transformer
from omnigraph.model.graph.clip_graph import GNN

class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    if len(pkl_files) == 0:
        print(f"No .pkl file found in {pretrain_model_path}")
        return model, args

    state_dict = torch.load(pkl_files[0], map_location='cpu')
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)

    return model, args

def transfer_param_tograph(clip_graph, gnn):
    print(clip_graph)
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn

def build_graph_tower(graph_tower_name, pretrain_graph_model_path):
    if graph_tower_name == 'MPNN': 
        # config.graph_hidden_size needs to be passed or assumed
        # Assuming args are available or hardcoded for now as this is a builder
        raise NotImplementedError("MPNN not supported in this builder yet")
    elif graph_tower_name == "clip_gcn_arxiv": 
        clip_graph, args= load_model_pretrained(CLIP, pretrain_graph_model_path)
        graph_tower = GNN(args)
        graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        return graph_tower
    elif graph_tower_name == "clip_gt":
        clip_graph, args= load_model_pretrained(CLIP, pretrain_graph_model_path) 
        graph_tower = graph_transformer(args)
        graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        return graph_tower
    elif graph_tower_name == "clip_gt_arxiv": 
        clip_graph, args= load_model_pretrained(CLIP, pretrain_graph_model_path) 
        graph_tower = graph_transformer(args)
        graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        return graph_tower
    elif graph_tower_name == "clip_gt_arxiv_pub": 
        clip_graph, args= load_model_pretrained(CLIP, pretrain_graph_model_path) 
        graph_tower = graph_transformer(args)
        graph_tower = transfer_param_tograph(clip_graph, graph_tower)
        return graph_tower
    else:
        # Fallback or error
        raise ValueError(f"Unknown graph tower: {graph_tower_name}")
