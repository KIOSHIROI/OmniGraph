import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from PIL import Image
import random
from torch_geometric.data import Data, Batch

from omnigraph.utils.env import setup_env
from omnigraph.model.OmniGraphModel import OmniGraphModel

setup_env()

class TriModalDataset(Dataset):
    def __init__(self, data_path, image_root="."):
        """
        Placeholder for Graph-Image-Text Dataset (Visual Genome / ART500K)
        """
        self.data_path = data_path
        self.image_root = image_root
        print(f"Loading Tri-Modal dataset from {data_path}...")
        
        # Mock Data Generation
        self.data = []
        for i in range(100):
            self.data.append({
                "id": f"sample_{i}",
                "image": "mock.jpg",
                "text": "A paper about graphs with a figure showing nodes.",
                "graph_nodes": 10
            })
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. Image
        pixel_values = torch.randn(3, 224, 224)
        
        # 2. Graph (Mock)
        num_nodes = item['graph_nodes']
        graph_node = torch.randn(num_nodes, 128) # Feature dim
        edge_index = torch.randint(0, num_nodes, (2, num_nodes * 2))
        graph_data = Data(graph_node=graph_node, edge_index=edge_index)
        
        # 3. Text
        text = item['text']
        
        return {
            "graph_data": graph_data,
            "pixel_values": pixel_values,
            "text": text
        }

def collate_fn_tri(batch):
    graph_data_list = [item['graph_data'] for item in batch]
    batch_graph = Batch.from_data_list(graph_data_list)
    
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        "graph_data": batch_graph,
        "pixel_values": pixel_values,
        "text": texts
    }

def train_stage3():
    # Config
    BATCH_SIZE = 2
    LR = 5e-5 # Lower LR for fine-tuning
    EPOCHS = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model (Load from Stage 2 checkpoint usually)
    model = OmniGraphModel(
        graph_model_name="clip_gt_arxiv_pub",
        llm_model_name="Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Load Stage 2 weights if available
    if os.path.exists("omnigraph_stage2.pt"):
        print("Loading Stage 2 weights...")
        model.load_state_dict(torch.load("omnigraph_stage2.pt"), strict=False)
        
    # 2. Freeze/Unfreeze
    # Stage 3: Joint Training of Projectors (and maybe Q-Formers?)
    # User said "further train GL-Projector and VL-Projector"
    
    for n, p in model.named_parameters():
        if "gl_projector" in n or "vl_projector" in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
            
    model.to(device)
    
    # 3. Data
    dataset = TriModalDataset("data/art500k.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_tri)
    
    # 4. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR)
    
    # 5. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 6. Loop
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            graph_data = batch['graph_data'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            text_input = batch['text']
            
            inputs = tokenizer(
                text_input, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(device)
            
            input_ids = inputs.input_ids
            labels = input_ids.clone()
            
            outputs = model(
                graph_data=graph_data,
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
    torch.save(model.state_dict(), "omnigraph_stage3.pt")
    print("Stage 3 Complete.")

if __name__ == "__main__":
    try:
        train_stage3()
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
