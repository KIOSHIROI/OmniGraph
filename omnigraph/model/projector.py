import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ProjectorConfig:
    input_dim: int
    output_dim: int
    hidden_dim: int = None
    dropout: float = 0.0
    type: str = "mlp"

class Projector(nn.Module):
    """
    Generic Projector module to map features from one dimension to another (e.g., to LLM embedding space).
    Can be a simple Linear layer or an MLP.
    """
    def __init__(self, config: ProjectorConfig = None, input_dim=None, output_dim=None, hidden_dim=None, dropout=0.0, type="mlp"):
        super().__init__()
        
        # Support initializing from config or individual arguments
        if config is not None:
            self.config = config
            input_dim = config.input_dim
            output_dim = config.output_dim
            hidden_dim = config.hidden_dim
            dropout = config.dropout
            type = config.type
        else:
            # Create config from args for consistency
            self.config = ProjectorConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                type=type
            )
        
        self.type = type
        
        if type == "linear":
            self.net = nn.Linear(input_dim, output_dim)
        elif type == "mlp":
            if hidden_dim is None:
                hidden_dim = output_dim
            
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"Unknown projector type: {type}")

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    print("Initializing Projector...")
    
    # Test parameters
    batch_size = 2
    seq_len = 10
    input_dim = 768
    output_dim = 4096 # e.g., LLaMA-7B hidden size
    
    # 1. Test MLP Projector with args
    print("\nTesting MLP Projector (args)...")
    projector_mlp = Projector(input_dim=input_dim, output_dim=output_dim, type="mlp")
    print(projector_mlp)
    
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output_mlp = projector_mlp(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"MLP Output shape: {output_mlp.shape}")
    assert output_mlp.shape == (batch_size, seq_len, output_dim)
    print("MLP Projector (args) test passed.")
    
    # 2. Test Linear Projector with Config
    print("\nTesting Linear Projector (config)...")
    config = ProjectorConfig(input_dim=input_dim, output_dim=output_dim, type="linear")
    projector_linear = Projector(config=config)
    print(projector_linear)
    
    output_linear = projector_linear(dummy_input)
    
    print(f"Linear Output shape: {output_linear.shape}")
    assert output_linear.shape == (batch_size, seq_len, output_dim)
    print("Linear Projector (config) test passed.")
