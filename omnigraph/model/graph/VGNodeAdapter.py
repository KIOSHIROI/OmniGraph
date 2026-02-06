import torch
import torch.nn as nn
import torch.nn.functional as F

class VGNodeAdapter(nn.Module):
    def __init__(self, num_obj: int, num_attr: int, out_dim: int = 128,
                 obj_dim: int = 96, attr_dim: int = 64, bbox_dim: int = 32):
        super().__init__()
        self.obj_emb = nn.Embedding(num_obj, obj_dim)
        self.attr_emb = nn.Embedding(num_attr, attr_dim)
        self.bbox_mlp = nn.Sequential(
            nn.Linear(4, bbox_dim),
            nn.GELU(),
            nn.Linear(bbox_dim, bbox_dim),
        )
        self.proj = nn.Sequential(
            nn.Linear(obj_dim + attr_dim + bbox_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, obj_id: torch.Tensor, attr_id: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        # obj_id: [N], attr_id: [N,A], bbox:[N,4]
        o = self.obj_emb(obj_id)                              # [N, obj_dim]
        a = self.attr_emb(attr_id)                            # [N, A, attr_dim]
        a = a.mean(dim=1)                                     # [N, attr_dim]（简单平均池化）
        b = self.bbox_mlp(bbox)                               # [N, bbox_dim]
        x = torch.cat([o, a, b], dim=-1)
        return self.proj(x)                                   # [N, 128]