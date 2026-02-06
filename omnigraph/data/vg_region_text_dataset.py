from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data import Dataset

class VGGraphRegionTextDataset(Dataset):
    def __init__(
        self,
        scene_graph_dataset,  # 你的 VGSceneGraphDataset 实例
        region_pairs: List[Tuple[int, str, Optional[Tuple[float,float,float,float]]]],
        prompt: str = "Describe the region.",
    ):
        self.sg = scene_graph_dataset
        self.prompt = prompt

        # 建 image_id -> idx 映射，保证 O(1) 取到对应 scene graph
        # 你的 VGSceneGraphDataset 里 items 是从 scene_graphs.json 读出来的 list
        self.image2idx: Dict[int, int] = {}
        for i, it in enumerate(self.sg.items):
            if isinstance(it, dict) and "image_id" in it:
                self.image2idx[int(it["image_id"])] = i

        # 过滤掉找不到 scene graph 的 region
        self.samples: List[Tuple[int, str, Optional[Tuple[float,float,float,float]]]] = []
        for image_id, phrase, bbox in region_pairs:
            if image_id in self.image2idx:
                self.samples.append((image_id, phrase, bbox))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id, phrase, _bbox = self.samples[idx]
        sg_idx = self.image2idx[image_id]

        # 直接复用 scene graph dataset 的建图逻辑（整图）
        sg_item = self.sg[sg_idx]  # 返回 dict: {id, graph_data, text, answer}

        return {
            "id": f"{image_id}_r{idx}",
            "graph_data": sg_item["graph_data"],
            "text": self.prompt,
            "answer": phrase,
        }