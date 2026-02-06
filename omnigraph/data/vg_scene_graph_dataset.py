import json
from typing import Any, Dict, List, Optional
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

class Vocab:
    def __init__(self, stoi: Dict[str, int]):
        self.stoi = stoi
        self.pad = stoi["<pad>"]
        self.unk = stoi["<unk>"]

    @classmethod
    def build(cls, tokens: List[str], min_freq: int = 2) -> "Vocab":
        from collections import Counter
        c = Counter(tokens)
        itos = ["<pad>", "<unk>"]
        for t, f in c.items():
            if f >= min_freq and t not in ("<pad>", "<unk>"):
                itos.append(t)
        return cls({t: i for i, t in enumerate(itos)})

    def id(self, tok: Optional[str]) -> int:
        if tok is None:
            return self.unk
        return self.stoi.get(tok, self.unk)

def build_vg_vocabs_from_file(path: str, min_freq: int = 2):
    items = json.load(open(path, "r", encoding="utf-8"))
    obj_t, pred_t, attr_t = [], [], []
    for it in items:
        for o in it.get("objects", []) or []:
            name = (o.get("names") or ["object"])[0]
            obj_t.append(str(name))
            for a in (o.get("attributes", []) or [])[:8]:
                attr_t.append(str(a))
        for r in it.get("relationships", []) or []:
            p = r.get("predicate", None)
            if p is not None:
                pred_t.append(str(p))
    return Vocab.build(obj_t, min_freq), Vocab.build(pred_t, min_freq), Vocab.build(attr_t, min_freq)

class VGSceneGraphDataset(Dataset):
    def __init__(
        self,
        scene_graphs_path: str,
        obj_vocab: Vocab,
        pred_vocab: Vocab,
        attr_vocab: Vocab,
        max_nodes: int = 80,
        max_attrs: int = 6,
        add_reverse_edges: bool = True,
        # VG 原始文件通常没有 width/height；如果没有，就用 bbox 的 max 做归一化
        use_bbox_max_norm: bool = True,
    ):
        self.items: List[Dict[str, Any]] = json.load(open(scene_graphs_path, "r", encoding="utf-8"))
        self.obj_vocab = obj_vocab
        self.pred_vocab = pred_vocab
        self.attr_vocab = attr_vocab
        self.max_nodes = max_nodes
        self.max_attrs = max_attrs
        self.add_reverse_edges = add_reverse_edges
        self.use_bbox_max_norm = use_bbox_max_norm

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        objs = list(it.get("objects", []) or [])
        rels = list(it.get("relationships", []) or [])
        sample_id = str(it.get("image_id", idx))

        if len(objs) == 0:
            data = Data(edge_index=torch.zeros((2, 0), dtype=torch.long))
            data.obj_id = torch.tensor([self.obj_vocab.unk], dtype=torch.long)
            data.attr_id = torch.full((1, self.max_attrs), self.attr_vocab.pad, dtype=torch.long)
            data.bbox = torch.zeros((1, 4), dtype=torch.float32)
            data.edge_pred_id = torch.zeros((0,), dtype=torch.long)
            data.num_nodes = 1
            return {"id": sample_id, "graph_data": data, "text": "Describe the scene graph.", "answer": ""}

        # 截断：按面积
        def area(o): return float(o.get("w", 0.0)) * float(o.get("h", 0.0))
        objs = sorted(objs, key=area, reverse=True)[: self.max_nodes]

        objid2idx = {}
        obj_ids, attr_ids, bboxes = [], [], []
        for i, o in enumerate(objs):
            oid = int(o.get("object_id", i))
            objid2idx[oid] = i

            name = (o.get("names") or ["object"])[0]
            obj_ids.append(self.obj_vocab.id(str(name)))

            attrs = (o.get("attributes", []) or [])[: self.max_attrs]
            a = [self.attr_vocab.id(str(x)) for x in attrs]
            if len(a) < self.max_attrs:
                a += [self.attr_vocab.pad] * (self.max_attrs - len(a))
            attr_ids.append(a)

            x = float(o.get("x", 0.0)); y = float(o.get("y", 0.0))
            w = float(o.get("w", 0.0)); h = float(o.get("h", 0.0))
            bboxes.append([x, y, w, h])

        bbox = torch.tensor(bboxes, dtype=torch.float32)  # [N,4]
        if self.use_bbox_max_norm and bbox.numel() > 0:
            # 用该图内 bbox 的 max 做归一化，避免没有图像宽高时尺度失控
            scale = bbox.max(dim=0).values.clamp_min(1.0)
            bbox = bbox / scale

        # edges + predicate ids
        src, dst, pids = [], [], []
        for r in rels:
            sid = r.get("subject_id", None)
            oid = r.get("object_id", None)
            if sid is None or oid is None:
                continue
            sid = int(sid); oid = int(oid)
            if sid not in objid2idx or oid not in objid2idx:
                continue
            s = objid2idx[sid]; o = objid2idx[oid]
            p = self.pred_vocab.id(str(r.get("predicate", None)))
            src.append(s); dst.append(o); pids.append(p)
            if self.add_reverse_edges:
                src.append(o); dst.append(s); pids.append(p)

        edge_index = torch.tensor([src, dst], dtype=torch.long) if len(src) else torch.zeros((2, 0), dtype=torch.long)

        data = Data(edge_index=edge_index)
        data.obj_id = torch.tensor(obj_ids, dtype=torch.long)            # [N]
        data.attr_id = torch.tensor(attr_ids, dtype=torch.long)          # [N,A]
        data.bbox = bbox                                                 # [N,4]
        data.edge_pred_id = torch.tensor(pids, dtype=torch.long)         # [E]
        data.num_nodes = len(obj_ids)

        return {
            "id": sample_id,
            "graph_data": data,
            "text": "Describe the scene graph.",
            "answer": "Describe objects and relations in the scene graph."
        }