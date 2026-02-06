from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

@dataclass
class Vocab:
    stoi: Dict[str, int]
    pad: int
    unk: int

    @classmethod
    def build(cls, tokens: List[str], min_freq: int = 1) -> "Vocab":
        from collections import Counter
        c = Counter(tokens)
        itos = ["<pad>", "<unk>"]
        for t, f in c.items():
            if f >= min_freq and t not in ("<pad>", "<unk>"):
                itos.append(t)
        stoi = {t: i for i, t in enumerate(itos)}
        return cls(stoi=stoi, pad=stoi["<pad>"], unk=stoi["<unk>"])

    def id(self, tok: Optional[str]) -> int:
        if tok is None:
            return self.unk
        return self.stoi.get(tok, self.unk)

def build_vg_vocabs(items: List[Dict[str, Any]], min_freq: int = 2):
    obj_tokens, pred_tokens, attr_tokens = [], [], []
    for it in items:
        for o in it.get("objects", []) or []:
            name = (o.get("names") or ["object"])[0]
            obj_tokens.append(str(name))
            for a in (o.get("attributes", []) or [])[:8]:
                attr_tokens.append(str(a))
        for r in it.get("relationships", []) or []:
            p = r.get("predicate", None)
            if p is not None:
                pred_tokens.append(str(p))
    return (
        Vocab.build(obj_tokens, min_freq=min_freq),
        Vocab.build(pred_tokens, min_freq=min_freq),
        Vocab.build(attr_tokens, min_freq=min_freq),
    )

class VGSceneGraphDataset(Dataset):
    """
    每个 item 形如:
    {
      "image_id": int,
      "width": int, "height": int,  # 可选但建议
      "objects": [ {object_id, names:[...], attributes:[...], x,y,w,h}, ... ],
      "relationships": [ {subject_id, object_id, predicate}, ... ]
    }
    """
    def __init__(
        self,
        items: List[Dict[str, Any]],
        obj_vocab: Vocab,
        pred_vocab: Vocab,
        attr_vocab: Vocab,
        max_nodes: int = 80,
        max_attrs: int = 6,
        add_reverse_edges: bool = True,
    ):
        self.items = items
        self.obj_vocab = obj_vocab
        self.pred_vocab = pred_vocab
        self.attr_vocab = attr_vocab
        self.max_nodes = max_nodes
        self.max_attrs = max_attrs
        self.add_reverse_edges = add_reverse_edges

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        objs = list(it.get("objects", []) or [])
        rels = list(it.get("relationships", []) or [])
        W = float(it.get("width", 1.0) or 1.0)
        H = float(it.get("height", 1.0) or 1.0)

        if len(objs) == 0:
            # dummy
            data = Data(edge_index=torch.zeros((2, 0), dtype=torch.long))
            data.obj_id = torch.tensor([self.obj_vocab.unk], dtype=torch.long)
            data.attr_id = torch.full((1, self.max_attrs), self.attr_vocab.pad, dtype=torch.long)
            data.bbox = torch.zeros((1, 4), dtype=torch.float32)
            data.edge_pred_id = torch.zeros((0,), dtype=torch.long)
            return {"id": str(it.get("image_id", idx)), "graph_data": data,
                    "text": "Describe the scene graph.", "answer": "A scene with several objects."}

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
            a_ids = [self.attr_vocab.id(str(a)) for a in attrs]
            if len(a_ids) < self.max_attrs:
                a_ids += [self.attr_vocab.pad] * (self.max_attrs - len(a_ids))
            attr_ids.append(a_ids)

            x = float(o.get("x", 0.0)); y = float(o.get("y", 0.0))
            w = float(o.get("w", 0.0)); h = float(o.get("h", 0.0))
            # 归一化 xywh
            bboxes.append([x / W, y / H, w / W, h / H])

        # edges + predicates
        src, dst, pids = [], [], []
        for r in rels:
            sid = r.get("subject_id"); oid = r.get("object_id")
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
        data.obj_id = torch.tensor(obj_ids, dtype=torch.long)                 # [N]
        data.attr_id = torch.tensor(attr_ids, dtype=torch.long)              # [N, A]
        data.bbox = torch.tensor(bboxes, dtype=torch.float32)                # [N, 4]
        data.edge_pred_id = torch.tensor(pids, dtype=torch.long)             # [E]

        # 你当前训练脚本需要这些键
        return {
            "id": str(it.get("image_id", idx)),
            "graph_data": data,
            "text": "Describe the scene graph.",
            "answer": "Describe objects and relations in the scene graph."
        }