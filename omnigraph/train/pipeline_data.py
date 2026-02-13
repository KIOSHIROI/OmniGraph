from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from omnigraph.data.vg_graph_qa import build_vg_graph_qa_records
from omnigraph.data.vg_scene_graph_dataset import (
    load_vg_scene_graph_items,
    merge_scene_graph_items,
    parse_scene_graph_paths,
)


def load_merged_scene_graph_items(
    scene_graphs_path: str,
    extra_scene_graphs: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str], Dict[str, int]]:
    """
    Load base + extra scene graphs, then merge by image_id (base first).

    Returns:
      base_items,
      merged_items,
      pseudo_items (flattened extras),
      extra_paths,
      merge_stats
    """
    base_items = load_vg_scene_graph_items(scene_graphs_path)
    extra_paths = parse_scene_graph_paths(extra_scene_graphs)

    extra_items_list: List[List[Dict[str, Any]]] = []
    pseudo_items: List[Dict[str, Any]] = []
    for p in extra_paths:
        items = load_vg_scene_graph_items(p)
        extra_items_list.append(items)
        pseudo_items.extend(items)

    merged_items, merge_stats = merge_scene_graph_items(base_items, extra_items_list)
    merge_stats = dict(merge_stats)
    merge_stats["extra_files"] = len(extra_paths)
    return base_items, merged_items, pseudo_items, extra_paths, merge_stats


def build_graph_qa_records_with_pseudo(
    *,
    disable_graph_qa: bool,
    base_scene_graph_items: Sequence[Dict[str, Any]],
    pseudo_scene_graph_items: Sequence[Dict[str, Any]],
    graph_qa_max_per_image: int,
    graph_qa_repeat: int,
    pseudo_graph_qa_max_per_image: int,
    pseudo_graph_qa_repeat: int,
    graph_qa_seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build synthetic graph-QA from base and pseudo scene graphs with fixed repeats.

    Returns:
      qa_records,
      summary {
        enabled, real_pairs, real_repeat, pseudo_pairs, pseudo_repeat,
        total_pairs, type_counts
      }
    """
    summary: Dict[str, Any] = {
        "enabled": bool(not disable_graph_qa),
        "real_pairs": 0,
        "real_repeat": max(1, int(graph_qa_repeat)),
        "pseudo_pairs": 0,
        "pseudo_repeat": max(1, int(pseudo_graph_qa_repeat)),
        "total_pairs": 0,
        "type_counts": {},
    }
    if bool(disable_graph_qa):
        return [], summary

    real_qa_records = build_vg_graph_qa_records(
        scene_graph_items=list(base_scene_graph_items),
        max_per_image=max(0, int(graph_qa_max_per_image)),
        seed=int(graph_qa_seed),
    )
    real_repeat = max(1, int(graph_qa_repeat))
    if real_repeat > 1 and len(real_qa_records) > 0:
        real_qa_records = real_qa_records * real_repeat

    pseudo_qa_records: List[Dict[str, Any]] = []
    pseudo_repeat = max(1, int(pseudo_graph_qa_repeat))
    if len(pseudo_scene_graph_items) > 0 and int(pseudo_graph_qa_max_per_image) > 0:
        pseudo_qa_records = build_vg_graph_qa_records(
            scene_graph_items=list(pseudo_scene_graph_items),
            max_per_image=int(pseudo_graph_qa_max_per_image),
            seed=int(graph_qa_seed) + 1009,
        )
        if pseudo_repeat > 1 and len(pseudo_qa_records) > 0:
            pseudo_qa_records = pseudo_qa_records * pseudo_repeat

    qa_records = real_qa_records + pseudo_qa_records
    type_counts: Dict[str, int] = {}
    for rec in qa_records:
        qa_type = str(rec.get("qa_type", "unknown"))
        type_counts[qa_type] = type_counts.get(qa_type, 0) + 1

    summary.update(
        {
            "real_pairs": int(len(real_qa_records)),
            "real_repeat": int(real_repeat),
            "pseudo_pairs": int(len(pseudo_qa_records)),
            "pseudo_repeat": int(pseudo_repeat),
            "total_pairs": int(len(qa_records)),
            "type_counts": type_counts,
        }
    )
    return qa_records, summary


def split_indices_by_image_id(
    samples: Sequence[Dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
    image_id_key: str = "image_id",
    fallback_when_train_empty: bool = True,
    require_non_empty_train: bool = False,
    require_non_empty_val: bool = False,
    error_prefix: str = "Split",
) -> Tuple[List[int], List[int], Set[int], Set[int]]:
    """
    Generic image_id-level split for leakage prevention.
    """
    if len(samples) == 0:
        if require_non_empty_train or require_non_empty_val:
            raise RuntimeError(f"{error_prefix} failed: empty samples.")
        return [], [], set(), set()

    image_ids = sorted({int(s[image_id_key]) for s in samples})
    rng = random.Random(int(seed))
    rng.shuffle(image_ids)

    if len(image_ids) <= 1:
        val_ids = set(image_ids)
    else:
        n_val = max(1, int(len(image_ids) * float(val_ratio)))
        val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids) - val_ids

    train_indices: List[int] = []
    val_indices: List[int] = []
    for idx, s in enumerate(samples):
        iid = int(s[image_id_key])
        if iid in val_ids:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    if not train_indices and fallback_when_train_empty and len(val_indices) > 1:
        train_indices = val_indices[:-1]
        val_indices = val_indices[-1:]
        train_ids = {int(samples[i][image_id_key]) for i in train_indices}
        val_ids = {int(samples[i][image_id_key]) for i in val_indices}

    if require_non_empty_train and not train_indices:
        raise RuntimeError(f"{error_prefix} failed: empty train set after image_id split.")
    if require_non_empty_val and not val_indices:
        raise RuntimeError(f"{error_prefix} failed: empty val set after image_id split.")

    return train_indices, val_indices, train_ids, val_ids
