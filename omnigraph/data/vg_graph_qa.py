from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Tuple


def _clean_text(s: Any) -> str:
    if s is None:
        return ""
    return " ".join(str(s).strip().split())


def _to_word(n: int) -> str:
    small = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
        16: "sixteen",
        17: "seventeen",
        18: "eighteen",
        19: "nineteen",
        20: "twenty",
    }
    return small.get(int(n), str(int(n)))


def _build_object_stats(items: List[Dict[str, Any]]) -> Tuple[List[str], Dict[int, Dict[int, str]]]:
    all_names = []
    per_image_obj_name: Dict[int, Dict[int, str]] = {}

    for it in items:
        try:
            image_id = int(it.get("image_id"))
        except Exception:
            continue

        id2name: Dict[int, str] = {}
        for o in (it.get("objects", []) or []):
            if not isinstance(o, dict):
                continue
            oid = o.get("object_id", None)
            if oid is None:
                continue
            try:
                oid = int(oid)
            except Exception:
                continue
            names = o.get("names", []) or []
            name = _clean_text(names[0] if len(names) > 0 else "object").lower()
            if not name:
                continue
            id2name[oid] = name
            all_names.append(name)
        per_image_obj_name[image_id] = id2name

    unique_names = sorted(set(all_names))
    return unique_names, per_image_obj_name


def build_vg_graph_qa_records(
    scene_graph_items: List[Dict[str, Any]],
    max_per_image: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build synthetic short-answer QA pairs from VG-style scene graphs.
    No GQA annotation is used.

    Output record:
      {"image_id": int, "question": str, "answer": str}
    """
    rng = random.Random(int(seed))
    max_per_image = max(1, int(max_per_image))

    global_names, image_obj_names = _build_object_stats(scene_graph_items)
    records: List[Dict[str, Any]] = []

    for it in scene_graph_items:
        try:
            image_id = int(it.get("image_id"))
        except Exception:
            continue

        objects = [o for o in (it.get("objects", []) or []) if isinstance(o, dict)]
        if len(objects) == 0:
            continue

        id2name = image_obj_names.get(image_id, {})
        obj_name_list = [n for n in id2name.values() if n]
        if len(obj_name_list) == 0:
            continue
        name_count = Counter(obj_name_list)

        qa_candidates: List[Tuple[str, str]] = []

        # 1) Existence positive
        pos_names = list(name_count.keys())
        rng.shuffle(pos_names)
        for n in pos_names[:2]:
            q = f"Is there a {n} in the scene? Answer yes or no."
            qa_candidates.append((q, "yes"))

        # 2) Existence negative (from global object names)
        neg_name = None
        if len(global_names) > 0:
            for _ in range(32):
                cand = rng.choice(global_names)
                if cand not in name_count:
                    neg_name = cand
                    break
        if neg_name:
            q = f"Is there a {neg_name} in the scene? Answer yes or no."
            qa_candidates.append((q, "no"))

        # 3) Attribute query
        attr_pairs: List[Tuple[str, str]] = []
        for o in objects:
            names = o.get("names", []) or []
            name = _clean_text(names[0] if len(names) > 0 else "object").lower()
            if not name:
                continue
            attrs = o.get("attributes", []) or []
            for a in attrs[:2]:
                attr = _clean_text(a).lower()
                if attr:
                    attr_pairs.append((name, attr))
        rng.shuffle(attr_pairs)
        for name, attr in attr_pairs[:2]:
            q = f"What is an attribute of the {name}?"
            qa_candidates.append((q, attr))

        # 4) Relation query
        rel_pairs: List[Tuple[str, str, str]] = []
        for r in (it.get("relationships", []) or []):
            if not isinstance(r, dict):
                continue
            sid = r.get("subject_id", None)
            oid = r.get("object_id", None)
            if sid is None or oid is None:
                continue
            try:
                sid = int(sid)
                oid = int(oid)
            except Exception:
                continue
            s_name = id2name.get(sid, "")
            o_name = id2name.get(oid, "")
            pred = _clean_text(r.get("predicate", "")).lower()
            if s_name and o_name and pred:
                rel_pairs.append((s_name, o_name, pred))
        rng.shuffle(rel_pairs)
        for s_name, o_name, pred in rel_pairs[:2]:
            q = f"What is the relation between the {s_name} and the {o_name}?"
            qa_candidates.append((q, pred))

        # 5) Counting query
        multi = [(n, c) for n, c in name_count.items() if c > 1]
        rng.shuffle(multi)
        if len(multi) > 0:
            n, c = multi[0]
            q = f"How many {n} are in the scene?"
            qa_candidates.append((q, _to_word(c)))

        # Deduplicate + cap
        dedup = []
        seen = set()
        for q, a in qa_candidates:
            q = _clean_text(q)
            a = _clean_text(a).lower()
            if not q or not a:
                continue
            key = (q.lower(), a.lower())
            if key in seen:
                continue
            seen.add(key)
            dedup.append((q, a))

        rng.shuffle(dedup)
        for q, a in dedup[:max_per_image]:
            records.append({"image_id": image_id, "question": q, "answer": a})

    return records
