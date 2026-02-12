from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple


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


def _build_predicate_stats(items: List[Dict[str, Any]]) -> List[str]:
    all_preds: List[str] = []
    for it in items:
        for r in (it.get("relationships", []) or []):
            if not isinstance(r, dict):
                continue
            pred = _clean_text(r.get("predicate", "")).lower()
            if pred:
                all_preds.append(pred)
    return sorted(set(all_preds))


def build_vg_graph_qa_records(
    scene_graph_items: List[Dict[str, Any]],
    max_per_image: int = 3,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build synthetic short-answer QA pairs from VG-style scene graphs.
    No GQA annotation is used.

    Output record:
      {"image_id": int, "question": str, "answer": str, "qa_type": str}
    """
    rng = random.Random(int(seed))
    max_per_image = max(1, int(max_per_image))

    global_names, image_obj_names = _build_object_stats(scene_graph_items)
    global_preds = _build_predicate_stats(scene_graph_items)
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

        qa_by_type: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        seen_qa: Set[Tuple[str, str]] = set()

        def add_qa(qa_type: str, q: str, a: str) -> None:
            q = _clean_text(q)
            a = _clean_text(a).lower()
            if not q or not a:
                return
            key = (q.lower(), a)
            if key in seen_qa:
                return
            seen_qa.add(key)
            qa_by_type[str(qa_type)].append((q, a))

        def sample_absent_name(present: Set[str], tries: int = 64) -> str:
            if not global_names:
                return ""
            for _ in range(int(tries)):
                cand = rng.choice(global_names)
                if cand and cand not in present:
                    return cand
            return ""

        pos_names = [n for n in name_count.keys() if n]
        rng.shuffle(pos_names)
        present_name_set = set(pos_names)

        # 1) verify existence (yes/no)
        for n in pos_names[:2]:
            add_qa("verify_exist", f"Is there a {n} in the scene? Answer yes or no.", "yes")
        neg_name = sample_absent_name(present_name_set)
        if neg_name:
            add_qa("verify_exist", f"Is there a {neg_name} in the scene? Answer yes or no.", "no")

        # 2) query attribute
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
            add_qa("query_attr", f"What is an attribute of the {name}?", attr)

        # 3) relation query + relation verification
        rel_pairs: List[Tuple[str, str, str]] = []
        pair2preds: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
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
            if s_name and o_name and pred and s_name != o_name:
                rel_pairs.append((s_name, o_name, pred))
                pair2preds[(s_name, o_name)].add(pred)

        rng.shuffle(rel_pairs)
        for s_name, o_name, pred in rel_pairs[:2]:
            add_qa("query_rel", f"What is the relation between the {s_name} and the {o_name}?", pred)
            add_qa("verify_rel", f"Is the {s_name} {pred} the {o_name}? Answer yes or no.", "yes")

            neg_pred = ""
            if global_preds:
                existing = pair2preds.get((s_name, o_name), set())
                for _ in range(32):
                    cand_pred = rng.choice(global_preds)
                    if cand_pred and cand_pred not in existing:
                        neg_pred = cand_pred
                        break
            if neg_pred:
                add_qa("verify_rel", f"Is the {s_name} {neg_pred} the {o_name}? Answer yes or no.", "no")

        # 4) counting/query + compare/choose
        count_pairs = list(name_count.items())
        rng.shuffle(count_pairs)
        for n, c in count_pairs[:2]:
            add_qa("query_count", f"How many {n} are in the scene?", _to_word(c))

        if len(count_pairs) >= 2:
            compare_pool = []
            for i in range(len(count_pairs)):
                for j in range(i + 1, len(count_pairs)):
                    n1, c1 = count_pairs[i]
                    n2, c2 = count_pairs[j]
                    if c1 != c2:
                        compare_pool.append((n1, c1, n2, c2))
            rng.shuffle(compare_pool)
            for n1, c1, n2, c2 in compare_pool[:2]:
                add_qa(
                    "compare_count",
                    f"Are there more instances of {n1} than {n2} in the scene? Answer yes or no.",
                    "yes" if c1 > c2 else "no",
                )
                add_qa(
                    "choose_count",
                    f"Which appears more often in the scene, {n1} or {n2}?",
                    n1 if c1 > c2 else n2,
                )

        # 5) logical (and/or existence)
        if len(pos_names) >= 2:
            a, b = pos_names[0], pos_names[1]
            add_qa("logical_and", f"Are there both a {a} and a {b} in the scene? Answer yes or no.", "yes")
            add_qa("logical_or", f"Is there either a {a} or a {b} in the scene? Answer yes or no.", "yes")

        neg_a = sample_absent_name(present_name_set)
        if pos_names and neg_a:
            a = pos_names[0]
            add_qa("logical_and", f"Are there both a {a} and a {neg_a} in the scene? Answer yes or no.", "no")
            add_qa("logical_or", f"Is there either a {a} or a {neg_a} in the scene? Answer yes or no.", "yes")

        neg_b = sample_absent_name(present_name_set.union({neg_a}) if neg_a else present_name_set)
        if neg_a and neg_b:
            add_qa("logical_or", f"Is there either a {neg_a} or a {neg_b} in the scene? Answer yes or no.", "no")

        # Balanced selection by qa type (avoid over-dominating query-only templates)
        category_order = [
            "logical_and",
            "logical_or",
            "verify_rel",
            "compare_count",
            "choose_count",
            "query_rel",
            "query_attr",
            "query_count",
            "verify_exist",
        ]
        for cat in list(qa_by_type.keys()):
            rng.shuffle(qa_by_type[cat])

        selected: List[Tuple[str, str, str]] = []
        while len(selected) < max_per_image:
            progressed = False
            for cat in category_order:
                bucket = qa_by_type.get(cat, [])
                if not bucket:
                    continue
                q, a = bucket.pop()
                selected.append((cat, q, a))
                progressed = True
                if len(selected) >= max_per_image:
                    break
            if not progressed:
                break

        for qa_type, q, a in selected:
            records.append({"image_id": image_id, "question": q, "answer": a, "qa_type": qa_type})

    return records
