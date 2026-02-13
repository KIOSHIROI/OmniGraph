from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _clean_text(x: Any) -> str:
    if x is None:
        return ""
    s = " ".join(str(x).strip().split())
    return s


def _clean_token(x: Any) -> str:
    return _clean_text(x).lower()


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _stable_int_id(x: Any) -> int:
    s = _clean_text(x)
    if not s:
        return 0
    try:
        return int(s)
    except Exception:
        digest = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
        # positive 31-bit int for better compatibility with legacy code paths.
        return int.from_bytes(digest, byteorder="little", signed=False) % 2147483647


def _extract_json_substring(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1)

    l = s.find("{")
    r = s.rfind("}")
    if l >= 0 and r > l:
        return s[l : r + 1]
    return s


def parse_graph_json(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if not isinstance(x, str):
        raise ValueError("graph_json must be dict or string")

    candidate = _extract_json_substring(x)
    if not candidate:
        raise ValueError("empty graph_json text")

    try:
        obj = json.loads(candidate)
    except Exception as e:
        raise ValueError(f"failed to parse graph_json: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError("graph_json root must be object")
    return obj


def _iter_object_records(raw_objects: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw_objects, list):
        for o in raw_objects:
            if isinstance(o, dict):
                yield o
    elif isinstance(raw_objects, dict):
        for k, o in raw_objects.items():
            if isinstance(o, dict):
                out = dict(o)
                if "object_id" not in out:
                    out["object_id"] = k
                yield out


def _iter_relationship_records(raw_rels: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw_rels, list):
        for r in raw_rels:
            if isinstance(r, dict):
                yield r


def _collect_relationships_from_objects(raw_objects: Any) -> List[Dict[str, Any]]:
    rels: List[Dict[str, Any]] = []
    for o in _iter_object_records(raw_objects):
        sid = o.get("object_id")
        for r in (o.get("relations", []) or []):
            if not isinstance(r, dict):
                continue
            rels.append(
                {
                    "subject_id": sid,
                    "object_id": r.get("object", r.get("object_id")),
                    "predicate": r.get("name", r.get("predicate", "")),
                }
            )
    return rels


def canonicalize_scene_graph(
    graph: Dict[str, Any] | str,
    *,
    image_id: Optional[Any] = None,
    image_path: Optional[str] = None,
    width: Optional[Any] = None,
    height: Optional[Any] = None,
    max_attrs_per_object: int = 8,
) -> Dict[str, Any]:
    obj = parse_graph_json(graph)

    raw_image_id = image_id if image_id is not None else obj.get("image_id")
    if raw_image_id is None and image_path:
        raw_image_id = image_path

    raw_objects = obj.get("objects", [])
    raw_rels = obj.get("relationships", [])

    objects: List[Dict[str, Any]] = []
    seen_oid: set[int] = set()
    next_oid = 1

    for o in _iter_object_records(raw_objects):
        oid = _as_int(o.get("object_id", next_oid), next_oid)
        while oid in seen_oid:
            oid += 1
        seen_oid.add(oid)
        next_oid = max(next_oid, oid + 1)

        names = o.get("names", []) or []
        if not names:
            fallback_name = o.get("name", "object")
            names = [fallback_name]
        name = _clean_token(names[0]) or "object"

        attrs = []
        seen_attr = set()
        for a in (o.get("attributes", []) or [])[: max(1, int(max_attrs_per_object))]:
            t = _clean_token(a)
            if not t or t in seen_attr:
                continue
            seen_attr.add(t)
            attrs.append(t)

        objects.append(
            {
                "object_id": int(oid),
                "names": [name],
                "attributes": attrs,
                "x": _as_float(o.get("x", 0.0), 0.0),
                "y": _as_float(o.get("y", 0.0), 0.0),
                "w": _as_float(o.get("w", 0.0), 0.0),
                "h": _as_float(o.get("h", 0.0), 0.0),
            }
        )

    valid_oids = {int(o["object_id"]) for o in objects}

    rel_candidates = list(_iter_relationship_records(raw_rels))
    if not rel_candidates:
        rel_candidates = _collect_relationships_from_objects(raw_objects)

    relationships: List[Dict[str, Any]] = []
    seen_rel = set()
    for r in rel_candidates:
        sid = _as_int(r.get("subject_id", -1), -1)
        oid = _as_int(r.get("object_id", -1), -1)
        pred = _clean_token(r.get("predicate", ""))
        if sid not in valid_oids or oid not in valid_oids:
            continue
        if not pred:
            continue
        key = (sid, pred, oid)
        if key in seen_rel:
            continue
        seen_rel.add(key)
        relationships.append(
            {
                "subject_id": int(sid),
                "object_id": int(oid),
                "predicate": pred,
            }
        )

    objects = sorted(objects, key=lambda x: int(x.get("object_id", 0)))
    relationships = sorted(
        relationships,
        key=lambda x: (int(x.get("subject_id", 0)), str(x.get("predicate", "")), int(x.get("object_id", 0))),
    )

    normalized_image_id = _stable_int_id(raw_image_id)
    out = {
        "image_id": int(normalized_image_id),
        "width": _as_int(width if width is not None else obj.get("width", 0), 0),
        "height": _as_int(height if height is not None else obj.get("height", 0), 0),
        "objects": objects,
        "relationships": relationships,
    }
    if str(raw_image_id) and str(raw_image_id).strip() != str(normalized_image_id):
        out["source_image_id"] = str(raw_image_id).strip()
    return out


def canonical_graph_hash(graph: Dict[str, Any]) -> str:
    blob = json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def graph_to_text(graph: Dict[str, Any], max_objects: int = 24, max_relationships: int = 32) -> str:
    id2name = {}
    obj_phrases: List[str] = []
    for o in (graph.get("objects", []) or [])[: max(1, int(max_objects))]:
        oid = _as_int(o.get("object_id", 0), 0)
        names = o.get("names", []) or ["object"]
        name = _clean_token(names[0]) or "object"
        id2name[oid] = name
        attrs = [
            _clean_token(a)
            for a in (o.get("attributes", []) or [])[:4]
            if _clean_token(a)
        ]
        if attrs:
            obj_phrases.append(f"{name} ({', '.join(attrs)})")
        else:
            obj_phrases.append(name)

    rel_phrases: List[str] = []
    for r in (graph.get("relationships", []) or [])[: max(1, int(max_relationships))]:
        sid = _as_int(r.get("subject_id", -1), -1)
        oid = _as_int(r.get("object_id", -1), -1)
        pred = _clean_token(r.get("predicate", ""))
        if not pred:
            continue
        sname = id2name.get(sid, "object")
        oname = id2name.get(oid, "object")
        rel_phrases.append(f"{sname} {pred} {oname}")

    a = "; ".join(obj_phrases)
    b = "; ".join(rel_phrases)
    if a and b:
        return f"Objects: {a}. Relations: {b}."
    if a:
        return f"Objects: {a}."
    if b:
        return f"Relations: {b}."
    return ""


def _tokenize_for_similarity(text: str) -> List[str]:
    s = _clean_token(text)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def token_jaccard_similarity(a: str, b: str) -> float:
    ta = set(_tokenize_for_similarity(a))
    tb = set(_tokenize_for_similarity(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta.intersection(tb))
    union = len(ta.union(tb))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def graph_structural_validity(
    graph: Dict[str, Any],
    *,
    min_nodes: int = 2,
    max_nodes: int = 36,
    min_rels: int = 1,
    max_rels: int = 72,
) -> Tuple[bool, List[str], Dict[str, int]]:
    reasons: List[str] = []
    objs = graph.get("objects", []) or []
    rels = graph.get("relationships", []) or []

    n_nodes = len(objs)
    n_rels = len(rels)
    dangling = 0

    if n_nodes < int(min_nodes):
        reasons.append("too_few_nodes")
    if n_nodes > int(max_nodes):
        reasons.append("too_many_nodes")
    if n_rels < int(min_rels):
        reasons.append("too_few_relationships")
    if n_rels > int(max_rels):
        reasons.append("too_many_relationships")

    valid_oids = set()
    for o in objs:
        if not isinstance(o, dict):
            reasons.append("invalid_object_record")
            continue
        oid = _as_int(o.get("object_id", -1), -1)
        if oid < 0:
            reasons.append("invalid_object_id")
        valid_oids.add(oid)
        names = o.get("names", []) or []
        if not names or not _clean_token(names[0]):
            reasons.append("empty_object_name")

    for r in rels:
        if not isinstance(r, dict):
            reasons.append("invalid_relationship_record")
            continue
        sid = _as_int(r.get("subject_id", -1), -1)
        oid = _as_int(r.get("object_id", -1), -1)
        pred = _clean_token(r.get("predicate", ""))
        if sid not in valid_oids or oid not in valid_oids:
            dangling += 1
        if not pred:
            reasons.append("empty_predicate")

    if dangling > 0:
        reasons.append("dangling_relationship")

    stats = {
        "num_nodes": int(n_nodes),
        "num_relationships": int(n_rels),
        "dangling_relationships": int(dangling),
    }
    return len(reasons) == 0, sorted(set(reasons)), stats


def confidence_from_avg_logprob(avg_logprob: float) -> float:
    x = float(avg_logprob)
    # map roughly [-2.5, 0] to [0, 1]
    y = (x + 2.5) / 2.5
    if y < 0.0:
        return 0.0
    if y > 1.0:
        return 1.0
    return float(y)


def aggregate_pseudo_score(avg_logprob: float, text_similarity: float, vision_similarity: float) -> float:
    conf = confidence_from_avg_logprob(avg_logprob)
    score = 0.45 * conf + 0.25 * float(text_similarity) + 0.30 * float(vision_similarity)
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)
