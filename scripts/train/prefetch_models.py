#!/usr/bin/env python3
from __future__ import annotations
import os
from typing import Optional
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

repo_root = _resolve_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from omnigraph.utils.env import setup_env  # pylint: disable=import-outside-toplevel

setup_env()

from huggingface_hub import snapshot_download

def _load_profiles(path: Path) -> Dict[str, Dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid profiles file: {path}")

    resolved: Dict[str, Dict[str, str]] = {}

    def resolve(name: str, stack: List[str]) -> Dict[str, str]:
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f"Profile inheritance cycle: {' -> '.join(stack + [name])}")
        node = raw.get(name)
        if not isinstance(node, dict):
            raise KeyError(f"Profile '{name}' not found in {path}")
        out: Dict[str, str] = {}
        base = node.get("_base")
        if base is not None:
            if not isinstance(base, str) or not base:
                raise ValueError(f"Invalid _base in profile '{name}': {base!r}")
            out.update(resolve(base, stack + [name]))
        for k, v in node.items():
            if not isinstance(k, str):
                continue
            if k.startswith("_"):
                continue
            out[k] = str(v)
        resolved[name] = out
        return out

    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        if key.startswith("_"):
            continue
        if not isinstance(value, dict):
            continue
        resolve(key, [])
    return resolved


def _default_models_from_profile(env: Dict[str, str]) -> List[str]:
    models: List[str] = []
    llm = str(env.get("LLM_MODEL", "")).strip()
    vision = str(env.get("VISION_MODEL", "")).strip()
    stage1_text = str(env.get("STAGE1_TEXT_MODEL_NAME", "")).strip()
    if llm:
        models.append(llm)
    if vision:
        models.append(vision)
    if stage1_text:
        models.append(stage1_text)
    return models


def main() -> int:
    ap = argparse.ArgumentParser(description="Prefetch OmniGraph Hugging Face model repos to local cache.")
    ap.add_argument("--profile", type=str, default="pro6000", help="Profile name from configs/train/infra_profiles.json")
    ap.add_argument("--profiles-file", type=str, default="configs/train/infra_profiles.json")
    ap.add_argument("--hf-cache", type=str, default="", help="Override OMNIGRAPH_HF_CACHE/HF_HOME")
    ap.add_argument("--hf-endpoint", type=str, default="", help="Override OMNIGRAPH_HF_ENDPOINT/HF_ENDPOINT")
    ap.add_argument("--model", action="append", default=[], help="Extra model repo_id to prefetch (repeatable).")
    ap.add_argument("--include-alt-llm", action="store_true", help="Also prefetch Qwen2.5-3B/7B pair for quick switching.")
    ap.add_argument("--include-legacy-qformer", action="store_true", help="Also prefetch bert-base-uncased for legacy qformer path.")
    ap.add_argument("--dry-run", action="store_true", help="Print model list only.")
    args = ap.parse_args()



    profiles_path = (repo_root / args.profiles_file).resolve()
    profiles = _load_profiles(profiles_path)
    if args.profile not in profiles:
        raise KeyError(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")
    env = profiles[args.profile]

    model_set: Set[str] = set(_default_models_from_profile(env))
    for m in args.model:
        mm = str(m).strip()
        if mm:
            model_set.add(mm)

    if args.include_alt_llm:
        # model_set.add("Qwen/Qwen2.5-3B-Instruct")
        model_set.add("Qwen/Qwen2.5-7B-Instruct")
    if args.include_legacy_qformer:
        model_set.add("bert-base-uncased")

    model_list = sorted(m for m in model_set if m)
    if not model_list:
        raise RuntimeError("No models resolved to prefetch.")

    print(f"[Prefetch] profile={args.profile}")
    print(f"[Prefetch] models={len(model_list)}")
    for idx, model_id in enumerate(model_list, start=1):
        print(f"  {idx}. {model_id}")

    if args.dry_run:
        print("[Prefetch] dry-run enabled; no download performed.")
        return 0

    for idx, model_id in enumerate(model_list, start=1):
        print(f"[Prefetch] ({idx}/{len(model_list)}) downloading: {model_id}")
        local = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            resume_download=True,
        )
        print(f"[Prefetch] done: {model_id} -> {local}")

    print("[Prefetch] all downloads finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

