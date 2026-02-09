#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Select checkpoint path from stage meta json.")
    ap.add_argument("--meta", required=True, help="Path to stage meta json.")
    ap.add_argument("--fallback", required=True, help="Fallback checkpoint path.")
    ap.add_argument("--key", default="best_ckpt", help="Meta field for preferred checkpoint path.")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    fallback = str(args.fallback)

    if not meta_path.exists():
        print(fallback)
        return 0

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        print(fallback)
        return 0

    preferred = str(meta.get(args.key, "") or "").strip()
    if preferred:
        print(preferred)
        return 0

    print(fallback)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
