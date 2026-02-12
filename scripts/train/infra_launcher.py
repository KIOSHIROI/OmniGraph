#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _parse_kv(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value (expected KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Invalid --set key: {item}")
        out[k] = v
    return out


def _build_run_id(prefix: str | None = None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = uuid4().hex[:8]
    if prefix:
        p = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in prefix.strip())
        p = p.strip("_")
        if p:
            return f"{p}_{ts}_{short}"
    return f"run_{ts}_{short}"


def _load_profiles(path: Path) -> dict[str, dict[str, str]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid profile file: {path}")
    out: dict[str, dict[str, str]] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not isinstance(v, dict):
            continue
        out[k] = {str(kk): str(vv) for kk, vv in v.items()}
    return out


def _resolved_paths(repo: Path, run_dir: Path) -> dict[str, str]:
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "REPO": str(repo),
        "STAGE1_EXPORT_PATH": str(ckpt_dir / "graph_qformer_stage1.pt"),
        "STAGE2A_DIR": str(ckpt_dir / "stage2A"),
        "STAGE2B_DIR": str(ckpt_dir / "stage2B"),
        "STAGE3_DIR": str(ckpt_dir / "stage3"),
        "STAGE2B_R2_DIR": str(ckpt_dir / "stage2B_round2"),
        "STAGE3_R2_DIR": str(ckpt_dir / "stage3_round2"),
        "GQA_PRED_PAPER": str(eval_dir / "pred_val_balanced_paper.jsonl"),
        "GQA_EVAL_PAPER": str(eval_dir / "eval_val_balanced_paper.txt"),
        "GQA_PRED_R2": str(eval_dir / "pred_val_balanced_round2.jsonl"),
        "GQA_EVAL_R2": str(eval_dir / "eval_val_balanced_round2.txt"),
    }


def _run_and_tee(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fp.write(line)
        return proc.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardized launcher for OmniGraph sprint pipeline.")
    parser.add_argument("--profile", type=str, default="4090", help="Profile name from configs/train/infra_profiles.json")
    parser.add_argument("--mode", type=str, default="full", choices=["stage1", "full", "stage2a", "stage2b", "stage3", "eval"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--profiles-file", type=str, default="configs/train/infra_profiles.json")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="Override env as KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    profiles_path = (repo / args.profiles_file).resolve()
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profiles_path}")

    profiles = _load_profiles(profiles_path)
    if args.profile not in profiles:
        raise KeyError(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")

    overrides = _parse_kv(args.sets)
    run_id = _build_run_id(args.run_name or args.profile)
    out_root = (repo / args.output_root).resolve()
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMNIGRAPH_HF_CACHE", "/media/disk/02drive/13hias/.cache")
    env.setdefault("OMNIGRAPH_HF_ENDPOINT", "https://hf-mirror.com")
    env.update(profiles[args.profile])
    env.update(_resolved_paths(repo, run_dir))
    env["GPU"] = str(args.gpu)
    env["PIPELINE_MODE"] = args.mode
    env["RUN_ID"] = run_id
    env.update(overrides)

    if args.mode == "stage1":
        command = ["bash", str((repo / "train_stage1.sh").resolve())]
    else:
        command = ["bash", str((repo / "scripts/train/run_4090_gqa_sprint.sh").resolve())]
    cmd_str = " ".join(shlex.quote(x) for x in command)

    manifest = {
        "run_id": run_id,
        "profile": args.profile,
        "mode": args.mode,
        "gpu": args.gpu,
        "repo": str(repo),
        "command": cmd_str,
        "profiles_file": str(profiles_path),
        "profile_env": profiles[args.profile],
        "overrides": overrides,
        "resolved_env": {k: env[k] for k in sorted(set(profiles[args.profile].keys()) | set(_resolved_paths(repo, run_dir).keys()) | set(overrides.keys()) | {"GPU", "PIPELINE_MODE", "RUN_ID"})},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "command.sh").write_text(cmd_str + "\n", encoding="utf-8")

    print(f"[Infra] run_id={run_id}")
    print(f"[Infra] profile={args.profile} mode={args.mode} gpu={args.gpu}")
    print(f"[Infra] run_dir={run_dir}")
    print(f"[Infra] manifest={manifest_path}")

    if args.dry_run:
        print("[Infra] dry-run enabled; command not executed.")
        return 0

    log_path = run_dir / "run.log"
    rc = _run_and_tee(command, env=env, log_path=log_path)

    status = {
        "run_id": run_id,
        "return_code": rc,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
        "manifest": str(manifest_path),
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Infra] exit_code={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
