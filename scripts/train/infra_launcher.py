#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from socket import gethostname
from uuid import uuid4

PROFILE_META_KEYS = {"_base", "_description"}
PROFILE_REQUIRED_KEYS = {
    "LLM_MODEL",
    "VISION_MODEL",
    "GRAPH_TOKENIZER_TYPE",
    "STAGE2A_BOOTSTRAP_MODE",
    "NODE_ENCODER_TYPE",
}
PROFILE_BOOL_KEYS = {
    "ISOLATE_GPU",
    "AUTO_BATCH_BY_VRAM",
    "AUTO_BATCH_RETRY_ON_OOM",
    "LOW_VRAM_4090",
    "USE_QA_TYPE_TOKEN",
    "ENABLE_GVL_ADAPTER",
    "ENABLE_GRAPH_AUX_HEAD",
    "ENABLE_XTC",
    "ENABLE_XTM",
    "RUN_ROUND2_ON_FAIL",
    "TOKENIZERS_PARALLELISM",
}
PROFILE_ENUM_KEYS = {
    "GRAPH_TOKENIZER_TYPE": {"perceiver", "qformer"},
    "STAGE2A_BOOTSTRAP_MODE": {"auto", "no_stage1", "legacy_stage1"},
    "NODE_ENCODER_TYPE": {"legacy_vg", "open_vocab", "hybrid"},
}
DATA_PATH_KEYS = {
    "OMNIGRAPH_DATA_ROOT",
    "VG_SCENE_GRAPHS",
    "VG_REGIONS",
    "VG_IMAGE_ROOT",
    "EXTRA_SCENE_GRAPHS",
    "STAGE1_QFORMER_CKPT",
    "GRAPH_TOKENIZER_PRETRAIN_CKPT",
    "GQA_QUESTIONS_JSON",
    "GQA_QUESTIONS_JSONL",
    "GQA_SCENE_RAW",
    "GQA_SCENE_VG",
    "GQA_IMAGE_ROOT",
    "GQA_PRED_PAPER",
    "GQA_EVAL_PAPER",
    "GQA_PRED_R2",
    "GQA_EVAL_R2",
}


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


def _sanitize_token(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip())
    return cleaned.strip("_")


def _read_json_no_conflict(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    for marker in ("<<<<<<<", "=======", ">>>>>>>"):
        if marker in text:
            raise ValueError(f"Unresolved merge conflict marker found in {path}: {marker}")
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid profile file: {path}")
    return obj


def _load_profiles(path: Path) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    obj = _read_json_no_conflict(path)
    raw_profiles: dict[str, dict] = {}
    for key, value in obj.items():
        if isinstance(key, str) and isinstance(value, dict):
            raw_profiles[key] = value

    resolved_cache: dict[str, dict[str, str]] = {}
    descriptions: dict[str, str] = {}

    def _resolve(name: str, stack: list[str]) -> dict[str, str]:
        if name in resolved_cache:
            return resolved_cache[name]
        if name in stack:
            raise ValueError(f"Profile inheritance cycle: {' -> '.join(stack + [name])}")
        if name not in raw_profiles:
            raise KeyError(f"Profile '{name}' not found")
        node = raw_profiles[name]
        base = node.get("_base")
        merged: dict[str, str] = {}
        if base is not None:
            if not isinstance(base, str) or not base.strip():
                raise ValueError(f"Profile '{name}' has invalid _base: {base!r}")
            merged.update(_resolve(base, stack + [name]))
        for key, value in node.items():
            if not isinstance(key, str):
                continue
            if key in PROFILE_META_KEYS:
                continue
            if key.startswith("_"):
                continue
            if isinstance(value, (dict, list)):
                raise ValueError(f"Profile '{name}' key '{key}' must be scalar, got {type(value).__name__}")
            if isinstance(value, bool):
                merged[key] = "1" if value else "0"
            else:
                merged[key] = str(value)
        resolved_cache[name] = merged
        return merged

    out: dict[str, dict[str, str]] = {}
    for name, node in raw_profiles.items():
        if name.startswith("_"):
            continue
        out[name] = _resolve(name, [])
        desc = node.get("_description")
        if isinstance(desc, str) and desc.strip():
            descriptions[name] = desc.strip()
    return out, descriptions


def _validate_profile_env(profile_name: str, env_map: dict[str, str]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for key in sorted(PROFILE_REQUIRED_KEYS):
        if not env_map.get(key):
            errors.append(f"[{profile_name}] missing required key: {key}")
    for key in sorted(PROFILE_BOOL_KEYS):
        if key not in env_map:
            continue
        value = env_map[key].strip().lower()
        if value not in {"0", "1", "true", "false"}:
            errors.append(f"[{profile_name}] {key} must be one of 0/1/true/false, got '{env_map[key]}'")
    for key, allowed in PROFILE_ENUM_KEYS.items():
        if key not in env_map:
            continue
        value = env_map[key].strip().lower()
        if value not in allowed:
            errors.append(f"[{profile_name}] {key} must be one of {sorted(allowed)}, got '{env_map[key]}'")

    bootstrap = env_map.get("STAGE2A_BOOTSTRAP_MODE", "").strip().lower()
    tokenizer = env_map.get("GRAPH_TOKENIZER_TYPE", "").strip().lower()
    if bootstrap == "legacy_stage1" and tokenizer != "qformer":
        errors.append(f"[{profile_name}] legacy_stage1 requires GRAPH_TOKENIZER_TYPE=qformer")
    if bootstrap == "no_stage1" and tokenizer == "qformer":
        warnings.append(f"[{profile_name}] qformer + no_stage1 will train tokenizer from scratch")
    return errors, warnings


def _load_data_overrides(path: Path) -> dict[str, str]:
    obj = _read_json_no_conflict(path)
    node = obj.get("env") if isinstance(obj.get("env"), dict) else obj
    if not isinstance(node, dict):
        raise ValueError(f"Invalid data config: {path}. Expected a JSON object (or an 'env' object).")

    out: dict[str, str] = {}
    unknown: list[str] = []
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        if key.startswith("_"):
            continue
        if key not in DATA_PATH_KEYS:
            unknown.append(key)
            continue
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        out[key] = text

    if unknown:
        raise ValueError(
            f"Unknown keys in data config {path}: {', '.join(sorted(unknown))}. "
            f"Allowed keys: {', '.join(sorted(DATA_PATH_KEYS))}"
        )
    return out


def _normalize_mode(mode: str) -> str:
    m = str(mode).strip().lower()
    alias = {
        "graph_tokenizer_pretrain": "stage1",
        "stage1": "stage1",
        "full": "full",
        "stage2a": "graph_bootstrap",
        "graph_bootstrap": "graph_bootstrap",
        "stage2b": "graph_refine",
        "graph_refine": "graph_refine",
        "stage3": "multimodal_tune",
        "multimodal_tune": "multimodal_tune",
        "eval": "evaluation",
        "evaluation": "evaluation",
    }
    if m not in alias:
        raise ValueError(f"Unsupported mode: {mode}")
    return alias[m]


def _resolved_paths(repo: Path, run_dir: Path) -> dict[str, str]:
    ckpt_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    stage2a_dir = ckpt_dir / "graph_bootstrap"
    stage2b_dir = ckpt_dir / "graph_refine"
    stage3_dir = ckpt_dir / "multimodal_tune"
    stage2b_r2_dir = ckpt_dir / "graph_refine_round2"
    stage3_r2_dir = ckpt_dir / "multimodal_tune_round2"

    return {
        "REPO": str(repo),
        "STAGE1_EXPORT_PATH": str(ckpt_dir / "graph_qformer_stage1.pt"),
        "STAGE2A_DIR": str(stage2a_dir),
        "STAGE2B_DIR": str(stage2b_dir),
        "STAGE3_DIR": str(stage3_dir),
        "STAGE2B_R2_DIR": str(stage2b_r2_dir),
        "STAGE3_R2_DIR": str(stage3_r2_dir),
        "GRAPH_BOOTSTRAP_DIR": str(stage2a_dir),
        "GRAPH_REFINE_DIR": str(stage2b_dir),
        "MULTIMODAL_TUNE_DIR": str(stage3_dir),
        "GRAPH_REFINE_R2_DIR": str(stage2b_r2_dir),
        "MULTIMODAL_TUNE_R2_DIR": str(stage3_r2_dir),
        "GQA_PRED_PAPER": str(eval_dir / "pred_val_balanced_paper.jsonl"),
        "GQA_EVAL_PAPER": str(eval_dir / "eval_val_balanced_paper.txt"),
        "GQA_PRED_R2": str(eval_dir / "pred_val_balanced_round2.jsonl"),
        "GQA_EVAL_R2": str(eval_dir / "eval_val_balanced_round2.txt"),
    }


def _run_and_tee(cmd: list[str], env: dict[str, str], log_path: Path, append: bool = False) -> int:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as fp:
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


def _git_info(repo: Path) -> dict[str, object]:
    def _git(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    commit = _git("rev-parse", "HEAD")
    short_commit = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain")
    dirty_files = [line for line in status.splitlines() if line.strip()]
    return {
        "branch": branch or None,
        "commit": commit or None,
        "short_commit": short_commit or None,
        "dirty": len(dirty_files) > 0,
        "dirty_file_count": len(dirty_files),
        "dirty_file_preview": dirty_files[:20],
    }


def _system_info() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": gethostname(),
    }


def _write_resolved_env_sh(path: Path, env_map: dict[str, str]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for key in sorted(env_map):
        lines.append(f"export {key}={shlex.quote(str(env_map[key]))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Standardized launcher for OmniGraph sprint pipeline.")
    parser.add_argument("--profile", type=str, default="4090", help="Profile name from configs/train/infra_profiles.json")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=[
            "graph_tokenizer_pretrain",
            "stage1",
            "full",
            "graph_bootstrap",
            "stage2a",
            "graph_refine",
            "stage2b",
            "multimodal_tune",
            "stage3",
            "evaluation",
            "eval",
        ],
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--run-id", type=str, default="", help="Explicit run id (for deterministic naming/resume)")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--profiles-file", type=str, default="configs/train/infra_profiles.json")
    parser.add_argument("--data-config", type=str, default="", help="JSON file with fine-grained data path overrides.")
    parser.add_argument("--set", dest="sets", action="append", default=[], help="Override env as KEY=VALUE")
    parser.add_argument("--list-profiles", action="store_true", help="List available profiles and exit")
    parser.add_argument("--validate-only", action="store_true", help="Validate profile and exit")
    parser.add_argument("--print-resolved-env", action="store_true", help="Print resolved env before execution")
    parser.add_argument("--resume", action="store_true", help="Append logs to an existing run directory")
    parser.add_argument("--require-clean-git", action="store_true", help="Fail if git working tree is dirty")
    parser.add_argument("--preflight", action="store_true", help="Run smoke preflight before launching pipeline")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    profiles_path = (repo / args.profiles_file).resolve()
    if not profiles_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profiles_path}")

    profiles, profile_descriptions = _load_profiles(profiles_path)
    if args.list_profiles:
        print("Available profiles:")
        for name in sorted(profiles):
            desc = profile_descriptions.get(name, "")
            suffix = f" - {desc}" if desc else ""
            print(f"  {name}{suffix}")
        return 0

    if args.profile not in profiles:
        raise KeyError(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")
    profile_env = profiles[args.profile]
    profile_errors, profile_warnings = _validate_profile_env(args.profile, profile_env)
    for warning in profile_warnings:
        print(f"[Infra][Warn] {warning}")
    if profile_errors:
        raise ValueError("Profile validation failed:\n" + "\n".join(f"- {msg}" for msg in profile_errors))

    data_overrides: dict[str, str] = {}
    data_config_path: Path | None = None
    if str(args.data_config).strip():
        raw_data_cfg = Path(str(args.data_config).strip())
        data_config_path = raw_data_cfg if raw_data_cfg.is_absolute() else (repo / raw_data_cfg)
        data_config_path = data_config_path.resolve()
        if not data_config_path.exists():
            raise FileNotFoundError(f"Data config not found: {data_config_path}")
        data_overrides = _load_data_overrides(data_config_path)

    if args.validate_only and not args.print_resolved_env and not args.dry_run:
        if data_config_path:
            print(f"[Infra] profile '{args.profile}' + data config validation passed.")
            print(f"[Infra] data_config={data_config_path}")
        else:
            print(f"[Infra] profile '{args.profile}' validation passed.")
        return 0

    overrides = _parse_kv(args.sets)
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id")

    if args.run_id:
        run_id = _sanitize_token(args.run_id)
        if not run_id:
            raise ValueError(f"Invalid --run-id: {args.run_id!r}")
    else:
        run_id = _build_run_id(args.run_name or args.profile)

    out_root = (repo / args.output_root).resolve()
    run_dir = out_root / run_id
    if args.resume:
        if not run_dir.exists():
            raise FileNotFoundError(f"--resume requested but run directory does not exist: {run_dir}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists and is not empty: {run_dir}. "
                f"Use --run-id with a new value, or pass --resume to append."
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    run_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("OMNIGRAPH_HF_CACHE", "/media/disk/02drive/13hias/.cache")
    env.setdefault("OMNIGRAPH_HF_ENDPOINT", "https://hf-mirror.com")
    env.update(profile_env)
    resolved_paths = _resolved_paths(repo, run_dir)
    env.update(resolved_paths)
    env.update(data_overrides)
    env["GPU"] = str(args.gpu)
    resolved_mode = _normalize_mode(args.mode)
    env["PIPELINE_MODE"] = resolved_mode
    env["RUN_ID"] = run_id
    env.update(overrides)

    git = _git_info(repo)
    if args.require_clean_git and git.get("dirty"):
        raise RuntimeError("Git working tree is dirty. Commit/stash changes or drop --require-clean-git.")

    if args.print_resolved_env:
        print("[Infra] resolved env:")
        for key in sorted(set(profile_env.keys()) | set(resolved_paths.keys()) | set(data_overrides.keys()) | set(overrides.keys()) | {"GPU", "PIPELINE_MODE", "RUN_ID"}):
            print(f"{key}={env[key]}")

    if resolved_mode == "stage1":
        command = ["bash", str((repo / "train_stage1.sh").resolve())]
    else:
        command = ["bash", str((repo / "scripts/train/run_4090_gqa_sprint.sh").resolve())]
    cmd_str = " ".join(shlex.quote(x) for x in command)

    tracked_env_keys = sorted(set(profile_env.keys()) | set(resolved_paths.keys()) | set(data_overrides.keys()) | set(overrides.keys()) | {"GPU", "PIPELINE_MODE", "RUN_ID"})
    now = datetime.now().isoformat(timespec="seconds")
    command_file = run_dir / ("command_resume.sh" if args.resume else "command.sh")
    manifest_path = run_dir / ("run_manifest_resume.json" if args.resume else "run_manifest.json")
    status_path = run_dir / ("run_status_resume.json" if args.resume else "run_status.json")
    env_sh_path = run_dir / "resolved_env.sh"

    manifest = {
        "run_id": run_id,
        "profile": args.profile,
        "mode": args.mode,
        "resolved_mode": resolved_mode,
        "gpu": args.gpu,
        "resume": args.resume,
        "repo": str(repo),
        "command": cmd_str,
        "profiles_file": str(profiles_path),
        "data_config_file": (str(data_config_path) if data_config_path else None),
        "profile_env": profile_env,
        "data_overrides": data_overrides,
        "overrides": overrides,
        "resolved_env": {k: env[k] for k in tracked_env_keys},
        "git": git,
        "system": _system_info(),
        "created_at": now,
    }

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    command_file.write_text(cmd_str + "\n", encoding="utf-8")
    _write_resolved_env_sh(env_sh_path, {k: env[k] for k in tracked_env_keys})

    print(f"[Infra] run_id={run_id}")
    print(f"[Infra] profile={args.profile} mode={args.mode} resolved_mode={resolved_mode} gpu={args.gpu}")
    print(f"[Infra] run_dir={run_dir}")
    print(f"[Infra] manifest={manifest_path}")
    print(f"[Infra] env_sh={env_sh_path}")

    if args.validate_only:
        print("[Infra] validate-only passed.")
        return 0

    if args.dry_run:
        print("[Infra] dry-run enabled; command not executed.")
        return 0

    if args.preflight:
        preflight_cmd = ["bash", str((repo / "scripts/train/check_pipeline_smoke.sh").resolve())]
        preflight_log = run_dir / "preflight.log"
        print(f"[Infra] preflight: {' '.join(shlex.quote(x) for x in preflight_cmd)}")
        preflight_rc = _run_and_tee(preflight_cmd, env=env, log_path=preflight_log, append=False)
        if preflight_rc != 0:
            raise RuntimeError(f"Preflight failed with rc={preflight_rc}. See {preflight_log}")

    log_path = run_dir / "run.log"
    rc = _run_and_tee(command, env=env, log_path=log_path, append=args.resume)

    status = {
        "run_id": run_id,
        "return_code": rc,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "log": str(log_path),
        "manifest": str(manifest_path),
    }
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Infra] exit_code={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
