# OmniGraph Standardized Infra Launcher

Use this launcher to run reproducible training/eval with profile inheritance, manifest logging, and resumable runs.
Default strict path is now `graph_bootstrap(no_stage1 + perceiver) -> graph_refine -> multimodal_tune -> GQA eval`.

## Entry

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --gpu 0
python scripts/train/infra_launcher.py --profile pro6000 --mode full --gpu 0
```

If data is not under `<repo>/data`, set one root and keep all VG/GQA defaults derived from it:

```bash
python scripts/train/infra_launcher.py --profile pro6000 --mode full --gpu 0 \
  --set OMNIGRAPH_DATA_ROOT=/absolute/path/to/data
```

For fine-grained per-file path control, use a JSON data config:

```bash
python scripts/train/infra_launcher.py --profile pro6000 --mode full --gpu 0 \
  --data-config configs/train/data_paths.sample.json
```

You can validate profile + data config without running:

```bash
python scripts/train/infra_launcher.py --profile pro6000 --validate-only \
  --data-config configs/train/data_paths.sample.json
```

## Prefetch models (new server)

Prefetch all model repos needed by current profile (LLM + vision + stage1 text encoder), plus optional alternates:

```bash
python scripts/train/prefetch_models.py \
  --profile pro6000 \
  --include-alt-llm \
  --include-legacy-qformer
```

List and validate profiles:

```bash
python scripts/train/infra_launcher.py --list-profiles
python scripts/train/infra_launcher.py --profile 4090 --validate-only
```

## Common modes

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode graph_tokenizer_pretrain --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode graph_bootstrap --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode graph_refine --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode multimodal_tune --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode evaluation --gpu 0
```

Legacy rollback (Stage1 + QFormer):

```bash
./train_infra.sh --profile 4090 --mode graph_bootstrap --gpu 0 \
  --set GRAPH_TOKENIZER_TYPE=qformer \
  --set STAGE2A_BOOTSTRAP_MODE=legacy_stage1 \
  --set STAGE1_QFORMER_CKPT=/absolute/path/to/runs/<run_id>/checkpoints/graph_qformer_stage1.pt
```

Optional Stage1 stop controls (legacy path only):

```bash
./train_infra.sh --profile 4090 --mode graph_tokenizer_pretrain --gpu 0 \
  --set STAGE1_MAX_STEPS=20000 \
  --set STAGE1_GTM_TEXT_SOURCE=qa \
  --set STAGE1_GTM_NEG_PER_POS=2 \
  --set STAGE1_GTM_WARMUP_STEPS=1000 \
  --set STAGE1_ENABLE_EARLY_STOP=1 \
  --set STAGE1_EARLY_STOP_PATIENCE=2 \
  --set STAGE1_EARLY_STOP_MIN_DELTA=0.001
```

## Override any env key

```bash
python scripts/train/infra_launcher.py \
  --profile 4090 \
  --mode full \
  --gpu 0 \
  --set S2A_MAX_LENGTH=128 \
  --set AUTO_BATCH_SCALE=1.2
```

Mid-run OOM auto recovery (reduce batch + resume from latest checkpoint):

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --gpu 0 \
  --set AUTO_RESUME_ON_OOM=1 \
  --set AUTO_BATCH_RETRY_ON_OOM=1 \
  --set S2A_CHECKPOINT_EVERY_N_STEPS=1000 \
  --set S2B_CHECKPOINT_EVERY_N_STEPS=1000 \
  --set S3_CHECKPOINT_EVERY_N_STEPS=1000
```

Print fully resolved env (profile + paths + overrides):

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --print-resolved-env --dry-run
```

Resume an existing run directory:

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode graph_refine --run-id my_exp_001 --resume --gpu 0
```

Require clean git state for strict reproducibility:

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --require-clean-git --gpu 0
```

Run CLI contract preflight before launching:

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --preflight --gpu 0
```

## Dry run

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --dry-run
```

## Artifacts

Each run creates `runs/<run_id>/` containing:
- `run_manifest.json`: full resolved config snapshot
- `command.sh`: exact command
- `run.log`: merged stdout/stderr
- `run_status.json`: return code + timestamps
- `resolved_env.sh`: export-able env snapshot for exact replay

When `--resume` is used:
- `run_manifest_resume.json`
- `command_resume.sh`
- `run_status_resume.json`
- `run.log` is appended

## Profile design

Profiles live in `configs/train/infra_profiles.json`.
- Use `_base` for inheritance to avoid duplication.
- Keys beginning with `_` are treated as metadata.
- Launcher validates required keys and enum/bool-like values before execution.
