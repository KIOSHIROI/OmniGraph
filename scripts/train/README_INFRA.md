# OmniGraph Standardized Infra Launcher

Use this launcher to run reproducible training/eval with profile + manifest logging.

## Entry

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode full --gpu 0
```

## Common modes

```bash
python scripts/train/infra_launcher.py --profile 4090 --mode stage1 --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode stage2a --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode stage2b --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode stage3 --gpu 0
python scripts/train/infra_launcher.py --profile 4090 --mode eval   --gpu 0
```

After Stage1 retraining, point Stage2A to the produced ckpt:

```bash
./train_infra.sh --profile 4090 --mode stage2a --gpu 0 \
  --set STAGE1_QFORMER_CKPT=/absolute/path/to/runs/<run_id>/checkpoints/graph_qformer_stage1.pt
```

Optional Stage1 stop controls:

```bash
./train_infra.sh --profile 4090 --mode stage1 --gpu 0 \
  --set STAGE1_MAX_STEPS=20000 \
  --set STAGE1_GTM_TEXT_SOURCE=qa \
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
