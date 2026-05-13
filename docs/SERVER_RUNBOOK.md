# Server Runbook

## Assumptions

- Use 6 GPUs.
- Run inside `tmux`.
- Run inside `.venv`.
- W&B is already configured on the server.
- Google Drive backup is available through `rclone` remote `gdrive:`.

## One-Time Setup

```bash
bash scripts/setup_venv.sh
```

## Data

Expected layout:

```text
/data/landmark-assistant/Dataset/
  landmark_id/
    labels.json
    images/
```

Create splits:

```bash
export DATA_ROOT=/data/landmark-assistant/Dataset
bash scripts/make_splits.sh
```

## Train One Fold

```bash
bash scripts/run_tmux_train.sh 0
```

## Train All Folds

```bash
bash scripts/run_cv_tmux.sh
```

## Monitor

```bash
tmux attach -t jongno-landmark-model-candidates
watch -n 2 nvidia-smi
```

## Backup

```bash
bash scripts/backup_to_gdrive.sh runs/<run-dir>
```
