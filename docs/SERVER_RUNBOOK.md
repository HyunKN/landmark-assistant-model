# Server Runbook

## Assumptions

- Use 4 GPUs for the current shared server lane: `GPUS=1,2,3,4 NPROC=4`.
- Keep `EXPORT_ONNX=0` during candidate comparison; export only the selected final run.
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
/workspace/landmark-assistant-model/Dataset/
  landmark_id/
    labels.json
    images/
```

Create splits:

```bash
export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
bash scripts/make_splits.sh
```

## Train MobileCLIP2-S4 One Fold

```bash
export DATA_ROOT=/workspace/landmark-assistant-model/Dataset
GPUS=1,2,3,4 NPROC=4 EXPORT_ONNX=0 bash scripts/run_candidate_tmux.sh mobileclip2_s4 0
```

## Train MobileCLIP2-S4 All Folds

```bash
for FOLD in 0 1 2 3 4; do
  GPUS=1,2,3,4 NPROC=4 EXPORT_ONNX=0 bash scripts/run_candidate_tmux.sh mobileclip2_s4 "$FOLD"
  echo "Started fold $FOLD. Wait for it to finish before starting the next fold."
done
```

## Monitor

```bash
tmux attach -t jongno-mobileclip2_s4-fold0
watch -n 2 nvidia-smi
```

## Backup

```bash
bash scripts/backup_to_gdrive.sh runs/<run-dir>
```
