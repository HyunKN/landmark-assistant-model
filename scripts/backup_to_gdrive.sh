#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?Usage: bash scripts/backup_to_gdrive.sh runs/<run-dir>}"
GDRIVE_BACKUP_ROOT="${GDRIVE_BACKUP_ROOT:-gdrive:landmark-assistant/runs}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required for Google Drive backup." >&2
  exit 1
fi

TARGET="${GDRIVE_BACKUP_ROOT}/$(basename "$PWD")/$(basename "$RUN_DIR")"
rclone copy "$RUN_DIR" "$TARGET" --progress
echo "Backed up $RUN_DIR to $TARGET"
