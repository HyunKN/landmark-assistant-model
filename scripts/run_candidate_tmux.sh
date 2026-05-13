#!/usr/bin/env bash
set -euo pipefail

CANDIDATE="${1:?Usage: bash scripts/run_candidate_tmux.sh <candidate_id> [fold]}"
FOLD="${2:-0}"
SESSION="${SESSION:-jongno-${CANDIDATE}-fold${FOLD}}"
GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC="${NPROC:-6}"
DATA_ROOT="${DATA_ROOT:-/data/landmark-assistant/Dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-landmark-assistant-sprint1}"
CONFIG="configs/candidates/${CANDIDATE}.yaml"

if [ ! -f "$CONFIG" ]; then
  echo "Missing candidate config: $CONFIG" >&2
  exit 1
fi

mkdir -p runs logs splits

tmux new-session -d -s "$SESSION" "cd '$PWD' && source .venv/bin/activate && export DATA_ROOT='$DATA_ROOT' WANDB_PROJECT='$WANDB_PROJECT' CUDA_VISIBLE_DEVICES='$GPUS' && python -m landmark_candidate.split_data --data-root '$DATA_ROOT' --out splits/kfold_seed20260513.json --seed 20260513 --folds 5 --test-ratio 0.15 && torchrun --nproc_per_node=$NPROC -m landmark_candidate.train --config '$CONFIG' --data-root '$DATA_ROOT' --split splits/kfold_seed20260513.json --fold '$FOLD' 2>&1 | tee logs/${SESSION}.log"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
