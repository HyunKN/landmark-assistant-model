#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-jongno-landmark-all-candidates}"
GPUS="${GPUS:-0,1,2,3,4,5}"
NPROC="${NPROC:-6}"
DATA_ROOT="${DATA_ROOT:-/data/landmark-assistant/Dataset}"
WANDB_PROJECT="${WANDB_PROJECT:-landmark-assistant-sprint1}"
CANDIDATES="${CANDIDATES:-mobileclip2_s4 mobileclip2_s3 mobileclip2_b mobilenetv4_hybrid_large mobilenetv4_conv_aa_large_in12k}"

mkdir -p runs logs splits

tmux new-session -d -s "$SESSION" "cd '$PWD' && source .venv/bin/activate && export DATA_ROOT='$DATA_ROOT' WANDB_PROJECT='$WANDB_PROJECT' CUDA_VISIBLE_DEVICES='$GPUS' && python -m landmark_candidate.split_data --data-root '$DATA_ROOT' --out splits/kfold_seed20260513.json --seed 20260513 --folds 5 --test-ratio 0.15 && for CANDIDATE in $CANDIDATES; do for FOLD in 0 1 2 3 4; do CONFIG=configs/candidates/\$CANDIDATE.yaml; echo '[run]' \$CANDIDATE fold \$FOLD; torchrun --nproc_per_node=$NPROC -m landmark_candidate.train --config \$CONFIG --data-root '$DATA_ROOT' --split splits/kfold_seed20260513.json --fold \$FOLD; done; done 2>&1 | tee logs/${SESSION}.log"

echo "Started tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
