#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data/landmark-assistant/Dataset}"
SEED="${SEED:-20260513}"
FOLDS="${FOLDS:-5}"
TEST_RATIO="${TEST_RATIO:-0.15}"

mkdir -p splits
python -m landmark_candidate.split_data \
  --data-root "$DATA_ROOT" \
  --out "splits/kfold_seed${SEED}.json" \
  --seed "$SEED" \
  --folds "$FOLDS" \
  --test-ratio "$TEST_RATIO"
