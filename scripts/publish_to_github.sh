#!/usr/bin/env bash
set -euo pipefail

OWNER="${1:?Usage: bash scripts/publish_to_github.sh <github-owner-or-org>}"
VISIBILITY="${VISIBILITY:-private}"
REPO="$(basename "$PWD")"

git init
git add .
git -c user.name="Codex" -c user.email="codex@local" commit -m "Initial unified candidate training repo" || true
git branch -M main

if command -v gh >/dev/null 2>&1; then
  gh repo create "${OWNER}/${REPO}" "--${VISIBILITY}" --source=. --remote=origin --push
else
  echo "gh CLI is not installed. Create ${OWNER}/${REPO} on GitHub, then run:"
  echo "git remote add origin git@github.com:${OWNER}/${REPO}.git"
  echo "git push -u origin main"
fi
