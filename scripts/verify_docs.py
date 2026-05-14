#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT
REQUIRED = [
    SITE / "index.html",
    SITE / "project-brief.html",
    SITE / "goals" / "sprint1-model-training-goal.html",
    SITE / "goals" / "sprint1-model-design-scorecard.html",
    SITE / "decisions" / "ADR-0001-model-architecture.html",
    SITE / "decisions" / "ADR-0002-training-model-design.html",
    SITE / "decisions" / "ADR-0003-text-encoder-natural-language-search.html",
    SITE / "decisions" / "ADR-0004-image-recognizer-model-selection.html",
    SITE / "decisions" / "ADR-0005-training-config-v2.html",
    SITE / "context" / "dataset-contract.html",
    SITE / "context" / "deep-interview-round1.html",
    SITE / "context" / "deep-interview-round2.html",
    SITE / "context" / "deep-interview-round3.html",
    SITE / "context" / "deep-interview-round4.html",
    SITE / "context" / "provided-json-audit.html",
    SITE / "context" / "bohyunsanshingak-scope-audit.html",
    SITE / "context" / "dataset-12class-audit-2026-05-14.html",
    SITE / "context" / "server-spec.html",
    SITE / "context" / "lightweight-model-reference-audit.html",
    SITE / "experiments" / "model-search-plan.html",
    SITE / "experiments" / "multimodal-model-selection-plan.html",
    SITE / "operations" / "training-runbook.html",
    SITE / "operations" / "experiment-tracking-backup.html",
    SITE / "operations" / "model-serving-contract.html",
    SITE / "operations" / "codex-workflow.html",
    SITE / "operations" / "candidate-repos.html",
    SITE / "operations" / "html-document-principles.html",
    SITE / "operations" / "sprint1-status-2026-05-14.html",
    SITE / "operations" / "vercel-docs-deploy.html",
]

SCRIPT_REQUIRED = []


def fail(message: str) -> None:
    print(f"[docs-check] FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    for path in REQUIRED:
        if not path.exists():
            fail(f"missing required document: {path.relative_to(ROOT)}")
        text = path.read_text(encoding="utf-8")
        if "<html" not in text.lower() or "</html>" not in text.lower():
            fail(f"not a complete html document: {path.relative_to(ROOT)}")
        if "TODO_DECISION" in text:
            fail(f"unresolved decision marker in {path.relative_to(ROOT)}")

    for path in SCRIPT_REQUIRED:
        if not path.exists():
            fail(f"missing required script: {path.relative_to(ROOT)}")

    href_pattern = re.compile(r'href="([^"]+\.html)"')
    for path in REQUIRED:
        text = path.read_text(encoding="utf-8")
        for href in href_pattern.findall(text):
            if href.startswith(("http://", "https://")):
                continue
            target = (path.parent / href).resolve()
            try:
                target.relative_to(ROOT)
            except ValueError:
                fail(f"link leaves repo: {path.relative_to(ROOT)} -> {href}")
            if not target.exists():
                fail(f"broken local link: {path.relative_to(ROOT)} -> {href}")

    print("[docs-check] OK")


if __name__ == "__main__":
    main()
