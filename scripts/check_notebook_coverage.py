#!/usr/bin/env python3
"""Verify notebook coverage and that the docs agree on the count.

Two properties are checked:

1. **Coverage** — every model directory (one containing ``train.py``) has at least
   one notebook, and every standalone ``ml/basics/<model>.py`` has a matching
   ``<model>.ipynb``.
2. **Count** — the numbers written in README.md and ROADMAP.md equal the number
   of tracked ``.ipynb`` files, so the docs cannot drift from the tree.

    python scripts/check_notebook_coverage.py        # exit 1 on violations
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ipynb_checkpoints", "runs", "node_modules"}

# Standalone basics modules that are infrastructure, not models.
BASICS_SKIP = {"data", "metrics", "__init__"}

README_PATTERN = re.compile(r"\*\*Notebooks:\*\*\s*(\d+)")
ROADMAP_PATTERN = re.compile(r"Jupyter Notebooks \((\d+)\)")


def model_dirs():
    """Directories that ship a training entry point."""
    return sorted(
        path.parent
        for path in REPO_ROOT.rglob("train.py")
        if not any(part in SKIP_DIRS for part in path.parts)
    )


def tracked_notebooks():
    out = subprocess.run(
        ["git", "ls-files", "*.ipynb"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def check_coverage(problems):
    for directory in model_dirs():
        rel = directory.relative_to(REPO_ROOT).as_posix()
        if not list(directory.glob("*.ipynb")):
            problems.append(f"{rel}: has train.py but no notebook")

    basics = REPO_ROOT / "ml" / "basics"
    for module in sorted(basics.glob("*.py")):
        if module.stem in BASICS_SKIP:
            continue
        if not (basics / f"{module.stem}.ipynb").exists():
            problems.append(f"ml/basics: {module.stem}.py has no {module.stem}.ipynb")


def check_counts(problems, actual):
    for doc, pattern in (("README.md", README_PATTERN), ("ROADMAP.md", ROADMAP_PATTERN)):
        text = (REPO_ROOT / doc).read_text()
        match = pattern.search(text)
        if not match:
            problems.append(f"{doc}: no notebook count marker matched {pattern.pattern!r}")
        elif int(match.group(1)) != actual:
            problems.append(f"{doc}: says {match.group(1)} notebooks, tree has {actual}")


def main():
    problems = []
    notebooks = tracked_notebooks()
    check_coverage(problems)
    check_counts(problems, len(notebooks))

    for problem in problems:
        print(f"FAIL {problem}")

    dirs = len(model_dirs())
    print(
        f"\nnotebooks={len(notebooks)} model_dirs={dirs} problems={len(problems)}",
        file=sys.stderr,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
