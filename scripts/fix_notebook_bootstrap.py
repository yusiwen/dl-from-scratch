#!/usr/bin/env python3
"""Insert the repository-root bootstrap cell into every notebook.

Idempotent: notebooks that already carry the bootstrap marker are left
untouched, so this is safe to run after any regeneration.

    python scripts/fix_notebook_bootstrap.py           # apply
    python scripts/fix_notebook_bootstrap.py --check   # report only, exit 1 if any missing
"""

import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_bootstrap import bootstrap_cell, bootstrap_index, is_current  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
# Skip virtualenvs and VCS internals; notebooks live next to their source.
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ipynb_checkpoints", "runs", "node_modules"}


def iter_notebooks():
    for path in sorted(REPO_ROOT.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def main(argv):
    check_only = "--check" in argv

    total = fixed = already = 0
    for path in iter_notebooks():
        total += 1
        nb = nbf.read(path, as_version=4)
        if is_current(nb):
            already += 1
            continue
        idx = bootstrap_index(nb)
        if check_only:
            state = "outdated" if idx is not None else "missing"
            print(f"{state}: {path.relative_to(REPO_ROOT)}")
            continue
        if idx is not None:
            # Refresh an older snippet in place so the cell keeps its position.
            nb.cells[idx] = bootstrap_cell()
        else:
            nb.cells.insert(0, bootstrap_cell())
        nbf.write(nb, path)
        fixed += 1
        print(f"fixed:   {path.relative_to(REPO_ROOT)}")

    print(f"\nnotebooks={total}  already_ok={already}  fixed={fixed}", file=sys.stderr)
    if check_only and (total - already) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
