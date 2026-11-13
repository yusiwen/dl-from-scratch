#!/usr/bin/env python3
"""Verify that notebooks take their hyperparameters from config.yaml.

``config.yaml`` is the single source of truth for a training run, so a notebook
next to one may not hardcode epochs / lr / batch size: it must load that config,
seed the RNGs from it, and read the values through ``cfg``. Without this check
the two drift apart silently (see issue #30).

    python scripts/check_notebook_config.py          # exit 1 on violations
    python scripts/check_notebook_config.py --verbose
"""

import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".venv", ".git", "__pycache__", ".ipynb_checkpoints", "runs", "node_modules"}

# Names that must come from config, never from a numeric literal.
HYPER_ASSIGN = re.compile(
    r"^\s*(NUM_EPOCHS|NUM_EPISODES|LR|MOMENTUM|WEIGHT_DECAY|BATCH_SIZE|BUFFER_SIZE"
    r"|TARGET_UPDATE|GAMMA|EPSILON_[A-Z_]+)\s*=\s*[0-9]",
    re.M,
)
NUMERIC_ARG = re.compile(r"\b(batch_size|image_size)\s*=\s*\d")
CFG_KEY = re.compile(r'cfg\["([^"]+)"\]|cfg\.get\("([^"]+)"')

# A literal `name=<number>` where `name` is also a config key is drift only when
# the two values disagree; architecture constants that match config are allowed.
# Defaults in `def __init__(..., max_len=128)` are skipped.
NAMED_NUMERIC_ARG = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(\d+(?:\.\d+)?)\b")

# Training knobs that must be read from config whenever config defines them.
REQUIRED_KEYS = ("seed", "lr", "batch_size", "num_epochs", "num_episodes")

# YAML 1.1 needs a dot before the exponent (5.0e-4); "5e-4" silently becomes a str.
NUMERIC_STRING = re.compile(r"^-?\d+(?:\.\d+)?[eE][-+]?\d+$")


def iter_notebooks():
    for path in sorted(REPO_ROOT.rglob("*.ipynb")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def code_of(notebook_path):
    import nbformat as nbf

    nb = nbf.read(notebook_path, as_version=4)
    return "\n".join(
        "".join(cell.get("source", "")) for cell in nb.cells if cell.get("cell_type") == "code"
    )


def check(notebook_path, config_path, verbose=False):
    problems = []
    config = yaml.safe_load(config_path.read_text()) or {}
    code = code_of(notebook_path)
    rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    config_rel = config_path.relative_to(REPO_ROOT).as_posix()

    if f'load_config("{config_rel}")' not in code:
        problems.append(f'does not call load_config("{config_rel}")')

    if "set_seed(" not in code:
        problems.append("does not call set_seed()")

    for match in HYPER_ASSIGN.finditer(code):
        problems.append(f"hardcodes {match.group(0).strip()}")

    for match in NUMERIC_ARG.finditer(code):
        problems.append(f"hardcodes {match.group(0)}")

    for line in code.splitlines():
        if "def " in line:
            continue
        for name, value in NAMED_NUMERIC_ARG.findall(line):
            if name in config and float(config[name]) != float(value):
                problems.append(
                    f"hardcodes {name}={value} but config has {name}={config[name]}")

    for key, value in config.items():
        if isinstance(value, str) and NUMERIC_STRING.match(value.strip()):
            problems.append(f'config value {key}="{value}" parses as a string, not a number')

    used_keys = {a or b for a, b in CFG_KEY.findall(code)}
    for key in sorted(used_keys - set(config)):
        problems.append(f'reads cfg["{key}"] but config has no such key')
    for key in REQUIRED_KEYS:
        if key in config and key not in used_keys:
            problems.append(f'config key "{key}" is not read from cfg')

    if verbose and not problems:
        print(f"ok   {rel}  (keys: {', '.join(sorted(used_keys))})")
    return rel, problems


def main(argv):
    verbose = "--verbose" in argv
    checked = skipped = 0
    failures = []

    for notebook in iter_notebooks():
        config = notebook.parent / "config.yaml"
        if not config.exists():
            skipped += 1
            continue
        checked += 1
        rel, problems = check(notebook, config, verbose)
        if problems:
            failures.append((rel, problems))

    for rel, problems in failures:
        print(f"FAIL {rel}")
        for problem in problems:
            print(f"     {problem}")

    print(f"\nchecked={checked} skipped={skipped} failed={len(failures)}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
