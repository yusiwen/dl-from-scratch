"""Shared first-cell bootstrap for generated notebooks.

Jupyter starts a kernel in the notebook's own directory, so a notebook in
``ml/basics/`` cannot ``import ml`` unless the repository root is on
``sys.path``. This cell locates the root by walking up to ``pyproject.toml``
and prepends it once, which also makes the notebooks runnable from any
directory (JupyterLab, ``jupyter nbconvert --execute``, or CI).

Keep this the single source of truth: both the generator scripts and
``fix_notebook_bootstrap.py`` import from here.
"""

import nbformat as nbf

MARKER = "# --- notebook bootstrap: repository root on sys.path ---"

BOOTSTRAP_SRC = f"""\
{MARKER}
# Jupyter runs the kernel in the notebook's own folder. The project resolves
# imports against the repository root and reads configs such as
# "ml/basics/knn.yaml" relative to it, so switch there and put it on sys.path.
# Locate the root by walking up to the directory that holds pyproject.toml.
import os
import pathlib
import sys

for _parent in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
    if (_parent / "pyproject.toml").exists():
        os.chdir(_parent)
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break
else:
    raise RuntimeError("repository root not found (no pyproject.toml above cwd)")
"""


def bootstrap_cell():
    """Return a fresh code cell carrying the bootstrap snippet."""
    return nbf.v4.new_code_cell(BOOTSTRAP_SRC)


def bootstrap_index(nb):
    """Index of the existing bootstrap cell, or None."""
    for idx, cell in enumerate(nb.cells):
        if cell.get("cell_type") == "code" and MARKER in "".join(cell.get("source", "")):
            return idx
    return None


def has_bootstrap(nb):
    """True when the notebook already contains the bootstrap cell."""
    return bootstrap_index(nb) is not None


def is_current(nb):
    """True when the bootstrap cell matches BOOTSTRAP_SRC exactly."""
    idx = bootstrap_index(nb)
    return idx is not None and "".join(nb.cells[idx].get("source", "")) == BOOTSTRAP_SRC


def config_cell(config_path):
    """Cell that loads ``config.yaml`` and seeds every RNG.

    Notebooks must not hardcode training hyperparameters: ``config.yaml`` is the
    single source of truth for a training run, so the notebook reads it here and
    every later cell takes epochs / lr / batch size from ``cfg``. That keeps a
    notebook run comparable with ``python -m <module>.train``.
    """
    source = f'''# --- 超参数唯一真源：{config_path} ---
from utils.config import load_config
from utils.seed import set_seed

cfg = load_config("{config_path}")
set_seed(cfg["seed"])
print(f"config: {{cfg}}")
'''
    return nbf.v4.new_code_cell(source)
