"""
Hyperparameter search: grid search wrapper for any train script.

Usage:
    uv run python scripts/hyperparameter_search.py \
        --model cv.simplecnn.train \
        --config cv/simplecnn/config.yaml \
        --param lr:0.0001,0.001,0.01 \
        --param batch_size:64,128 \
        --epochs 10

Output: results/hyperparameter_search.csv
"""

import argparse
import copy
import csv
import itertools
import os
import re
import sys
import io
import time
import importlib
import yaml
from pathlib import Path

# Run as a plain script from a checkout that is not pip-installed: the training
# modules are imported as `cv.*` / `ml.*`, so the repository root has to be on
# sys.path before the first project import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.config as config_module


def parse_param(param_str):
    key, vals = param_str.split(":", 1)
    values = []
    for v in vals.split(","):
        v = v.strip()
        for cast in (int, float):
            try:
                values.append(cast(v))
                break
            except (ValueError, TypeError):
                continue
        else:
            values.append(v)
    return key, values


def extract_metrics(text):
    """Extract final loss and accuracy from training output."""
    loss = re.findall(r"Loss:\s*([\d.]+)", text)
    acc = re.findall(r"Acc:\s*([\d.]+)%", text)
    return (float(loss[-1]) if loss else None,
            float(acc[-1]) if acc else None)


def resolve_target(spec):
    """Resolve ``--model`` to a callable.

    Both spellings have to work, because the docstring advertises the first one:

        cv.simplecnn.train        -> the train() function in cv/simplecnn/train.py
        cv.simplecnn.train.train  -> the same function, spelled out
    """
    try:
        module = importlib.import_module(spec)
    except ImportError:
        module = None

    if module is not None:
        for name in ("train", "main"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn
        raise SystemExit(f"{spec!r} is a module without a train()/main() to call")

    module_path, func_name = spec.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, func_name, None)
    if callable(fn):
        return fn
    raise SystemExit(f"cannot resolve --model {spec!r} to a callable")


def main():
    parser = argparse.ArgumentParser(description="Grid search hyperparameter optimization")
    parser.add_argument("--model", required=True, help="e.g. cv.simplecnn.train")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--param", action="append", required=True, help="key:val1,val2,...")
    parser.add_argument("--epochs", type=int, default=10, help="Override num_epochs")
    args = parser.parse_args()

    param_keys, param_grids = [], []
    for p in args.param:
        key, values = parse_param(p)
        param_keys.append(key)
        param_grids.append(values)

    total = 1
    for g in param_grids:
        total *= len(g)

    print(f"Hyperparameter Grid Search")
    print(f"  Model:   {args.model}")
    print(f"  Config:  {args.config}")
    print(f"  Params:  {param_keys} = {param_grids}")
    print(f"  Combos:  {total}")
    print(f"  Epochs:  {args.epochs}")
    print()

    results_file = "results/hyperparameter_search.csv"
    os.makedirs("results", exist_ok=True)

    with open(args.config) as f:
        base_config = yaml.safe_load(f)

    # The train scripts bind `load_config` when they are imported
    # (``from utils.config import load_config``), so the patch has to be in place
    # *before* the import below. Every trial then reads an in-memory dict and the
    # user's config.yaml is never written -- an interrupted search can no longer
    # leave the grid's last values behind.
    trial_config = {}
    config_module.load_config = lambda path: copy.deepcopy(trial_config)

    train_fn = resolve_target(args.model)

    with open(results_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["model"] + param_keys + ["final_loss", "final_acc", "total_time_s"])

        for combo_idx, values in enumerate(itertools.product(*param_grids), 1):
            cfg = copy.deepcopy(base_config)
            for k, v in zip(param_keys, values):
                cfg[k] = v
            cfg["num_epochs"] = args.epochs
            trial_config.clear()
            trial_config.update(cfg)

            combo_str = ", ".join(f"{k}={v}" for k, v in zip(param_keys, values))
            print(f"[{combo_idx}/{total}] {combo_str}", flush=True)

            # Run training with stdout capture.
            t0 = time.time()
            captured = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured
            try:
                train_fn()
            finally:
                sys.stdout = old_stdout

            elapsed = time.time() - t0
            output = captured.getvalue()
            print(output, end="")

            final_loss, final_acc = extract_metrics(output)
            print(f"  → Loss: {final_loss}, Acc: {final_acc}%, Time: {elapsed:.0f}s", flush=True)

            writer.writerow([args.model] + list(values) +
                            [f"{final_loss:.4f}" if final_loss else "",
                             f"{final_acc:.2f}" if final_acc else "",
                             f"{elapsed:.0f}"])
            csvfile.flush()

    print(f"\nDone. Results saved to {results_file}")
    print(f"View with: column -t -s, {results_file}")


if __name__ == "__main__":
    main()
